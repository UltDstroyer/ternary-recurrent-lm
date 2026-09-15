"""Ternary weight projection with a straight-through gradient estimator."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def make_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    weight_mode: str,
    threshold_factor: float,
    quantization_levels: int = 3,
) -> nn.Linear:
    if weight_mode == "full":
        return nn.Linear(in_features, out_features, bias=bias)
    if weight_mode == "ternary":
        return TernaryLinear(
            in_features, out_features, bias=bias, threshold_factor=threshold_factor
        )
    if weight_mode == "quantized":
        return QuantizedLinear(
            in_features,
            out_features,
            bias=bias,
            threshold_factor=threshold_factor,
            quantization_levels=quantization_levels,
        )
    raise ValueError(f"unknown weight mode: {weight_mode}")


def hard_ternary(weight: Tensor, threshold_factor: float = 0.7) -> Tensor:
    if weight.ndim < 2:
        raise ValueError("ternary projection expects at least a matrix")
    reduce_dims = tuple(range(1, weight.ndim))
    threshold = threshold_factor * weight.abs().mean(dim=reduce_dims, keepdim=True)
    active = weight.abs() > threshold
    count = active.sum(dim=reduce_dims, keepdim=True).clamp_min(1)
    alpha = (weight.abs() * active).sum(dim=reduce_dims, keepdim=True) / count
    return weight.sign() * active.to(weight.dtype) * alpha


def hard_quantize(
    weight: Tensor,
    quantization_levels: int,
    threshold_factor: float = 0.7,
) -> Tensor:
    """Project each output row onto a symmetric, uniformly spaced value alphabet."""
    if weight.ndim < 2:
        raise ValueError("quantized projection expects at least a matrix")
    if quantization_levels < 3:
        raise ValueError("quantization_levels must be at least 3")
    if quantization_levels == 3:
        return hard_ternary(weight, threshold_factor)

    reduce_dims = tuple(range(1, weight.ndim))
    scale = weight.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(
        torch.finfo(weight.dtype).eps
    )
    alphabet = torch.linspace(
        -1.0,
        1.0,
        quantization_levels,
        dtype=weight.dtype,
        device=weight.device,
    )
    assigned = torch.zeros_like(weight)
    for _ in range(2):
        normalized = (weight / scale).clamp(-1.0, 1.0)
        indices = ((normalized + 1.0) * (quantization_levels - 1) / 2.0).round().long()
        assigned = alphabet[indices]
        denominator = assigned.square().sum(dim=reduce_dims, keepdim=True).clamp_min(1.0)
        scale = (weight * assigned).sum(dim=reduce_dims, keepdim=True) / denominator
        scale = scale.abs().clamp_min(torch.finfo(weight.dtype).eps)
    return assigned * scale


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        *args: object,
        threshold_factor: float = 0.7,
        quantization_levels: int = 3,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if quantization_levels < 3:
            raise ValueError("quantization_levels must be at least 3")
        self.threshold_factor = threshold_factor
        self.quantization_levels = quantization_levels
        self.quantization_strength = 1.0

    def projected_weight(self) -> Tensor:
        return hard_quantize(
            self.weight,
            self.quantization_levels,
            self.threshold_factor,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.projected_weight()
        target = torch.lerp(self.weight, projected, self.quantization_strength)
        effective_weight = self.weight + (target - self.weight).detach()
        return F.linear(inputs, effective_weight, self.bias)

    def set_quantization_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("quantization strength must be in [0, 1]")
        self.quantization_strength = strength

    @torch.no_grad()
    def sign_fractions(self) -> dict[str, float]:
        projected = self.projected_weight()
        return {
            "negative": float((projected < 0).float().mean()),
            "zero": float((projected == 0).float().mean()),
            "positive": float((projected > 0).float().mean()),
        }


class TernaryLinear(QuantizedLinear):
    def __init__(self, *args: object, threshold_factor: float = 0.7, **kwargs: object) -> None:
        super().__init__(
            *args,
            threshold_factor=threshold_factor,
            quantization_levels=3,
            **kwargs,
        )

    def ternary_fractions(self) -> dict[str, float]:
        return self.sign_fractions()
