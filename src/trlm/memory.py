"""A sequence-local scratchpad that suppresses redundant loop summaries."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from trlm.ternary import make_linear


class ScratchpadMemory(nn.Module):
    def __init__(
        self,
        d_model: int,
        slots: int,
        threshold_factor: float = 0.7,
        weight_mode: str = "ternary",
        quantization_levels: int = 3,
        quantization_scheme: str = "uniform",
        novelty_power: float = 0.5,
        merge_threshold: float = 0.15,
        chunks: int = 1,
    ) -> None:
        super().__init__()
        self.slots = slots
        self.novelty_power = novelty_power
        self.merge_threshold = merge_threshold
        self.chunks = chunks
        linear = lambda output, bias=False: make_linear(
            d_model,
            output,
            bias=bias,
            weight_mode=weight_mode,
            threshold_factor=threshold_factor,
            quantization_levels=quantization_levels,
            quantization_scheme=quantization_scheme,
        )
        self.read_query = linear(d_model)
        self.read_output = linear(d_model)
        self.write_value = linear(d_model)
        self.write_gate = linear(1, bias=True)

    def empty(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        values = torch.zeros(
            batch_size,
            self.slots,
            self.read_query.in_features,
            device=device,
            dtype=dtype,
        )
        occupied = torch.zeros(batch_size, self.slots, device=device, dtype=torch.bool)
        return values, occupied

    def read(self, hidden: Tensor, values: Tensor, occupied: Tensor) -> Tensor:
        query = F.normalize(self.read_query(hidden), dim=-1)
        keys = F.normalize(values, dim=-1)
        scores = torch.einsum("btd,bsd->bts", query, keys)
        scores = scores.masked_fill(~occupied[:, None, :], -1e4)
        weights = scores.softmax(dim=-1) * occupied[:, None, :].to(hidden.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context = torch.einsum("bts,bsd->btd", weights, values)
        return self.read_output(context)

    def write(
        self,
        hidden: Tensor,
        values: Tensor,
        occupied: Tensor,
        loop_index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        summaries = torch.stack(
            [chunk.mean(dim=1) for chunk in torch.tensor_split(hidden, self.chunks, dim=1)],
            dim=1,
        )
        raw_novelties: list[Tensor] = []
        novelties: list[Tensor] = []
        strengths: list[Tensor] = []
        for chunk_index in range(self.chunks):
            values, occupied, raw, novelty, strength = self._write_summary(
                summaries[:, chunk_index],
                values,
                occupied,
                loop_index * self.chunks + chunk_index,
            )
            raw_novelties.append(raw)
            novelties.append(novelty)
            strengths.append(strength)
        return (
            values,
            occupied,
            torch.stack(raw_novelties, dim=1).mean(dim=1),
            torch.stack(novelties, dim=1).mean(dim=1),
            torch.stack(strengths, dim=1).mean(dim=1),
        )

    def _write_summary(
        self,
        pooled: Tensor,
        values: Tensor,
        occupied: Tensor,
        write_index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        candidate = torch.tanh(self.write_value(pooled))
        similarities = torch.einsum(
            "bd,bsd->bs", F.normalize(candidate, dim=-1), F.normalize(values, dim=-1)
        )
        similarities = similarities.masked_fill(~occupied, -1.0)
        had_memory = occupied.any(dim=-1)
        maximum = similarities.max(dim=-1).values.clamp(min=0.0, max=1.0)
        raw_novelty = torch.where(had_memory, 1.0 - maximum, torch.ones_like(maximum))
        novelty = raw_novelty.pow(self.novelty_power)
        strength = torch.sigmoid(self.write_gate(pooled)).squeeze(-1) * novelty
        round_robin = torch.full(
            (pooled.shape[0],),
            write_index % self.slots,
            device=pooled.device,
            dtype=torch.long,
        )
        closest = similarities.argmax(dim=-1)
        target = torch.where(
            had_memory & (raw_novelty <= self.merge_threshold), closest, round_robin
        )
        selector = F.one_hot(target, num_classes=self.slots).to(pooled.dtype)
        blend = strength[:, None, None] * selector[:, :, None]
        updated_values = values * (1.0 - blend) + candidate[:, None, :] * blend
        return updated_values, occupied | selector.bool(), raw_novelty, novelty, strength
