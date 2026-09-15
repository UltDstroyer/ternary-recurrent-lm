import torch

from trlm.ternary import QuantizedLinear, TernaryLinear, hard_quantize, hard_ternary


def test_projection_has_at_most_three_values_per_row() -> None:
    weights = torch.tensor([[0.1, -2.0, 3.0, 0.2], [1.0, -1.0, 0.0, 0.1]])
    projected = hard_ternary(weights)
    for row in projected:
        assert torch.unique(row).numel() <= 3
        assert 0.0 in row


def test_projection_preserves_shape() -> None:
    weights = torch.randn(7, 5)
    assert hard_ternary(weights).shape == weights.shape


def test_straight_through_estimator_passes_gradient() -> None:
    layer = TernaryLinear(4, 2, bias=False)
    layer(torch.ones(3, 4)).sum().backward()
    assert layer.weight.grad is not None
    assert torch.count_nonzero(layer.weight.grad) > 0


def test_quantization_strength_interpolates_from_latent_to_ternary() -> None:
    layer = TernaryLinear(3, 2, bias=False)
    inputs = torch.randn(4, 3)
    layer.set_quantization_strength(0.0)
    assert torch.allclose(layer(inputs), torch.nn.functional.linear(inputs, layer.weight))
    layer.set_quantization_strength(1.0)
    assert torch.allclose(
        layer(inputs), torch.nn.functional.linear(inputs, layer.projected_weight())
    )


def test_four_level_projection_uses_symmetric_nonzero_alphabet() -> None:
    weights = torch.tensor([[-3.0, -1.0, 1.0, 3.0]])
    projected = hard_quantize(weights, 4)
    assert torch.allclose(projected, weights)
    assert not bool((projected == 0).any())


def test_five_level_projection_includes_zero() -> None:
    weights = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    projected = hard_quantize(weights, 5)
    assert torch.allclose(projected, weights)
    assert bool((projected == 0).any())


def test_four_level_zero_groupings_include_zero() -> None:
    weights = torch.tensor([[-2.0, -0.1, 0.0, 0.5, 2.0]])
    for scheme in ("zero_positive", "zero_negative", "zero_adaptive"):
        projected = hard_quantize(weights, 4, quantization_scheme=scheme)
        assert bool((projected == 0).any()), scheme
        assert torch.unique(projected).numel() <= 4


def test_row_adaptive_zero_grouping_uses_both_orientations() -> None:
    weights = torch.tensor(
        [
            [-1.0, 0.0, 0.4, 0.8],
            [-0.8, -0.4, 0.0, 1.0],
        ]
    )
    projected = hard_quantize(weights, 4, quantization_scheme="zero_adaptive")
    assert bool((projected[0] == 0).any())
    assert bool((projected[1] == 0).any())
    assert (projected[0] > 0).sum() >= (projected[0] < 0).sum()
    assert (projected[1] < 0).sum() >= (projected[1] > 0).sum()


def test_general_quantized_linear_preserves_straight_through_gradients() -> None:
    layer = QuantizedLinear(4, 2, bias=False, quantization_levels=5)
    layer(torch.ones(3, 4)).sum().backward()
    assert layer.weight.grad is not None
