from dataclasses import replace

import torch

from trlm.config import ModelConfig
from trlm.model import TernaryRecurrentLM


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        d_model=16,
        n_heads=4,
        d_ff=32,
        max_seq_len=12,
        max_loops=4,
        min_loops=2,
        memory_slots=4,
    )


def test_forward_shapes_and_halting_mass() -> None:
    output = TernaryRecurrentLM(tiny_config())(torch.randint(0, 32, (3, 10)))
    assert output.logits.shape == (3, 10, 32)
    assert output.exit_weights.shape == (3, 4)
    assert torch.allclose(output.exit_weights.sum(dim=1), torch.ones(3), atol=1e-5)


def test_loss_backpropagates_through_reused_core() -> None:
    model = TernaryRecurrentLM(tiny_config())
    inputs = torch.randint(0, 32, (2, 8))
    loss, metrics = model.loss(inputs, torch.randint(0, 32, (2, 8)), 1e-3)
    loss.backward()
    assert model.core.up.weight.grad is not None
    assert metrics["expected_loops"] >= 2.0


def test_fixed_depth_places_exit_mass_on_last_loop() -> None:
    config = replace(
        tiny_config(),
        max_loops=3,
        min_loops=1,
        weight_mode="full",
        tie_core=False,
        memory_enabled=False,
        adaptive_halting=False,
        redundancy_halting=False,
    )
    model = TernaryRecurrentLM(config)
    output = model(torch.randint(0, 32, (2, 8)))
    assert torch.allclose(output.exit_weights[:, :-1], torch.zeros(2, 2))
    assert torch.allclose(output.exit_weights[:, -1], torch.ones(2))
    assert len(model.additional_cores) == 2


def test_weight_tying_reduces_parameters() -> None:
    shared = TernaryRecurrentLM(tiny_config())
    untied = TernaryRecurrentLM(replace(tiny_config(), tie_core=False))
    assert sum(p.numel() for p in shared.parameters()) < sum(p.numel() for p in untied.parameters())


def test_loop_specialization_emits_prediction_at_every_depth() -> None:
    config = replace(tiny_config(), loop_embeddings=True, loop_norms=True, loop_adapter_rank=4)
    model = TernaryRecurrentLM(config)
    output = model(torch.randint(0, 32, (2, 8)))
    assert output.loop_logits.shape == (2, 4, 8, 32)
    assert len(model.loop_norms) == len(model.loop_adapters) == 4


def test_progressive_and_distillation_losses_are_active() -> None:
    model = TernaryRecurrentLM(tiny_config())
    inputs = torch.randint(0, 32, (2, 8))
    targets = torch.randint(0, 32, (2, 8))
    loss, metrics = model.loss(
        inputs,
        targets,
        intermediate_loss_weight=0.2,
        improvement_loss_weight=0.1,
        teacher_logits=torch.randn(2, 8, 32),
        distillation_weight=0.5,
    )
    loss.backward()
    assert metrics["intermediate_loss"] > 0.0
    assert metrics["distillation_loss"] > 0.0


def test_prediction_stability_can_stop_evaluation_early() -> None:
    config = replace(
        tiny_config(),
        adaptive_halting=False,
        redundancy_halting=False,
        stability_halting=True,
        stability_kl_threshold=1e6,
        stability_patience=1,
    )
    output = TernaryRecurrentLM(config).eval()(torch.randint(0, 32, (2, 8)))
    assert output.loops_executed == 2
    assert output.stability_triggered[:, 1].all()
    assert torch.allclose(output.expected_loops, torch.full((2,), 2.0))
