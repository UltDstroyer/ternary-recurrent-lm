from dataclasses import replace

import pytest

from trlm.config import ModelConfig, TrainingConfig


def test_default_configs_are_valid() -> None:
    ModelConfig()
    TrainingConfig()


def test_width_must_be_divisible_by_heads() -> None:
    with pytest.raises(ValueError):
        ModelConfig(d_model=10, n_heads=4)


def test_loop_bounds_are_validated() -> None:
    with pytest.raises(ValueError):
        ModelConfig(min_loops=7, max_loops=6)


def test_memory_chunks_fit_in_slots() -> None:
    with pytest.raises(ValueError):
        ModelConfig(memory_slots=2, memory_chunks=3)


def test_new_specialization_fields_can_be_enabled() -> None:
    config = replace(ModelConfig(), loop_embeddings=True, loop_norms=True, loop_adapter_rank=8)
    assert config.loop_embeddings and config.loop_norms and config.loop_adapter_rank == 8


def test_quantization_level_validation() -> None:
    with pytest.raises(ValueError):
        ModelConfig(weight_mode="quantized", quantization_levels=2)
    with pytest.raises(ValueError):
        ModelConfig(weight_mode="ternary", quantization_levels=4)
    assert ModelConfig(weight_mode="quantized", quantization_levels=5).quantization_levels == 5
