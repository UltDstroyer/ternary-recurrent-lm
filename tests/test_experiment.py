import torch

from trlm.config import ModelConfig
from trlm.experiment import (
    four_level_grouping_variants,
    improvement_variants,
    long_training_variants,
    model_config_for,
    quantization_level_variants,
    quantization_strength_for,
    size_matched_variants,
    storage_metrics,
    validation_windows,
    variants_for_suite,
)
from trlm.model import TernaryRecurrentLM


def test_validation_windows_are_next_byte_aligned() -> None:
    inputs, targets = validation_windows(torch.arange(20), length=4, stride=4)
    assert torch.equal(inputs[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(targets[0], torch.tensor([1, 2, 3, 4]))


def test_improvement_matrix_isolates_mechanisms_and_combines_winners() -> None:
    matrix = {variant.name: variant for variant in improvement_variants()}
    assert len(matrix) == 13
    assert matrix["b_loop_conditioning"].loop_embeddings
    assert matrix["c_low_rank_adapters"].loop_adapter_rank == 8
    assert matrix["d_chunk_memory"].memory_chunks == 4
    assert matrix["e_progressive_supervision"].intermediate_loss_weight > 0
    assert matrix["g_gradual_ternarization"].ternary_ramp_fraction > 0
    assert matrix["h_teacher_distillation"].distillation_weight > 0
    assert matrix["l_quality_combo_stability_halt"].stability_halting


def test_gradual_ternarization_has_warmup_ramp_and_projected_end() -> None:
    variant = next(v for v in improvement_variants() if v.name == "g_gradual_ternarization")
    assert quantization_strength_for(0.0, variant) == 0.0
    assert quantization_strength_for(variant.ternary_warmup_fraction, variant) == 0.0
    assert 0.0 < quantization_strength_for(0.5, variant) < 1.0
    assert quantization_strength_for(1.0, variant) == 1.0


def test_long_training_matrix_contains_controls_and_prior_winners() -> None:
    variants = long_training_variants()
    matrix = {variant.name: variant for variant in variants}
    assert list(matrix) == [
        "full_untied_fixed",
        "full_tied_fixed",
        "a_memory_v2_fixed",
        "g_gradual_ternarization",
        "k_quality_combo",
        "l_quality_combo_stability_halt",
    ]
    assert matrix["full_untied_fixed"].weight_mode == "full"
    assert not matrix["full_untied_fixed"].tie_core
    assert matrix["full_tied_fixed"].weight_mode == "full"
    assert matrix["full_tied_fixed"].tie_core
    assert matrix["g_gradual_ternarization"].ternary_ramp_fraction > 0
    assert matrix["l_quality_combo_stability_halt"].stability_halting


def test_suite_selection_rejects_unknown_names() -> None:
    assert len(variants_for_suite("improvements")) == 13
    assert len(variants_for_suite("long_training")) == 6
    assert len(variants_for_suite("size_matched")) == 5
    assert len(variants_for_suite("quantization_levels")) == 3
    assert len(variants_for_suite("four_level_groupings")) == 6
    try:
        variants_for_suite("missing")
    except ValueError as error:
        assert str(error) == "unknown experiment suite: missing"
    else:
        raise AssertionError("unknown suite should fail")


def test_size_matched_models_use_the_untied_storage_budget() -> None:
    base = ModelConfig(
        d_model=64,
        n_heads=4,
        d_ff=160,
        max_seq_len=64,
        memory_slots=12,
    )
    variants = size_matched_variants()
    sizes = {}
    for variant in variants:
        model = TernaryRecurrentLM(model_config_for(base, variant))
        sizes[variant.name] = storage_metrics(model)["estimated_packed_inference_bytes"]

    target = sizes["full_untied_fixed"]
    assert target == 1_234_176
    for name, size in sizes.items():
        assert abs(size - target) / target < 0.001, (name, size, target)


def test_quantization_level_suite_changes_only_weight_alphabet() -> None:
    variants = quantization_level_variants()
    assert [variant.quantization_levels for variant in variants] == [3, 4, 5]
    assert all(
        (variant.d_model, variant.n_heads, variant.d_ff) == (352, 8, 1888)
        for variant in variants
    )


def test_four_level_grouping_suite_changes_only_grouping() -> None:
    variants = four_level_grouping_variants()
    assert len(variants) == 6
    assert {variant.quantization_scheme for variant in variants} == {
        "uniform",
        "symmetric_narrow",
        "symmetric_wide",
        "zero_positive",
        "zero_negative",
        "zero_adaptive",
    }
    assert all(variant.quantization_levels == 4 for variant in variants)
    assert all(
        (variant.d_model, variant.n_heads, variant.d_ff) == (352, 8, 1888)
        for variant in variants
    )
