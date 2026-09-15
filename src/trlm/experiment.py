"""Run controlled multi-seed architecture and training ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from trlm.config import ModelConfig, TrainingConfig, load_config
from trlm.model import TernaryRecurrentLM
from trlm.ternary import QuantizedLinear
from trlm.train import sample_batch


@dataclass(frozen=True)
class Variant:
    name: str
    weight_mode: str
    tie_core: bool
    memory_enabled: bool
    adaptive_halting: bool
    ponder_cost: float = 0.0
    memory_novelty_power: float = 0.5
    memory_merge_threshold: float = 0.15
    redundancy_halting: bool = False
    redundancy_halt_threshold: float = 0.15
    redundancy_patience: int = 2
    loop_embeddings: bool = False
    loop_norms: bool = False
    loop_adapter_rank: int = 0
    memory_chunks: int = 1
    intermediate_loss_weight: float = 0.0
    improvement_loss_weight: float = 0.0
    memory_start_fraction: float = 0.0
    ternary_warmup_fraction: float = 0.0
    ternary_ramp_fraction: float = 0.0
    distillation_weight: float = 0.0
    distillation_temperature: float = 2.0
    stability_halting: bool = False
    stability_kl_threshold: float = 0.02
    stability_patience: int = 2
    halting_start_fraction: float = 0.0
    d_model: int | None = None
    n_heads: int | None = None
    d_ff: int | None = None
    quantization_levels: int = 3


def improvement_variants() -> list[Variant]:
    untied = Variant("full_untied_fixed", "full", False, False, False)
    current = Variant("a_memory_v2_fixed", "ternary", True, True, False)
    conditioned = replace(
        current, name="b_loop_conditioning", loop_embeddings=True, loop_norms=True
    )
    adapted = replace(current, name="c_low_rank_adapters", loop_adapter_rank=8)
    chunked = replace(current, name="d_chunk_memory", memory_chunks=4)
    progressive = replace(
        current,
        name="e_progressive_supervision",
        intermediate_loss_weight=0.1,
        improvement_loss_weight=0.05,
    )
    staged = replace(current, name="f_staged_memory", memory_start_fraction=0.2)
    gradual = replace(
        current,
        name="g_gradual_ternarization",
        ternary_warmup_fraction=0.15,
        ternary_ramp_fraction=0.7,
    )
    distilled = replace(
        current,
        name="h_teacher_distillation",
        distillation_weight=0.25,
        distillation_temperature=2.0,
    )
    stable = replace(
        current,
        name="i_prediction_stability_halt",
        stability_halting=True,
        stability_kl_threshold=0.02,
        stability_patience=2,
        halting_start_fraction=0.75,
    )
    adaptive = replace(
        stable,
        name="j_adaptive_stability_halt",
        adaptive_halting=True,
        ponder_cost=0.01,
    )
    quality_combo = replace(
        current,
        name="k_quality_combo",
        loop_adapter_rank=8,
        intermediate_loss_weight=0.1,
        improvement_loss_weight=0.05,
        ternary_warmup_fraction=0.15,
        ternary_ramp_fraction=0.7,
    )
    combo_halt = replace(
        quality_combo,
        name="l_quality_combo_stability_halt",
        stability_halting=True,
        stability_kl_threshold=0.02,
        stability_patience=2,
        halting_start_fraction=0.75,
    )
    return [
        untied,
        current,
        conditioned,
        adapted,
        chunked,
        progressive,
        staged,
        gradual,
        distilled,
        stable,
        adaptive,
        quality_combo,
        combo_halt,
    ]


def long_training_variants() -> list[Variant]:
    """Return the focused controls and winners for the million-token comparison."""
    improvements = {variant.name: variant for variant in improvement_variants()}
    full_tied = Variant("full_tied_fixed", "full", True, False, False)
    return [
        improvements["full_untied_fixed"],
        full_tied,
        improvements["a_memory_v2_fixed"],
        improvements["g_gradual_ternarization"],
        improvements["k_quality_combo"],
        improvements["l_quality_combo_stability_halt"],
    ]


def size_matched_variants() -> list[Variant]:
    """Allocate the untied control's packed-byte budget to compact models."""
    untied = improvement_variants()[0]
    gradual = Variant(
        "m_size_matched_gradual",
        "ternary",
        True,
        True,
        False,
        ternary_warmup_fraction=0.15,
        ternary_ramp_fraction=0.7,
        d_model=352,
        n_heads=8,
        d_ff=1888,
    )

    def enhanced(name: str, width: int, feed_forward: int, rank: int) -> Variant:
        return Variant(
            name,
            "ternary",
            True,
            True,
            False,
            loop_adapter_rank=rank,
            intermediate_loss_weight=0.1,
            improvement_loss_weight=0.05,
            ternary_warmup_fraction=0.15,
            ternary_ramp_fraction=0.7,
            d_model=width,
            n_heads=8,
            d_ff=feed_forward,
        )

    return [
        untied,
        gradual,
        enhanced("n_size_matched_quality_r8", 352, 1392, 8),
        enhanced("o_size_matched_quality_r16", 336, 1152, 16),
        enhanced("p_size_matched_quality_r32", 288, 1072, 32),
    ]


def quantization_level_variants() -> list[Variant]:
    """Compare 3-, 4-, and 5-level weights in the winning wide architecture."""
    winner = size_matched_variants()[1]
    return [
        replace(
            winner,
            name=f"q{levels}_level_shared_capacity",
            weight_mode="quantized",
            quantization_levels=levels,
        )
        for levels in (3, 4, 5)
    ]


def variants_for_suite(name: str) -> list[Variant]:
    if name == "improvements":
        return improvement_variants()
    if name == "long_training":
        return long_training_variants()
    if name == "size_matched":
        return size_matched_variants()
    if name == "quantization_levels":
        return quantization_level_variants()
    raise ValueError(f"unknown experiment suite: {name}")


def model_config_for(base: ModelConfig, variant: Variant) -> ModelConfig:
    return replace(
        base,
        d_model=variant.d_model if variant.d_model is not None else base.d_model,
        n_heads=variant.n_heads if variant.n_heads is not None else base.n_heads,
        d_ff=variant.d_ff if variant.d_ff is not None else base.d_ff,
        weight_mode=variant.weight_mode,
        quantization_levels=variant.quantization_levels,
        tie_core=variant.tie_core,
        memory_enabled=variant.memory_enabled,
        adaptive_halting=variant.adaptive_halting,
        memory_novelty_power=variant.memory_novelty_power,
        memory_merge_threshold=variant.memory_merge_threshold,
        redundancy_halting=variant.redundancy_halting,
        redundancy_halt_threshold=variant.redundancy_halt_threshold,
        redundancy_patience=variant.redundancy_patience,
        loop_embeddings=variant.loop_embeddings,
        loop_norms=variant.loop_norms,
        loop_adapter_rank=variant.loop_adapter_rank,
        memory_chunks=variant.memory_chunks,
        stability_halting=variant.stability_halting,
        stability_kl_threshold=variant.stability_kl_threshold,
        stability_patience=variant.stability_patience,
    )


def quantization_strength_for(progress: float, variant: Variant) -> float:
    if variant.ternary_ramp_fraction <= 0.0:
        return 1.0
    return max(
        0.0,
        min(
            1.0,
            (progress - variant.ternary_warmup_fraction) / variant.ternary_ramp_fraction,
        ),
    )


def load_corpus(path: Path, sequence_length: int) -> Tensor:
    raw = path.read_bytes().strip()
    if len(raw) <= sequence_length:
        raise ValueError(f"corpus split needs more than {sequence_length} bytes")
    return torch.tensor(list(raw), dtype=torch.long)


def validation_windows(tokens: Tensor, length: int, stride: int) -> tuple[Tensor, Tensor]:
    starts = range(0, tokens.numel() - length - 1, stride)
    inputs = torch.stack([tokens[start : start + length] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + length + 1] for start in starts])
    return inputs, targets


@torch.no_grad()
def evaluate(
    model: TernaryRecurrentLM,
    inputs: Tensor,
    targets: Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    expected_loops: list[Tensor] = []
    width = model.config.max_loops
    exits = torch.zeros(width)
    raw_novelty = torch.zeros(width)
    novelty = torch.zeros(width)
    writes = torch.zeros(width)
    redundancy = torch.zeros(width)
    stability = torch.zeros(width)
    divergence = torch.zeros(width)
    examples = 0
    maximum_executed = 0
    started = time.perf_counter()
    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].to(device)
        batch_targets = targets[start : start + batch_size].to(device)
        output = model(batch_inputs)
        count = batch_targets.numel()
        total_loss += float(
            F.cross_entropy(output.logits.flatten(0, 1), batch_targets.flatten(), reduction="sum")
        )
        total_correct += int((output.logits.argmax(dim=-1) == batch_targets).sum())
        total_tokens += count
        current_examples = batch_inputs.shape[0]
        examples += current_examples
        expected_loops.append(output.expected_loops.cpu())
        executed = output.loops_executed
        maximum_executed = max(maximum_executed, executed)
        exits[:executed] += output.exit_weights.sum(dim=0).cpu()
        raw_novelty[:executed] += output.memory_raw_novelty.sum(dim=0).cpu()
        novelty[:executed] += output.memory_novelty.sum(dim=0).cpu()
        writes[:executed] += output.memory_write_strength.sum(dim=0).cpu()
        redundancy[:executed] += output.redundancy_triggered.sum(dim=0).cpu()
        stability[:executed] += output.stability_triggered.sum(dim=0).cpu()
        divergence[:executed] += output.prediction_divergence.sum(dim=0).cpu()
    elapsed = time.perf_counter() - started
    mean_loss = total_loss / total_tokens
    loop_tensor = torch.cat(expected_loops)
    return {
        "cross_entropy": mean_loss,
        "perplexity": math.exp(mean_loss),
        "next_byte_accuracy": total_correct / total_tokens,
        "expected_loops": float(loop_tensor.mean()),
        "expected_loops_std": float(loop_tensor.std(unbiased=False)),
        "actual_loops": maximum_executed,
        "tokens_per_second": total_tokens / elapsed,
        "exit_mass_by_loop": (exits / examples).tolist(),
        "memory_raw_novelty_by_loop": (raw_novelty / examples).tolist(),
        "memory_novelty_by_loop": (novelty / examples).tolist(),
        "memory_write_by_loop": (writes / examples).tolist(),
        "redundancy_halt_by_loop": (redundancy / examples).tolist(),
        "stability_halt_by_loop": (stability / examples).tolist(),
        "prediction_divergence_by_loop": (divergence / examples).tolist(),
    }


def storage_metrics(model: TernaryRecurrentLM) -> dict[str, Any]:
    quantized_weights = {
        id(module.weight): module
        for module in model.modules()
        if isinstance(module, QuantizedLinear)
    }
    full_bytes = 0
    packed_bytes = 0.0
    counts = {"negative": 0, "zero": 0, "positive": 0}
    for parameter in model.parameters():
        full_bytes += parameter.numel() * parameter.element_size()
        layer = quantized_weights.get(id(parameter))
        if layer is None:
            packed_bytes += parameter.numel() * parameter.element_size()
        else:
            bits = math.ceil(math.log2(layer.quantization_levels))
            packed_bytes += parameter.numel() * bits / 8 + layer.out_features * 4
            projected = layer.projected_weight().detach()
            counts["negative"] += int((projected < 0).sum())
            counts["zero"] += int((projected == 0).sum())
            counts["positive"] += int((projected > 0).sum())
    total = sum(counts.values())
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "full_precision_bytes": full_bytes,
        "estimated_packed_inference_bytes": math.ceil(packed_bytes),
        "quantization_levels": sorted(
            {layer.quantization_levels for layer in quantized_weights.values()}
        ),
        "bits_per_quantized_weight": sorted(
            {math.ceil(math.log2(layer.quantization_levels)) for layer in quantized_weights.values()}
        ),
        "ternary_fractions": {key: value / total for key, value in counts.items()}
        if total
        else None,
    }


def train_variant(
    variant: Variant,
    base_model: ModelConfig,
    training: TrainingConfig,
    train_tokens: Tensor,
    validation_inputs: Tensor,
    validation_targets: Tensor,
    validation_batch_size: int,
    device: torch.device,
    teacher_model: TernaryRecurrentLM | None = None,
) -> tuple[dict[str, Any], TernaryRecurrentLM]:
    torch.manual_seed(training.seed)
    generator = random.Random(training.seed)
    model = TernaryRecurrentLM(model_config_for(base_model, variant)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    initial = evaluate(model, validation_inputs, validation_targets, validation_batch_size, device)
    model.train()
    started = time.perf_counter()
    metrics: dict[str, float] = {}
    for step in range(1, training.steps + 1):
        progress = (step - 1) / max(training.steps - 1, 1)
        model.set_ternary_strength(quantization_strength_for(progress, variant))
        inputs, targets = sample_batch(
            train_tokens,
            training.batch_size,
            training.sequence_length,
            generator,
            device,
        )
        teacher_logits = None
        if variant.distillation_weight > 0.0:
            if teacher_model is None:
                raise ValueError("distillation variant requires a trained teacher")
            teacher_model.eval()
            with torch.no_grad():
                teacher_logits = teacher_model(inputs).logits
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(
            inputs,
            targets,
            variant.ponder_cost,
            intermediate_loss_weight=variant.intermediate_loss_weight,
            improvement_loss_weight=variant.improvement_loss_weight,
            teacher_logits=teacher_logits,
            distillation_weight=variant.distillation_weight,
            distillation_temperature=variant.distillation_temperature,
            allow_halting=progress >= variant.halting_start_fraction,
            use_memory=progress >= variant.memory_start_fraction,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % training.log_every == 0:
            print(
                f"{variant.name} step={step:04d} "
                f"language_loss={metrics['language_loss']:.4f} "
                f"loops={metrics['expected_loops']:.3f}",
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    model.set_ternary_strength(1.0)
    validation = evaluate(
        model, validation_inputs, validation_targets, validation_batch_size, device
    )
    return {
        "seed": training.seed,
        "variant": asdict(variant),
        "model": asdict(model.config),
        "storage": storage_metrics(model),
        "initial_validation": initial,
        "final_training_batch": metrics,
        "validation": validation,
        "training_seconds": training_seconds,
        "training_tokens": training.steps * training.batch_size * training.sequence_length,
    }, model


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "seed",
        "variant",
        "parameters",
        "packed_bytes",
        "validation_cross_entropy",
        "validation_perplexity",
        "validation_accuracy",
        "expected_loops",
        "actual_loops",
        "tokens_per_second",
        "stability_halt_rate",
        "training_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            validation = result["validation"]
            writer.writerow(
                {
                    "seed": result["seed"],
                    "variant": result["variant"]["name"],
                    "parameters": result["storage"]["parameters"],
                    "packed_bytes": result["storage"]["estimated_packed_inference_bytes"],
                    "validation_cross_entropy": validation["cross_entropy"],
                    "validation_perplexity": validation["perplexity"],
                    "validation_accuracy": validation["next_byte_accuracy"],
                    "expected_loops": validation["expected_loops"],
                    "actual_loops": validation["actual_loops"],
                    "tokens_per_second": validation["tokens_per_second"],
                    "stability_halt_rate": sum(validation["stability_halt_by_loop"]),
                    "training_seconds": result["training_seconds"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/improvements.json"))
    parser.add_argument("--output", type=Path, default=Path("results/improvements.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    base_model, training = load_config(args.config)
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    experiment = payload["experiment"]
    torch.set_num_threads(int(experiment["torch_threads"]))
    device = torch.device(args.device)
    train_path = Path(experiment["train_corpus"])
    validation_path = Path(experiment["validation_corpus"])
    train_tokens = load_corpus(train_path, training.sequence_length)
    validation_tokens = load_corpus(validation_path, training.sequence_length)
    validation_inputs, validation_targets = validation_windows(
        validation_tokens, training.sequence_length, int(experiment["validation_stride"])
    )
    selected = variants_for_suite(str(experiment.get("suite", "improvements")))
    if args.variants:
        requested = set(args.variants)
        selected = [variant for variant in selected if variant.name in requested]
        missing = requested - {variant.name for variant in selected}
        if missing:
            raise ValueError(f"unknown variants: {', '.join(sorted(missing))}")
    seeds = args.seeds or [int(value) for value in experiment["seeds"]]
    print(
        f"device={device} train_bytes={train_tokens.numel()} "
        f"validation_bytes={validation_tokens.numel()} "
        f"validation_windows={validation_inputs.shape[0]} runs={len(selected) * len(seeds)}"
    )
    results: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        results = previous["results"]
    completed = {
        (int(result["seed"]), str(result["variant"]["name"])) for result in results
    }

    def checkpoint() -> None:
        document = {
            "corpus": {"train": str(train_path), "validation": str(validation_path)},
            "corpus_bytes": train_tokens.numel() + validation_tokens.numel(),
            "train_bytes": train_tokens.numel(),
            "validation_bytes": validation_tokens.numel(),
            "validation_windows": validation_inputs.shape[0],
            "configuration": payload,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        write_csv(args.output.with_suffix(".csv"), results)

    for seed in seeds:
        teacher: TernaryRecurrentLM | None = None
        for variant in selected:
            key = (seed, variant.name)
            needs_teacher = variant.name == "full_untied_fixed" and any(
                candidate.distillation_weight > 0.0
                and (seed, candidate.name) not in completed
                for candidate in selected
            )
            if key in completed and not needs_teacher:
                print(f"skipping completed seed={seed} variant={variant.name}", flush=True)
                continue
            result, trained = train_variant(
                variant,
                base_model,
                replace(training, seed=seed),
                train_tokens,
                validation_inputs,
                validation_targets,
                int(experiment["validation_batch_size"]),
                device,
                teacher,
            )
            if variant.name == "full_untied_fixed":
                teacher = trained
            if key not in completed:
                results.append(result)
                completed.add(key)
                checkpoint()
                print(
                    f"checkpointed {len(results)}/{len(selected) * len(seeds)} runs",
                    flush=True,
                )
    checkpoint()
    print(f"wrote {args.output} and {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
