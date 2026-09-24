"""Three-model next-token comparison sourced from the repository's four-level core.

Every target is the token *after* the complete supplied prefix. In particular,
sequence-wide scratchpad summaries cannot see any scored target token.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from trlm.config import ModelConfig
from trlm.model import RMSNorm, RecurrentCore, TernaryRecurrentLM
from trlm.ternary import QuantizedLinear


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


@dataclass
class BranchStats:
    splits: int = 0
    merges: int = 0
    core_calls: int = 0
    peak_threads: int = 1
    threads_by_pass: list[int] = field(default_factory=list)


class SerialLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.model = TernaryRecurrentLM(config)

    def forward(self, x: Tensor) -> tuple[Tensor, BranchStats]:
        output = self.model(x, allow_halting=False)
        return output.logits[:, -1], BranchStats(
            core_calls=output.loops_executed,
            threads_by_pass=[1] * output.loops_executed,
        )

    def set_quantization_strength(self, strength: float) -> None:
        self.model.set_ternary_strength(strength)


class BranchLM(nn.Module):
    """One quantized core, bounded parallel ideas, and one shared scratchpad.

    Two new ideas can spawn each pass from the most novel parents. All live
    ideas read one memory snapshot before their writes are committed together
    for the next pass. Similar ideas can merge after two passes of development.
    Split and merge controls never inspect the target token.
    """
    def __init__(
        self,
        config: ModelConfig,
        max_threads: int = 8,
        children_per_pass: int = 2,
        split_threshold: float = 0.15,
        merge_threshold: float = 0.995,
    ) -> None:
        super().__init__()
        if max_threads < 1 or children_per_pass < 0:
            raise ValueError("max_threads must be positive and children_per_pass nonnegative")
        self.base = TernaryRecurrentLM(config)
        self.config = config
        self.max_threads = max_threads
        self.children_per_pass = children_per_pass
        self.split_threshold = split_threshold
        self.merge_threshold = merge_threshold
        self.idea_offsets = nn.Parameter(torch.randn(max_threads - 1, config.d_model) * 0.15)
        self.idea_gate = nn.Linear(config.d_model, 1, bias=False)

    def set_quantization_strength(self, strength: float) -> None:
        self.base.set_ternary_strength(strength)

    def forward(self, x: Tensor) -> tuple[Tensor, BranchStats]:
        base = self.base
        memory = base.memory
        assert memory is not None
        batch, length = x.shape
        position = torch.arange(length, device=x.device)
        hidden = base.input_norm(
            base.token_embedding(x) + base.position_embedding(position)[None]
        )
        values, occupied = memory.empty(batch, device=hidden.device, dtype=hidden.dtype)
        stats = BranchStats(threads_by_pass=[1])
        hidden = base.core(hidden, torch.zeros_like(hidden))
        stats.core_calls += 1
        values, occupied, _, novelty, _ = memory.write(hidden, values, occupied, 0)
        branches = [hidden]
        branch_novelties = [novelty]
        branch_ages = [1]
        spawn_index = 0
        for loop_index in range(1, self.config.max_loops):
            candidates = sorted(
                range(len(branches)),
                key=lambda i: float(branch_novelties[i].mean().detach()),
                reverse=True,
            )
            candidates = [
                i for i in candidates
                if bool(branch_novelties[i].mean().detach() > self.split_threshold)
            ]
            to_spawn = min(self.children_per_pass, self.max_threads - len(branches))
            for child_index in range(to_spawn):
                if not candidates:
                    break
                parent = candidates[child_index % len(candidates)]
                offset = self.idea_offsets[spawn_index % len(self.idea_offsets)][None, None]
                branches.append(branches[parent] + offset)
                branch_novelties.append(branch_novelties[parent])
                branch_ages.append(0)
                spawn_index += 1
                stats.splits += 1
            stats.peak_threads = max(stats.peak_threads, len(branches))
            threads = len(branches)
            stats.threads_by_pass.append(threads)
            stacked = torch.stack(branches, dim=1).flatten(0, 1)
            shared_values = values[:, None].expand(-1, threads, -1, -1).reshape(
                batch * threads, memory.slots, self.config.d_model
            )
            shared_occupied = occupied[:, None].expand(-1, threads, -1).reshape(
                batch * threads, memory.slots
            )
            context = memory.read(stacked, shared_values, shared_occupied)
            parallel = base.core(stacked, context)
            advanced = parallel.reshape(batch, threads, length, -1).unbind(dim=1)
            stats.core_calls += threads
            current_novelties = []
            for thread_index, state in enumerate(advanced):
                values, occupied, _, current_novelty, _ = memory.write(
                    state, values, occupied,
                    loop_index * self.max_threads + thread_index,
                )
                current_novelties.append(current_novelty)
            branches, branch_novelties, new_ages = [], [], []
            for state, current_novelty, age in zip(
                advanced, current_novelties, branch_ages
            ):
                age += 1
                nearest, best_similarity = None, -1.0
                if loop_index >= 3 and age >= 2:
                    for candidate_index, candidate in enumerate(branches):
                        if new_ages[candidate_index] < 2:
                            continue
                        similarity = float(F.cosine_similarity(
                            state[:, -1].detach(), candidate[:, -1].detach(), dim=-1
                        ).mean())
                        if similarity > best_similarity:
                            nearest, best_similarity = candidate_index, similarity
                if nearest is not None and best_similarity > self.merge_threshold:
                    branches[nearest] = (branches[nearest] + state) / 2
                    branch_novelties[nearest] = (
                        branch_novelties[nearest] + current_novelty
                    ) / 2
                    new_ages[nearest] = max(new_ages[nearest], age)
                    stats.merges += 1
                else:
                    branches.append(state)
                    branch_novelties.append(current_novelty)
                    new_ages.append(age)
            branch_ages = new_ages
        leaked = torch.stack([
            base.leak_projection(base.leak_norm(state))[:, -1]
            for state in branches
        ], dim=1)
        weights = self.idea_gate(leaked).softmax(dim=1)
        combined = (weights * leaked).sum(dim=1)
        return base.lm_head(base.output_norm(combined)), stats


class UntiedTransformer(nn.Module):
    """Six independent, full-precision causal Transformer blocks, no scratchpad."""
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.input_norm = RMSNorm(config.d_model)
        self.blocks = nn.ModuleList(RecurrentCore(config) for _ in range(config.max_loops))
        self.output_norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, BranchStats]:
        positions = torch.arange(x.shape[1], device=x.device)
        hidden = self.input_norm(
            self.token_embedding(x) + self.position_embedding(positions)[None]
        )
        zero = torch.zeros_like(hidden)
        for block in self.blocks:
            hidden = block(hidden, zero)
        return self.head(self.output_norm(hidden[:, -1])), BranchStats(
            core_calls=len(self.blocks), threads_by_pass=[1] * len(self.blocks)
        )


def synthetic_data(seed: int, count: int, length: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(2, 16, (count, length), generator=generator)
    marker = torch.randint(0, 2, (count,), generator=generator)
    x[:, 0] = marker
    y = torch.where(marker.bool(), x[:, length // 2], x[:, 2]).clone()
    return x, y


def wikitext_data(path: Path, count: int, length: int, seed: int) -> tuple[Tensor, Tensor]:
    raw = path.read_bytes()
    if len(raw) < length + 1:
        raise ValueError(f"corpus is too short: {path}")
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(raw) - length, (count,), generator=generator)
    x = torch.tensor(
        [list(raw[int(start):int(start) + length]) for start in starts], dtype=torch.long
    )
    y = torch.tensor([raw[int(start) + length] for start in starts], dtype=torch.long)
    return x, y


@torch.no_grad()
def evaluate(model: nn.Module, x: Tensor, y: Tensor, device: torch.device, batch: int) -> dict:
    model.eval()
    total_loss = total_correct = total = 0
    stats = BranchStats()
    threads_by_pass: list[float] = []
    for first in range(0, len(x), batch):
        logits, current = model(x[first:first + batch].to(device))
        targets = y[first:first + batch].to(device)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        total_correct += (logits.argmax(-1) == targets).sum().item()
        total += len(targets)
        stats.splits += current.splits * len(targets)
        stats.merges += current.merges * len(targets)
        stats.core_calls += current.core_calls * len(targets)
        stats.peak_threads = max(stats.peak_threads, current.peak_threads)
        if not threads_by_pass:
            threads_by_pass = [0.0] * len(current.threads_by_pass)
        for pass_index, threads in enumerate(current.threads_by_pass):
            threads_by_pass[pass_index] += threads * len(targets)
    return {
        "cross_entropy": total_loss / total,
        "accuracy": total_correct / total,
        "splits_per_example": stats.splits / total,
        "merges_per_example": stats.merges / total,
        "core_calls_per_example": stats.core_calls / total,
        "peak_threads": stats.peak_threads,
        "mean_threads_by_pass": [threads / total for threads in threads_by_pass],
        "scored_targets": total,
    }


@torch.no_grad()
def inference_ms(model: nn.Module, x: Tensor, device: torch.device, batch: int) -> float:
    model.eval()
    inputs = x[:batch].to(device)
    for _ in range(2):
        model(inputs)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(5):
        model(inputs)
    synchronize(device)
    return (time.perf_counter() - start) * 1000 / 5


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "cpu":
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    width = args.width if args.width is not None else (352 if args.full_size else 64)
    loops = args.passes if args.passes is not None else (6 if args.full_size else 3)
    length = args.prefix_length if args.prefix_length is not None else (32 if args.full_size else 12)
    ff = args.feed_forward if args.feed_forward is not None else (1888 if args.full_size else 192)
    batch = args.batch_size if args.batch_size is not None else (8 if args.full_size else 2)
    steps = args.steps if args.steps is not None else (200 if args.full_size else 2)
    train_count = args.train_count if args.train_count is not None else (4096 if args.full_size else 64)
    valid_count = args.validation_count if args.validation_count is not None else (512 if args.full_size else 24)
    common = dict(
        vocab_size=256 if args.dataset == "wikitext2" else 16,
        d_model=width, n_heads=8 if width % 8 == 0 else 4, d_ff=ff,
        max_seq_len=length, max_loops=loops, min_loops=loops,
        memory_slots=12, adaptive_halting=False, redundancy_halting=False,
        dropout=0.0, memory_enabled=True,
    )
    recurrent_config = ModelConfig(
        **common, tie_core=True, weight_mode="quantized",
        quantization_levels=4, quantization_scheme="uniform"
    )
    full_config = ModelConfig(
        **{**common, "memory_enabled": False},
        tie_core=False, weight_mode="full"
    )
    if args.dataset == "wikitext2":
        root = Path("data/external/wikitext-2")
        train_x, train_y = wikitext_data(root / "train.txt", train_count, length, args.seed)
        val_x, val_y = wikitext_data(root / "valid.txt", valid_count, length, args.seed + 1)
    else:
        train_x, train_y = synthetic_data(args.seed, train_count, length)
        val_x, val_y = synthetic_data(args.seed + 1, valid_count, length)
    generator = torch.Generator().manual_seed(args.seed + 2)
    batches = [torch.randint(0, train_count, (batch,), generator=generator)
               for _ in range(steps)]
    models = {
        "serial": lambda: SerialLM(recurrent_config),
        "branches": lambda: BranchLM(
            recurrent_config, max_threads=args.max_threads,
            children_per_pass=args.children_per_pass,
            split_threshold=args.split_threshold,
            merge_threshold=args.merge_threshold,
        ),
        "untied_full_transformer": lambda: UntiedTransformer(full_config),
    }
    results = {}
    for name, make_model in models.items():
        torch.manual_seed(args.seed)
        model = make_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
        model.train()
        synchronize(device)
        start = time.perf_counter()
        for step, index in enumerate(batches):
            if hasattr(model, "set_quantization_strength"):
                progress = step / max(steps - 1, 1)
                strength = max(0.0, min(1.0, (progress - 0.15) / 0.7))
                model.set_quantization_strength(strength)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(train_x[index].to(device))
            loss = F.cross_entropy(logits, train_y[index].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        synchronize(device)
        training_seconds = time.perf_counter() - start
        if hasattr(model, "set_quantization_strength"):
            model.set_quantization_strength(1.0)
        valid = evaluate(model, val_x, val_y, device, batch)
        inference = inference_ms(model, val_x, device, batch)
        quantized = sum(m.weight.numel() for m in model.modules()
                        if isinstance(m, QuantizedLinear))
        results[name] = {
            "parameters": parameters(model),
            "quantized_weight_parameters": quantized,
            "validation": valid,
            "training_seconds": training_seconds,
            "inference_ms_per_batch": inference,
            "inference_batch_size": batch,
        }
        print(name, json.dumps(results[name]), flush=True)
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": str(device),
            "cpu_threads": torch.get_num_threads() if device.type == "cpu" else None,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "revision": revision,
        },
        "settings": {
            "dataset": args.dataset, "full_size": args.full_size, "width": width,
            "feed_forward": ff, "passes": loops, "prefix_length": length,
            "steps": steps, "batch_size": batch, "train_examples": train_count,
            "validation_examples": valid_count, "seed": args.seed,
            "max_threads": args.max_threads,
            "children_per_pass": args.children_per_pass,
            "split_threshold": args.split_threshold,
            "merge_threshold": args.merge_threshold,
            "target_position": "immediately after entire prefix",
            "quantization_schedule": "warmup 0.15, ramp 0.70, fully quantized validation",
        },
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--full-size", action="store_true")
    parser.add_argument("--passes", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--feed-forward", type=int)
    parser.add_argument("--prefix-length", type=int)
    parser.add_argument("--dataset", choices=["synthetic", "wikitext2"], default="synthetic")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--validation-count", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-threads", type=int, default=8)
    parser.add_argument("--children-per-pass", type=int, default=2)
    parser.add_argument("--split-threshold", type=float, default=0.15)
    parser.add_argument("--merge-threshold", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, default=Path("results/four_level_branch_probe.json"))
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
