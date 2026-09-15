"""Configuration objects with inexpensive validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 384
    max_seq_len: int = 256
    max_loops: int = 6
    min_loops: int = 2
    memory_slots: int = 6
    dropout: float = 0.0
    ternary_threshold: float = 0.7
    halt_epsilon: float = 0.01
    weight_mode: Literal["full", "ternary", "quantized"] = "ternary"
    quantization_levels: int = 3
    tie_core: bool = True
    memory_enabled: bool = True
    adaptive_halting: bool = True
    memory_novelty_power: float = 0.5
    memory_merge_threshold: float = 0.15
    redundancy_halting: bool = True
    redundancy_halt_threshold: float = 0.15
    redundancy_patience: int = 2
    loop_embeddings: bool = False
    loop_norms: bool = False
    loop_adapter_rank: int = 0
    memory_chunks: int = 1
    stability_halting: bool = False
    stability_kl_threshold: float = 0.02
    stability_patience: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.d_model <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.d_ff <= 0 or self.max_seq_len <= 1:
            raise ValueError("d_ff must be positive and max_seq_len must exceed 1")
        if not 1 <= self.min_loops <= self.max_loops:
            raise ValueError("require 1 <= min_loops <= max_loops")
        if self.memory_slots < 1:
            raise ValueError("memory_slots must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.ternary_threshold < 0.0:
            raise ValueError("ternary_threshold must be non-negative")
        if not 0.0 < self.halt_epsilon < 1.0:
            raise ValueError("halt_epsilon must be in (0, 1)")
        if self.weight_mode not in {"full", "ternary", "quantized"}:
            raise ValueError("weight_mode must be 'full', 'ternary', or 'quantized'")
        if not 3 <= self.quantization_levels <= 256:
            raise ValueError("quantization_levels must be in [3, 256]")
        if self.weight_mode == "ternary" and self.quantization_levels != 3:
            raise ValueError("ternary weight mode requires exactly 3 quantization levels")
        if not 0.0 < self.memory_novelty_power <= 1.0:
            raise ValueError("memory_novelty_power must be in (0, 1]")
        if not 0.0 <= self.memory_merge_threshold <= 1.0:
            raise ValueError("memory_merge_threshold must be in [0, 1]")
        if not 0.0 <= self.redundancy_halt_threshold <= 1.0:
            raise ValueError("redundancy_halt_threshold must be in [0, 1]")
        if self.redundancy_patience < 1:
            raise ValueError("redundancy_patience must be positive")
        if self.loop_adapter_rank < 0:
            raise ValueError("loop_adapter_rank must be non-negative")
        if not 1 <= self.memory_chunks <= self.memory_slots:
            raise ValueError("memory_chunks must be in [1, memory_slots]")
        if self.stability_kl_threshold < 0.0:
            raise ValueError("stability_kl_threshold must be non-negative")
        if self.stability_patience < 1:
            raise ValueError("stability_patience must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    steps: int = 300
    batch_size: int = 16
    sequence_length: int = 96
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    ponder_cost: float = 1e-3
    log_every: int = 20

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size, self.sequence_length, self.log_every) < 1:
            raise ValueError("step, batch, sequence, and logging counts must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0 or self.ponder_cost < 0.0:
            raise ValueError("optimizer values and ponder_cost must be non-negative")


def load_config(path: str | Path) -> tuple[ModelConfig, TrainingConfig]:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return ModelConfig(**payload.get("model", {})), TrainingConfig(**payload.get("training", {}))


def config_dict(model: ModelConfig, training: TrainingConfig) -> dict[str, dict[str, Any]]:
    return {"model": asdict(model), "training": asdict(training)}
