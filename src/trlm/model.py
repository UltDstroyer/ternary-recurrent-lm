"""Configurable recurrent causal language model for controlled ablations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from trlm.config import ModelConfig
from trlm.memory import ScratchpadMemory
from trlm.ternary import QuantizedLinear, make_linear


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt() * self.scale


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_width = config.d_model // config.n_heads
        kwargs = {
            "bias": False,
            "weight_mode": config.weight_mode,
            "threshold_factor": config.ternary_threshold,
            "quantization_levels": config.quantization_levels,
        }
        self.qkv = make_linear(config.d_model, 3 * config.d_model, **kwargs)
        self.output = make_linear(config.d_model, config.d_model, **kwargs)
        self.dropout = config.dropout

    def forward(self, hidden: Tensor) -> Tensor:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.n_heads, self.head_width)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        mask = torch.ones(length, length, device=hidden.device, dtype=torch.bool).triu(1)
        weights = F.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attended = (weights @ value).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)


class RecurrentCore(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        kwargs = {
            "bias": False,
            "weight_mode": config.weight_mode,
            "threshold_factor": config.ternary_threshold,
            "quantization_levels": config.quantization_levels,
        }
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalAttention(config)
        self.memory_norm = RMSNorm(config.d_model)
        self.mlp_norm = RMSNorm(config.d_model)
        self.up = make_linear(config.d_model, 2 * config.d_ff, **kwargs)
        self.down = make_linear(config.d_ff, config.d_model, **kwargs)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor, memory_context: Tensor) -> Tensor:
        hidden = hidden + self.dropout(self.attention(self.attention_norm(hidden)))
        hidden = hidden + memory_context
        gate, value = self.up(self.mlp_norm(hidden)).chunk(2, dim=-1)
        return hidden + self.dropout(self.down(F.silu(gate) * value))


class LoopAdapter(nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.up(F.silu(self.down(hidden)))


@dataclass
class ModelOutput:
    logits: Tensor
    exit_weights: Tensor
    halt_probabilities: Tensor
    expected_loops: Tensor
    memory_raw_novelty: Tensor
    memory_novelty: Tensor
    memory_write_strength: Tensor
    redundancy_triggered: Tensor
    stability_triggered: Tensor
    prediction_divergence: Tensor
    loop_logits: Tensor
    loops_executed: int


class TernaryRecurrentLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.input_norm = RMSNorm(config.d_model)
        self.core = RecurrentCore(config)
        self.additional_cores = (
            nn.ModuleList(RecurrentCore(config) for _ in range(config.max_loops - 1))
            if not config.tie_core
            else nn.ModuleList()
        )
        self.memory = (
            ScratchpadMemory(
                config.d_model,
                config.memory_slots,
                config.ternary_threshold,
                config.weight_mode,
                config.quantization_levels,
                config.memory_novelty_power,
                config.memory_merge_threshold,
                config.memory_chunks,
            )
            if config.memory_enabled
            else None
        )
        self.leak_norm = RMSNorm(config.d_model)
        self.leak_projection = make_linear(
            config.d_model,
            config.d_model,
            bias=False,
            weight_mode=config.weight_mode,
            threshold_factor=config.ternary_threshold,
            quantization_levels=config.quantization_levels,
        )
        self.halt_head = nn.Linear(config.d_model, 1) if config.adaptive_halting else None
        self.loop_embedding = (
            nn.Embedding(config.max_loops, config.d_model) if config.loop_embeddings else None
        )
        self.loop_norms = (
            nn.ModuleList(RMSNorm(config.d_model) for _ in range(config.max_loops))
            if config.loop_norms
            else nn.ModuleList()
        )
        self.loop_adapters = (
            nn.ModuleList(
                LoopAdapter(config.d_model, config.loop_adapter_rank)
                for _ in range(config.max_loops)
            )
            if config.loop_adapter_rank > 0
            else nn.ModuleList()
        )
        self.output_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        if self.halt_head is not None:
            nn.init.constant_(self.halt_head.bias, -1.0)
        if self.loop_embedding is not None:
            nn.init.normal_(self.loop_embedding.weight, mean=0.0, std=0.02)

    def core_for_loop(self, loop_index: int) -> RecurrentCore:
        return (
            self.core
            if self.config.tie_core or loop_index == 0
            else self.additional_cores[loop_index - 1]
        )

    def set_ternary_strength(self, strength: float) -> None:
        for module in self.modules():
            if isinstance(module, QuantizedLinear):
                module.set_quantization_strength(strength)

    def forward(
        self,
        input_ids: Tensor,
        *,
        allow_halting: bool = True,
        use_memory: bool = True,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError(f"sequence length {length} exceeds max_seq_len")
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.input_norm(
            self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        )
        memory_values: Tensor | None = None
        occupied: Tensor | None = None
        if self.memory is not None:
            memory_values, occupied = self.memory.empty(
                batch, device=hidden.device, dtype=hidden.dtype
            )
        remaining = torch.ones(batch, device=hidden.device, dtype=hidden.dtype)
        leaked_sum = torch.zeros_like(hidden)
        exits: list[Tensor] = []
        halt_probs: list[Tensor] = []
        raw_novelties: list[Tensor] = []
        novelties: list[Tensor] = []
        strengths: list[Tensor] = []
        redundancy_triggers: list[Tensor] = []
        stability_triggers: list[Tensor] = []
        divergences: list[Tensor] = []
        loop_logits: list[Tensor] = []
        redundancy_streak = torch.zeros(batch, device=hidden.device, dtype=torch.long)
        stability_streak = torch.zeros_like(redundancy_streak)
        previous_probabilities: Tensor | None = None

        for loop_index in range(self.config.max_loops):
            if self.memory is not None and use_memory:
                assert memory_values is not None and occupied is not None
                memory_context = self.memory.read(hidden, memory_values, occupied)
            else:
                memory_context = torch.zeros_like(hidden)
            if self.loop_embedding is not None:
                hidden = hidden + self.loop_embedding.weight[loop_index][None, None]
            if self.loop_norms:
                hidden = self.loop_norms[loop_index](hidden)
            hidden = self.core_for_loop(loop_index)(hidden, memory_context)
            if self.loop_adapters:
                hidden = hidden + self.loop_adapters[loop_index](hidden)
            leaked = self.leak_projection(self.leak_norm(hidden))
            provisional = self.lm_head(self.output_norm(leaked))
            loop_logits.append(provisional)
            probabilities = F.softmax(provisional, dim=-1)
            if previous_probabilities is None:
                divergence = torch.zeros(batch, device=hidden.device, dtype=hidden.dtype)
                stable = torch.zeros(batch, device=hidden.device, dtype=torch.bool)
            else:
                divergence = (
                    F.kl_div(
                        F.log_softmax(provisional, dim=-1),
                        previous_probabilities,
                        reduction="none",
                    )
                    .sum(dim=-1)
                    .mean(dim=-1)
                )
                stable = divergence <= self.config.stability_kl_threshold
            divergences.append(divergence)
            stability_streak = torch.where(
                stable, stability_streak + 1, torch.zeros_like(stability_streak)
            )
            previous_probabilities = probabilities.detach()

            if self.memory is not None and use_memory:
                assert memory_values is not None and occupied is not None
                memory_values, occupied, raw, novelty, strength = self.memory.write(
                    hidden, memory_values, occupied, loop_index
                )
                recurrent = (loop_index > 0) & (raw <= self.config.redundancy_halt_threshold)
                redundancy_streak = torch.where(
                    recurrent,
                    redundancy_streak + 1,
                    torch.zeros_like(redundancy_streak),
                )
            else:
                raw = torch.zeros(batch, device=hidden.device, dtype=hidden.dtype)
                novelty = torch.zeros_like(raw)
                strength = torch.zeros_like(raw)
                redundancy_streak.zero_()

            can_halt = loop_index + 1 >= self.config.min_loops
            is_last = loop_index + 1 == self.config.max_loops
            adaptive = self.config.adaptive_halting and allow_halting
            if not adaptive:
                halt_probability = (
                    torch.ones_like(remaining) if is_last else torch.zeros_like(remaining)
                )
                halt_mass = remaining if is_last else torch.zeros_like(remaining)
            else:
                assert self.halt_head is not None
                halt_probability = torch.sigmoid(self.halt_head(hidden.mean(dim=1))).squeeze(-1)
                halt_mass = (
                    torch.zeros_like(remaining)
                    if not can_halt
                    else remaining
                    if is_last
                    else remaining * halt_probability
                )
            redundancy_trigger = (
                (redundancy_streak >= self.config.redundancy_patience)
                & (remaining > self.config.halt_epsilon)
                if self.config.redundancy_halting and allow_halting and can_halt
                else torch.zeros(batch, device=hidden.device, dtype=torch.bool)
            )
            stability_trigger = (
                (stability_streak >= self.config.stability_patience)
                & (remaining > self.config.halt_epsilon)
                if self.config.stability_halting and allow_halting and can_halt
                else torch.zeros(batch, device=hidden.device, dtype=torch.bool)
            )
            forced = redundancy_trigger | stability_trigger
            halt_probability = torch.where(
                forced, torch.ones_like(halt_probability), halt_probability
            )
            halt_mass = torch.where(forced, remaining, halt_mass)
            leaked_sum = leaked_sum + halt_mass[:, None, None] * leaked
            remaining = (remaining - halt_mass).clamp_min(0.0)
            exits.append(halt_mass)
            halt_probs.append(halt_probability)
            raw_novelties.append(raw)
            novelties.append(novelty)
            strengths.append(strength)
            redundancy_triggers.append(redundancy_trigger)
            stability_triggers.append(stability_trigger)
            if (
                not self.training
                and allow_halting
                and (adaptive or self.config.redundancy_halting or self.config.stability_halting)
                and can_halt
                and not is_last
                and bool(torch.all(remaining <= self.config.halt_epsilon))
            ):
                leaked_sum = leaked_sum + remaining[:, None, None] * leaked
                exits[-1] = exits[-1] + remaining
                remaining = torch.zeros_like(remaining)
                break

        exit_tensor = torch.stack(exits, dim=1)
        loop_numbers = torch.arange(
            1, exit_tensor.shape[1] + 1, device=hidden.device, dtype=hidden.dtype
        )
        expected_loops = (exit_tensor * loop_numbers[None]).sum(dim=1)
        return ModelOutput(
            logits=self.lm_head(self.output_norm(leaked_sum)),
            exit_weights=exit_tensor,
            halt_probabilities=torch.stack(halt_probs, dim=1),
            expected_loops=expected_loops,
            memory_raw_novelty=torch.stack(raw_novelties, dim=1),
            memory_novelty=torch.stack(novelties, dim=1),
            memory_write_strength=torch.stack(strengths, dim=1),
            redundancy_triggered=torch.stack(redundancy_triggers, dim=1),
            stability_triggered=torch.stack(stability_triggers, dim=1),
            prediction_divergence=torch.stack(divergences, dim=1),
            loop_logits=torch.stack(loop_logits, dim=1),
            loops_executed=exit_tensor.shape[1],
        )

    def loss(
        self,
        input_ids: Tensor,
        targets: Tensor,
        ponder_cost: float = 0.0,
        *,
        intermediate_loss_weight: float = 0.0,
        improvement_loss_weight: float = 0.0,
        teacher_logits: Tensor | None = None,
        distillation_weight: float = 0.0,
        distillation_temperature: float = 2.0,
        allow_halting: bool = True,
        use_memory: bool = True,
    ) -> tuple[Tensor, dict[str, float]]:
        output = self(input_ids, allow_halting=allow_halting, use_memory=use_memory)
        language_loss = F.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]), targets.reshape(-1)
        )
        loop_losses = torch.stack(
            [
                F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                for logits in output.loop_logits.unbind(dim=1)
            ]
        )
        weights = torch.arange(
            1, loop_losses.numel() + 1, device=loop_losses.device, dtype=loop_losses.dtype
        )
        intermediate_loss = (loop_losses * weights).sum() / weights.sum()
        improvement_loss = (
            F.relu(loop_losses[1:] - loop_losses[:-1].detach()).mean()
            if loop_losses.numel() > 1
            else torch.zeros_like(language_loss)
        )
        if teacher_logits is None or distillation_weight == 0.0:
            distillation_loss = torch.zeros_like(language_loss)
        else:
            temperature = distillation_temperature
            distillation_loss = (
                F.kl_div(
                    F.log_softmax(output.logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction="batchmean",
                )
                * (temperature**2)
                / targets.shape[1]
            )
        ponder_loss = output.expected_loops.mean()
        total = (
            language_loss
            + intermediate_loss_weight * intermediate_loss
            + improvement_loss_weight * improvement_loss
            + distillation_weight * distillation_loss
            + ponder_cost * ponder_loss
        )
        return total, {
            "loss": float(total.detach()),
            "language_loss": float(language_loss.detach()),
            "expected_loops": float(ponder_loss.detach()),
            "intermediate_loss": float(intermediate_loss.detach()),
            "improvement_loss": float(improvement_loss.detach()),
            "distillation_loss": float(distillation_loss.detach()),
            "memory_novelty": float(output.memory_novelty.mean().detach()),
            "write_strength": float(output.memory_write_strength.mean().detach()),
            "redundancy_halt_rate": float(output.redundancy_triggered.float().mean().detach()),
        }
