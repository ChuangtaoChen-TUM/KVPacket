"""Reconstruct packetized student inputs and optimize their wrapper parameters.

Training samples contain full-attention teacher trajectories and source spans but
no wrapper assignment. ``PacketTrainer`` resolves those spans through its registry,
replaces each source body with a wrapped embedding sequence, and recreates the
packet-isolated attention graph used by serving.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from .._internal.persistence import atomic_torch_save
from .._internal.validation import string_mapping
from ..backend import TrainingBackend
from ..errors import BackendCompatibilityError, WrapperFormatError
from ..fingerprint import canonical_json_bytes
from ..packet.registry import WrapperRegistry
from ..session.sample import TrainingSample
from .mask import batch_attention_masks, packet_attention_mask

__all__ = ["TRAINER_CHECKPOINT_FORMAT", "PacketTrainer", "TrainerMetrics"]

TRAINER_CHECKPOINT_FORMAT = "kvpacket.trainer_checkpoint/1"
OptimizeSelection = Literal["all", "none"] | Sequence[str]
LossReduction = Literal["token_mean", "sample_mean", "legacy_sum"]
TrainingObjective = Literal["teacher_kl", "teacher_ce"]


@dataclass(frozen=True)
class TrainerMetrics:
    """Token-normalized result of one student forward or optimizer step.

    ``loss`` remains token-normalized for stable reporting even when the trainer
    uses ``legacy_sum`` for backward compatibility with the reference trainer.

    Attributes:
        loss: Reported objective averaged according to the configured reduction.
        num_tokens: Number of generated target tokens evaluated in the batch.
        wrapper_ids: Sorted wrapper IDs referenced by the evaluated samples.
    """

    loss: float
    num_tokens: int
    wrapper_ids: tuple[str, ...]


def _json_mapping(value: Mapping[str, object] | None, *, where: str) -> dict[str, object]:
    """Deep-copy a mapping after verifying checkpoint-safe JSON compatibility."""
    copied = deepcopy(dict(value or {}))
    try:
        canonical_json_bytes(copied)
    except ValueError as error:
        raise ValueError(f"{where} must be JSON-compatible: {error}") from error
    return copied


def _adamw_config(value: Mapping[str, object] | None) -> dict[str, object]:
    """Normalize the supported AdamW options into a serializable configuration."""
    raw = _json_mapping(value or {"lr": 1e-3}, where="optimizer_kwargs")
    allowed = {"lr", "betas", "eps", "weight_decay", "amsgrad", "maximize"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unsupported AdamW options: {sorted(unknown)}")

    def number(name: str, default: float) -> float:
        item = raw.get(name, default)
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"AdamW {name} must be a number")
        return float(item)

    raw_betas = raw.get("betas", [0.9, 0.999])
    if not (
        isinstance(raw_betas, (list, tuple))
        and len(raw_betas) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in raw_betas)
    ):
        raise ValueError("AdamW betas must contain two numbers")
    amsgrad = raw.get("amsgrad", False)
    maximize = raw.get("maximize", False)
    if not isinstance(amsgrad, bool) or not isinstance(maximize, bool):
        raise ValueError("AdamW amsgrad and maximize must be booleans")
    return {
        "lr": number("lr", 1e-3),
        "betas": [
            float(cast(int | float, raw_betas[0])),
            float(cast(int | float, raw_betas[1])),
        ],
        "eps": number("eps", 1e-8),
        "weight_decay": number("weight_decay", 1e-2),
        "amsgrad": amsgrad,
        "maximize": maximize,
    }


class PacketTrainer:
    """Train selected wrappers against full-attention teacher outputs.

    ``objective="teacher_kl"`` minimizes full-distribution KL and requires
    recorded teacher logits. ``objective="teacher_ce"`` minimizes cross-entropy
    against the teacher-generated token IDs and accepts compact samples without logits.

    ``optimize`` accepts ``"all"``, ``"none"`` (evaluation), or an explicit
    wrapper-ID allowlist.  Every trainable wrapper owns an independent AdamW
    optimizer, which makes optimizer state stable and addressable by wrapper ID.

    Args:
        backend: Frozen-model backend used for embeddings and differentiable student
            forwards.
        registry: Named wrappers compatible with the backend model.
        default_wrapper_id: Fallback wrapper for chunks without an explicit route.
        wrapper_assignments: Per-chunk overrides from recorded source IDs to wrapper
            IDs.
        optimize: Wrappers to update. ``"none"`` evaluates without creating
            optimizers.
        objective: ``"teacher_kl"`` for recorded distributions or ``"teacher_ce"``
            for teacher-generated token IDs.
        loss_reduction: Statistical unit used to normalize gradients. See
            ``docs/guides/training.md`` for the exact reductions.
        optimizer_kwargs: Supported AdamW keyword arguments.
        provenance: JSON-compatible user metadata saved in checkpoints.

    Raises:
        TypeError: If the backend or optimize allowlist has an invalid type.
        ValueError: If routing, objective, reduction, optimizer, or provenance
            configuration is invalid.
        WrapperFormatError: If routing or optimization references an unknown wrapper.
        BackendCompatibilityError: If the registry targets another backend model.
    """

    def __init__(
        self,
        backend: TrainingBackend,
        registry: WrapperRegistry,
        *,
        default_wrapper_id: str | None = None,
        wrapper_assignments: Mapping[str, str] | None = None,
        optimize: OptimizeSelection = "all",
        objective: TrainingObjective = "teacher_kl",
        loss_reduction: LossReduction = "token_mean",
        optimizer_kwargs: Mapping[str, object] | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(backend, TrainingBackend):
            raise TypeError("backend does not implement the TrainingBackend capability")
        registry.validate_compatibility(backend.compatibility)
        self.backend = backend
        self.registry = registry

        assignments: dict[str, str] = dict(wrapper_assignments or {})
        if not all(isinstance(key, str) and key for key in assignments):
            raise ValueError("wrapper assignment chunk IDs must be non-empty strings")
        unknown_routes = set(assignments.values()) - set(registry.ids())
        if default_wrapper_id is not None and default_wrapper_id not in registry:
            unknown_routes.add(default_wrapper_id)
        if unknown_routes:
            raise WrapperFormatError(f"unknown routed wrappers: {sorted(unknown_routes)}")
        if default_wrapper_id is None and not assignments:
            raise ValueError("provide default_wrapper_id or wrapper_assignments")
        self.default_wrapper_id = default_wrapper_id
        self.wrapper_assignments = assignments

        if objective not in ("teacher_kl", "teacher_ce"):
            raise ValueError("objective must be 'teacher_kl' or 'teacher_ce'")
        self.objective: TrainingObjective = objective
        if loss_reduction not in ("token_mean", "sample_mean", "legacy_sum"):
            raise ValueError("loss_reduction must be 'token_mean', 'sample_mean', or 'legacy_sum'")
        self.loss_reduction: LossReduction = loss_reduction
        self.optimizer_kwargs = _adamw_config(optimizer_kwargs)
        self.user_provenance = _json_mapping(provenance, where="provenance")
        self.step_count = 0

        if optimize in ("all", "none"):
            selected = registry.ids() if optimize == "all" else []
            self.optimize_mode = cast(str, optimize)
        else:
            selected = list(optimize)
            if not all(isinstance(item, str) for item in selected):
                raise TypeError("optimize allowlist must contain wrapper IDs")
            if len(set(selected)) != len(selected):
                raise ValueError("optimize allowlist contains duplicate wrapper IDs")
            self.optimize_mode = "allowlist"
        unknown = set(selected) - set(registry.ids())
        if unknown:
            raise WrapperFormatError(f"unknown optimized wrappers: {sorted(unknown)}")
        self.optimized_wrapper_ids = tuple(sorted(selected))

        selected_set = set(self.optimized_wrapper_ids)
        for wrapper_id in registry.ids():
            wrapper = registry[wrapper_id].wrapper.to(device=backend.device)
            for parameter in wrapper.parameters():
                parameter.requires_grad_(wrapper_id in selected_set)
        backend.prepare_for_training()
        betas = cast(list[float], self.optimizer_kwargs["betas"])
        self.optimizers: dict[str, torch.optim.Optimizer] = {
            wrapper_id: torch.optim.AdamW(
                registry[wrapper_id].wrapper.parameters(),
                lr=cast(float, self.optimizer_kwargs["lr"]),
                betas=(betas[0], betas[1]),
                eps=cast(float, self.optimizer_kwargs["eps"]),
                weight_decay=cast(float, self.optimizer_kwargs["weight_decay"]),
                amsgrad=cast(bool, self.optimizer_kwargs["amsgrad"]),
                maximize=cast(bool, self.optimizer_kwargs["maximize"]),
            )
            for wrapper_id in self.optimized_wrapper_ids
        }

    def _wrapper_id(self, chunk_id: str) -> str:
        """Resolve one recorded chunk through trainer-owned routing."""
        wrapper_id = self.wrapper_assignments.get(chunk_id, self.default_wrapper_id)
        if wrapper_id is None:
            raise WrapperFormatError(f"chunk {chunk_id!r} has no wrapper assignment")
        return wrapper_id

    def _validate_sample(self, sample: TrainingSample) -> tuple[str, ...]:
        """Validate teacher targets and resolve all wrappers used by a sample.

        Returns:
            Sorted wrapper IDs referenced by the sample's chunk spans.

        Raises:
            ValueError: If targets are absent, generated positions are invalid, or
                reusable spans overlap.
            WrapperFormatError: If a chunk cannot be routed to a wrapper.
        """
        if self.objective == "teacher_kl" and sample.logits is None:
            raise ValueError("KL training requires teacher logits in every TrainingSample")
        if sample.num_generated == 0:
            raise ValueError("training sample has no generated tokens")
        if bool(sample.generated_mask[0]):
            raise ValueError("the first trajectory token cannot be a generated target")

        wrapper_ids: set[str] = set()
        occupied: list[tuple[int, int, str]] = []
        for chunk_id, occurrences in sample.chunk_spans.items():
            wrapper_ids.add(self._wrapper_id(chunk_id))
            for start, end in occurrences:
                if bool(sample.generated_mask[start:end].any()):
                    raise ValueError(f"chunk {chunk_id!r} overlaps generated tokens")
                occupied.append((start, end, chunk_id))
        occupied.sort()
        for left, right in zip(occupied, occupied[1:]):
            if right[0] < left[1]:
                raise ValueError(
                    f"chunk spans overlap: {left[2]!r} {left[:2]} and {right[2]!r} {right[:2]}"
                )
        return tuple(sorted(wrapper_ids))

    def _student_inputs(self, sample: TrainingSample) -> tuple[Tensor, Tensor, Tensor]:
        """Rebuild the embeddings and attention graph consumed by the student.

        The final trajectory token is a prediction target and is not forwarded.
        Every registered source span is replaced with its wrapper's
        header/body/trailer sequence. Generated targets are then mapped back to the
        preceding student positions whose logits predict them.

        Args:
            sample: Full-attention trajectory with generated targets and source spans.

        Returns:
            Wrapped input embeddings, a 4-D boolean attention mask, and positions
            whose logits predict the generated targets.

        Raises:
            ValueError: If a source span reaches beyond the forwarded trajectory or
                no student input can be formed.
        """
        device = self.backend.device
        input_ids = sample.token_ids[:-1].to(device).unsqueeze(0)
        with torch.no_grad():
            base = self.backend.embed(input_ids).detach()

        occurrences: list[tuple[int, int, str]] = []
        for chunk_id, spans in sample.chunk_spans.items():
            occurrences.extend((start, end, chunk_id) for start, end in spans)
        occurrences.sort()

        pieces: list[Tensor] = []
        packet_spans: list[tuple[int, int]] = []
        original_to_student: dict[int, int] = {}
        original_cursor = 0
        student_cursor = 0
        for start, end, chunk_id in occurrences:
            # The final trajectory token is a target only and is not forwarded.
            if end > input_ids.shape[1]:
                raise ValueError(f"chunk {chunk_id!r} reaches beyond the student input")
            if start > original_cursor:
                plain = base[:, original_cursor:start, :]
                pieces.append(plain)
                for position in range(original_cursor, start):
                    original_to_student[position] = student_cursor + position - original_cursor
                student_cursor += int(plain.shape[1])
            wrapper_id = self._wrapper_id(chunk_id)
            wrapped = self.registry[wrapper_id].wrapper(base[:, start:end, :])
            pieces.append(wrapped)
            packet_start = student_cursor
            body_start = packet_start + self.registry[wrapper_id].wrapper.header_len
            for position in range(start, end):
                original_to_student[position] = body_start + position - start
            student_cursor += int(wrapped.shape[1])
            packet_spans.append((packet_start, student_cursor))
            original_cursor = end
        if original_cursor < input_ids.shape[1]:
            plain = base[:, original_cursor:, :]
            pieces.append(plain)
            for position in range(original_cursor, int(input_ids.shape[1])):
                original_to_student[position] = student_cursor + position - original_cursor
            student_cursor += int(plain.shape[1])
        if not pieces:
            raise ValueError("training sample is too short to form a student input")

        embeds = torch.cat(pieces, dim=1)
        mask = packet_attention_mask(
            student_cursor,
            packet_spans,
            sliding_window=self.backend.sliding_window,
            device=device,
        )

        target_positions = sample.generated_mask.nonzero(as_tuple=False).flatten().tolist()
        logit_positions = torch.tensor(
            [original_to_student[int(position) - 1] for position in target_positions],
            dtype=torch.long,
            device=device,
        )
        return embeds, mask, logit_positions

    def _batched_loss_sums(
        self, samples: Sequence[TrainingSample]
    ) -> list[tuple[Tensor, tuple[str, ...]]]:
        """Evaluate a padded forward and retain one differentiable sum per sample.

        Losses are computed in FP32 even when the model forward uses a lower
        precision. Keeping them as separate sums allows ``step_batch`` to apply its
        reduction over the complete optimizer batch rather than independently per
        forward microbatch.

        Args:
            samples: Non-empty forward microbatch.

        Returns:
            Per-sample objective sums paired with their resolved wrapper IDs.

        Raises:
            ValueError: If samples, backend logits, teacher targets, or vocabulary
                indices are invalid.
        """
        if not samples:
            raise ValueError("forward microbatch must not be empty")
        prepared: list[tuple[Tensor, Tensor, Tensor, tuple[str, ...]]] = []
        for sample in samples:
            wrapper_ids = self._validate_sample(sample)
            embeds, attention_mask, positions = self._student_inputs(sample)
            prepared.append((embeds, attention_mask, positions, wrapper_ids))

        max_length = max(int(item[0].shape[1]) for item in prepared)
        padded_items: list[Tensor] = []
        for embeds, _attention_mask, _positions, _wrapper_ids in prepared:
            length = int(embeds.shape[1])
            padded_items.append(F.pad(embeds, (0, 0, 0, max_length - length)))
        padded_embeds = torch.cat(padded_items, dim=0)
        batched_mask = batch_attention_masks([item[1] for item in prepared])

        logits = self.backend.training_forward(
            inputs_embeds=padded_embeds, attention_mask=batched_mask
        )
        if logits.dim() != 3 or logits.shape[:2] != padded_embeds.shape[:2]:
            raise ValueError("training backend returned malformed logits")
        results: list[tuple[Tensor, tuple[str, ...]]] = []
        for index, (embeds, _attention_mask, positions, wrapper_ids) in enumerate(prepared):
            student = logits[index].index_select(0, positions.to(logits.device))
            sample = samples[index]
            # Models may run in BF16, but full-vocabulary objectives are
            # numerically sensitive. Compute the loss in FP32 while allowing
            # autograd to cast the gradient back through the model.
            if self.objective == "teacher_kl":
                teacher = sample.logits
                assert teacher is not None
                teacher = teacher.to(device=student.device).detach()
                if teacher.shape != student.shape:
                    raise ValueError(
                        "teacher/student logits shape mismatch: "
                        f"{tuple(teacher.shape)} vs {tuple(student.shape)}"
                    )
                loss_sum = F.kl_div(
                    F.log_softmax(student.float(), dim=-1),
                    F.log_softmax(teacher.float(), dim=-1),
                    reduction="sum",
                    log_target=True,
                )
            else:
                teacher_tokens = sample.token_ids[sample.generated_mask].to(student.device)
                if teacher_tokens.shape != (student.shape[0],):
                    raise ValueError("teacher token IDs do not align with student logits")
                invalid = (teacher_tokens < 0) | (teacher_tokens >= student.shape[-1])
                if bool(invalid.any()):
                    raise ValueError("teacher token ID is outside the student vocabulary")
                loss_sum = F.cross_entropy(
                    student.float(),
                    teacher_tokens,
                    reduction="sum",
                )
            results.append((loss_sum, wrapper_ids))
        return results

    def _loss(self, sample: TrainingSample) -> tuple[Tensor, tuple[str, ...]]:
        """Apply the configured single-sample reduction to one objective sum."""
        loss, wrapper_ids = self._batched_loss_sums([sample])[0]
        if self.loss_reduction in ("token_mean", "sample_mean"):
            loss = loss / sample.num_generated
        return loss, wrapper_ids

    def evaluate(self, sample: TrainingSample) -> TrainerMetrics:
        """Evaluate one recorded trajectory without updating wrapper parameters.

        Args:
            sample: Teacher trajectory to reconstruct and score.

        Returns:
            Token-normalized loss and the wrappers referenced by the sample.
        """
        with torch.no_grad():
            loss, wrapper_ids = self._loss(sample)
        reported_loss = float(loss)
        if self.loss_reduction == "legacy_sum":
            reported_loss /= sample.num_generated
        return TrainerMetrics(reported_loss, sample.num_generated, wrapper_ids)

    def step(self, sample: TrainingSample) -> TrainerMetrics:
        """Run one optimizer step using a single teacher trajectory.

        This is the single-sample convenience form of ``step_batch``.

        Args:
            sample: Teacher trajectory to reconstruct and optimize against.

        Returns:
            Loss and wrapper-use metrics for the step.
        """
        return self.step_batch([sample])

    def step_batch(
        self,
        samples: Sequence[TrainingSample],
        *,
        forward_batch_size: int = 1,
        sample_weights: Sequence[float] | None = None,
    ) -> TrainerMetrics:
        """Accumulate an optimizer batch, optionally using padded 4-D forwards.

        ``forward_batch_size=1`` is the memory-efficient default. Larger values
        right-pad several wrapped sequences, combine their packet masks as
        ``[B, 1, L, L]``, and run one differentiable backend forward per microbatch.
        The objective is normalized over the complete optimizer batch, never per
        forward microbatch. Samples containing only frozen wrappers are reported but
        do not dilute the gradient denominator. ``step_count`` advances only when at
        least one optimized wrapper is used.

        Args:
            samples: Non-empty optimizer batch of teacher trajectories.
            forward_batch_size: Maximum trajectories combined in each padded model
                forward.
            sample_weights: Optional positive weights for token-mean training.

        Returns:
            Reported batch loss, generated-token count, and all referenced wrappers.

        Raises:
            ValueError: If the batch, microbatch size, weights, or samples are invalid.
        """
        if not samples:
            raise ValueError("training batch must not be empty")
        if forward_batch_size <= 0:
            raise ValueError("forward_batch_size must be positive")
        if sample_weights is None:
            weights = [1.0] * len(samples)
        else:
            weights = [float(weight) for weight in sample_weights]
            if len(weights) != len(samples):
                raise ValueError("sample_weights must match samples")
            if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
                raise ValueError("sample_weights must be finite and positive")
            if self.loss_reduction != "token_mean":
                raise ValueError("sample_weights currently require token_mean loss reduction")

        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        num_tokens = sum(sample.num_generated for sample in samples)
        if num_tokens == 0:
            raise ValueError("training batch has no generated tokens")
        token_loss_sum = 0.0
        sample_mean_sum = 0.0
        used: set[str] = set()
        all_wrapper_ids: set[str] = set()
        optimized = set(self.optimized_wrapper_ids)
        active = [bool(set(self._validate_sample(sample)) & optimized) for sample in samples]
        # Frozen-only samples remain visible in metrics, but including them in this
        # denominator would shrink gradients without contributing trainable paths.
        active_tokens = sum(
            sample.num_generated * weight
            for sample, weight, is_active in zip(
                samples,
                weights,
                active,
                strict=True,
            )
            if is_active
        )
        active_samples = sum(active)
        for offset in range(0, len(samples), forward_batch_size):
            microbatch = samples[offset : offset + forward_batch_size]
            losses = self._batched_loss_sums(microbatch)
            backward_terms: list[Tensor] = []
            microbatch_uses_optimized = False
            for local_index, (sample, (loss_sum, wrapper_ids)) in enumerate(
                zip(microbatch, losses)
            ):
                sample_is_active = active[offset + local_index]
                sample_weight = weights[offset + local_index]
                detached_sum = float(loss_sum.detach())
                token_loss_sum += detached_sum * sample_weight
                sample_mean_sum += detached_sum / sample.num_generated
                sample_used = set(wrapper_ids) & optimized
                used.update(sample_used)
                all_wrapper_ids.update(wrapper_ids)
                microbatch_uses_optimized |= bool(sample_used)
                if self.loss_reduction == "token_mean":
                    backward_terms.append(
                        loss_sum * sample_weight / active_tokens
                        if sample_is_active
                        else loss_sum * 0
                    )
                elif self.loss_reduction == "sample_mean":
                    backward_terms.append(
                        loss_sum / (sample.num_generated * active_samples)
                        if sample_is_active
                        else loss_sum * 0
                    )
                else:
                    backward_terms.append(loss_sum if sample_is_active else loss_sum * 0)
            if microbatch_uses_optimized:
                torch.stack(backward_terms).sum().backward()
        for wrapper_id, optimizer in self.optimizers.items():
            if wrapper_id in used:
                optimizer.step()
        if used:
            self.step_count += 1
        reported_loss = (
            sample_mean_sum / len(samples)
            if self.loss_reduction == "sample_mean"
            else token_loss_sum
            / sum(
                sample.num_generated * weight
                for sample, weight in zip(samples, weights, strict=True)
            )
        )
        return TrainerMetrics(reported_loss, num_tokens, tuple(sorted(all_wrapper_ids)))

    def fit(
        self,
        samples: Iterable[TrainingSample],
        *,
        epochs: int = 1,
        batch_size: int = 1,
        target_tokens_per_step: int | None = None,
        forward_batch_size: int = 1,
        drop_last: bool = False,
    ) -> list[TrainerMetrics]:
        """Train over a re-iterable sample source and return optimizer-step metrics.

        ``token_mean`` and ``sample_mean`` may set ``target_tokens_per_step``;
        sample count then varies until the token target is reached, with
        ``batch_size`` acting as the maximum number of samples in one optimizer
        step. ``sample_mean`` still gives every trajectory in that optimizer batch
        equal weight regardless of its generated length.

        Args:
            samples: Re-iterable teacher-sample source. A one-shot iterator supports
                only one epoch.
            epochs: Number of complete passes over the sample source.
            batch_size: Maximum samples in one optimizer step.
            target_tokens_per_step: Optional approximate active generated-token budget.
            forward_batch_size: Maximum samples evaluated by one padded model forward.
            drop_last: Whether to discard an incomplete final optimizer batch.

        Returns:
            Metrics in optimizer-step order across all epochs.

        Raises:
            ValueError: If sizes are invalid, token budgeting is incompatible with the
                reduction, or a one-shot iterator is requested for multiple epochs.
        """
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if forward_batch_size <= 0:
            raise ValueError("forward_batch_size must be positive")
        if target_tokens_per_step is not None:
            if target_tokens_per_step <= 0:
                raise ValueError("target_tokens_per_step must be positive")
            if self.loss_reduction == "legacy_sum":
                raise ValueError("target_tokens_per_step is not valid with legacy_sum")
        if epochs > 1 and iter(samples) is samples:
            raise ValueError("epochs > 1 requires a re-iterable sample registry")
        metrics: list[TrainerMetrics] = []
        for _ in range(epochs):
            batch: list[TrainingSample] = []
            batch_tokens = 0
            optimized = set(self.optimized_wrapper_ids)
            for sample in samples:
                batch.append(sample)
                sample_is_active = bool(set(self._validate_sample(sample)) & optimized)
                if sample_is_active:
                    batch_tokens += sample.num_generated
                reached_target = (
                    target_tokens_per_step is not None and batch_tokens >= target_tokens_per_step
                )
                if reached_target or len(batch) == batch_size:
                    metrics.append(self.step_batch(batch, forward_batch_size=forward_batch_size))
                    batch = []
                    batch_tokens = 0
            if batch and not drop_last:
                metrics.append(self.step_batch(batch, forward_batch_size=forward_batch_size))
        return metrics

    def state_dict(self) -> dict[str, object]:
        """Serialize trainer configuration, progress, routing, and optimizer states.

        Wrapper tensors and compatibility metadata are owned by the registry and are
        added by ``save`` rather than duplicated here.
        """
        return {
            "step_count": self.step_count,
            "objective": self.objective,
            "loss_reduction": self.loss_reduction,
            "optimize_mode": self.optimize_mode,
            "optimized_wrapper_ids": list(self.optimized_wrapper_ids),
            "default_wrapper_id": self.default_wrapper_id,
            "wrapper_assignments": dict(self.wrapper_assignments),
            "optimizer_type": "AdamW",
            "optimizer_kwargs": deepcopy(self.optimizer_kwargs),
            "optimizers": {
                wrapper_id: optimizer.state_dict()
                for wrapper_id, optimizer in self.optimizers.items()
            },
            "backend_compatibility_fingerprint": self.backend.compatibility.fingerprint,
            "provenance": deepcopy(self.user_provenance),
        }

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically save the full registry, trainer state, and provenance.

        Args:
            path: Destination checkpoint path. Parent directories are created by the
                persistence helper when needed.
        """
        atomic_torch_save(
            {
                "format": TRAINER_CHECKPOINT_FORMAT,
                "registry": self.registry.state_dict(),
                "trainer": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        backend: TrainingBackend,
        device: torch.device | str | None = None,
    ) -> "PacketTrainer":
        """Restore a complete trainer checkpoint and ID-keyed optimizer states.

        Registry tensors are moved to ``device`` when supplied, otherwise to the
        backend device. Checkpoints created before objective metadata existed are
        interpreted as ``teacher_kl`` checkpoints.

        Args:
            path: Trainer checkpoint created by ``save``.
            backend: Training backend for the restored registry and future forwards.
            device: Optional destination for restored wrapper tensors.

        Returns:
            Trainer with registry tensors, routing, optimizer moments, and step count
            restored.

        Raises:
            ValueError: If the checkpoint schema or trainer state is invalid.
            BackendCompatibilityError: If the checkpoint targets another model.
        """
        raw = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint = string_mapping(raw, where="trainer checkpoint")
        expected = {"format", "registry", "trainer"}
        if set(checkpoint) != expected or checkpoint["format"] != TRAINER_CHECKPOINT_FORMAT:
            raise ValueError("invalid or unsupported trainer checkpoint")
        registry = WrapperRegistry.from_state_dict(
            checkpoint["registry"], device=device or backend.device, requires_grad=True
        )
        registry.validate_compatibility(backend.compatibility)
        state = string_mapping(checkpoint["trainer"], where="trainer state")
        required = {
            "step_count",
            "objective",
            "loss_reduction",
            "optimize_mode",
            "optimized_wrapper_ids",
            "default_wrapper_id",
            "wrapper_assignments",
            "optimizer_type",
            "optimizer_kwargs",
            "optimizers",
            "backend_compatibility_fingerprint",
            "provenance",
        }
        compatible_keys = (required, required - {"objective"})
        if set(state) not in compatible_keys or state["optimizer_type"] != "AdamW":
            raise ValueError("invalid or unsupported trainer state")
        # Checkpoints written before the compact CE objective existed are KL checkpoints.
        objective = state.get("objective", "teacher_kl")
        if objective not in ("teacher_kl", "teacher_ce"):
            raise ValueError("trainer checkpoint has an invalid objective")
        loss_reduction = state["loss_reduction"]
        if loss_reduction not in ("token_mean", "sample_mean", "legacy_sum"):
            raise ValueError("trainer checkpoint has an invalid loss reduction")
        if state["backend_compatibility_fingerprint"] != backend.compatibility.fingerprint:
            raise BackendCompatibilityError("trainer checkpoint targets a different backend")
        optimized = state["optimized_wrapper_ids"]
        if not isinstance(optimized, list) or not all(isinstance(item, str) for item in optimized):
            raise ValueError("optimized_wrapper_ids must be a list of strings")
        default_wrapper_id = state["default_wrapper_id"]
        if default_wrapper_id is not None and not isinstance(default_wrapper_id, str):
            raise ValueError("default_wrapper_id must be a string or None")
        assignments = string_mapping(state["wrapper_assignments"], where="wrapper assignments")
        if not all(isinstance(value, str) for value in assignments.values()):
            raise ValueError("wrapper assignment values must be strings")
        trainer = cls(
            backend,
            registry,
            default_wrapper_id=default_wrapper_id,
            wrapper_assignments=cast(dict[str, str], assignments),
            optimize=cast(list[str], optimized),
            objective=cast(TrainingObjective, objective),
            loss_reduction=cast(LossReduction, loss_reduction),
            optimizer_kwargs=string_mapping(state["optimizer_kwargs"], where="optimizer kwargs"),
            provenance=string_mapping(state["provenance"], where="provenance"),
        )
        step_count = state["step_count"]
        if not isinstance(step_count, int) or isinstance(step_count, bool) or step_count < 0:
            raise ValueError("step_count must be a non-negative integer")
        optimizer_states = string_mapping(state["optimizers"], where="optimizer states")
        if set(optimizer_states) != set(trainer.optimizers):
            raise ValueError("optimizer state wrapper IDs do not match optimized wrappers")
        for wrapper_id, optimizer in trainer.optimizers.items():
            optimizer.load_state_dict(cast(dict[str, object], optimizer_states[wrapper_id]))
        trainer.step_count = step_count
        trainer.optimize_mode = cast(str, state["optimize_mode"])
        return trainer
