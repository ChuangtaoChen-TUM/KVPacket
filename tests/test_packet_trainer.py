"""D7 tests for heterogeneous wrapper training and atomic trainer checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
import torch
from torch import Tensor

from kvpacket.backend import BackendCompatibility, ForwardOutput, TrainingBackend
from kvpacket.fingerprint import fingerprint_json
from kvpacket.packet import PacketWrapper, WrapperEntry, WrapperRegistry
from kvpacket.session import TrainingSample
from kvpacket.training import PacketTrainer, create_wrapper


class _TrainingBackend:
    def __init__(
        self,
        hidden_size: int = 4,
        vocab_size: int = 7,
        sliding_window: int | None = None,
    ) -> None:
        self._device = torch.device("cpu")
        self._compatibility = BackendCompatibility(
            backend="test-train",
            model_id="test/student",
            model_config_fingerprint=fingerprint_json(
                {"hidden_size": hidden_size}, domain="tests.trainer.model/1"
            ),
            hidden_size=hidden_size,
            model_artifact_fingerprint=fingerprint_json(
                "test/student", domain="tests.trainer.artifact/1"
            ),
        )
        generator = torch.Generator().manual_seed(7)
        self.embedding = torch.randn(32, hidden_size, generator=generator)
        self.projection = torch.randn(hidden_size, vocab_size, generator=generator)
        self.last_attention_mask: Tensor | None = None
        self.prepared = False
        self._sliding_window = sliding_window

    @property
    def compatibility(self) -> BackendCompatibility:
        return self._compatibility

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def rotary_emb(self):
        return None

    @property
    def nope_dim(self) -> int | None:
        return None

    @property
    def is_rerotatable(self) -> bool:
        return True

    @property
    def sliding_window(self) -> int | None:
        return self._sliding_window

    def assert_rerotatable(self) -> None:
        return None

    def shift_cache(self, cache, offset):
        return cache.shift(self.rotary_emb, offset, nope_dim=self.nope_dim)

    def shift_cache_flops(self, cache) -> int:
        return cache.rerotate_flops(nope_dim=self.nope_dim)

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.embedding[input_ids]

    def forward(self, **kwargs) -> ForwardOutput:
        raise NotImplementedError

    def prefill_caches(self, **kwargs):
        raise NotImplementedError

    def prepare_for_training(self) -> None:
        self.prepared = True

    def embedding_statistics(self) -> tuple[float, float]:
        return float(self.embedding.mean()), float(self.embedding.std())

    def training_forward(self, *, inputs_embeds: Tensor, attention_mask: Tensor) -> Tensor:
        self.last_attention_mask = attention_mask.detach().clone()
        # A tiny causal surrogate: every later position depends on all wrapper tokens.
        return inputs_embeds.cumsum(dim=1) @ self.projection


def _single_registry(
    backend: TrainingBackend,
    wrapper: PacketWrapper,
    wrapper_id: str = "document",
) -> WrapperRegistry:
    return WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={wrapper_id: WrapperEntry(wrapper)},
    )


def _collection(backend: TrainingBackend) -> WrapperRegistry:
    registry = _single_registry(backend, PacketWrapper(1, 1, 4))
    registry.add("skill", PacketWrapper(2, 1, 4))
    return registry


def _trainer(backend: TrainingBackend, registry: WrapperRegistry, **kwargs) -> PacketTrainer:
    assignments = {"doc": "document"}
    if "skill" in registry:
        assignments["tool"] = "skill"
    return PacketTrainer(backend, registry, wrapper_assignments=assignments, **kwargs)


def _sample(registry: WrapperRegistry) -> TrainingSample:
    return TrainingSample(
        token_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        generated_mask=torch.tensor([False] * 6 + [True, True]),
        chunk_spans={"doc": [(0, 2)], "tool": [(3, 5)]},
        logits=torch.randn(2, 7, generator=torch.Generator().manual_seed(11)),
    )


def _variable_sample(
    registry: WrapperRegistry,
    num_generated: int,
    *,
    seed: int,
) -> TrainingSample:
    prefix_length = 6
    length = prefix_length + num_generated
    return TrainingSample(
        token_ids=torch.arange(1, length + 1),
        generated_mask=torch.tensor([False] * prefix_length + [True] * num_generated),
        chunk_spans={"doc": [(0, 2)], "tool": [(3, 5)]},
        logits=torch.randn(
            num_generated,
            7,
            generator=torch.Generator().manual_seed(seed),
        ),
    )


def test_trainer_reconstructs_different_wrappers_and_updates_all():
    backend = _TrainingBackend()
    registry = _collection(backend)
    sample = _sample(registry)
    before = {
        wrapper_id: registry[wrapper_id].wrapper.header.detach().clone()
        for wrapper_id in registry.ids()
    }
    trainer = _trainer(backend, registry, optimizer_kwargs={"lr": 0.05, "weight_decay": 0.0})
    result = trainer.step(sample)

    assert backend.prepared
    assert result.num_tokens == 2
    assert result.wrapper_ids == ("document", "skill")
    assert trainer.step_count == 1
    assert set(trainer.optimizers) == {"document", "skill"}
    assert not torch.equal(before["document"], registry["document"].wrapper.header)
    assert not torch.equal(before["skill"], registry["skill"].wrapper.header)

    mask = backend.last_attention_mask
    assert mask is not None
    # The second packet starts after the first packet and one plain token. Its rows
    # cannot attend that prefix, while the later query rows can.
    assert not bool(mask[0, 0, 5, 4])
    assert bool(mask[0, 0, -1, 0])


def test_allowlist_and_none_modes_do_not_update_unselected_wrappers():
    backend = _TrainingBackend()
    registry = _collection(backend)
    sample = _sample(registry)
    document_before = registry["document"].wrapper.header.detach().clone()
    skill_before = registry["skill"].wrapper.header.detach().clone()
    trainer = _trainer(backend, registry, optimize=["document"])
    trainer.step(sample)
    assert not torch.equal(document_before, registry["document"].wrapper.header)
    assert torch.equal(skill_before, registry["skill"].wrapper.header)

    frozen = {
        wrapper_id: registry[wrapper_id].wrapper.header.detach().clone()
        for wrapper_id in registry.ids()
    }
    evaluator = _trainer(
        backend,
        registry,
        optimize="none",
    )
    metrics = evaluator.step(sample)
    assert metrics.num_tokens == 2
    assert evaluator.step_count == 0
    for wrapper_id in registry.ids():
        assert torch.equal(frozen[wrapper_id], registry[wrapper_id].wrapper.header)


def test_checkpoint_restores_collection_provenance_and_optimizer_ids(tmp_path: Path):
    backend = _TrainingBackend()
    registry = _collection(backend)
    sample = _sample(registry)
    trainer = _trainer(
        backend,
        registry,
        optimize=["document"],
        provenance={"dataset": "unit-test"},
    )
    trainer.step(sample)
    path = tmp_path / "trainer.pt"
    trainer.save(path)

    restored = PacketTrainer.load(path, backend=backend)
    assert restored.step_count == 1
    assert set(restored.optimizers) == {"document"}
    assert restored.user_provenance == {"dataset": "unit-test"}
    assert restored.loss_reduction == "token_mean"
    assert restored.objective == "teacher_kl"
    assert restored.wrapper_assignments == trainer.wrapper_assignments
    # Original RECORD samples remain valid after restoring already-trained tensors.
    restored.step(sample)
    assert restored.step_count == 2


def test_pre_objective_checkpoint_defaults_to_teacher_kl(tmp_path: Path):
    backend = _TrainingBackend()
    registry = _collection(backend)
    trainer = _trainer(backend, registry)
    path = tmp_path / "pre-objective.pt"
    trainer.save(path)

    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["trainer"].pop("objective")
    torch.save(raw, path)

    restored = PacketTrainer.load(path, backend=backend)
    assert restored.objective == "teacher_kl"


def test_trainer_rejects_missing_logits_and_overlapping_spans():
    backend = _TrainingBackend()
    registry = _collection(backend)
    sample = _sample(registry)
    trainer = _trainer(backend, registry, optimize="none")
    sample.logits = None
    with pytest.raises(ValueError, match="teacher logits"):
        trainer.evaluate(sample)

    overlap = _sample(registry)
    overlap.chunk_spans = {"doc": [(0, 3)], "tool": [(2, 5)]}
    with pytest.raises(ValueError, match="overlap"):
        trainer.evaluate(overlap)


def test_teacher_ce_uses_generated_token_ids_without_logits():
    backend = _TrainingBackend()
    registry = _collection(backend)
    sample = TrainingSample(
        token_ids=torch.tensor([1, 2, 3, 4, 5, 1, 2]),
        generated_mask=torch.tensor([False] * 5 + [True, True]),
        chunk_spans={"doc": [(0, 2)], "tool": [(3, 5)]},
        logits=None,
    )
    before = registry["document"].wrapper.header.detach().clone()
    trainer = _trainer(
        backend,
        registry,
        objective="teacher_ce",
        optimizer_kwargs={"lr": 0.05, "weight_decay": 0.0},
    )

    embeds, attention_mask, positions = trainer._student_inputs(sample)
    student = backend.training_forward(inputs_embeds=embeds, attention_mask=attention_mask)[
        0
    ].index_select(0, positions)
    expected = torch.nn.functional.cross_entropy(
        student.float(), torch.tensor([1, 2]), reduction="mean"
    )
    loss_sum, _wrapper_ids = trainer._batched_loss_sums([sample])[0]
    metrics = trainer.evaluate(sample)

    assert loss_sum.dtype == torch.float32
    assert metrics.loss == pytest.approx(float(expected.detach()))
    assert trainer.objective == "teacher_ce"
    trainer.step(sample)
    assert not torch.equal(before, registry["document"].wrapper.header)


def test_teacher_ce_checkpoint_roundtrip_and_objective_validation(tmp_path: Path):
    backend = _TrainingBackend()
    registry = _collection(backend)
    trainer = _trainer(backend, registry, objective="teacher_ce")
    path = tmp_path / "teacher-ce.pt"
    trainer.save(path)

    restored = PacketTrainer.load(path, backend=backend)
    assert restored.objective == "teacher_ce"
    assert restored.state_dict()["objective"] == "teacher_ce"
    with pytest.raises(ValueError, match="objective"):
        _trainer(backend, registry, objective="cross_entropy")


def test_hf_backend_exposes_a_real_differentiable_training_path():
    from transformers import LlamaConfig, LlamaForCausalLM

    from kvpacket import HFBackend

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
    )
    backend = HFBackend(model)
    registry = _single_registry(backend, PacketWrapper(1, 1, 16))
    sample = TrainingSample(
        token_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
        generated_mask=torch.tensor([False, False, False, False, True, True]),
        chunk_spans={"doc": [(1, 3)]},
        logits=torch.randn(2, 32),
    )
    metrics = _trainer(backend, registry).step(sample)
    assert metrics.num_tokens == 2
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_hf_backend_accepts_variable_length_4d_training_batch():
    from transformers import LlamaConfig, LlamaForCausalLM

    from kvpacket import HFBackend

    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
    )
    backend = HFBackend(model)
    registry = _single_registry(backend, PacketWrapper(1, 2, 16))
    samples = [
        TrainingSample(
            token_ids=torch.arange(1, length + 1),
            generated_mask=torch.tensor([False] * (length - generated) + [True] * generated),
            chunk_spans={"doc": [(1, 3)]},
            logits=torch.randn(generated, 32),
        )
        for length, generated in ((6, 2), (8, 3))
    ]
    metrics = _trainer(backend, registry).step_batch(samples, forward_batch_size=2)
    assert metrics.num_tokens == 5


def test_homogeneous_trainer_routes_all_chunks_to_default_wrapper():
    backend = _TrainingBackend()
    wrapper = PacketWrapper(1, 1, 4)
    registry = _single_registry(backend, wrapper)
    sample = TrainingSample(
        token_ids=torch.tensor([1, 2, 3, 4]),
        generated_mask=torch.tensor([False, False, False, True]),
        chunk_spans={"doc": [(0, 2)]},
        logits=torch.randn(1, 7),
    )
    trainer = _trainer(backend, registry, optimize="none")
    assert trainer.evaluate(sample).wrapper_ids == ("document",)


def test_trainer_sliding_window_uses_exact_transformers_boundary():
    backend = _TrainingBackend(sliding_window=3)
    registry = _collection(backend)
    trainer = _trainer(backend, registry, optimize="none")
    trainer.evaluate(_sample(registry))
    mask = backend.last_attention_mask
    assert mask is not None
    for row in range(mask.shape[-2]):
        allowed = mask[0, 0, row].nonzero(as_tuple=False).flatten().tolist()
        assert all(column > row - 3 for column in allowed)
        assert row in allowed


def test_streaming_batch_accumulation_steps_once_and_handles_frozen_only_sample():
    backend = _TrainingBackend()
    registry = _collection(backend)
    sample = _sample(registry)
    trainer = _trainer(backend, registry, optimize=["document"])
    metrics = trainer.step_batch([sample, sample])
    assert metrics.num_tokens == 4
    assert trainer.step_count == 1

    frozen_only = TrainingSample(
        token_ids=torch.tensor([1, 2, 3, 4]),
        generated_mask=torch.tensor([False, False, False, True]),
        chunk_spans={"tool": [(0, 2)]},
        logits=torch.randn(1, 7),
    )
    trainer.step(frozen_only)
    assert trainer.step_count == 1


@pytest.mark.parametrize("loss_reduction", ["token_mean", "sample_mean"])
def test_frozen_only_samples_do_not_dilute_training_normalization(
    loss_reduction: Literal["token_mean", "sample_mean"],
):
    backend_single = _TrainingBackend()
    source = _collection(backend_single)
    active = _sample(source)
    frozen_only = TrainingSample(
        token_ids=torch.tensor([1, 2, 3, 4]),
        generated_mask=torch.tensor([False, False, False, True]),
        chunk_spans={"tool": [(0, 2)]},
        logits=torch.randn(1, 7),
    )
    backend_mixed = _TrainingBackend()
    single_collection = WrapperRegistry.from_state_dict(source.state_dict())
    mixed_collection = WrapperRegistry.from_state_dict(source.state_dict())
    kwargs = {"lr": 0.01, "weight_decay": 0.0}
    single = _trainer(
        backend_single,
        single_collection,
        optimize=["document"],
        loss_reduction=loss_reduction,
        optimizer_kwargs=kwargs,
    )
    mixed = _trainer(
        backend_mixed,
        mixed_collection,
        optimize=["document"],
        loss_reduction=loss_reduction,
        optimizer_kwargs=kwargs,
    )
    single.step(active)
    mixed.step_batch([active, frozen_only], forward_batch_size=2)
    for name in ("header", "trailer"):
        torch.testing.assert_close(
            getattr(single_collection["document"].wrapper, name),
            getattr(mixed_collection["document"].wrapper, name),
        )


def test_loss_reductions_have_expected_gradient_scale_and_report_same_mean():
    backend_mean = _TrainingBackend()
    collection_mean = _collection(backend_mean)
    sample_mean = _sample(collection_mean)
    initial = collection_mean.state_dict()

    backend_sum = _TrainingBackend()
    collection_sum = WrapperRegistry.from_state_dict(initial, requires_grad=True)
    sample_sum = _sample(collection_sum)
    kwargs = {"lr": 0.0, "weight_decay": 0.0}
    mean_trainer = _trainer(
        backend_mean,
        collection_mean,
        loss_reduction="token_mean",
        optimizer_kwargs=kwargs,
    )
    sum_trainer = _trainer(
        backend_sum,
        collection_sum,
        loss_reduction="legacy_sum",
        optimizer_kwargs=kwargs,
    )

    mean_metrics = mean_trainer.step(sample_mean)
    sum_metrics = sum_trainer.step(sample_sum)
    assert sum_metrics.loss == pytest.approx(mean_metrics.loss)
    for wrapper_id in collection_mean.ids():
        mean_grad = collection_mean[wrapper_id].wrapper.header.grad
        sum_grad = collection_sum[wrapper_id].wrapper.header.grad
        assert mean_grad is not None and sum_grad is not None
        torch.testing.assert_close(sum_grad, mean_grad * sample_mean.num_generated)


def test_token_mean_accepts_positive_sample_weights():
    backend = _TrainingBackend()
    registry = _collection(backend)
    samples = [
        _variable_sample(registry, 2, seed=31),
        _variable_sample(registry, 3, seed=32),
    ]
    trainer = _trainer(
        backend,
        registry,
        optimizer_kwargs={"lr": 0.0, "weight_decay": 0.0},
    )
    loss_sums = [float(loss.detach()) for loss, _wrapper_ids in trainer._batched_loss_sums(samples)]
    weights = [1.0, 3.0]
    expected = (loss_sums[0] * weights[0] + loss_sums[1] * weights[1]) / (
        samples[0].num_generated * weights[0] + samples[1].num_generated * weights[1]
    )
    metrics = trainer.step_batch(samples, sample_weights=weights)
    assert metrics.loss == pytest.approx(expected)

    with pytest.raises(ValueError, match="match samples"):
        trainer.step_batch(samples, sample_weights=[1.0])
    with pytest.raises(ValueError, match="finite and positive"):
        trainer.step_batch(samples, sample_weights=[1.0, 0.0])


def test_kl_is_computed_in_float32_for_bfloat16_model_logits():
    backend = _TrainingBackend()
    backend.embedding = backend.embedding.to(torch.bfloat16)
    backend.projection = backend.projection.to(torch.bfloat16)
    registry = _collection(backend)
    sample = _sample(registry)
    assert sample.logits is not None
    sample.logits = sample.logits.to(torch.bfloat16)
    trainer = _trainer(
        backend,
        registry,
        optimizer_kwargs={"lr": 0.0, "weight_decay": 0.0},
    )

    loss, _wrapper_ids = trainer._batched_loss_sums([sample])[0]

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_hf_training_forward_converts_boolean_allow_mask_to_additive():
    """The custom 4-D allow mask must reproduce an ordinary causal forward.

    Transformers 5.x does not interpret a 4-D boolean tensor as an allow/block
    mask.  This regression check exercises HFBackend's explicit additive-mask
    conversion on a tiny local Llama model.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    from kvpacket import HFBackend

    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    ).eval()
    backend = HFBackend(model)
    input_ids = torch.tensor([[1, 7, 11, 5, 3]], dtype=torch.long)
    embeds = model.model.embed_tokens(input_ids)
    length = int(input_ids.shape[1])
    allow_mask = torch.tril(torch.ones((length, length), dtype=torch.bool)).view(
        1, 1, length, length
    )

    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits
        actual = backend.training_forward(
            inputs_embeds=embeds,
            attention_mask=allow_mask,
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_legacy_sum_checkpoint_roundtrip(tmp_path: Path):
    backend = _TrainingBackend()
    registry = _collection(backend)
    trainer = _trainer(
        backend,
        registry,
        loss_reduction="legacy_sum",
        optimizer_kwargs={"lr": 0.01, "weight_decay": 0.0},
    )
    trainer.step(_sample(registry))
    path = tmp_path / "legacy-sum-trainer.pt"
    trainer.save(path)

    restored = PacketTrainer.load(path, backend=backend)
    assert restored.loss_reduction == "legacy_sum"
    assert restored.state_dict()["loss_reduction"] == "legacy_sum"


def test_sample_mean_averages_samples_and_supports_token_budget_batches():
    backend = _TrainingBackend()
    source = _collection(backend)
    samples = [
        _variable_sample(source, generated, seed=20 + generated) for generated in (1, 3, 2, 4, 1)
    ]
    token_collection = WrapperRegistry.from_state_dict(source.state_dict(), requires_grad=True)
    sample_collection = WrapperRegistry.from_state_dict(source.state_dict(), requires_grad=True)
    token_trainer = _trainer(backend, token_collection, optimize="none")
    sample_trainer = _trainer(
        backend,
        sample_collection,
        optimize="none",
        loss_reduction="sample_mean",
    )

    individual = [sample_trainer.evaluate(sample).loss for sample in samples[:2]]
    sample_metrics = sample_trainer.step_batch(samples[:2])
    token_metrics = token_trainer.step_batch(samples[:2])
    assert sample_metrics.loss == pytest.approx(sum(individual) / 2)
    expected_token_mean = sum(
        metric * sample.num_generated for metric, sample in zip(individual, samples[:2])
    ) / sum(sample.num_generated for sample in samples[:2])
    assert token_metrics.loss == pytest.approx(expected_token_mean)

    fit_metrics = sample_trainer.fit(samples, batch_size=2)
    assert [metric.num_tokens for metric in fit_metrics] == [4, 6, 1]
    dropped = sample_trainer.fit(samples, batch_size=2, drop_last=True)
    assert [metric.num_tokens for metric in dropped] == [4, 6]
    budget_collection = WrapperRegistry.from_state_dict(source.state_dict(), requires_grad=True)
    budget_trainer = _trainer(
        backend,
        budget_collection,
        loss_reduction="sample_mean",
        optimizer_kwargs={"lr": 0.0, "weight_decay": 0.0},
    )
    token_budgeted = budget_trainer.fit(samples, batch_size=4, target_tokens_per_step=5)
    assert [metric.num_tokens for metric in token_budgeted] == [6, 5]


def test_token_budget_batches_have_variable_sample_counts():
    backend = _TrainingBackend()
    registry = _collection(backend)
    samples = [
        _variable_sample(registry, generated, seed=30 + generated) for generated in (2, 3, 4, 1, 6)
    ]
    trainer = _trainer(
        backend,
        registry,
        optimizer_kwargs={"lr": 0.0, "weight_decay": 0.0},
    )
    metrics = trainer.fit(
        samples,
        batch_size=4,
        target_tokens_per_step=5,
    )
    assert [metric.num_tokens for metric in metrics] == [5, 5, 6]

    capped = trainer.fit(
        samples[:3],
        batch_size=2,
        target_tokens_per_step=100,
    )
    assert [metric.num_tokens for metric in capped] == [5, 4]


@pytest.mark.parametrize("loss_reduction", ["token_mean", "sample_mean", "legacy_sum"])
def test_4d_forward_microbatch_matches_sequential_updates(
    loss_reduction: Literal["token_mean", "sample_mean", "legacy_sum"],
):
    backend_sequential = _TrainingBackend()
    source = _collection(backend_sequential)
    samples = [_variable_sample(source, generated, seed=40 + generated) for generated in (1, 4, 2)]
    backend_batched = _TrainingBackend()
    sequential_collection = WrapperRegistry.from_state_dict(source.state_dict(), requires_grad=True)
    batched_collection = WrapperRegistry.from_state_dict(source.state_dict(), requires_grad=True)
    kwargs = {"lr": 0.01, "weight_decay": 0.0}
    sequential = _trainer(
        backend_sequential,
        sequential_collection,
        loss_reduction=loss_reduction,
        optimizer_kwargs=kwargs,
    )
    batched = _trainer(
        backend_batched,
        batched_collection,
        loss_reduction=loss_reduction,
        optimizer_kwargs=kwargs,
    )

    sequential_metrics = sequential.step_batch(samples, forward_batch_size=1)
    batched_metrics = batched.step_batch(samples, forward_batch_size=3)
    assert batched_metrics.loss == pytest.approx(sequential_metrics.loss, rel=1e-6, abs=1e-7)
    for wrapper_id in source.ids():
        for name in ("header", "trailer"):
            torch.testing.assert_close(
                getattr(sequential_collection[wrapper_id].wrapper, name),
                getattr(batched_collection[wrapper_id].wrapper, name),
                rtol=1e-5,
                atol=1e-6,
            )
    mask = backend_batched.last_attention_mask
    assert mask is not None and mask.shape[0] == 3
    lengths = [batched._student_inputs(sample)[0].shape[1] for sample in samples]
    for index, length in enumerate(lengths):
        assert not bool(mask[index, 0, :length, length:].any())
        for padding_position in range(length, mask.shape[-1]):
            assert bool(mask[index, 0, padding_position, padding_position])


def test_sample_mean_checkpoint_roundtrip(tmp_path: Path):
    backend = _TrainingBackend()
    registry = _collection(backend)
    trainer = _trainer(backend, registry, loss_reduction="sample_mean")
    path = tmp_path / "sample-mean.pt"
    trainer.save(path)
    assert PacketTrainer.load(path, backend=backend).loss_reduction == "sample_mean"


def test_configurable_wrapper_initialization():
    backend = _TrainingBackend()
    torch.manual_seed(101)
    fixed = create_wrapper(
        backend,
        header_len=2,
        trailer_len=3,
        dtype=torch.float32,
        initialization={"strategy": "fixed", "mean": 1.5, "std": 0.0},
    )
    assert torch.equal(fixed.header, torch.full_like(fixed.header, 1.5))
    assert torch.equal(fixed.trailer, torch.full_like(fixed.trailer, 1.5))

    torch.manual_seed(102)
    from_model = create_wrapper(
        backend,
        header_len=64,
        trailer_len=64,
        dtype=torch.float32,
        initialization={"strategy": "model-embedding"},
    )
    expected_mean, expected_std = backend.embedding_statistics()
    assert from_model.mean == pytest.approx(expected_mean)
    assert from_model.std == pytest.approx(expected_std)
    with pytest.raises(ValueError, match="does not accept explicit"):
        create_wrapper(
            backend,
            header_len=1,
            trailer_len=1,
            dtype=torch.float32,
            initialization={
                "strategy": "model-embedding",
                "std": 0.1,
            },
        )


def test_collection_load_owns_wrapper_tensor_storage():
    backend = _TrainingBackend()
    source = _collection(backend)
    state = source.state_dict()
    first = WrapperRegistry.from_state_dict(state)
    second = WrapperRegistry.from_state_dict(state)
    first_header = first["document"].wrapper.header
    second_header = second["document"].wrapper.header
    assert first_header.data_ptr() != second_header.data_ptr()
    with torch.no_grad():
        first_header.add_(1)
    assert not torch.equal(first_header, second_header)
