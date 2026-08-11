"""Adapt Hugging Face causal language models to KV Packet backend protocols.

``HFBackend`` exposes token embeddings, model forwards, native continuation,
offline cache prefill, and differentiable training through framework-neutral KV Packet
types. Native ``DynamicCache`` objects are confined to the conversion boundary in this
module.

Support is capability-based rather than tied to model-type names. A compatible model
must expose shared token and rotary embeddings at the expected Hugging Face locations,
use a repositionable RoPE variant, and return standard-attention cache layers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache

from ...kvcache.cache import KeyValue, KVCache
from ...fingerprint import fingerprint_json
from ..base import (
    BackendCompatibility,
    BackendGenerationOutput,
    ComputeBackend,
    ForwardOutput,
    RerotationError,
)

__all__ = ["HFBackend", "RerotationError"]

_LOCATION_KEYS = {
    "_name_or_path",
    "name_or_path",
    "tokenizer_file",
    "vocab_file",
    "merges_file",
}


def _json_component(value: object) -> object:
    """Normalize HF config values without retaining Python object reprs."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Mapping):
        return {
            str(key): _json_component(item)
            for key, item in value.items()
            if str(key) not in _LOCATION_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_component(item) for item in value]
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _model_config_fingerprint_data(config: object) -> dict[str, object]:
    """Extract stable effective model fields used to fingerprint a configuration."""
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        raw = to_dict()
    else:
        raw = {
            key: item
            for owner in reversed(type(config).mro())
            for key, item in vars(owner).items()
            if not key.startswith("_") and not callable(item) and not isinstance(item, property)
        }
        raw.update(vars(config))
    if not isinstance(raw, Mapping):
        raise TypeError(f"{type(config).__name__}.to_dict() must return a mapping")
    fingerprint_data = {str(key): _json_component(item) for key, item in raw.items()}
    # These describe where/how the config was loaded, not the effective architecture.
    for unstable in ("_name_or_path", "name_or_path", "transformers_version"):
        fingerprint_data.pop(unstable, None)
    return fingerprint_data


class HFBackend(ComputeBackend):
    """Hugging Face implementation of inference and training backend capabilities.

    The backend keeps the supplied model by reference. It computes a compatibility
    fingerprint during construction and rejects models that do not expose shared input
    and rotary embeddings.

    Args:
        model: Hugging Face causal language model.
    Raises:
        AttributeError: If the model does not expose the required embeddings.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        *,
        model_artifact_id: str | None = None,
    ) -> None:
        """Bind a Hugging Face model and establish its persistent identity.

        Args:
            model: Loaded causal language model.
            model_artifact_id: Stable revision, content digest, or deployment artifact
                identifier for the exact weights. When omitted, the backend combines
                the model ID with Hugging Face revision metadata when available.
        """
        self.model: PreTrainedModel = model
        self.config = model.config

        base = getattr(self.model, "model", None)
        if base is None or not hasattr(base, "embed_tokens"):
            raise AttributeError(
                "HFBackend requires model.model.embed_tokens; this model does not "
                "expose token embeddings in the expected place."
            )
        if not hasattr(base, "rotary_emb"):
            raise AttributeError(
                "HFBackend requires a shared model.model.rotary_emb. Models that compute "
                "RoPE per-attention-layer are not yet wired up."
            )

        model_id = str(
            getattr(self.config, "_name_or_path", "")
            or getattr(self.model, "name_or_path", "")
            or type(self.model).__qualname__
        )
        revision = (
            model_artifact_id
            or getattr(self.config, "_commit_hash", None)
            or getattr(self.model, "_commit_hash", None)
        )
        artifact_identity = {
            "model_id": model_id,
            "revision": str(revision) if revision is not None else None,
        }
        self._compatibility = BackendCompatibility(
            backend="hf",
            model_id=model_id,
            model_config_fingerprint=fingerprint_json(
                _model_config_fingerprint_data(self.config),
                domain="kvpacket.hf_model_config/1",
            ),
            hidden_size=self.hidden_size,
            model_artifact_fingerprint=fingerprint_json(
                artifact_identity,
                domain="kvpacket.hf_model_artifact/1",
            ),
        )

    @property
    def compatibility(self) -> BackendCompatibility:
        """Stable identity of the model configuration."""
        return self._compatibility

    @property
    def device(self) -> torch.device:
        """Device reported by the underlying model."""
        return self.model.device

    @property
    def hidden_size(self) -> int:
        """Input-embedding width declared by the model configuration."""
        return int(self.config.hidden_size)

    @property
    def num_layers(self) -> int:
        """Transformer layer count declared by the model configuration."""
        return int(self.config.num_hidden_layers)

    @property
    def rotary_emb(self):
        """Shared rotary embedding exposed by the base model."""
        return self.model.model.rotary_emb

    @property
    def nope_dim(self) -> int | None:
        """Leading non-rotary key width used by partial-RoPE models."""
        v = getattr(self.config, "qk_nope_head_dim", None)
        return int(v) if isinstance(v, int) else None

    @property
    def sliding_window(self) -> int | None:
        """Configured causal attention window, if present."""
        v = getattr(self.config, "sliding_window", None)
        return int(v) if isinstance(v, int) else None

    @property
    def attention_scaling(self) -> float:
        """Constant amplitude applied by the model's rotary embedding."""
        return float(getattr(self.rotary_emb, "attention_scaling", 1.0))

    @property
    def rope_scaling_type(self) -> str | None:
        """Configured RoPE scaling strategy, or ``None`` for plain RoPE."""
        rs = getattr(self.config, "rope_scaling", None)
        if not rs:
            return None
        return rs.get("rope_type") or rs.get("type")

    @property
    def is_rerotatable(self) -> bool:
        """Whether the configured RoPE frequencies remain position-shiftable."""
        return self.rope_scaling_type not in {"dynamic"}

    def assert_rerotatable(self) -> None:
        """Reject dynamic RoPE frequencies that cannot be shifted exactly.

        Raises:
            RerotationError: If runtime sequence length changes the RoPE frequencies.
        """
        if not self.is_rerotatable:
            raise RerotationError(
                f"rope_scaling type '{self.rope_scaling_type}' recomputes RoPE frequencies "
                "from the runtime sequence length, so cached keys cannot be re-rotated to a "
                "new position strictly. Use a static scaling config or plain RoPE."
            )

    def shift_cache(self, cache: KVCache, offset: int) -> KVCache:
        """Reposition cached keys without exposing Hugging Face RoPE internals."""
        self.assert_rerotatable()
        return cache.shift(self.rotary_emb, offset, nope_dim=self.nope_dim)

    def shift_cache_flops(self, cache: KVCache) -> int:
        """Analytic re-rotation cost for one cache."""
        return cache.rerotate_flops(nope_dim=self.nope_dim)

    def embed(self, input_ids: Tensor) -> Tensor:
        """Map token IDs through the model's input embedding table."""
        return self.model.model.embed_tokens(input_ids)

    def prepare_for_training(self) -> None:
        """Freeze the Hugging Face model; wrapper embeddings remain trainable."""
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def embedding_statistics(self) -> tuple[float, float]:
        """Measure native-dtype input-embedding mean and standard deviation."""
        weight = self.model.model.embed_tokens.weight
        return float(weight.mean()), float(weight.std())

    def training_forward(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Run a cache-free differentiable student forward for wrapper training.

        Boolean four-dimensional masks use KV Packet's allow-mask convention and are
        converted to the additive bias expected by Hugging Face attention.

        Args:
            inputs_embeds: Wrapped student embeddings.
            attention_mask: Packet-isolated attention topology.

        Returns:
            Differentiable logits shaped ``(batch, sequence, vocabulary)``.
        """
        # PacketTrainer expresses its block-causal mask as a boolean *allow* mask.
        # Transformers 5.x accepts a 4-D mask here but treats boolean tensors as
        # numeric attention bias rather than applying allow/block semantics.  Convert
        # explicitly to the additive representation expected by HF attention:
        # allowed entries receive zero bias and blocked entries receive -infinity.
        if attention_mask.dim() == 4 and attention_mask.dtype == torch.bool:
            additive_mask = torch.zeros(
                attention_mask.shape,
                dtype=inputs_embeds.dtype,
                device=attention_mask.device,
            )
            attention_mask = additive_mask.masked_fill(
                ~attention_mask, torch.finfo(inputs_embeds.dtype).min
            )
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = output.logits
        if not isinstance(logits, Tensor):
            raise TypeError("Hugging Face causal LM did not return tensor logits")
        return logits

    @torch.no_grad()
    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        position_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        cache: KVCache | None = None,
    ) -> ForwardOutput:
        """Run inference and convert the extended native cache back to ``KVCache``.

        Explicit positions should be supplied when extending a repositioned cache,
        because a native ``DynamicCache`` does not retain logical position IDs.

        Args:
            input_ids: New token IDs, mutually exclusive with ``inputs_embeds``.
            inputs_embeds: New embeddings, mutually exclusive with ``input_ids``.
            position_ids: Logical positions assigned to the new sequence.
            attention_mask: Attention representation accepted by the model.
            cache: Optional KV Packet prefix converted internally to ``DynamicCache``.

        Returns:
            Model logits and the extended framework-neutral cache.
        """
        past = self._to_dynamic(cache) if cache is not None else None
        out = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        cache_positions = None
        if position_ids is not None:
            cache_positions = position_ids
            if cache is not None:
                cache_positions = torch.cat(
                    [
                        cache.layers[0].logical_positions.to(position_ids.device),
                        position_ids,
                    ],
                    dim=1,
                )
        return ForwardOutput(
            logits=out.logits,
            cache=self._from_dynamic(
                out.past_key_values,
                position_ids=cache_positions,
            ),
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None = None,
        cache: KVCache | None = None,
        max_new_tokens: int = 64,
        eos_token_id: int | list[int] | None = None,
        capture_logits: bool = False,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
    ) -> BackendGenerationOutput:
        """Continue generation with native Hugging Face decoding.

        The session normally supplies its already selected first token as ``input_ids``.
        Hugging Face forwards that token and owns every later token-selection step, so
        its generation config and RNG behavior remain authoritative after TTFT. Only
        batch size one and two-dimensional attention masks are supported.

        Returns:
            Generated IDs excluding ``input_ids``, optional raw logits preceding each
            generated ID, a reusable cache, and the next logical position.
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("native HF generation requires one non-empty prompt")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match the prompt input_ids shape")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        past_len = 0 if cache is None else cache.seq_len
        expected_mask_len = past_len + int(input_ids.shape[1])
        if attention_mask is None:
            attention_mask = torch.ones(
                (1, expected_mask_len), dtype=torch.long, device=input_ids.device
            )
        elif attention_mask.dim() != 2 or attention_mask.shape != (1, expected_mask_len):
            raise ValueError(
                "native HF generation requires a 2-D attention mask covering cache and prompt"
            )

        generation_kwargs: dict[str, object] = {
            "past_key_values": self._to_dynamic(cache) if cache is not None else None,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_logits": capture_logits,
        }
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if do_sample:
            generation_kwargs.update(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            if min_p > 0.0:
                generation_kwargs["min_p"] = min_p

        generate = getattr(self.model, "generate", None)
        if not callable(generate):
            raise TypeError("Hugging Face model does not expose generate()")
        output = generate(inputs=input_ids, **generation_kwargs)

        sequences = getattr(output, "sequences", None)
        native_cache = getattr(output, "past_key_values", None)
        if not isinstance(sequences, Tensor) or sequences.dim() != 2 or sequences.shape[0] != 1:
            raise TypeError("Hugging Face generation did not return one token sequence")
        if not isinstance(native_cache, Cache):
            raise TypeError("Hugging Face generation did not return a native cache")
        prompt_len = int(input_ids.shape[1])
        token_ids = [int(token_id) for token_id in sequences[0, prompt_len:].tolist()]
        captured_logits: list[Tensor] = []
        if capture_logits:
            raw_logits = getattr(output, "logits", None)
            if not isinstance(raw_logits, tuple) or len(raw_logits) != len(token_ids):
                raise TypeError("Hugging Face generation did not return one logit tensor per token")
            for step_logits in raw_logits:
                if not isinstance(step_logits, Tensor) or step_logits.dim() != 2:
                    raise TypeError("Hugging Face generation returned malformed token logits")
                captured_logits.append(step_logits[0].detach().clone())
        if not token_ids:
            raise RuntimeError("Hugging Face generation returned no continuation tokens")

        native_len = int(native_cache.get_seq_length())
        new_cached = native_len - past_len
        expected_cached = prompt_len + len(token_ids) - 1
        if new_cached != expected_cached:
            raise RuntimeError(
                "unexpected native generation cache length: "
                f"expected {expected_cached} new positions, found {new_cached}"
            )
        extra_cached = new_cached - prompt_len
        new_positions = position_ids
        if extra_cached:
            generated_positions = torch.arange(
                int(position_ids[0, -1]) + 1,
                int(position_ids[0, -1]) + 1 + extra_cached,
                device=position_ids.device,
            ).unsqueeze(0)
            new_positions = torch.cat([new_positions, generated_positions], dim=1)
        cache_positions = new_positions
        if cache is not None:
            cache_positions = torch.cat(
                [cache.layers[0].logical_positions.to(position_ids.device), new_positions], dim=1
            )
        persistent_cache = self._from_dynamic(native_cache, position_ids=cache_positions)

        last_position = int(position_ids[0, -1]) + len(token_ids)
        committed = self.forward(
            input_ids=torch.tensor([[token_ids[-1]]], dtype=torch.long, device=self.device),
            position_ids=torch.tensor([[last_position]], dtype=torch.long, device=self.device),
            cache=persistent_cache,
        )
        persistent_cache = committed.cache

        final_offset = int(persistent_cache.layers[0].logical_positions[0, -1]) + 1
        return BackendGenerationOutput(
            token_ids=token_ids,
            logits=captured_logits,
            cache=persistent_cache,
            final_offset=final_offset,
        )

    @torch.no_grad()
    def prefill_caches(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> list[KVCache]:
        """Prefill a batch and split its native cache into padding-free items.

        Args:
            input_ids: Batched token IDs, mutually exclusive with ``inputs_embeds``.
            inputs_embeds: Batched embeddings, mutually exclusive with ``input_ids``.
            attention_mask: Valid-token mask. All positions are valid when omitted.
            position_ids: Optional explicit source positions.

        Returns:
            Independent per-example caches in batch order.

        Raises:
            ValueError: If inputs are absent, or compression shortens a padded batch.
                Compressed sources with padding must be materialized independently.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("provide input_ids or inputs_embeds")

        source = input_ids if input_ids is not None else inputs_embeds
        assert source is not None
        batch, seq_len = source.shape[0], source.shape[1]
        device = self.device

        if attention_mask is None:
            attention_mask = torch.ones((batch, seq_len), dtype=torch.long, device=device)

        base_positions = (
            position_ids
            if position_ids is not None
            else torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        )
        positions = [
            base_positions[index].to(device)[attention_mask[index].bool()].unsqueeze(0)
            for index in range(batch)
        ]

        full = self.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=base_positions,
        ).cache
        is_compressed = full.seq_len != seq_len
        if is_compressed and not bool(torch.all(attention_mask)):
            raise ValueError(
                "compressed batched prefill does not support padding; "
                "materialize each source independently"
            )

        caches: list[KVCache] = []
        for index in range(batch):
            item_mask = attention_mask[index].bool()
            layers: list[KeyValue] = []
            for layer in full.layers:
                if is_compressed:
                    key = layer.key[index : index + 1]
                    value = layer.value[index : index + 1]
                else:
                    key = layer.key[index : index + 1, :, item_mask, :]
                    value = layer.value[index : index + 1, :, item_mask, :]
                layers.append(KeyValue(key, value, positions[index]))
            caches.append(KVCache(layers))
        return caches

    # The transformers ``DynamicCache`` does not carry position ids; ``KVCache`` owns
    # them. This is the only place in the codebase that knows the HF cache shape.
    def _from_dynamic(self, cache: Cache, position_ids: Tensor | None = None) -> KVCache:
        """Convert a native cache, defaulting missing positions to contiguous IDs."""
        layers: list[KeyValue] = []
        for layer in cache.layers:
            # HF types `layers` as a union that includes linear-attention layers with no
            # keys/values; we only ever receive standard-attention caches (asserted next).
            key, value = layer.keys, layer.values  # ty: ignore[unresolved-attribute]
            assert isinstance(key, Tensor) and isinstance(value, Tensor)
            if position_ids is None:
                pos = torch.arange(key.size(2), dtype=torch.long, device=key.device).unsqueeze(0)
            else:
                pos = position_ids.to(key.device)
            layers.append(KeyValue(key, value, pos))
        return KVCache(layers)

    def _to_dynamic(self, kv: KVCache) -> DynamicCache:
        """Wrap KV Packet key/value tensors in a native ``DynamicCache``.

        Conversion drops logical position IDs and reuses key/value tensors without
        copying. The caller must therefore pass explicit positions when extending a
        cache whose logical positions differ from contiguous cache indices.
        """
        hf = DynamicCache(config=self.config)
        for layer_idx, layer in enumerate(kv.layers):
            hf.update(key_states=layer.key, value_states=layer.value, layer_idx=layer_idx)
        return hf

    def __repr__(self) -> str:
        return (
            f"HFBackend(type={getattr(self.config, 'model_type', '?')}, "
            f"layers={self.num_layers}, hidden={self.hidden_size}, nope_dim={self.nope_dim}, "
            f"sliding_window={self.sliding_window}, rerotatable={self.is_rerotatable})"
        )
