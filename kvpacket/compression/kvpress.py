"""Apply supported KVPress token-pruning methods during packet materialization."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import Tensor

from ..backend import ComputeBackend, HFBackend
from ..fingerprint import FingerprintValue
from .base import CompressionSpec

__all__ = ["SUPPORTED_KVPRESS_METHODS", "KVPressCompressor"]

_KVPRESS_CLASSES = {
    "expected_attention": "ExpectedAttentionPress",
    "key_diff": "KeyDiffPress",
    "knorm": "KnormPress",
    "lagkv": "LagKVPress",
    "random": "RandomPress",
    "snapkv": "SnapKVPress",
    "streaming_llm": "StreamingLLMPress",
}
SUPPORTED_KVPRESS_METHODS = frozenset(_KVPRESS_CLASSES)


class KVPressCompressor:
    """Configure one query-independent KVPress method for offline packet caches.

    KVPress removes a fraction of token positions from each layer during the isolated
    document prefill. The full wrapped-document span remains packet metadata; only the
    physical key/value sequence is shortened. No header or trailer positions receive
    special treatment.

    Args:
        method: Supported KVPress method name.
        compression_ratio: Fraction of KV positions removed, in ``[0, 1)``.
        options: Additional JSON-compatible constructor arguments for the press.

    Raises:
        ImportError: If the optional KVPress dependency is unavailable.
        ValueError: If the method, ratio, or options are invalid.
    """

    def __init__(
        self,
        method: str,
        compression_ratio: float,
        *,
        options: Mapping[str, FingerprintValue] | None = None,
    ) -> None:
        if method not in _KVPRESS_CLASSES:
            raise ValueError(
                f"unsupported KVPress method {method!r}; "
                f"choose one of {sorted(SUPPORTED_KVPRESS_METHODS)}"
            )
        if isinstance(compression_ratio, bool) or not 0 <= compression_ratio < 1:
            raise ValueError("compression_ratio must be in [0, 1)")
        try:
            implementation_version = version("kvpress")
        except PackageNotFoundError as exc:
            raise ImportError(
                "KVPress compression requires the optional dependency; install kvpacket[compress]"
            ) from exc

        raw_options = {} if options is None else dict(options)
        if "compression_ratio" in raw_options:
            raise ValueError("pass compression_ratio directly, not through options")
        self._spec = CompressionSpec(
            provider="kvpress",
            method=method,
            implementation_version=implementation_version,
            config={
                "compression_ratio": float(compression_ratio),
                "options": raw_options,
            },
        )

    @property
    def spec(self) -> CompressionSpec:
        """Persistent KVPress method, version, and constructor parameters."""
        return self._spec

    @contextmanager
    def activate(self, backend: ComputeBackend) -> Iterator[None]:
        """Install KVPress hooks around one Hugging Face offline prefill.

        A fresh press is constructed for each packet so stateful methods cannot leak
        observations from one source into another.

        Raises:
            TypeError: If the backend is not ``HFBackend`` or KVPress exposes an
                incompatible press implementation.
        """
        if not isinstance(backend, HFBackend):
            raise TypeError("KVPressCompressor currently requires HFBackend")
        if backend.sliding_window is not None:
            raise ValueError(
                "KVPress packet compression currently requires global attention; "
                "sliding-window cache visibility has different position semantics"
            )
        module = import_module("kvpress")
        class_name = _KVPRESS_CLASSES[self.spec.method]
        press_type = getattr(module, class_name, None)
        if not callable(press_type):
            raise TypeError(f"installed KVPress does not expose callable {class_name}")
        raw_options = self.spec.config["options"]
        if not isinstance(raw_options, dict):
            raise TypeError("normalized KVPress options are not a mapping")
        press = press_type(
            compression_ratio=self.spec.config["compression_ratio"],
            **raw_options,
        )

        # Transformers 5.3 stopped forwarding ``cache_position`` to attention
        # modules, while KVPress 0.5.x still uses it to recognize the prefill.
        # Supply the ordinary contiguous cache coordinate only while its hooks are
        # active; logical packet positions remain owned by KV Packet metadata.
        def supply_cache_position(module, args, kwargs):
            del module
            if "cache_position" not in kwargs:
                hidden_states = kwargs.get("hidden_states")
                if not isinstance(hidden_states, Tensor):
                    hidden_states = args[0] if args else None
                if not isinstance(hidden_states, Tensor):
                    raise TypeError("KVPress attention hook did not receive hidden states")
                kwargs["cache_position"] = torch.arange(
                    hidden_states.shape[1],
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            return args, kwargs

        base = backend.model.model
        language_model = getattr(base, "language_model", base)
        layers = getattr(language_model, "layers", None)
        if layers is None:
            raise TypeError("KVPress requires model.model.layers")
        pre_hooks = [
            layer.self_attn.register_forward_pre_hook(
                supply_cache_position,
                with_kwargs=True,
            )
            for layer in layers
        ]
        try:
            with press(backend.model):
                yield
        finally:
            for hook in pre_hooks:
                hook.remove()
