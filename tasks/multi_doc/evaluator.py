"""RAG (multi-doc) evaluation — all three methods on the one engine.

The reference's separate ``kv_packet_eval`` / ``no_recompute`` / ``full_recompute``
functions collapse into a single :class:`PacketSession` configured three ways:

* **FULL_RECOMPUTE** (oracle bound) — no chunks registered, so the whole context is
  forwarded fresh with full causal attention. FLOPs = the document prefill cost.
* **NO_RECOMPUTE** (naive bound) — documents registered as chunks with **no wrapper**:
  raw per-doc caches concatenated and re-rotated to contiguous positions. FLOPs =
  re-rotation only.
* **KV_PACKET** (the method) — documents registered as chunks with a trained
  ``PacketWrapper``: each doc served from its packet (header+body+trailer),
  re-rotated to its offset. FLOPs = re-rotation only.
Only these two reference bounds are kept — no competing baselines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from kvpacket.backend import HFBackend
from kvpacket.packet import PacketWrapper, RawText, TextSourceEncoder, WrapperEntry, WrapperRegistry
from kvpacket.preprocessing import PacketPreprocessor
from kvpacket.session import MixedPrefillPolicy, Mode, PacketSession

from ..common.metrics import TransformerFlops, f1_states
from .data.base import AnswerPostprocess, RetrievalSample

__all__ = ["Method", "SampleResult", "evaluate_sample", "evaluate_dataset"]


class Method(str, Enum):
    FULL_RECOMPUTE = "full_recompute"
    NO_RECOMPUTE = "no_recompute"
    KV_PACKET = "kv_packet"


@dataclass
class SampleResult:
    method: str
    tp: int
    fp: int
    fn: int
    ttft: float
    flops: int
    online_total_flops: int
    baseline_gpu_bytes: int
    peak_gpu_bytes: int
    peak_gpu_delta_bytes: int
    raw_pred: str
    pred: str
    gold: str
    num_generated: int

    @property
    def exact(self) -> bool:
        return self.pred.strip() == self.gold.strip() and bool(self.gold.strip())


def _packet_store(sample, backend, tokenizer, wrapper: PacketWrapper):
    """Build offline document packets for one evaluation sample."""
    registry = WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={"document": WrapperEntry(wrapper)},
    )
    preprocessor = PacketPreprocessor(
        backend=backend,
        wrapper_registry=registry,
        source_encoder=TextSourceEncoder(tokenizer),
    )
    for index, document in enumerate(sample.documents):
        source_id = f"doc{index}"
        if isinstance(document, str):
            preprocessor.preprocess({RawText(document, source_id=source_id): "document"})
        else:
            source_text = tokenizer.decode(document, skip_special_tokens=False)
            preprocessor.preprocess_pretokenized(
                RawText(source_text, source_id=source_id),
                document,
                wrapper_id="document",
            )
    return preprocessor.store


def evaluate_sample(
    sample: RetrievalSample,
    backend: HFBackend,
    tokenizer,
    method: Method,
    *,
    wrapper: PacketWrapper | None = None,
    max_new_tokens: int = 32,
    answer_postprocess: AnswerPostprocess | None = None,
    match: str = "token",
    cache_preamble_before_ttft: bool = False,
    mixed_prefill_policy: MixedPrefillPolicy | str = MixedPrefillPolicy.SEQUENTIAL,
    decoding_kwargs: Mapping[str, Any] | None = None,
) -> SampleResult:
    """Evaluate one already-formatted retrieval sample through a session."""
    method = Method(method)
    use_wrapper = None
    store = None
    if method is not Method.FULL_RECOMPUTE:
        if method is Method.KV_PACKET:
            if wrapper is None:
                raise ValueError("KV_PACKET evaluation requires a wrapper")
            use_wrapper = wrapper
        else:
            use_wrapper = PacketWrapper(0, 0, backend.hidden_size, requires_grad=False)
        store = _packet_store(sample, backend, tokenizer, use_wrapper)
    sess = PacketSession(
        backend=backend,
        store=store,
        mode=Mode.SERVE,
        match=match,
        tokenizer=tokenizer,
        mixed_prefill_policy=mixed_prefill_policy,
    )
    if store is not None:
        sess.register_store_chunks()
    if cache_preamble_before_ttft and sample.preamble:
        # A stable system/history prefix is part of the reusable historical cache,
        # not the online request whose TTFT we are measuring.
        sess.push(sample.preamble, match=False)
        sess.process()
    elif sample.preamble:
        sess.push(sample.preamble)
    for doc in sample.documents:
        # FULL_RECOMPUTE: no chunks -> these are forwarded fresh (oracle).
        sess.push(doc)
    sess.push(sample.task_prompt)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        baseline_gpu_bytes = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_gpu_bytes = 0
    res = sess.generate(
        max_new_tokens=max_new_tokens,
        **dict(decoding_kwargs or {}),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
    else:
        peak_gpu_bytes = 0
    raw_pred = res.text
    pred, gold = raw_pred, sample.answer
    if answer_postprocess is not None:
        pred, gold = answer_postprocess(pred, gold)

    tp, fp, fn = f1_states(gold.split(), pred.split())
    flops = res.flops
    if method is Method.FULL_RECOMPUTE:
        flops = _doc_prefill_flops(backend, tokenizer, sample)
    online_total_flops = _online_total_flops(
        backend,
        tokenizer,
        sample,
        method,
        wrapper=use_wrapper,
        rerotation_flops=res.flops,
        cache_preamble_before_ttft=cache_preamble_before_ttft,
    )

    return SampleResult(
        method=method.value,
        tp=tp,
        fp=fp,
        fn=fn,
        ttft=res.ttft,
        flops=int(flops),
        online_total_flops=online_total_flops,
        baseline_gpu_bytes=baseline_gpu_bytes,
        peak_gpu_bytes=peak_gpu_bytes,
        peak_gpu_delta_bytes=max(0, peak_gpu_bytes - baseline_gpu_bytes),
        raw_pred=raw_pred,
        pred=pred,
        gold=gold,
        num_generated=len(res.token_ids),
    )


def evaluate_dataset(
    samples,
    backend: HFBackend,
    tokenizer,
    method: Method,
    *,
    wrapper: PacketWrapper | None = None,
    max_new_tokens: int = 32,
    answer_postprocess: AnswerPostprocess | None = None,
    match: str = "token",
    cache_preamble_before_ttft: bool = False,
    mixed_prefill_policy: MixedPrefillPolicy | str = MixedPrefillPolicy.SEQUENTIAL,
    decoding_kwargs: Mapping[str, Any] | None = None,
):
    """Yield a :class:`SampleResult` per sample (drive an accumulator upstream)."""
    for sample in samples:
        yield evaluate_sample(
            sample,
            backend,
            tokenizer,
            method,
            wrapper=wrapper,
            max_new_tokens=max_new_tokens,
            answer_postprocess=answer_postprocess,
            match=match,
            cache_preamble_before_ttft=cache_preamble_before_ttft,
            mixed_prefill_policy=mixed_prefill_policy,
            decoding_kwargs=decoding_kwargs,
        )


def _doc_prefill_flops(backend: HFBackend, tokenizer, s: RetrievalSample) -> int:
    """Analytic cost of prefilling the document tokens (what KV Packet avoids)."""
    doc_len = sum(
        len(tokenizer(d, add_special_tokens=False)["input_ids"]) if isinstance(d, str) else len(d)
        for d in s.documents
    )
    pre_len = len(tokenizer(s.preamble, add_special_tokens=False)["input_ids"]) if s.preamble else 0
    return TransformerFlops(backend.config).total_flops(seq_len=doc_len, cache_len=pre_len)


def _online_total_flops(
    backend: HFBackend,
    tokenizer,
    sample: RetrievalSample,
    method: Method,
    *,
    wrapper: PacketWrapper | None,
    rerotation_flops: int,
    cache_preamble_before_ttft: bool,
) -> int:
    """Count the actual online forwards plus cache re-rotation."""
    calculator = TransformerFlops(backend.config)
    preamble_len = (
        len(tokenizer(sample.preamble, add_special_tokens=False)["input_ids"])
        if sample.preamble
        else 0
    )
    document_lengths = [
        len(tokenizer(document, add_special_tokens=False)["input_ids"])
        if isinstance(document, str)
        else len(document)
        for document in sample.documents
    ]
    task_len = len(tokenizer(sample.task_prompt, add_special_tokens=False)["input_ids"])
    if method is Method.FULL_RECOMPUTE:
        document_len = sum(document_lengths)
        prefix_len = preamble_len + document_len
        total = (
            calculator.total_flops(
                seq_len=document_len,
                cache_len=preamble_len if cache_preamble_before_ttft else 0,
            )
            if document_len
            else 0
        )
        if preamble_len and not cache_preamble_before_ttft:
            total = calculator.total_flops(seq_len=prefix_len)
        return total + calculator.total_flops(
            seq_len=task_len,
            cache_len=prefix_len,
        )

    wrapped_extra = 0
    if method is Method.KV_PACKET:
        if wrapper is None:
            raise ValueError("KV Packet online FLOPs require a wrapper")
        wrapped_extra = len(document_lengths) * (wrapper.header_len + wrapper.trailer_len)
    cache_len = preamble_len + sum(document_lengths) + wrapped_extra
    total = (
        calculator.total_flops(seq_len=preamble_len)
        if preamble_len and not cache_preamble_before_ttft
        else 0
    )
    total += calculator.total_flops(seq_len=task_len, cache_len=cache_len)
    return total + int(rerotation_flops)
