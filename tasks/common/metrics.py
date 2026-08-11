"""Token-F1, aggregation, and an analytic transformer-FLOPs calculator.

``f1_states`` / ``calculate_metrics`` are the reference bag-of-words token metric.
``MetricAccumulator`` sums per-sample counts and latency/FLOPs into a summary.
``TransformerFlops`` is a config-driven forward-pass FLOPs estimate (Llama/Qwen
family), used to quantify the ``full_recompute`` document-prefill cost that KV
Packet avoids.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

__all__ = ["f1_states", "calculate_metrics", "MetricAccumulator", "TransformerFlops"]


def f1_states(gold_tokens: list[str], pred_tokens: list[str]) -> tuple[int, int, int]:
    """Bag-of-words (tp, fp, fn) between gold and predicted tokens."""
    common = Counter(gold_tokens) & Counter(pred_tokens)
    tp = sum(common.values())
    return tp, len(pred_tokens) - tp, len(gold_tokens) - tp


def calculate_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


@dataclass
class MetricAccumulator:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n: int = 0
    ttft_sum: float = 0.0
    flops_sum: int = 0
    exact: int = 0
    _f1s: list[float] = field(default_factory=list)

    def add(
        self, tp: int, fp: int, fn: int, ttft: float = 0.0, flops: int = 0, exact: bool = False
    ) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.n += 1
        self.ttft_sum += ttft
        self.flops_sum += flops
        self.exact += int(exact)
        self._f1s.append(calculate_metrics(tp, fp, fn)[2])

    def summary(self) -> dict:
        micro_p, micro_r, micro_f1 = calculate_metrics(self.tp, self.fp, self.fn)
        macro_f1 = sum(self._f1s) / len(self._f1s) if self._f1s else 0.0
        return {
            "n": self.n,
            "f1": micro_f1,  # micro (token-pooled) F1
            "macro_f1": macro_f1,  # mean per-sample F1
            "precision": micro_p,
            "recall": micro_r,
            "exact_match": (self.exact / self.n) if self.n else 0.0,
            "ttft_ms": (self.ttft_sum / self.n * 1e3) if self.n else 0.0,
            "flops": (self.flops_sum / self.n) if self.n else 0.0,
        }


class TransformerFlops:
    """Config-driven forward-pass FLOP estimate for a decoder layer stack.

    Counts one prefill of ``seq_len`` new tokens attending over ``cache_len`` cached
    positions (the reference accounting). Includes the Qwen3 QK-norm term when the
    model exposes per-head q/k norms.
    """

    def __init__(self, config):
        self.c = config

    def _get(self, name, default=None):
        v = getattr(self.c, name, default)
        if v is None and default is None:
            raise ValueError(f"config missing {name!r}")
        return v

    def decoder_layer_flops(self, batch_size: int, seq_len: int, cache_len: int = 0) -> int:
        H = int(self._get("hidden_size"))
        nh = int(self._get("num_attention_heads"))
        hd = int(getattr(self.c, "head_dim", None) or H // nh)
        nkv = int(self._get("num_key_value_heads", nh))
        inter = int(self._get("intermediate_size"))

        q = 2 * H * nh * hd * seq_len
        k = 2 * H * nkv * hd * seq_len
        v = k
        o = q
        rope = 3 * seq_len * hd * (nh + nkv)
        attn = 4 * seq_len * (seq_len + cache_len) * hd * nh
        mlp = 6 * seq_len * H * inter
        other = 10 * seq_len * H

        flops = q + k + v + o + rope + attn + mlp + other
        mtype = str(getattr(self.c, "model_type", ""))
        if mtype.startswith("qwen3"):  # QK RMSNorm applied post-projection
            flops += 4 * seq_len * (nh + nkv) * hd
        return flops * batch_size

    def total_flops(self, seq_len: int, cache_len: int = 0, batch_size: int = 1) -> int:
        L = int(self._get("num_hidden_layers"))
        return self.decoder_layer_flops(batch_size, seq_len, cache_len) * L
