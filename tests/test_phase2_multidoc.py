"""Phase 2 Deliverable 2 — multi-doc task + library pieces.

* ``uv run pytest tests/test_phase2_multidoc.py``   (light: parity vs reference, metrics, data, config)
* ``python tests/test_phase2_multidoc.py``          (also runs the 3-method live eval)
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpacket.kvcache.rope import rerotate_flops, rerotate_keys
from tasks.common.metrics import MetricAccumulator, TransformerFlops, calculate_metrics, f1_states
from tasks.multi_doc.data import (
    apply_chat_template,
    get_dataset,
    get_ret_eval_generator,
    synthetic_generator,
)


def _small_rotary(head_dim: int = 16, device: str = "cpu"):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    config = LlamaConfig(
        hidden_size=head_dim * 4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        head_dim=head_dim,
    )
    return LlamaRotaryEmbedding(config).to(device)


def test_rerotate_parity_with_reference():
    """Our re-rotation == the (vendored) reference ``rerotate_embeddings`` bitwise."""
    from tests.reference.rerotate_reference import rerotate_embeddings

    torch.manual_seed(0)
    B, H, S, D = 1, 4, 7, 16
    rot = _small_rotary(head_dim=D)
    k = torch.randn(B, H, S, D)
    old = torch.arange(S).unsqueeze(0)
    new = old + 13
    mine = rerotate_keys(k, rot, old, new)
    ref = rerotate_embeddings(k, rot, old, new, nope_dim=None)
    max_diff = (mine - ref).abs().max().item()
    assert max_diff == 0.0, f"re-rotation differs from reference: {max_diff}"

    # FLOPs formula parity (reference cal_layer_flops uses the identical expression)
    assert rerotate_flops((B, H, S, D), nope_dim=None) == B * (4 + 3 * H) * S * D
    print(f"[ok] rerotate parity vs reference (max_diff={max_diff})")


def test_metrics():
    assert f1_states(["a", "b", "c"], ["a", "b", "x"]) == (2, 1, 1)
    p, r, f1 = calculate_metrics(2, 1, 1)
    assert abs(f1 - 2 / 3) < 1e-9
    acc = MetricAccumulator()
    acc.add(2, 1, 1, ttft=0.01, flops=100, exact=False)
    acc.add(3, 0, 0, ttft=0.02, flops=200, exact=True)
    s = acc.summary()
    assert s["n"] == 2 and s["exact_match"] == 0.5
    assert s["flops"] == 150 and abs(s["ttft_ms"] - 15.0) < 1e-6


def test_transformer_flops_scales():
    class Cfg:
        hidden_size = 1536
        num_attention_heads = 12
        num_key_value_heads = 2
        head_dim = 128
        intermediate_size = 8960
        num_hidden_layers = 28
        model_type = "qwen2"

    tf = TransformerFlops(Cfg())
    f100 = tf.total_flops(seq_len=100, cache_len=0)
    f200 = tf.total_flops(seq_len=200, cache_len=0)
    assert f100 > 0 and f200 > 2 * f100  # attention grows super-linearly in seq_len


def test_synthetic_generator_answer_in_docs():
    for s in synthetic_generator(num_samples=6, num_documents=4, seed=1):
        assert len(s.documents) == 4
        assert any(s.answer in d for d in s.documents), "gold answer must appear in a doc"


def test_apply_chat_template_preserves_packet_boundaries():
    from typing import cast

    from tasks.multi_doc.data.base import RetrievalSample
    from tasks.multi_doc.data.templates import TokenizerType

    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            tokenize,
            add_generation_prompt,
            enable_thinking,
        ):
            assert messages == [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "{DOCUMENT_PLACEHOLDER}"},
            ]
            assert tokenize is False and add_generation_prompt is True
            assert enable_thinking is False
            return "<chat>{DOCUMENT_PLACEHOLDER}</chat><assistant>"

    s = RetrievalSample(
        preamble="P",
        documents=["d1"],
        task_prompt="Q",
        query="query",
        answer="answer",
        meta={"source": "test"},
    )
    tokenizer = cast(TokenizerType, FakeTokenizer())
    q = apply_chat_template(
        s,
        tokenizer,
        system_prompt="sys",
        chat_template_kwargs={"enable_thinking": False},
    )
    assert q.preamble == "<chat>P"
    assert q.documents == ["d1"]
    assert q.task_prompt == "Q</chat><assistant>"
    assert q.query == "query" and q.answer == "answer" and q.meta == {"source": "test"}


def test_generator_applies_chat_template_when_tokenizer_is_provided():
    from typing import cast

    from tasks.multi_doc.data.templates import TokenizerType

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "{DOCUMENT_PLACEHOLDER}"}]
            assert tokenize is False and add_generation_prompt is True
            return "<user>{DOCUMENT_PLACEHOLDER}</user><assistant>"

    tokenizer = cast(TokenizerType, FakeTokenizer())
    samples = list(
        get_ret_eval_generator(
            name="synthetic",
            num_samples=2,
            num_documents=3,
            num_shots=0,
            subset="default",
            split="test",
            seed=7,
            tokenizer=tokenizer,
        )
    )
    assert len(samples) == 2
    assert all(sample.preamble.startswith("<user>") for sample in samples)
    assert all(sample.task_prompt.endswith("</user><assistant>") for sample in samples)
    assert all(len(sample.documents) == 3 for sample in samples)


def _run_light():
    test_rerotate_parity_with_reference()
    test_metrics()
    print("[ok] metrics + accumulator")
    test_transformer_flops_scales()
    print("[ok] transformer flops scaling")
    test_synthetic_generator_answer_in_docs()
    print("[ok] synthetic generator")
    test_apply_chat_template_preserves_packet_boundaries()
    print("[ok] chat scaffolding")
    test_generator_applies_chat_template_when_tokenizer_is_provided()
    print("[ok] generator-level chat formatting")


def _run_live():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvpacket.backend import HFBackend
    from kvpacket.packet.wrapper import PacketWrapper
    from tasks.multi_doc.evaluator import Method, evaluate_sample

    name = "Qwen/Qwen2.5-1.5B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[live] loading {name} ...")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
    model.eval()
    adapter = HFBackend(model)

    spec = get_dataset("synthetic")
    samples = list(spec.generator(num_samples=3, num_documents=3, seed=0))
    wrapper = PacketWrapper(
        header_len=4,
        trailer_len=4,
        dim=adapter.hidden_size,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=False,
    )

    per_method = {}
    for method in [Method.FULL_RECOMPUTE, Method.NO_RECOMPUTE, Method.KV_PACKET]:
        acc = MetricAccumulator()
        for s in samples:
            r = evaluate_sample(
                s,
                adapter,
                tok,
                method,
                wrapper=wrapper,
                max_new_tokens=6,
                answer_postprocess=spec.answer_postprocess,
            )
            acc.add(r.tp, r.fp, r.fn, ttft=r.ttft, flops=r.flops, exact=r.exact)
        per_method[method.value] = acc.summary()
        print(f"[live] {method.value:<15} {per_method[method.value]}")

    # engine plumbing: all three ran; the FLOPs claim holds by orders of magnitude.
    assert per_method["full_recompute"]["flops"] > 100 * per_method["kv_packet"]["flops"]
    assert per_method["kv_packet"]["n"] == 3 and per_method["no_recompute"]["n"] == 3
    print("[live] D2 multi-doc: all methods ran; full_recompute FLOPs >> kv_packet FLOPs.")


if __name__ == "__main__":
    _run_light()
    if "--no-model" not in sys.argv:
        _run_live()
    print("\nAll Phase 2 multi-doc checks passed.")
