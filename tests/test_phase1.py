"""Phase 1 verification for the KV Packet core primitives.

Runnable two ways:
  * ``uv run pytest tests/test_phase1.py``          (light tests; no model download)
  * ``python tests/test_phase1.py``                 (also runs the live-model check)

The light tests build a tiny rotary embedding from a config (no weights) and
assert the re-rotation identity exactly; the ``__main__`` block additionally loads
the locally cached ``Qwen/Qwen2.5-1.5B`` for an end-to-end plumbing check.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpacket.kvcache.cache import KeyValue, KVCache, concat_caches
from kvpacket.kvcache.positions import cumsum_pos, shifted_cumsum_pos
from kvpacket.kvcache.rope import apply_rotary, rerotate_flops, rerotate_keys
from kvpacket.packet.wrapper import PacketWrapper


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


def test_rerotate_identity():
    """R(Δ)·R(p_old) == R(p_new): re-rotating equals rotating at the new positions."""
    torch.manual_seed(0)
    device = "cpu"
    B, H, S, D = 2, 4, 6, 16
    rot = _small_rotary(head_dim=D, device=device)

    k0 = torch.randn(B, H, S, D, dtype=torch.float32, device=device)  # unrotated
    old_pos = torch.tensor([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]], device=device)
    new_pos = torch.tensor([[7, 8, 9, 10, 11, 12], [0, 1, 2, 3, 4, 5]], device=device)

    cos_o, sin_o = rot(k0, position_ids=old_pos)
    cos_n, sin_n = rot(k0, position_ids=new_pos)
    k_old = apply_rotary(k0, cos_o, sin_o)
    k_new = apply_rotary(k0, cos_n, sin_n)

    k_rerot = rerotate_keys(k_old, rot, old_pos, new_pos)
    max_err = (k_rerot - k_new).abs().max().item()
    assert max_err < 1e-4, f"re-rotation identity error too large: {max_err}"


def test_rerotate_identity_scaled():
    """Strict shift under RoPE amplitude scaling (YaRN mscale != 1)."""
    torch.manual_seed(0)
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    D = 16
    config = LlamaConfig(
        hidden_size=D * 4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        head_dim=D,
        rope_theta=10000.0,  # ty: ignore[unknown-argument]
        max_position_embeddings=2048,
        rope_scaling={  # ty: ignore[unknown-argument]
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 512,
        },
    )
    rot = LlamaRotaryEmbedding(config)
    a = float(getattr(rot, "attention_scaling", 1.0))
    assert abs(a - 1.0) > 1e-6, f"test misconfigured: expected scaling != 1, got {a}"

    B, H, S = 2, 4, 6
    k0 = torch.randn(B, H, S, D, dtype=torch.float32)
    old_pos = torch.tensor([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]])
    new_pos = torch.tensor([[7, 8, 9, 10, 11, 12], [0, 1, 2, 3, 4, 5]])
    k_old = apply_rotary(k0, *rot(k0, position_ids=old_pos))  # a·R(old)·k0
    k_new = apply_rotary(k0, *rot(k0, position_ids=new_pos))  # a·R(new)·k0

    k_rerot = rerotate_keys(k_old, rot, old_pos, new_pos)  # divides out `a` internally
    err = (k_rerot - k_new).abs().max().item()
    assert err < 1e-4, f"scaled re-rotation error too large: {err} (a={a})"


def test_rerotate_flops():
    assert rerotate_flops((1, 8, 100, 128), nope_dim=None) == 1 * (4 + 3 * 8) * 100 * 128
    assert rerotate_flops((1, 8, 100, 128), nope_dim=64) == 1 * (4 + 3 * 8) * 100 * 64


def test_positions():
    assert cumsum_pos([3, 5, 2]).tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 0, 1]
    assert shifted_cumsum_pos([3, 5, 2], [4, 6, 3]).tolist() == [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]
    # cache_lens == doc_lens -> contiguous (the no-filler / no-recompute case)
    assert shifted_cumsum_pos([3, 5, 2], [3, 5, 2]).tolist() == list(range(10))


def test_wrapper_roundtrip():
    w = PacketWrapper(header_len=8, trailer_len=8, dim=32, mean=0.1, std=0.02)
    assert isinstance(w, torch.nn.Module)
    assert dict(w.named_parameters()).keys() == {"header", "trailer"}
    x = torch.randn(2, 5, 32)
    wrapped = w(x)
    assert wrapped.shape == (2, 5 + 16, 32)
    # 2-D path
    assert w(torch.randn(5, 32)).shape == (5 + 16, 32)
    # Standard nn.Module state preserves values when geometry is known.
    sd = w.state_dict()
    w2 = PacketWrapper(8, 8, 32)
    w2.load_state_dict(sd)
    assert torch.equal(w.header, w2.header) and torch.equal(w.trailer, w2.trailer)
    assert w2.filler_len == 16


def test_concat_caches():
    def dummy(seq, pos_start):
        layers = [
            KeyValue(
                torch.zeros(1, 2, seq, 8),
                torch.zeros(1, 2, seq, 8),
                torch.arange(pos_start, pos_start + seq).unsqueeze(0),
            )
            for _ in range(3)
        ]
        return KVCache(layers)

    c = concat_caches([dummy(3, 0), dummy(5, 0)])
    assert c.num_layers == 3 and c.seq_len == 8
    assert c.layers[0].logical_positions.tolist() == [[0, 1, 2, 0, 1, 2, 3, 4]]


def _run_light():
    test_rerotate_identity()
    print("[ok] rerotate identity")
    test_rerotate_identity_scaled()
    print("[ok] rerotate identity (YaRN scaling)")
    test_rerotate_flops()
    print("[ok] rerotate flops")
    test_positions()
    print("[ok] positions")
    test_wrapper_roundtrip()
    print("[ok] wrapper roundtrip")
    test_concat_caches()
    print("[ok] concat caches")


def _run_live_model():
    """End-to-end plumbing check on the locally cached Qwen2.5-1.5B."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvpacket.backend import HFBackend

    name = "Qwen/Qwen2.5-1.5B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[live] loading {name} on {device} ...")
    tok = AutoTokenizer.from_pretrained(name)
    assert tok is not None
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
    model.eval()
    adapter = HFBackend(model)
    print(f"[live] {adapter}")

    docs = ["The capital of France is Paris.", " Water boils at 100 degrees Celsius."]
    caches = []
    for d in docs:
        ids = tok([d], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        caches.append(adapter.prefill_caches(input_ids=ids)[0])
    lens = [c.doc_len for c in caches]
    print(f"[live] built {len(caches)} doc caches, lens={lens}")

    stitched = concat_caches(caches)
    old = cumsum_pos(lens, device).unsqueeze(0)
    new = shifted_cumsum_pos(lens, lens, device).unsqueeze(0)  # contiguous
    rotated = stitched.rerotate(adapter.rotary_emb, old, new, nope_dim=adapter.nope_dim)
    flops = rotated.rerotate_flops(nope_dim=adapter.nope_dim)
    print(f"[live] stitched+rerotated: {rotated}  rerotate_flops={flops:,}")

    # feed a query against the stitched cache (naive no-recompute serving path)
    q = tok(
        ["Q: What is the capital of France? A:"], return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    total = sum(lens)
    hf = adapter._to_dynamic(rotated)  # white-box: feed the raw HF model directly
    pos = torch.arange(total, total + q.size(1), device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=q, past_key_values=hf, position_ids=pos)
    assert torch.isfinite(out.logits).all(), "non-finite logits from stitched cache"
    nxt = tok.decode(out.logits[0, -1].argmax())
    print(f"[live] forward against stitched cache OK; logits finite; next-token guess={nxt!r}")
    print("[live] plumbing check passed.")


if __name__ == "__main__":
    _run_light()
    if "--no-model" not in sys.argv:
        _run_live_model()
    print("\nAll Phase 1 checks passed.")
