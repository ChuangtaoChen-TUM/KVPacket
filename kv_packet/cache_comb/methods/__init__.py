""" KV Combination Methods """
from ..abc import EvalCombFunc
from .default import full_context_eval, no_cache_eval, full_recompute, single_cache
from .no_recompute import no_recompute_eval
from .cache_blend import cache_blend_eval
from .epic import epic_eval
from .sink import sink_eval
from .rand_recompute import rand_recompute_eval
from .kv_packet import kv_packet_eval
from .sam_kv import sam_kv_eval
from .a3 import a3_eval

CACHE_COMB_FUNC_DICT: dict[str, EvalCombFunc] = {
    "full_context": full_context_eval,
    "no_cache": no_cache_eval,
    "full_recompute": full_recompute,
    "no_recompute": no_recompute_eval,
    "cache_blend": cache_blend_eval,
    "epic": epic_eval,
    "sink": sink_eval,
    "rand_recompute": rand_recompute_eval,
    "kv_packet": kv_packet_eval,
    "sam_kv": sam_kv_eval,
    "a3": a3_eval,
    "single_cache": single_cache,
}

def get_cache_comb_func(name: str) -> EvalCombFunc:
    if name not in CACHE_COMB_FUNC_DICT:
        raise ValueError(f"Unsupported cache combination method: {name}")
    return CACHE_COMB_FUNC_DICT[name]


__all__ = [
    "CACHE_COMB_FUNC_DICT",
    "get_cache_comb_func",
]