""" KV compression methods and utilities """
from torch import Tensor
from dowhen import when
from dowhen.handler import EventHandler
from kvpress import (
    ScorerPress,
    RandomPress,
    KnormPress,
    SnapKVPress,
    ExpectedAttentionPress,
    StreamingLLMPress,
    TOVAPress,
    ObservedAttentionPress,
    QFilterPress,
    PyramidKVPress,
    LagKVPress,
    KeyDiffPress,
    NonCausalAttnPress,
    LeverageScorePress,
    CompactorPress,
    CURPress,
    KVzapPress,
)

PRESS_CLASSES: dict[str, type[ScorerPress]] = {
    "random": RandomPress,
    "knorm": KnormPress,
    "snapkv": SnapKVPress,
    "expected_attention": ExpectedAttentionPress,
    "streaming_llm": StreamingLLMPress,
    "tova": TOVAPress,
    "observed_attention": ObservedAttentionPress,
    "qfilter": QFilterPress,
    "pyramidkv": PyramidKVPress,
    "lagkv": LagKVPress,
    "keydiff": KeyDiffPress,
    "non_causal_attn": NonCausalAttnPress,
    "leverage_score": LeverageScorePress,
    "compactor": CompactorPress,
    "cur": CURPress,
    "kvzap": KVzapPress,
}


class KeepIndex:
    """ A utility class to keep certain indices during compression by setting their scores to inf
    Usage:
        with KeepIndex(indices_to_keep=[0, 5, 10]):
            # During this block, the specified indices will be kept during compression
            ...
        After the block, the original compression behavior is restored.
    """
    def __init__(self, indices_to_keep: list[int]|None):
        self._inst_line = "k_len = keys.shape[2]"
        self._handler: EventHandler|None = None
        self.indices_to_keep = indices_to_keep
        self._fix_func = self.get_keep_func()


    def bind(self):
        if self._handler is not None:
            raise RuntimeError("KeepIndex is already bound.")

        self._fix_handler = when(
            ScorerPress.compress,
            self._inst_line,
        ).do(self._fix_func)


    def unbind(self):
        if self._fix_handler is None:
            raise RuntimeError("KeepIndex is not bound.")
        self._fix_handler.remove()
        self._fix_handler = None
    

    def get_keep_func(self):
        def fix_func(scores: Tensor) -> dict:
            if self.indices_to_keep is not None:
                scores[..., self.indices_to_keep] = float("inf")
            return {"scores": scores}
        return fix_func


    def __del__(self):
        if self._fix_handler is not None:
            self.unbind()
    

    def __enter__(self):
        self.bind()
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.unbind()
        pass
