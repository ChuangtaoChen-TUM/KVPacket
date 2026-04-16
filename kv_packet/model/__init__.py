from transformers import (
    LlamaForCausalLM,
    Qwen3ForCausalLM,
)
from typing import TypeAlias

SupportedModel: TypeAlias = \
    LlamaForCausalLM | \
    Qwen3ForCausalLM
