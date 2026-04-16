from transformers import (
    LlamaForCausalLM,
    Qwen3ForCausalLM,
)
from ...model import SupportedModel

class BaseFlopsCalculator:
    def decoder_layer_flops(
        self,
        batch_size: int,
        seq_len: int,
        cache_len: int=0
    ) -> int:
        raise NotImplementedError()


    def total_flops(
        self,
        batch_size: int,
        seq_len: int,
        cache_len: int=0
    ) -> int:
        raise NotImplementedError()


class AutoFlopsCalculator(BaseFlopsCalculator):
    def __init__(self, model: SupportedModel):
        if isinstance(model, LlamaForCausalLM):
            self.calculator = LlamaFlopsCalculator(model)
        elif isinstance(model, Qwen3ForCausalLM):
            self.calculator = Qwen3FlopsCalculator(model)
        else:
            raise ValueError(f"Unsupported model type for FLOPS calculation: {type(model)}")


    def decoder_layer_flops(
        self,
        batch_size: int,
        seq_len: int,
        cache_len: int=0
    ) -> int:
        return self.calculator.decoder_layer_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len
        )


    def total_flops(
        self,
        batch_size: int,
        seq_len: int,
        cache_len: int=0
    ) -> int:
        return self.calculator.total_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len
        )


def get_int(value: int|None) -> int:
    if value is None:
        raise ValueError("Expected an integer value, but got None.")
    return value


class LlamaFlopsCalculator(BaseFlopsCalculator):
    def __init__(self, model: LlamaForCausalLM):
        self.config = model.config

    def decoder_layer_flops(
        self,
        batch_size: int,
        seq_len: int,
        cache_len: int=0
    ) -> int:
        num_flops: int = 0

        hidden_size = get_int(self.config.hidden_size)
        num_attention_heads = get_int(self.config.num_attention_heads)
        head_dim = get_int(self.config.head_dim)

        num_key_value_heads = get_int(self.config.num_key_value_heads)
        intermediate_size = get_int(self.config.intermediate_size)

        # Q, K, V projections
        q_proj_flops = 2 * hidden_size * num_attention_heads * head_dim * seq_len
        k_proj_flops = 2 * hidden_size * num_key_value_heads * head_dim * seq_len
        v_proj_flops = k_proj_flops
        o_proj_flops = q_proj_flops

        rope_flops = 3 * seq_len * head_dim * (
            num_attention_heads + num_key_value_heads
        )

        attn_flops = 4 * seq_len * (seq_len + cache_len) \
            * head_dim * num_attention_heads
        
        mlp_flops = 6 * seq_len * hidden_size * intermediate_size

        # layer norm and residual
        other_flops = 8 * seq_len * hidden_size + 2 * seq_len * hidden_size

        num_flops += (
            q_proj_flops + k_proj_flops + v_proj_flops + o_proj_flops
            + rope_flops + attn_flops + mlp_flops + other_flops
        ) * batch_size

        return num_flops


    def total_flops(
        self,
        batch_size: int,
        seq_len: int,
        cache_len: int=0
    ) -> int:
        total_flops: int = 0
        num_layers = get_int(self.config.num_hidden_layers)

        total_flops += self.decoder_layer_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len
        ) * num_layers

        return total_flops


class Qwen3FlopsCalculator(BaseFlopsCalculator):
    def __init__(self, model: Qwen3ForCausalLM):
        self.config = model.config

    def decoder_layer_flops(
        self,
        batch_size: int,
        seq_len: int,
        cache_len: int=0
    ) -> int:
        num_flops: int = 0

        hidden_size = get_int(self.config.hidden_size)
        num_attention_heads = get_int(self.config.num_attention_heads)

        # Qwen3 logic: head_dim is derived if not explicit
        if hasattr(self.config, "head_dim") and self.config.head_dim is not None:
            head_dim = get_int(self.config.head_dim)
        else:
            head_dim = hidden_size // num_attention_heads
        assert isinstance(head_dim, int)

        num_key_value_heads = get_int(self.config.num_key_value_heads)
        assert isinstance(num_key_value_heads, int)

        intermediate_size = self.config.intermediate_size
        assert isinstance(intermediate_size, int)

        # 1. Q, K, V Projections
        q_proj_flops = 2 * hidden_size * num_attention_heads * head_dim * seq_len
        k_proj_flops = 2 * hidden_size * num_key_value_heads * head_dim * seq_len
        v_proj_flops = k_proj_flops
        o_proj_flops = q_proj_flops

        # 2. QK Norm (RMSNorm) - Specific to Qwen 3
        # Applied to Q and K *after* projection but *before* RoPE.
        # Estimate: ~4 FLOPs per element (square, sum, sqrt, mult)
        q_norm_flops = 4 * seq_len * num_attention_heads * head_dim
        k_norm_flops = 4 * seq_len * num_key_value_heads * head_dim

        # 3. RoPE (Rotary Positional Embeddings)
        # Applied to Q and K
        rope_flops = 3 * seq_len * head_dim * (
            num_attention_heads + num_key_value_heads
        )

        # 4. Attention (Flash/SDPA logic)
        # Score calculation + Value aggregation
        attn_flops = 4 * seq_len * (seq_len + cache_len) \
            * head_dim * num_attention_heads
        
        # 5. MLP (SwiGLU)
        # 3 Linear layers (Gate, Up, Down): 3 * (2 * H * I) * L
        mlp_flops = 6 * seq_len * hidden_size * intermediate_size

        # 6. Layer Norms (Input + Post-Attn) and Residuals
        # 2 RMSNorms per layer (4 ops per element) + 2 Residual adds
        other_flops = 8 * seq_len * hidden_size + 2 * seq_len * hidden_size

        num_flops += (
            q_proj_flops + k_proj_flops + v_proj_flops + o_proj_flops
            + q_norm_flops + k_norm_flops  # Added Qwen3 specific cost
            + rope_flops + attn_flops + mlp_flops + other_flops
        ) * batch_size

        return num_flops

    def total_flops(
        self,
        batch_size: int,
        seq_len: int,
        cache_len: int=0
    ) -> int:
        total_flops: int = 0
        num_layers = get_int(self.config.num_hidden_layers)

        total_flops += self.decoder_layer_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            cache_len=cache_len
        ) * num_layers

        return total_flops