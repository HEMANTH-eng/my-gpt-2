from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """GPT Model Hyperparameter Configuration.

    Attributes:
        vocab_size: Total vocabulary size.
        max_seq_len: Maximum context window sequence length (block size).
        n_layer: Number of stacked transformer blocks (depth).
        n_head: Number of parallel attention query heads.
        n_kv_head: Optional number of key/value heads for Grouped Query Attention (GQA).
        d_model: Hidden feature dimension (embedding dimension).
        d_ff: Feed-forward inner hidden dimension (defaults to 4 * d_model or 8/3 for SwiGLU).
        dropout: Dropout regularization probability.
        bias: Whether linear layers and LayerNorm include additive bias parameters.
        use_rope: If True, uses Rotary Position Embeddings (RoPE); else learned/sinusoidal.
        rope_theta: Base frequency for RoPE rotary embeddings (default: 10000.0).
        norm_type: Normalization type ('rmsnorm' or 'layernorm').
        mlp_type: Feed-forward activation type ('swiglu' or 'gelu').
        weight_tying: If True, ties token embedding weight matrix with LM head weight matrix.
        use_gradient_checkpointing: Enables gradient checkpointing for memory efficiency.
        lora_r: Rank for LoRA adaptation (0 disables LoRA).
        lora_alpha: Scaling parameter for LoRA.
    """

    vocab_size: int = 50257
    max_seq_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None
    d_model: int = 768
    d_ff: Optional[int] = None
    dropout: float = 0.1
    bias: bool = True
    learned_pos: bool = True
    use_rope: bool = True
    rope_theta: float = 10000.0
    norm_type: str = "rmsnorm"  # 'rmsnorm' or 'layernorm'
    mlp_type: str = "swiglu"    # 'swiglu' or 'gelu'
    weight_tying: bool = True
    use_gradient_checkpointing: bool = False
    lora_r: int = 0
    lora_alpha: float = 16.0

    @classmethod
    def gpt_micro(cls, vocab_size: int = 500) -> "GPTConfig":
        """Micro configuration for quick unit testing and local training."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=128,
            n_layer=4,
            n_head=4,
            n_kv_head=4,
            d_model=128,
            dropout=0.0,
            use_rope=True,
            norm_type="rmsnorm",
            mlp_type="swiglu",
        )

    @classmethod
    def gpt2_small(cls) -> "GPTConfig":
        """Standard GPT-2 / Llama-style 124M parameter configuration."""
        return cls(
            vocab_size=50257,
            max_seq_len=1024,
            n_layer=12,
            n_head=12,
            n_kv_head=12,
            d_model=768,
            use_rope=True,
            norm_type="rmsnorm",
            mlp_type="swiglu",
        )

    @classmethod
    def gpt2_medium(cls) -> "GPTConfig":
        """Standard 350M parameter configuration."""
        return cls(
            vocab_size=50257,
            max_seq_len=1024,
            n_layer=24,
            n_head=16,
            n_kv_head=8,  # GQA (8 KV heads for 16 Query heads)
            d_model=1024,
            use_rope=True,
            norm_type="rmsnorm",
            mlp_type="swiglu",
        )


