from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """GPT Model Hyperparameter Configuration.

    Attributes:
        vocab_size: Total vocabulary size.
        max_seq_len: Maximum context window sequence length (block size).
        n_layer: Number of stacked transformer blocks (depth).
        n_head: Number of parallel attention heads.
        d_model: Hidden feature dimension (embedding dimension).
        d_ff: Feed-forward inner hidden dimension (defaults to 4 * d_model).
        dropout: Dropout regularization probability.
        bias: Whether linear layers and LayerNorm include additive bias parameters.
        learned_pos: If True, uses learned positional embeddings; else sinusoidal.
        weight_tying: If True, ties token embedding weight matrix with LM head weight matrix.
    """

    vocab_size: int = 50257
    max_seq_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    d_ff: Optional[int] = None
    dropout: float = 0.1
    bias: bool = True
    learned_pos: bool = True
    weight_tying: bool = True

    @classmethod
    def gpt_micro(cls, vocab_size: int = 500) -> "GPTConfig":
        """Micro configuration for quick unit testing and local training."""
        return cls(
            vocab_size=vocab_size,
            max_seq_len=128,
            n_layer=4,
            n_head=4,
            d_model=128,
            dropout=0.0,
        )

    @classmethod
    def gpt2_small(cls) -> "GPTConfig":
        """Standard GPT-2 124M parameter configuration."""
        return cls(
            vocab_size=50257,
            max_seq_len=1024,
            n_layer=12,
            n_head=12,
            d_model=768,
        )

    @classmethod
    def gpt2_medium(cls) -> "GPTConfig":
        """Standard GPT-2 350M parameter configuration."""
        return cls(
            vocab_size=50257,
            max_seq_len=1024,
            n_layer=24,
            n_head=16,
            d_model=1024,
        )

