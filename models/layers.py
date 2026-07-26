from typing import Optional
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (MLP) for Transformer architectures.

    Computes:
        FFN(x) = Dropout( Linear_2( GELU( Linear_1(x) ) ) )

    Structure & Mathematical Steps:
        1. Up-Projection: Linear_1 maps (B, T, d_model) -> (B, T, d_ff) where d_ff = 4 * d_model.
        2. GELU Activation: Non-linear Gaussian Error Linear Unit activation.
        3. Down-Projection: Linear_2 maps (B, T, d_ff) -> (B, T, d_model).
        4. Dropout: Applied to final output representation.
        5. Residual Support: Optional additive skip connection x + FFN(x).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        approximate_gelu: bool = True,
    ) -> None:
        """Initializes the FeedForward block.

        Args:
            d_model: Input/output feature dimension size (e.g., 768).
            d_ff: Hidden layer dimension size (default: 4 * d_model).
            dropout: Dropout probability.
            bias: Whether linear layers include additive bias terms.
            approximate_gelu: If True, uses 'tanh' approximation for GELU (GPT-2 standard).
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model

        self.c_fc = nn.Linear(self.d_model, self.d_ff, bias=bias)
        self.gelu = nn.GELU(approximate="tanh" if approximate_gelu else "none")
        self.c_proj = nn.Linear(self.d_ff, self.d_model, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        # Standard normal distribution weight initialization (std=0.02)
        nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        if bias:
            nn.init.zeros_(self.c_fc.bias)
            nn.init.zeros_(self.c_proj.bias)

    def forward(self, x: torch.Tensor, residual: bool = False) -> torch.Tensor:
        """Forward pass for FeedForward block.

        Args:
            x: Input embeddings tensor with shape (batch_size, seq_len, d_model).
            residual: If True, adds input x as a residual skip connection: x + FFN(x).

        Returns:
            Output tensor with shape (batch_size, seq_len, d_model).
        """
        if x.dim() != 3:
            raise ValueError(f"Input tensor x must be a 3D tensor of shape (B, T, d_model), got {x.dim()}D")

        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) does not match expected d_model ({self.d_model})."
            )

        h = self.c_fc(x)
        h = self.gelu(h)
        h = self.c_proj(h)
        h = self.dropout(h)

        if residual:
            return x + h
        return h


class LayerNorm(nn.Module):
    """Layer Normalization implemented from scratch.

    Computes:
        y = ( (x - mean) / sqrt(var + eps) ) * weight + bias
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, bias: bool = True) -> None:
        """Initializes LayerNorm parameters.

        Args:
            normalized_shape: Dimensionality of feature space to normalize across (d_model).
            eps: Epsilon value added to variance for numerical stability.
            bias: Whether layer norm includes learnable additive bias parameter.
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for LayerNorm.

        Args:
            x: Tensor with shape (..., normalized_shape).

        Returns:
            Normalized tensor with identical shape (..., normalized_shape).
        """
        if x.size(-1) != self.normalized_shape:
            raise ValueError(
                f"Expected feature dimension {self.normalized_shape}, got {x.size(-1)}"
            )
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.bias is not None:
            return x_norm * self.weight + self.bias
        return x_norm * self.weight


class TransformerBlock(nn.Module):
    """Reusable GPT Transformer Block with Pre-LayerNorm architecture.

    Combines Multi-Head Causal Self-Attention, Feed-Forward MLP, Layer Normalization,
    and Residual Skip Connections into a single transformer decoder layer.

    Pre-LN Architecture & Mathematical Steps:
        1. Sub-layer 1 (Causal Self-Attention):
           x = x + MultiHeadCausalAttention( LayerNorm_1(x) )
        2. Sub-layer 2 (Position-wise Feed-Forward):
           x = x + FeedForward( LayerNorm_2(x) )
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initializes TransformerBlock.

        Args:
            d_model: Model feature dimension size (e.g. 768).
            n_head: Number of attention heads (e.g. 12).
            d_ff: Feed-forward hidden dimension size (default: 4 * d_model).
            dropout: Dropout probability.
            bias: Whether linear layers and LayerNorm use bias parameters.
        """
        super().__init__()
        from models.attention import MultiHeadCausalAttention

        self.d_model = d_model
        self.n_head = n_head

        self.ln_1 = LayerNorm(d_model, bias=bias)
        self.attn = MultiHeadCausalAttention(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
            bias=bias,
        )
        self.ln_2 = LayerNorm(d_model, bias=bias)
        self.mlp = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for TransformerBlock.

        Args:
            x: Input embeddings tensor with shape (batch_size, seq_len, d_model).
            return_attn_weights: If True, also returns attention weights tensor (batch_size, n_head, seq_len, seq_len).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
            Optionally returns (output, attn_weights) if return_attn_weights is True.
        """
        if x.dim() != 3:
            raise ValueError(f"Input tensor x must be a 3D tensor of shape (B, T, d_model), got {x.dim()}D")

        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) does not match expected d_model ({self.d_model})."
            )

        if return_attn_weights:
            attn_out, attn_weights = self.attn(self.ln_1(x), return_attn_weights=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


