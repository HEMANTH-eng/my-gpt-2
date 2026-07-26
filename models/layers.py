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

