import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention implemented from scratch.

    Computes mathematically:
        Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + M ) @ V

    Step-by-step mathematical flow:
        1. Matrix Multiplication: Q @ K^T calculates similarity scores between Queries and Keys.
        2. Scaling: Division by sqrt(d_k) stabilizes gradients for large dimensions.
        3. Masking (Optional): Applies causal or padding mask M by assigning -1e9 to masked entries.
        4. Softmax: Converts raw similarity scores into probability distribution over keys (dim=-1).
        5. Dropout: Regularizes attention weight matrix.
        6. Context Calculation: Weighted sum of Values V using attention probabilities.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        """Initializes attention module.

        Args:
            dropout: Dropout probability applied to attention weights.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for Scaled Dot-Product Attention.

        Args:
            q: Query tensor with shape (..., T_q, d_k).
            k: Key tensor with shape (..., T_k, d_k).
            v: Value tensor with shape (..., T_k, d_v).
            mask: Optional mask tensor broadcastable to (..., T_q, T_k).
                  If boolean: True indicates positions to be MASKED OUT (-1e9).
                  If integer/float: 0 indicates positions to be MASKED OUT (-1e9).

        Returns:
            Tuple containing:
                - output: Context matrix with shape (..., T_q, d_v).
                - attn_weights: Attention probability weights with shape (..., T_q, T_k).
        """
        # Step 1: Shape Validation
        if q.dim() < 2 or k.dim() < 2 or v.dim() < 2:
            raise ValueError(
                f"Query, Key, and Value tensors must have at least 2 dimensions. "
                f"Got Q: {q.dim()}D, K: {k.dim()}D, V: {v.dim()}D"
            )

        d_k = q.size(-1)
        if k.size(-1) != d_k:
            raise ValueError(
                f"Query feature dimension ({d_k}) must match Key feature dimension ({k.size(-1)})."
            )

        if k.size(-2) != v.size(-2):
            raise ValueError(
                f"Key sequence length ({k.size(-2)}) must match Value sequence length ({v.size(-2)})."
            )

        # Step 2: Compute unscaled similarity dot products (Q @ K^T)
        # Shape: (..., T_q, d_k) @ (..., d_k, T_k) -> (..., T_q, T_k)
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Step 3: Scale dot-product scores by sqrt(d_k)
        scaling_factor = math.sqrt(d_k)
        scores = scores / scaling_factor

        # Step 4: Apply optional attention mask (e.g. Causal or Padding mask)
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, -1e9)
            else:
                scores = scores.masked_fill(mask == 0, -1e9)

        # Step 5: Softmax along key sequence length dimension (-1)
        attn_weights = F.softmax(scores, dim=-1)

        # Step 6: Apply dropout regularization to attention probabilities
        attn_weights_dropped = self.dropout(attn_weights)

        # Step 7: Compute context matrix by weighting Values: (attn_weights @ V)
        # Shape: (..., T_q, T_k) @ (..., T_k, d_v) -> (..., T_q, d_v)
        output = torch.matmul(attn_weights_dropped, v)

        return output, attn_weights


class SingleHeadCausalSelfAttention(nn.Module):
    """Single-Head Causal Self-Attention block for GPT models."""

    def __init__(self, d_model: int, d_k: int, dropout: float = 0.0) -> None:
        """Initializes single-head causal self-attention.

        Args:
            d_model: Input feature dimension.
            d_k: Hidden projection dimension for Query/Key/Value.
            dropout: Attention dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k

        # Linear projections W_q, W_k, W_v, W_o
        self.q_proj = nn.Linear(d_model, d_k, bias=False)
        self.k_proj = nn.Linear(d_model, d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_k, bias=False)
        self.out_proj = nn.Linear(d_k, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for causal self-attention.

        Args:
            x: Input embeddings tensor with shape (batch_size, seq_len, d_model).

        Returns:
            Output context tensor with shape (batch_size, seq_len, d_model).
        """
        if x.dim() != 3:
            raise ValueError(f"Input x must be a 3D tensor of shape (B, T, d_model), got {x.dim()}D")

        batch_size, seq_len, d_model = x.size()
        if d_model != self.d_model:
            raise ValueError(f"Input d_model ({d_model}) does not match expected d_model ({self.d_model})")

        # 1. Project input X into Q, K, V
        q = self.q_proj(x)  # (B, T, d_k)
        k = self.k_proj(x)  # (B, T, d_k)
        v = self.v_proj(x)  # (B, T, d_k)

        # 2. Construct upper-triangular causal mask (True for positions j > i to be masked out)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)

        # 3. Compute scaled dot-product attention
        context, _ = self.attention(q, k, v, mask=causal_mask)

        # 4. Final linear output projection
        return self.out_proj(context)


class MultiHeadCausalAttention(nn.Module):
    """Multi-Head Causal Self-Attention layer for GPT architectures.

    Splits the feature dimension d_model into n_head parallel heads of dimension
    d_head = d_model // n_head, allowing the model to jointly attend to information
    from different representation subspaces at different positions.

    Computes mathematically:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
        where head_i = Attention(Q @ W_i^Q, K @ W_i^K, V @ W_i^V)

    Key Features & Optimizations:
        - Parallel tensor operations: Reshapes (B, T, d_model) to (B, n_head, T, d_head).
        - Batched dot-product attention computed across all heads concurrently.
        - Causal masking: Prevents token position i from attending to future positions j > i.
        - Configurable linear projection weights and dropout.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        """Initializes MultiHeadCausalAttention.

        Args:
            d_model: Model feature dimension size (e.g., 768).
            n_head: Number of parallel attention heads (e.g., 12).
            dropout: Attention dropout probability.
            bias: Whether linear projections include additive bias parameters.
        """
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_head ({n_head}). "
                f"Remainder: {d_model % n_head}"
            )

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        # Linear projections for Query, Key, Value, and Output
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for multi-head causal self-attention.

        Args:
            x: Input embeddings tensor with shape (batch_size, seq_len, d_model).
            return_attn_weights: If True, also returns attention weights of shape (batch_size, n_head, seq_len, seq_len).

        Returns:
            Output context tensor of shape (batch_size, seq_len, d_model).
            Optionally returns (output, attn_weights) if return_attn_weights is True.
        """
        if x.dim() != 3:
            raise ValueError(f"Input x must be a 3D tensor of shape (B, T, d_model), got {x.dim()}D")

        batch_size, seq_len, d_model = x.size()
        if d_model != self.d_model:
            raise ValueError(f"Input d_model ({d_model}) does not match expected d_model ({self.d_model})")

        # Step 1: Project inputs to Query, Key, and Value spaces: (B, T, d_model) -> (B, T, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Step 2: Reshape and transpose for multi-head parallel computation:
        # (B, T, d_model) -> (B, T, n_head, d_head) -> (B, n_head, T, d_head)
        q = q.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)

        # Step 3: Create 4D upper-triangular causal mask: (1, 1, seq_len, seq_len)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # Step 4: Scaled dot-product attention across all heads in parallel
        # context: (B, n_head, T, d_head), attn_weights: (B, n_head, T, T)
        context, attn_weights = self.attention(q, k, v, mask=causal_mask)

        # Step 5: Concatenate head outputs: (B, n_head, T, d_head) -> (B, T, n_head, d_head) -> (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Step 6: Output linear projection
        out = self.out_proj(context)

        if return_attn_weights:
            return out, attn_weights
        return out


