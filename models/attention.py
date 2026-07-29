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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats Key/Value tensor heads for Grouped Query Attention (GQA).

    Args:
        x: Key/Value tensor of shape (batch_size, n_kv_head, seq_len, head_dim).
        n_rep: Number of times to replicate each KV head to match Query head count.

    Returns:
        Expanded tensor of shape (batch_size, n_kv_head * n_rep, seq_len, head_dim).
    """
    if n_rep == 1:
        return x
    batch_size, n_kv_head, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch_size, n_kv_head, n_rep, seq_len, head_dim)
        .reshape(batch_size, n_kv_head * n_rep, seq_len, head_dim)
    )


class MultiHeadCausalAttention(nn.Module):
    """Multi-Head & Grouped-Query Causal Attention (MHA/GQA) with RoPE and KV-Cache support."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_head ({n_head})."
            )

        self.d_model = d_model
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_rep = self.n_head // self.n_kv_head
        self.d_head = d_model // n_head

        # Projections
        self.q_proj = nn.Linear(d_model, self.n_head * self.d_head, bias=bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_head * self.d_head, bias=bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_head * self.d_head, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        if x.dim() != 3:
            raise ValueError(f"Input x must be 3D (B, T, d_model), got {x.dim()}D")

        batch_size, seq_len, d_model = x.size()
        if d_model != self.d_model:
            raise ValueError(f"Input d_model ({d_model}) does not match expected ({self.d_model})")

        # 1. Linear Projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_head, self.d_head).transpose(1, 2)

        # 2. Apply RoPE if provided
        if rope_cos is not None and rope_sin is not None:
            from models.layers import apply_rotary_pos_emb
            pos_offset = kv_cache[0].size(-2) if (kv_cache is not None and kv_cache[0] is not None) else 0
            cos = rope_cos[pos_offset : pos_offset + seq_len]
            sin = rope_sin[pos_offset : pos_offset + seq_len]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. Key-Value Caching
        if kv_cache is not None and kv_cache[0] is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        new_kv_cache = (k, v)

        # 4. Grouped Query Attention (repeat KV heads if GQA)
        keys_repeat = repeat_kv(k, self.n_rep)
        values_repeat = repeat_kv(v, self.n_rep)

        # 5. Causal Masking
        total_seq_len = keys_repeat.size(-2)
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones((seq_len, total_seq_len), device=x.device, dtype=torch.bool),
                diagonal=total_seq_len - seq_len + 1,
            ).unsqueeze(0).unsqueeze(0)
        else:
            causal_mask = None

        # 6. Attention computation
        context, attn_weights = self.attention(q, keys_repeat, values_repeat, mask=causal_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        out = self.out_proj(context)

        if use_cache or kv_cache is not None:
            return out, new_kv_cache

        if return_attn_weights:
            return out, attn_weights
        return out




