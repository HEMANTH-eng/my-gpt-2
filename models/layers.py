from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (RMSNorm) from Llama/Mistral architectures.

    Computes:
        y = (x / sqrt(mean(x^2) + eps)) * weight
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.normalized_shape:
            raise ValueError(
                f"Expected feature dimension {self.normalized_shape}, got {x.size(-1)}"
            )
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU (Swish Gated Linear Unit) Feed-Forward Network.

    Computes:
        SwiGLU(x) = Dropout( (SiLU(x @ W_gate) * (x @ W_up)) @ W_down )
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        # SwiGLU hidden dimension standard is 2/3 * 4 * d_model (or 8/3 d_model rounded up to multiple of 256)
        if d_ff is None:
            hidden_dim = int(2 * (4 * d_model) / 3)
            self.d_ff = 256 * ((hidden_dim + 255) // 256)
        else:
            self.d_ff = d_ff

        self.w_gate = nn.Linear(d_model, self.d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, self.d_ff, bias=bias)
        self.w_down = nn.Linear(self.d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.normal_(self.w_gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_down.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, residual: bool = False) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Input tensor x must be 3D (B, T, d_model), got {x.dim()}D")

        h = F.silu(self.w_gate(x)) * self.w_up(x)
        out = self.dropout(self.w_down(h))

        if residual:
            return x + out
        return out


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for relative sequence position encoding."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # inv_freq shape: (dim // 2,)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin cached tables
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        # Combine [freqs, freqs] along last dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of tensor x."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) to query and key tensors."""
    # q, k shape: (batch_size, num_heads, seq_len, head_dim)
    # cos, sin shape: (seq_len, head_dim) -> reshape to (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) linear wrapper layer for parameter-efficient SFT."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, base_layer.in_features)))
            self.lora_B = nn.Parameter(torch.zeros((base_layer.out_features, r)))
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B)

            # Freeze base layer parameters
            self.base_layer.weight.requires_grad = False
            if self.base_layer.bias is not None:
                self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.r > 0:
            lora_out = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T
            result = result + lora_out * self.scaling
        return result


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network supporting both GELU and SwiGLU activations."""

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        mlp_type: str = "gelu",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mlp_type = mlp_type

        if mlp_type == "swiglu":
            self.net = SwiGLU(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias)
        else:
            self.d_ff = d_ff if d_ff is not None else 4 * d_model
            self.c_fc = nn.Linear(self.d_model, self.d_ff, bias=bias)
            self.gelu = nn.GELU(approximate="tanh")
            self.c_proj = nn.Linear(self.d_ff, self.d_model, bias=bias)
            self.dropout = nn.Dropout(p=dropout)

            nn.init.normal_(self.c_fc.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
            if bias:
                nn.init.zeros_(self.c_fc.bias)
                nn.init.zeros_(self.c_proj.bias)


    def forward(self, x: torch.Tensor, residual: bool = False) -> torch.Tensor:
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) does not match expected d_model ({self.d_model})."
            )

        if self.mlp_type == "swiglu":
            return self.net(x, residual=residual)

        h = self.c_fc(x)
        h = self.gelu(h)
        h = self.c_proj(h)
        h = self.dropout(h)

        if residual:
            return x + h
        return h

    @property
    def c_fc(self) -> nn.Module:
        if self.mlp_type == "swiglu":
            return self.net.w_gate
        return self._c_fc

    @c_fc.setter
    def c_fc(self, value: nn.Module) -> None:
        self._c_fc = value

    @property
    def c_proj(self) -> nn.Module:
        if self.mlp_type == "swiglu":
            return self.net.w_down
        return self._c_proj

    @c_proj.setter
    def c_proj(self, value: nn.Module) -> None:
        self._c_proj = value



class LayerNorm(nn.Module):
    """Layer Normalization wrapper supporting standard LayerNorm and RMSNorm."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layernorm",
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.norm_type = norm_type

        if norm_type == "rmsnorm":
            self.norm = RMSNorm(normalized_shape, eps=eps)
        else:
            self.eps = eps
            self._weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None

    @property
    def weight(self) -> torch.Tensor:
        if self.norm_type == "rmsnorm":
            return self.norm.weight
        return self._weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        self._weight = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "rmsnorm":
            return self.norm(x)

        if x.size(-1) != self.normalized_shape:
            raise ValueError(f"Expected feature dimension {self.normalized_shape}, got {x.size(-1)}")
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.bias is not None:
            return x_norm * self.weight + self.bias
        return x_norm * self.weight




class TransformerBlock(nn.Module):
    """Reusable GPT Transformer Block with Pre-LN / Pre-RMSNorm architecture.

    Combines Multi-Head / Grouped-Query Attention, Feed-Forward MLP (GELU/SwiGLU),
    Layer Normalization (LayerNorm/RMSNorm), RoPE, and Residual Skip Connections.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        norm_type: str = "layernorm",
        mlp_type: str = "gelu",
    ) -> None:
        super().__init__()
        from models.attention import MultiHeadCausalAttention

        self.d_model = d_model
        self.n_head = n_head

        self.ln_1 = LayerNorm(d_model, bias=bias, norm_type=norm_type)
        self.attn = MultiHeadCausalAttention(
            d_model=d_model,
            n_head=n_head,
            n_kv_head=n_kv_head,
            dropout=dropout,
            bias=bias,
        )
        self.ln_2 = LayerNorm(d_model, bias=bias, norm_type=norm_type)
        self.mlp = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            mlp_type=mlp_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() != 3:
            raise ValueError(f"Input tensor x must be 3D (B, T, d_model), got {x.dim()}D")

        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) does not match expected d_model ({self.d_model})."
            )

        normed_1 = self.ln_1(x)

        if use_cache or kv_cache is not None:
            attn_out, new_kv_cache = self.attn(
                normed_1,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                kv_cache=kv_cache,
                use_cache=use_cache,
            )
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, new_kv_cache

        if return_attn_weights:
            attn_out, attn_weights = self.attn(
                normed_1,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                return_attn_weights=True,
            )
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, attn_weights

        attn_out = self.attn(normed_1, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x




