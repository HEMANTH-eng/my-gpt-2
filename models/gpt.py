import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import GPTConfig
from models.embedding import GPTEmbedding
from models.layers import LayerNorm, LoRALinear, RotaryEmbedding, TransformerBlock
from utils.logger import get_logger

logger = get_logger(__name__)


class GPT(nn.Module):
    """Production-grade decoder-only LLM architecture supporting RoPE, RMSNorm, SwiGLU, GQA, and KV Caching."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Input Embeddings (Token + optional Position)
        self.embedding = GPTEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            learned_pos=config.learned_pos and not getattr(config, "use_rope", False),
        )

        # 2. Rotary Position Embeddings (RoPE)
        self.use_rope = getattr(config, "use_rope", True)
        if self.use_rope:
            head_dim = config.d_model // config.n_head
            self.rope = RotaryEmbedding(
                dim=head_dim,
                max_seq_len=config.max_seq_len,
                theta=getattr(config, "rope_theta", 10000.0),
            )

        # 3. Transformer Decoder Blocks Stack
        norm_type = getattr(config, "norm_type", "rmsnorm")
        mlp_type = getattr(config, "mlp_type", "swiglu")
        n_kv_head = getattr(config, "n_kv_head", config.n_head)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_head=config.n_head,
                n_kv_head=n_kv_head,
                d_ff=config.d_ff,
                dropout=config.dropout,
                bias=config.bias,
                norm_type=norm_type,
                mlp_type=mlp_type,
            )
            for _ in range(config.n_layer)
        ])

        # 4. Final Normalization Layer (LayerNorm or RMSNorm)
        self.ln_f = LayerNorm(config.d_model, bias=config.bias, norm_type=norm_type)

        # 5. Language Modeling Output Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 6. Weight Tying
        if config.weight_tying:
            self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        # 7. Custom Parameter Initialization
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("w_down.weight") or pn.endswith("out_proj.weight") or pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # 8. Inject LoRA adapters if lora_r > 0
        if getattr(config, "lora_r", 0) > 0:
            self.inject_lora(r=config.lora_r, lora_alpha=getattr(config, "lora_alpha", 16.0))

        logger.info(
            f"Initialized SOTA GPT model with {self.get_num_params() / 1e6:.2f}M parameters "
            f"(n_layer={config.n_layer}, n_head={config.n_head}, norm={norm_type}, mlp={mlp_type})"
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def inject_lora(self, r: int = 8, lora_alpha: float = 16.0) -> None:
        """Injects LoRA adaptation layers into query, key, value, and output linear projections."""
        for block in self.blocks:
            block.attn.q_proj = LoRALinear(block.attn.q_proj, r=r, lora_alpha=lora_alpha)
            block.attn.v_proj = LoRALinear(block.attn.v_proj, r=r, lora_alpha=lora_alpha)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding and hasattr(self.embedding, "pos_embedding") and hasattr(self.embedding.pos_embedding, "embedding"):
            n_params -= self.embedding.pos_embedding.embedding.weight.numel()
        return n_params


    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]]:
        batch_size, seq_len = idx.size()

        if seq_len > self.config.max_seq_len and kv_caches is None:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds model max_seq_len ({self.config.max_seq_len})."
            )

        # 1. Compute Embeddings
        x = self.embedding(idx)

        # 2. Prepare RoPE frequencies if enabled
        rope_cos, rope_sin = None, None
        if self.use_rope:
            total_len = (
                (kv_caches[0][0].size(-2) + seq_len)
                if (kv_caches is not None and kv_caches[0] is not None)
                else seq_len
            )
            rope_cos, rope_sin = self.rope(x, total_len)


        # 3. Pass through Decoder Blocks
        is_cached_gen = (kv_caches is not None)
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            past_kv = kv_caches[i] if is_cached_gen else None
            if is_cached_gen:
                x, new_kv = block(x, rope_cos=rope_cos, rope_sin=rope_sin, kv_cache=past_kv, use_cache=True)
                new_kv_caches.append(new_kv)
            else:
                x = block(x, rope_cos=rope_cos, rope_sin=rope_sin)


        # 4. Final Normalization
        x = self.ln_f(x)

        # 5. LM Head Logits
        logits = self.lm_head(x)

        # 6. Compute Loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        if kv_caches is not None:
            return logits, loss, new_kv_caches

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generates max_new_tokens with optional KV Caching for fast inference."""
        kv_caches = [None] * len(self.blocks) if use_cache else None

        for step in range(max_new_tokens):
            if use_cache and step > 0:
                idx_cond = idx[:, -1:]
            else:
                idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len :]

            if use_cache:
                logits, _, kv_caches = self.forward(idx_cond, kv_caches=kv_caches)
            else:
                logits, _ = self.forward(idx_cond)

            logits = logits[:, -1, :] / max(temperature, 1e-5)


            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


