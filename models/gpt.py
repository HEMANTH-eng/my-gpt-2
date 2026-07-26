import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import GPTConfig
from models.embedding import GPTEmbedding
from models.layers import LayerNorm, TransformerBlock
from utils.logger import get_logger

logger = get_logger(__name__)


class GPT(nn.Module):
    """Complete GPT Autoregressive Large Language Model built from scratch.

    Architecture Details:
        - Input Embeddings: GPTEmbedding (Token Embedding + Positional Embedding + Dropout).
        - Transformer Backbone: Stack of n_layer TransformerBlocks (Pre-LN Multi-Head Causal Attention + Feed-Forward MLP).
        - Final Normalization: LayerNorm(d_model).
        - LM Output Head: Linear(d_model, vocab_size, bias=False).
        - Weight Tying: Shared parameter weights between token embedding and LM head.
    """

    def __init__(self, config: GPTConfig) -> None:
        """Initializes the complete GPT model.

        Args:
            config: GPTConfig dataclass instance specifying hyperparameters.
        """
        super().__init__()
        self.config = config

        # 1. Input Embeddings (Token + Position)
        self.embedding = GPTEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            learned_pos=config.learned_pos,
        )

        # 2. Transformer Decoder Blocks Stack
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_head=config.n_head,
                d_ff=config.d_ff,
                dropout=config.dropout,
                bias=config.bias,
            )
            for _ in range(config.n_layer)
        ])

        # 3. Final Layer Normalization
        self.ln_f = LayerNorm(config.d_model, bias=config.bias)

        # 4. Final Language Modeling Output Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 5. Weight Tying: Tie embedding weights with LM output head (GPT-2 standard)
        if config.weight_tying:
            self.lm_head.weight = self.embedding.token_embedding.embedding.weight

        # 6. Apply custom weight initialization across all parameters
        self.apply(self._init_weights)

        # Apply scaled initialization to residual output projections (1 / sqrt(2 * N))
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("out_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        logger.info(
            f"Initialized GPT model with {self.get_num_params() / 1e6:.2f}M parameters "
            f"(n_layer={config.n_layer}, n_head={config.n_head}, d_model={config.d_model})"
        )

    def _init_weights(self, module: nn.Module) -> None:
        """Applies Gaussian normal initialization (mean=0.0, std=0.02) to linear & embedding weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Returns the total number of trainable parameters in the model.

        Args:
            non_embedding: If True, subtracts position embedding parameter count.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding and self.config.learned_pos:
            n_params -= self.embedding.pos_embedding.embedding.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for training and inference.

        Args:
            idx: Integer token tensor of shape (batch_size, seq_len).
            targets: Optional integer token tensor of shape (batch_size, seq_len).

        Returns:
            Tuple of (logits, loss):
                - logits: Tensor of shape (batch_size, seq_len, vocab_size).
                - loss: Cross entropy loss scalar if targets provided, else None.
        """
        batch_size, seq_len = idx.size()

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds model max_seq_len ({self.config.max_seq_len})."
            )

        # 1. Compute joint token + position embeddings: (B, T) -> (B, T, d_model)
        x = self.embedding(idx)

        # 2. Sequential pass through Transformer Decoder Blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final Layer Normalization
        x = self.ln_f(x)

        # 4. Project hidden states to vocabulary logits: (B, T, d_model) -> (B, T, vocab_size)
        logits = self.lm_head(x)

        # 5. Compute Cross Entropy Loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively generates max_new_tokens given prompt sequence idx.

        Args:
            idx: Initial prompt token IDs tensor of shape (batch_size, seq_len).
            max_new_tokens: Number of new tokens to generate.
            temperature: Softmax sampling temperature (1.0 = standard, <1.0 = confident).
            top_k: Optional top-k probability truncation threshold.

        Returns:
            Tensor of token IDs of shape (batch_size, seq_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop prompt to max_seq_len if context exceeds window limit
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len :]

            # Forward pass to obtain logits for the last sequence position
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optionally filter logits to top-k candidates
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Softmax to get sampling probability distribution
            probs = F.softmax(logits, dim=-1)

            # Sample next token ID from distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Concatenate sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

