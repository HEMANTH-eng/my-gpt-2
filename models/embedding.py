import math
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token Embedding layer mapping discrete token IDs to continuous d_model vectors."""

    def __init__(self, vocab_size: int, d_model: int, std: float = 0.02) -> None:
        """Initializes token embedding weights.

        Args:
            vocab_size: Total size of token vocabulary.
            d_model: Dimensionality of embedding vector space.
            std: Standard deviation for weight initialization.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for token embeddings.

        Args:
            x: Tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            Embedded tensor with shape (batch_size, seq_len, d_model).
        """
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """Learned Positional Embedding layer (standard GPT-2/3 architecture)."""

    def __init__(self, max_seq_len: int, d_model: int, std: float = 0.02) -> None:
        """Initializes learned position embeddings up to max_seq_len.

        Args:
            max_seq_len: Maximum supported sequence context length.
            d_model: Dimensionality of embedding vector space.
            std: Standard deviation for weight initialization.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating position embeddings for input tokens.

        Args:
            x: Tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            Positional embedding tensor with shape (seq_len, d_model),
            broadcastable to (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum sequence length ({self.max_seq_len})."
            )
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        return self.embedding(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed Sinusoidal Positional Embedding layer (Vaswani et al. Transformer)."""

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        """Initializes fixed sine/cosine position encoding lookup matrix.

        Args:
            max_seq_len: Maximum supported sequence context length.
            d_model: Feature dimension size.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for sinusoidal position embeddings.

        Args:
            x: Tensor of token IDs with shape (batch_size, seq_len).

        Returns:
            Positional embedding tensor with shape (seq_len, d_model).
        """
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds maximum sequence length ({self.max_seq_len})."
            )
        return self.pe[:seq_len, :]


class GPTEmbedding(nn.Module):
    """Combines Token Embeddings, Positional Embeddings, and Dropout into a single layer."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        learned_pos: bool = True,
    ) -> None:
        """Initializes joint GPT embedding module.

        Args:
            vocab_size: Vocabulary size.
            d_model: Model feature dimension.
            max_seq_len: Maximum supported context window length.
            dropout: Dropout probability applied to sum of embeddings.
            learned_pos: If True, uses learned positional embeddings; else sinusoidal.
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        if learned_pos:
            self.pos_embedding = PositionalEmbedding(max_seq_len, d_model)
        else:
            self.pos_embedding = SinusoidalPositionalEmbedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass summing token and position embeddings followed by dropout.

        Args:
            x: Integer token tensor of shape (batch_size, seq_len).

        Returns:
            Float embedding tensor of shape (batch_size, seq_len, d_model).
        """
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(x)
        return self.dropout(tok_emb + pos_emb)

