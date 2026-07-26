import pytest
import torch

from models.embedding import (
    GPTEmbedding,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TokenEmbedding,
)


def test_token_embedding_forward_and_grad():
    batch_size, seq_len, vocab_size, d_model = 2, 10, 1000, 64
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    tok_emb = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
    out = tok_emb(x)

    assert out.shape == (batch_size, seq_len, d_model)
    assert out.dtype == torch.float32

    # Verify backpropagation gradient flow
    loss = out.sum()
    loss.backward()
    assert tok_emb.embedding.weight.grad is not None
    assert tok_emb.embedding.weight.grad.shape == (vocab_size, d_model)


def test_positional_embedding_forward():
    seq_len, max_seq_len, d_model = 16, 128, 64
    x = torch.randint(0, 100, (2, seq_len))

    pos_emb = PositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)
    out = pos_emb(x)

    assert out.shape == (seq_len, d_model)
    assert out.dtype == torch.float32

    # Gradient check
    loss = out.sum()
    loss.backward()
    assert pos_emb.embedding.weight.grad is not None


def test_sinusoidal_positional_embedding():
    seq_len, max_seq_len, d_model = 20, 256, 32
    x = torch.randint(0, 50, (2, seq_len))

    sin_pos = SinusoidalPositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)
    out = sin_pos(x)

    assert out.shape == (seq_len, d_model)
    assert out.dtype == torch.float32


def test_gpt_embedding_combination():
    batch_size, seq_len, vocab_size, d_model, max_seq_len = 4, 32, 500, 128, 256
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    gpt_emb_learned = GPTEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        dropout=0.1,
        learned_pos=True,
    )
    out_learned = gpt_emb_learned(x)
    assert out_learned.shape == (batch_size, seq_len, d_model)

    gpt_emb_sinusoidal = GPTEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        dropout=0.0,
        learned_pos=False,
    )
    out_sin = gpt_emb_sinusoidal(x)
    assert out_sin.shape == (batch_size, seq_len, d_model)


def test_exceed_max_seq_len_error():
    max_seq_len = 16
    x = torch.randint(0, 100, (2, 32))  # seq_len 32 > max_seq_len 16

    pos_emb = PositionalEmbedding(max_seq_len=max_seq_len, d_model=64)
    with pytest.raises(ValueError):
        pos_emb(x)

