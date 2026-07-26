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


from models.attention import ScaledDotProductAttention, SingleHeadCausalSelfAttention


def test_scaled_dot_product_attention_shapes():
    batch_size, seq_len, d_k, d_v = 3, 10, 32, 64
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_v)

    attention = ScaledDotProductAttention(dropout=0.0)
    output, attn_weights = attention(q, k, v)

    assert output.shape == (batch_size, seq_len, d_v)
    assert attn_weights.shape == (batch_size, seq_len, seq_len)

    # Verify softmax property: rows sum to 1.0 along key dimension (-1)
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_causal_masking_property():
    batch_size, seq_len, d_k = 2, 8, 16
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_k)

    # Upper triangular mask (True for positions j > i)
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)

    attention = ScaledDotProductAttention(dropout=0.0)
    _, attn_weights = attention(q, k, v, mask=causal_mask)

    # For all j > i, attn_weights[b, i, j] must be 0.0
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert attn_weights[b, i, j].item() == 0.0


def test_attention_shape_validation():
    q = torch.randn(2, 8, 32)
    k = torch.randn(2, 8, 16)  # Mismatched d_k (16 vs 32)
    v = torch.randn(2, 8, 32)

    attention = ScaledDotProductAttention()
    with pytest.raises(ValueError):
        attention(q, k, v)


def test_single_head_causal_self_attention_forward_grad():
    batch_size, seq_len, d_model, d_k = 4, 12, 64, 32
    x = torch.randn(batch_size, seq_len, d_model)

    sa = SingleHeadCausalSelfAttention(d_model=d_model, d_k=d_k)
    out = sa(x)

    assert out.shape == (batch_size, seq_len, d_model)

    loss = out.sum()
    loss.backward()

    assert sa.q_proj.weight.grad is not None
    assert sa.k_proj.weight.grad is not None
    assert sa.v_proj.weight.grad is not None
    assert sa.out_proj.weight.grad is not None


