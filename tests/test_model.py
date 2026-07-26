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


from models.attention import MultiHeadCausalAttention


def test_multi_head_causal_attention_shapes():
    batch_size, seq_len, d_model, n_head = 2, 32, 768, 12
    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadCausalAttention(d_model=d_model, n_head=n_head)
    out = mha(x)
    assert out.shape == (batch_size, seq_len, d_model)

    out_with_weights, attn_weights = mha(x, return_attn_weights=True)
    assert out_with_weights.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, n_head, seq_len, seq_len)


def test_multi_head_invalid_dimension_error():
    with pytest.raises(ValueError):
        MultiHeadCausalAttention(d_model=100, n_head=3)  # 100 % 3 != 0


def test_multi_head_causal_masking():
    batch_size, seq_len, d_model, n_head = 2, 8, 64, 4
    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadCausalAttention(d_model=d_model, n_head=n_head)
    _, attn_weights = mha(x, return_attn_weights=True)

    # For all batches b, heads h, and positions j > i, attn_weights[b, h, i, j] must be 0.0
    for b in range(batch_size):
        for h in range(n_head):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert attn_weights[b, h, i, j].item() == 0.0


def test_multi_head_gradient_flow():
    batch_size, seq_len, d_model, n_head = 3, 16, 128, 8
    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadCausalAttention(d_model=d_model, n_head=n_head)
    out = mha(x)

    loss = out.sum()
    loss.backward()

    assert mha.q_proj.weight.grad is not None
    assert mha.k_proj.weight.grad is not None
    assert mha.v_proj.weight.grad is not None
    assert mha.out_proj.weight.grad is not None


from models.layers import FeedForward


def test_feed_forward_shape():
    batch_size, seq_len, d_model = 4, 16, 128
    x = torch.randn(batch_size, seq_len, d_model)

    ff = FeedForward(d_model=d_model)
    out = ff(x)

    assert out.shape == (batch_size, seq_len, d_model)
    assert ff.c_fc.out_features == 4 * d_model


def test_feed_forward_custom_d_ff():
    batch_size, seq_len, d_model, d_ff = 2, 8, 64, 256
    x = torch.randn(batch_size, seq_len, d_model)

    ff = FeedForward(d_model=d_model, d_ff=d_ff)
    out = ff(x)

    assert out.shape == (batch_size, seq_len, d_model)
    assert ff.c_fc.out_features == d_ff


def test_feed_forward_residual():
    batch_size, seq_len, d_model = 2, 10, 32
    x = torch.randn(batch_size, seq_len, d_model)

    ff = FeedForward(d_model=d_model, dropout=0.0)
    out_no_res = ff(x, residual=False)
    out_res = ff(x, residual=True)

    assert torch.allclose(out_res, x + out_no_res, atol=1e-6)


def test_feed_forward_gradient_flow():
    batch_size, seq_len, d_model = 3, 12, 64
    x = torch.randn(batch_size, seq_len, d_model)

    ff = FeedForward(d_model=d_model)
    out = ff(x)

    loss = out.sum()
    loss.backward()

    assert ff.c_fc.weight.grad is not None
    assert ff.c_proj.weight.grad is not None
    assert ff.c_fc.bias.grad is not None
    assert ff.c_proj.bias.grad is not None


def test_feed_forward_dimension_error():
    x = torch.randn(2, 10, 32)  # d_model=32 mismatch with d_model=64
    ff = FeedForward(d_model=64)
    with pytest.raises(ValueError):
        ff(x)


from models.layers import LayerNorm, TransformerBlock


def test_layer_norm():
    batch_size, seq_len, d_model = 2, 10, 64
    x = torch.randn(batch_size, seq_len, d_model) * 5 + 3

    ln = LayerNorm(normalized_shape=d_model)
    out = ln(x)

    assert out.shape == (batch_size, seq_len, d_model)
    # Check normalized zero mean and unit variance per feature dimension
    assert torch.allclose(out.mean(dim=-1), torch.zeros(batch_size, seq_len), atol=1e-4)
    assert torch.allclose(out.std(dim=-1, unbiased=False), torch.ones(batch_size, seq_len), atol=1e-3)



def test_transformer_block_shape():
    batch_size, seq_len, d_model, n_head = 3, 16, 64, 4
    x = torch.randn(batch_size, seq_len, d_model)

    block = TransformerBlock(d_model=d_model, n_head=n_head)
    out = block(x)
    assert out.shape == (batch_size, seq_len, d_model)

    out_with_weights, attn_weights = block(x, return_attn_weights=True)
    assert out_with_weights.shape == (batch_size, seq_len, d_model)
    assert attn_weights.shape == (batch_size, n_head, seq_len, seq_len)


def test_transformer_block_stacking():
    batch_size, seq_len, d_model, n_head, num_layers = 2, 12, 64, 4, 4
    x = torch.randn(batch_size, seq_len, d_model)

    blocks = torch.nn.ModuleList([
        TransformerBlock(d_model=d_model, n_head=n_head)
        for _ in range(num_layers)
    ])

    h = x
    for block in blocks:
        h = block(h)

    assert h.shape == (batch_size, seq_len, d_model)


def test_transformer_block_gradient_flow():
    batch_size, seq_len, d_model, n_head = 2, 8, 32, 4
    x = torch.randn(batch_size, seq_len, d_model)

    block = TransformerBlock(d_model=d_model, n_head=n_head)
    out = block(x)

    loss = out.sum()
    loss.backward()

    assert block.ln_1.weight.grad is not None
    assert block.ln_2.weight.grad is not None
    assert block.attn.q_proj.weight.grad is not None
    assert block.mlp.c_fc.weight.grad is not None





