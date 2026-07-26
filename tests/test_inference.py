import pytest
import torch

from config.model_config import GPTConfig
from models.gpt import GPT
from models.inference import GPTGenerator, sample_next_token
from tokenizer.bpe_tokenizer import BPETokenizer


def test_sample_next_token_greedy():
    logits = torch.tensor([[1.0, 5.0, 2.0], [8.0, 3.0, 4.0]])
    next_tokens = sample_next_token(logits, greedy=True)

    assert next_tokens.shape == (2, 1)
    assert next_tokens[0, 0].item() == 1  # index of max value 5.0
    assert next_tokens[1, 0].item() == 0  # index of max value 8.0


def test_sample_next_token_temperature_and_truncation():
    logits = torch.randn(4, 100)
    next_tokens = sample_next_token(logits, temperature=0.7, top_k=10, top_p=0.9)

    assert next_tokens.shape == (4, 1)
    assert (next_tokens >= 0).all() and (next_tokens < 100).all()


def test_generator_greedy_decoding():
    config = GPTConfig.gpt_micro(vocab_size=200)
    model = GPT(config)
    generator = GPTGenerator(model=model, device="cpu")

    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)
    out1 = generator.generate(prompt, max_new_tokens=6, greedy=True)
    out2 = generator.generate(prompt, max_new_tokens=6, greedy=True)

    assert out1.shape == (1, 3 + 6)
    assert torch.equal(out1, out2)  # Greedy decoding must be 100% deterministic


def test_generator_top_k_top_p_sampling():
    config = GPTConfig.gpt_micro(vocab_size=200)
    model = GPT(config)
    generator = GPTGenerator(model=model, device="cpu")

    prompt = torch.tensor([[5, 15, 25]], dtype=torch.long)
    out = generator.generate(
        prompt,
        max_new_tokens=8,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
    )

    assert out.shape == (1, 3 + 8)


def test_generator_beam_search_decoding():
    config = GPTConfig.gpt_micro(vocab_size=150)
    model = GPT(config)
    generator = GPTGenerator(model=model, device="cpu")

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out_beam = generator.generate(
        prompt,
        max_new_tokens=5,
        num_beams=3,
    )

    assert out_beam.shape == (1, 3 + 5)


def test_generator_streaming_token_generation():
    config = GPTConfig.gpt_micro(vocab_size=150)
    model = GPT(config)
    generator = GPTGenerator(model=model, device="cpu")

    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    max_new_tokens = 6

    stream = generator.generate_stream(prompt, max_new_tokens=max_new_tokens, greedy=True)
    streamed_tokens = list(stream)

    assert len(streamed_tokens) == max_new_tokens
    for t in streamed_tokens:
        assert isinstance(t, int)
        assert 0 <= t < 150


def test_end_to_end_text_inference():
    text_corpus = "Building a custom GPT model completely from scratch in PyTorch!"
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train(text_corpus)

    config = GPTConfig.gpt_micro(vocab_size=len(tokenizer.vocab))
    model = GPT(config)
    generator = GPTGenerator(model=model, tokenizer=tokenizer, device="cpu")

    generated_text = generator.generate(
        prompt="Building a",
        max_new_tokens=5,
        greedy=True,
    )

    assert isinstance(generated_text, str)
    assert generated_text.startswith("Building a")
