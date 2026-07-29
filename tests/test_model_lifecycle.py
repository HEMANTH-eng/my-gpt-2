import pytest
import torch

from config.model_config import GPTConfig
from dataset.curator import SFTDataset, SFTExample, load_domain_dataset
from models.gpt import GPT
from tokenizer.bpe_tokenizer import BPETokenizer
from training.continued_pretrain import ContinuedPretrainer
from training.sft_trainer import SFTTrainer
from utils.benchmarks import run_domain_benchmarks


def test_dataset_curator_and_sft_dataset():
    examples = load_domain_dataset("coding")
    assert len(examples) > 0

    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("def fibonacci(n): return n")

    sft_dataset = SFTDataset(examples=examples, tokenizer=tokenizer, max_seq_len=64)
    assert len(sft_dataset) == len(examples)

    x, y = sft_dataset[0]
    assert x.shape == (64,)
    assert y.shape == (64,)
    # Prompt tokens must be masked with ignore_index -100
    assert (y[:3] == -100).all()


def test_sft_trainer_fine_tuning(tmp_path):
    from config.train_config import TrainConfig
    config = GPTConfig.gpt_micro(vocab_size=260)
    model = GPT(config)

    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("def fibonacci(n): return n")

    train_config = TrainConfig.micro_config()
    train_config.checkpoint_dir = str(tmp_path)

    sft_trainer = SFTTrainer(model=model, tokenizer=tokenizer, config=train_config, device="cpu")
    best_loss = sft_trainer.fine_tune_domain(domain="coding", epochs=1, batch_size=1)

    assert best_loss > 0.0


def test_continued_pretrainer():
    config = GPTConfig.gpt_micro(vocab_size=260)
    model = GPT(config)

    dummy_tokens = torch.randint(0, 260, size=(100,))
    pretrainer = ContinuedPretrainer(model=model, device="cpu")

    loss = pretrainer.train_on_corpus(token_ids=dummy_tokens, seq_len=16, epochs=1)
    assert loss >= 0.0


def test_domain_benchmark_suite():
    config = GPTConfig.gpt_micro(vocab_size=260)
    model = GPT(config)

    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("def fibonacci(n): return n")

    results = run_domain_benchmarks(model=model, tokenizer=tokenizer, device="cpu")

    assert "coding_score" in results
    assert "medical_score" in results
    assert "legal_score" in results
    assert "math_score" in results
    assert "overall_benchmark_avg" in results
    assert 0.0 <= results["overall_benchmark_avg"] <= 100.0

