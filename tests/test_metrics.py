import math
import numpy as np
import pytest
import torch

from config.model_config import GPTConfig
from dataset.dataloader import create_dataloaders
from dataset.dataset import GPTDataset
from models.gpt import GPT
from scripts.evaluate import run_evaluation
from utils.metrics import (
    benchmark_generation_speed,
    compute_perplexity,
    evaluate_loss_and_perplexity,
)


def test_compute_perplexity():
    assert compute_perplexity(0.0) == 1.0
    assert abs(compute_perplexity(1.0) - math.exp(1.0)) < 1e-5
    assert compute_perplexity(float("inf")) == float("inf")
    assert compute_perplexity(float("nan")) == float("inf")


def test_evaluate_loss_and_perplexity():
    config = GPTConfig.gpt_micro(vocab_size=100)
    model = GPT(config)

    tokens = np.random.randint(0, 100, size=200, dtype=np.int64)
    dataset = GPTDataset(tokens, seq_len=16)
    loader, _ = create_dataloaders(dataset, batch_size=4)

    results = evaluate_loss_and_perplexity(model, loader, device="cpu", max_batches=5)

    assert "val_loss" in results
    assert "perplexity" in results
    assert results["val_loss"] > 0.0
    assert results["perplexity"] >= 1.0
    assert results["num_batches"] == 5
    assert results["num_tokens"] == 5 * 4 * 16


def test_benchmark_generation_speed():
    config = GPTConfig.gpt_micro(vocab_size=100)
    model = GPT(config)

    prompt = torch.randint(0, config.vocab_size, (1, 8))
    bench = benchmark_generation_speed(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=10,
        num_runs=2,
        device="cpu",
    )

    assert "tokens_per_sec" in bench
    assert "ms_per_token" in bench
    assert bench["tokens_per_sec"] > 0.0
    assert bench["ms_per_token"] > 0.0
    assert bench["generated_tokens"] == 10.0


def test_evaluation_script_execution(tmp_path):
    report_file = tmp_path / "test_report.md"
    run_evaluation(
        seq_len=16,
        max_new_tokens=10,
        output_report=str(report_file),
    )

    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "Perplexity" in content
    assert "Throughput" in content
