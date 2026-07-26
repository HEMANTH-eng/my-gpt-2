import math
import time
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import get_logger

logger = get_logger(__name__)


def compute_perplexity(loss: float) -> float:
    """Computes perplexity PPL = exp(loss) with numerical overflow protection.

    Args:
        loss: Cross entropy loss float value.

    Returns:
        Perplexity float value.
    """
    if math.isnan(loss) or math.isinf(loss):
        return float("inf")
    capped_loss = min(loss, 20.0)  # Avoid math.exp overflow
    return math.exp(capped_loss)


def evaluate_loss_and_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluates average cross entropy loss and perplexity on a dataset DataLoader.

    Args:
        model: PyTorch GPT model instance.
        dataloader: DataLoader instance for evaluation.
        device: Device string ('cuda', 'cpu').
        max_batches: Optional limit on number of batches to evaluate.

    Returns:
        Dictionary containing 'val_loss', 'perplexity', 'num_batches', and 'num_tokens'.
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_batches = 0
    total_tokens = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            _, loss = model(x, targets=y)

            total_loss += loss.item()
            total_batches += 1
            total_tokens += x.numel()

    avg_loss = total_loss / max(1, total_batches)
    perplexity = compute_perplexity(avg_loss)

    logger.info(
        f"Evaluation complete: Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f} "
        f"over {total_batches} batches ({total_tokens} tokens)."
    )

    return {
        "val_loss": avg_loss,
        "perplexity": perplexity,
        "num_batches": total_batches,
        "num_tokens": total_tokens,
    }


def benchmark_generation_speed(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 50,
    num_runs: int = 3,
    device: str = "cpu",
) -> Dict[str, float]:
    """Benchmarks inference throughput (tokens/sec and latency per token).

    Args:
        model: PyTorch GPT model instance.
        prompt_ids: 2D integer tensor of prompt token IDs with shape (1, seq_len).
        max_new_tokens: Number of tokens to generate per run.
        num_runs: Number of warm runs to average speed metrics.
        device: Device string ('cuda', 'cpu').

    Returns:
        Dictionary containing 'tokens_per_sec', 'ms_per_token', and 'total_time_sec'.
    """
    model.eval()
    model.to(device)
    prompt_ids = prompt_ids.to(device)

    from models.inference import GPTGenerator
    generator = GPTGenerator(model=model, device=device)

    # Warmup run
    generator.generate(prompt_ids, max_new_tokens=5, greedy=True)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = 0.0
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = generator.generate(prompt_ids, max_new_tokens=max_new_tokens, greedy=True)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

    avg_time_sec = total_time / num_runs
    tokens_per_sec = max_new_tokens / max(avg_time_sec, 1e-6)
    ms_per_token = (avg_time_sec * 1000.0) / max_new_tokens

    logger.info(
        f"Inference Benchmark: {tokens_per_sec:.2f} tokens/sec | "
        f"{ms_per_token:.2f} ms/token (avg over {num_runs} runs)."
    )

    return {
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": ms_per_token,
        "total_time_sec": avg_time_sec,
        "generated_tokens": float(max_new_tokens),
    }
