import argparse
from pathlib import Path
import sys
import torch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.model_config import GPTConfig
from dataset.dataloader import create_dataloaders
from dataset.dataset import GPTDataset
from models.gpt import GPT
from utils.logger import get_logger
from utils.metrics import benchmark_generation_speed, evaluate_loss_and_perplexity

logger = get_logger("evaluate_script")


def run_evaluation(
    checkpoint_path: str = "",
    seq_len: int = 64,
    max_new_tokens: int = 50,
    output_report: str = "docs/evaluation_report.md",
) -> None:
    """Runs evaluation benchmark and generates a structured report."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting evaluation benchmark on device '{device}'...")

    # Instantiate model
    config = GPTConfig.gpt_micro(vocab_size=300)
    model = GPT(config)

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading checkpoint weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Create dummy evaluation dataset
    dummy_tokens = torch.randint(0, config.vocab_size, (200,)).numpy()
    eval_dataset = GPTDataset(dummy_tokens, seq_len=seq_len)
    eval_loader, _ = create_dataloaders(eval_dataset, batch_size=4)

    # 1. Measure Loss & Perplexity
    eval_results = evaluate_loss_and_perplexity(model, eval_loader, device=device)

    # 2. Measure Generation Speed Throughput
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    bench_results = benchmark_generation_speed(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=max_new_tokens,
        num_runs=3,
        device=device,
    )

    # 3. Format Evaluation Report
    report_content = f"""# GPT Model Evaluation & Benchmark Report

## 1. Executive Summary
- **Model Parameters**: {model.get_num_params() / 1e6:.2f}M
- **Device**: `{device}`
- **Context Length (`seq_len`)**: {seq_len}
- **Target Vocabulary Size**: {config.vocab_size}

## 2. Accuracy & Language Modeling Metrics
| Metric | Value |
| :--- | :--- |
| **Validation Loss** | `{eval_results['val_loss']:.4f}` |
| **Perplexity (PPL)** | `{eval_results['perplexity']:.2f}` |
| **Evaluated Batches** | `{eval_results['num_batches']}` |
| **Evaluated Tokens** | `{eval_results['num_tokens']}` |

## 3. Inference Throughput & Latency Benchmarks
| Metric | Value |
| :--- | :--- |
| **Generation Throughput** | `{bench_results['tokens_per_sec']:.2f} tokens/sec` |
| **Latency per Token** | `{bench_results['ms_per_token']:.2f} ms/token` |
| **Total Generation Time** | `{bench_results['total_time_sec']:.4f} sec` |
| **Generated Tokens** | `{bench_results['generated_tokens']}` |
"""

    if output_report:
        report_path = Path(output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_content, encoding="utf-8")
        logger.info(f"Evaluation report written to {report_path}")

    print("\n" + report_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and Benchmark GPT Language Model")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint .pt")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence context length")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Tokens to generate for benchmark")
    parser.add_argument("--output", type=str, default="docs/evaluation_report.md", help="Report output path")
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        seq_len=args.seq_len,
        max_new_tokens=args.max_new_tokens,
        output_report=args.output,
    )
