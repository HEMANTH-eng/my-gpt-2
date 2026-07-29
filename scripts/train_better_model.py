import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.model_config import GPTConfig
from config.train_config import TrainConfig
from dataset.curator import DOMAIN_CORPORA, load_domain_dataset
from models.gpt import GPT
from tokenizer.bpe_tokenizer import BPETokenizer
from training.continued_pretrain import ContinuedPretrainer
from training.sft_trainer import SFTTrainer
from utils.benchmarks import run_domain_benchmarks
from utils.logger import get_logger

logger = get_logger("train_better_model")


def main():
    parser = argparse.ArgumentParser(description="Train Better GPT Models: Domain SFT, Continued Pretraining & Benchmarks")
    parser.add_argument("--mode", type=str, choices=["sft", "pretrain_continue", "benchmark"], default="sft", help="Execution mode")
    parser.add_argument("--domain", type=str, choices=["all", "general", "coding", "medical", "legal", "math"], default="all", help="Target domain preset")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--checkpoint", type=str, default="", help="Input checkpoint path to resume/fine-tune")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train lightweight base tokenizer on rich domain text
    corpus_parts = [
        "Building high quality domain specific GPT language models for coding, medical, legal, and mathematics.",
        "System: You are a helpful AI assistant. User: hi Assistant: Hello! How can I help you today?",
        "User: Explain python functions. Assistant: A Python function is defined using the def keyword followed by function name and parameters.",
    ]
    for domain_examples in DOMAIN_CORPORA.values():
        for ex in domain_examples:
            corpus_parts.append(ex.prompt)
            corpus_parts.append(ex.response)

    corpus = "\n".join(corpus_parts)
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(corpus)

    actual_vocab_size = len(tokenizer.vocab)
    config = GPTConfig.gpt_micro(vocab_size=actual_vocab_size)
    model = GPT(config)

    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Loading base checkpoint weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "sft":
        logger.info(f"=== Running Supervised Fine-Tuning (SFT) for Domain: {args.domain} ===")
        sft_trainer = SFTTrainer(model=model, tokenizer=tokenizer, device=device)
        best_loss = sft_trainer.fine_tune_domain(domain=args.domain, epochs=args.epochs)
        logger.info(f"Domain '{args.domain}' SFT complete. Best Response Loss: {best_loss:.4f}")

        # Save trained checkpoint and tokenizer
        best_path = ckpt_dir / "best.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": config}, best_path)
        tokenizer.save(ckpt_dir / "tokenizer")
        logger.info(f"Saved trained checkpoint to '{best_path}' and tokenizer to '{ckpt_dir / 'tokenizer'}'.")

    elif args.mode == "pretrain_continue":
        logger.info("=== Running Continued Pretraining Pipeline ===")
        pretrainer = ContinuedPretrainer(model=model, device=device)
        dummy_tokens = torch.randint(0, config.vocab_size, (200,))
        loss = pretrainer.train_on_corpus(token_ids=dummy_tokens, epochs=args.epochs)
        logger.info(f"Continued pretraining complete. Loss: {loss:.4f}")

        best_path = ckpt_dir / "best.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": config}, best_path)
        tokenizer.save(ckpt_dir / "tokenizer")
        logger.info(f"Saved trained checkpoint to '{best_path}' and tokenizer to '{ckpt_dir / 'tokenizer'}'.")

    elif args.mode == "benchmark":
        logger.info("=== Running Domain Benchmark Matrix ===")
        results = run_domain_benchmarks(model=model, tokenizer=tokenizer, device=device)
        print("\n=== Domain Benchmark Matrix Results ===")
        for domain_name, score in results.items():
            print(f"  • {domain_name.replace('_', ' ').title()}: {score:.1f}%")


if __name__ == "__main__":
    main()
