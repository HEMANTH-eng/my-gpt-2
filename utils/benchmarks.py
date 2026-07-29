from typing import Any, Dict
import torch

from utils.logger import get_logger

logger = get_logger("domain_benchmarks")


def run_domain_benchmarks(
    model: Any,
    tokenizer: Any,
    device: str = "cpu",
) -> Dict[str, float]:
    """Runs automated benchmark evaluation suite across domains (Coding, Medical, Legal, Math, Instruction).

    Args:
        model: PyTorch GPT model instance.
        tokenizer: BPETokenizer instance.
        device: Execution target device string ('cuda', 'cpu').

    Returns:
        Dictionary of score percentages per domain (0.0 to 100.0%).
    """
    from models.inference import GPTGenerator

    logger.info("Executing Domain Benchmark Evaluation Suite...")
    generator = GPTGenerator(model=model, tokenizer=tokenizer, device=device)

    scores = {}

    # 1. Coding Benchmark
    coding_prompts = [
        "Write a Python function to compute factorial:",
        "Implement a binary search function:",
    ]
    coding_score = 0
    for prompt in coding_prompts:
        out = generator.generate(prompt, max_new_tokens=30, greedy=True)
        if isinstance(out, str) and ("def " in out or "return " in out or "while " in out or "for " in out):
            coding_score += 50
    scores["coding_score"] = float(coding_score)

    # 2. Medical Benchmark
    medical_prompts = [
        "What are symptoms of acute myocardial infarction?",
        "Explain ACE inhibitor mechanism:",
    ]
    medical_score = 0
    for prompt in medical_prompts:
        out = generator.generate(prompt, max_new_tokens=30, greedy=True)
        if isinstance(out, str) and any(w in out.lower() for w in ["symptom", "pain", "inhibitor", "blood", "pressure"]):
            medical_score += 50
    scores["medical_score"] = float(medical_score)

    # 3. Legal Benchmark
    legal_prompts = [
        "What constitutes a legally binding contract?",
        "Define stare decisis doctrine:",
    ]
    legal_score = 0
    for prompt in legal_prompts:
        out = generator.generate(prompt, max_new_tokens=30, greedy=True)
        if isinstance(out, str) and any(w in out.lower() for w in ["contract", "court", "precedent", "legal", "law"]):
            legal_score += 50
    scores["legal_score"] = float(legal_score)

    # 4. Math Benchmark
    math_prompts = [
        "Solve 2x + 5 = 17 step by step:",
        "Calculate the derivative of f(x) = 3x^2:",
    ]
    math_score = 0
    for prompt in math_prompts:
        out = generator.generate(prompt, max_new_tokens=30, greedy=True)
        if isinstance(out, str) and any(w in out.lower() for w in ["step", "=", "x", "answer", "6"]):
            math_score += 50
    scores["math_score"] = float(math_score)

    # 5. Overall Instruction Following Benchmark Average
    scores["overall_benchmark_avg"] = sum(scores.values()) / len(scores)

    logger.info(
        f"Benchmark Results: Coding: {scores['coding_score']:.1f}% | "
        f"Medical: {scores['medical_score']:.1f}% | Legal: {scores['legal_score']:.1f}% | "
        f"Math: {scores['math_score']:.1f}% | Overall: {scores['overall_benchmark_avg']:.1f}%"
    )

    return scores
