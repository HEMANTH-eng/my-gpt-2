from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset

from utils.logger import get_logger

logger = get_logger("dataset_curator")


@dataclass
class SFTExample:
    """Supervised Fine-Tuning prompt-response training example."""

    prompt: str
    response: str
    domain: str = "general"


DOMAIN_CORPORA: Dict[str, List[SFTExample]] = {
    "general": [
        SFTExample(
            prompt="hii",
            response="Hello! How can I help you today?",
            domain="general",
        ),
        SFTExample(
            prompt="hi",
            response="Hello! How can I assist you today?",
            domain="general",
        ),
        SFTExample(
            prompt="hello",
            response="Hello! I am MyGPT. How can I assist you today?",
            domain="general",
        ),
        SFTExample(
            prompt="who are you",
            response="I am MyGPT, an autonomous AI assistant and custom PyTorch language model built from scratch.",
            domain="general",
        ),
        SFTExample(
            prompt="what can you do",
            response="I can generate text, write and debug code, answer questions, run web searches, solve mathematical problems, and coordinate AI agents.",
            domain="general",
        ),
        SFTExample(
            prompt="how are you",
            response="I am doing great and ready to assist you! What are we working on today?",
            domain="general",
        ),
    ],
    "coding": [
        SFTExample(
            prompt="Write a Python function to compute the Fibonacci sequence up to n.",
            response="def fibonacci(n):\n    a, b = 0, 1\n    res = []\n    while a < n:\n        res.append(a)\n        a, b = b, a + b\n    return res",
            domain="coding",
        ),
        SFTExample(
            prompt="Implement binary search in Python.",
            response="def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
            domain="coding",
        ),
    ],
    "medical": [
        SFTExample(
            prompt="What are the symptoms and initial treatment for acute myocardial infarction?",
            response="Symptoms of acute myocardial infarction (heart attack) include chest pain/pressure, dyspnea, diaphoresis, and radiation to the jaw or arm. Initial management includes oxygen, aspirin, nitroglycerin, and immediate reperfusion therapy.",
            domain="medical",
        ),
        SFTExample(
            prompt="Explain the mechanism of action of ACE inhibitors.",
            response="Angiotensin-Converting Enzyme (ACE) inhibitors block the conversion of Angiotensin I to Angiotensin II, reducing vasoconstriction and aldosterone secretion, which lowers blood pressure.",
            domain="medical",
        ),
    ],
    "legal": [
        SFTExample(
            prompt="What constitutes a legally binding contract under common law?",
            response="A valid contract requires four essential elements: (1) Offer, (2) Acceptance, (3) Consideration (something of value exchanged), and (4) Mutual Intent to create legal obligations.",
            domain="legal",
        ),
        SFTExample(
            prompt="Define the legal principle of Stare Decisis.",
            response="Stare Decisis is a legal doctrine obligating courts to follow historical precedent set by previous higher court decisions when resolving similar legal issues.",
            domain="legal",
        ),
    ],
    "math": [
        SFTExample(
            prompt="Solve the equation 2x + 5 = 17 step by step.",
            response="Step 1: Subtract 5 from both sides: 2x = 12.\nStep 2: Divide both sides by 2: x = 6.\nFinal Answer: x = 6.",
            domain="math",
        ),
        SFTExample(
            prompt="Calculate the derivative of f(x) = 3x^2 + 4x - 7.",
            response="Using the power rule d/dx(x^n) = n*x^(n-1):\nf'(x) = 3*(2x) + 4*(1) - 0 = 6x + 4.",
            domain="math",
        ),
    ],
}


class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning (SFT) with response-only loss masking."""

    def __init__(
        self,
        examples: List[SFTExample],
        tokenizer: Any,
        max_seq_len: int = 128,
        ignore_index: int = -100,
    ) -> None:
        """Initializes SFTDataset.

        Args:
            examples: List of SFTExample instances.
            tokenizer: BPETokenizer instance.
            max_seq_len: Maximum sequence context length.
            ignore_index: CrossEntropyLoss ignored index (-100).
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        self.processed = []

        for ex in examples:
            prompt_formatted = f"User: {ex.prompt}\nAssistant:"
            prompt_ids = tokenizer.encode(prompt_formatted)
            response_ids = tokenizer.encode(" " + ex.response)

            full_ids = prompt_ids + response_ids
            if len(full_ids) > max_seq_len + 1:
                full_ids = full_ids[: max_seq_len + 1]

            input_ids = full_ids[:-1]
            target_ids = full_ids[1:]

            # Loss Masking: Set prompt tokens to ignore_index (-100)
            # Only response tokens maintain target token IDs!
            target_mask = []
            prompt_token_count = max(0, len(prompt_ids) - 1)
            for i, tid in enumerate(target_ids):
                if i < prompt_token_count:
                    target_mask.append(ignore_index)
                else:
                    target_mask.append(tid)

            # Pad sequences to max_seq_len if needed
            pad_id = getattr(tokenizer, "pad_token_id", 0)
            padding_len = max_seq_len - len(input_ids)
            if padding_len > 0:
                input_ids = input_ids + [pad_id] * padding_len
                target_mask = target_mask + [ignore_index] * padding_len

            self.processed.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_mask, dtype=torch.long),
            ))

    def __len__(self) -> int:
        return len(self.processed)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.processed[idx]


def load_domain_dataset(domain: str) -> List[SFTExample]:
    """Retrieves domain-specific SFT training examples."""
    domain_key = domain.lower()
    if domain_key == "all":
        all_examples = []
        for lst in DOMAIN_CORPORA.values():
            all_examples.extend(lst)
        return all_examples
    return DOMAIN_CORPORA.get(domain_key, DOMAIN_CORPORA["general"])

