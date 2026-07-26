import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer random seed (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Returns 'cuda' if CUDA GPU is available, else 'cpu'.

    Returns:
        Device string ('cuda' or 'cpu').
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

