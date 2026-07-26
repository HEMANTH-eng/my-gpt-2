from typing import Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    """PyTorch Dataset for autoregressive language model training.

    Slices 1D token ID sequences into (input x, target y) pairs of length seq_len,
    where target y is shifted by 1 token position.
    Supports in-memory NumPy arrays, PyTorch Tensors, and disk-backed np.memmap files.
    """

    def __init__(
        self,
        tokens: Union[np.ndarray, np.memmap, torch.Tensor],
        seq_len: int = 1024,
    ) -> None:
        """Initializes the GPT dataset.

        Args:
            tokens: 1D array/tensor containing pre-tokenized token IDs.
            seq_len: Context window size (sequence length).
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be a positive integer, got {seq_len}")

        if isinstance(tokens, torch.Tensor):
            self.tokens = tokens.numpy()
        else:
            self.tokens = tokens

        self.seq_len = seq_len
        self.num_tokens = len(self.tokens)

        if self.num_tokens <= seq_len:
            raise ValueError(
                f"Total tokens ({self.num_tokens}) must be greater than sequence length ({seq_len})."
            )

    def __len__(self) -> int:
        """Returns total number of valid sequence windows."""
        return self.num_tokens - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetches the i-th input sequence x and target sequence y.

        Args:
            idx: Index of sequence window.

        Returns:
            Tuple (x, y) where:
                x: Tensor of shape (seq_len,) containing token IDs.
                y: Tensor of shape (seq_len,) containing next token IDs (shifted right by 1).
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        chunk = self.tokens[idx : idx + self.seq_len + 1]
        x_np = chunk[:-1].astype(np.int64)
        y_np = chunk[1:].astype(np.int64)

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        return x, y

