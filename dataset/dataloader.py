from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import DataLoader

from dataset.dataset import GPTDataset
from dataset.preprocessor import TextPreprocessor
from tokenizer.bpe_tokenizer import BPETokenizer
from utils.logger import get_logger

logger = get_logger(__name__)


def create_dataloaders(
    train_dataset: GPTDataset,
    val_dataset: Optional[GPTDataset] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Creates PyTorch DataLoaders for training and validation datasets.

    Args:
        train_dataset: Instantiated GPTDataset for training.
        val_dataset: Optional instantiated GPTDataset for validation.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle training dataset batches.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If True, copies tensors to CUDA pinned memory before returning.
        drop_last: Set to True to drop incomplete last batch.

    Returns:
        Tuple of (train_loader, val_loader). val_loader is None if val_dataset is None.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader: Optional[DataLoader] = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader


def prepare_gpt_dataloaders(
    files_or_text: Union[List[str], str],
    tokenizer: BPETokenizer,
    output_dir: Union[str, Path],
    seq_len: int = 512,
    batch_size: int = 16,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """High-level end-to-end dataset pipeline.

    Processes raw text, tokenizes, splits into train/val, caches binary token files,
    constructs memory-mapped GPTDatasets, and returns ready-to-use PyTorch DataLoaders.

    Args:
        files_or_text: Input raw text or list of file paths.
        tokenizer: Trained BPETokenizer instance.
        output_dir: Directory for cached preprocessed binary files.
        seq_len: Sequence length context window.
        batch_size: Batch size for DataLoaders.
        val_ratio: Fraction of tokens for validation split.
        num_workers: Number of DataLoader worker threads.
        pin_memory: CUDA pin memory flag.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    preprocessor = TextPreprocessor(tokenizer)
    train_bin, val_bin = preprocessor.process_and_save(
        files_or_text=files_or_text,
        output_dir=output_dir,
        val_ratio=val_ratio,
    )

    vocab_size = len(tokenizer.vocab)
    dtype = np.uint16 if vocab_size < 65535 else np.uint32

    train_tokens = TextPreprocessor.load_memmap(train_bin, dtype=dtype)
    train_dataset = GPTDataset(train_tokens, seq_len=seq_len)

    val_dataset: Optional[GPTDataset] = None
    val_tokens = TextPreprocessor.load_memmap(val_bin, dtype=dtype)
    if len(val_tokens) > seq_len:
        val_dataset = GPTDataset(val_tokens, seq_len=seq_len)

    return create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

