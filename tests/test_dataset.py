import numpy as np
import pytest
import torch

from dataset.dataloader import create_dataloaders, prepare_gpt_dataloaders
from dataset.dataset import GPTDataset
from dataset.preprocessor import TextPreprocessor
from tokenizer.bpe_tokenizer import BPETokenizer


def test_gpt_dataset_shapes():
    seq_len = 16
    total_tokens = 100
    tokens = np.arange(total_tokens, dtype=np.int64)

    dataset = GPTDataset(tokens, seq_len=seq_len)
    assert len(dataset) == total_tokens - seq_len

    x, y = dataset[0]
    assert x.shape == (seq_len,)
    assert y.shape == (seq_len,)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64

    # Verify target shifting: y[i] == x[i + 1]
    assert torch.equal(x[1:], y[:-1])
    assert y[-1].item() == tokens[seq_len]


def test_dataset_invalid_seq_len():
    tokens = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError):
        GPTDataset(tokens, seq_len=20)


def test_preprocessor_and_memmap(tmp_path):
    tokenizer = BPETokenizer(vocab_size=280)
    corpus = "The quick brown fox jumps over the lazy dog. " * 20
    tokenizer.train(corpus)

    preprocessor = TextPreprocessor(tokenizer)
    train_bin, val_bin = preprocessor.process_and_save(
        files_or_text=corpus,
        output_dir=tmp_path,
        val_ratio=0.2,
    )

    assert train_bin.exists()
    assert val_bin.exists()

    train_memmap = TextPreprocessor.load_memmap(train_bin, dtype=np.uint16)
    val_memmap = TextPreprocessor.load_memmap(val_bin, dtype=np.uint16)

    total_tokens = len(train_memmap) + len(val_memmap)
    assert total_tokens > 0
    assert abs((len(val_memmap) / total_tokens) - 0.2) < 0.05


def test_dataloader_batching():
    seq_len = 8
    batch_size = 4
    tokens = np.random.randint(0, 100, size=200, dtype=np.int64)

    dataset = GPTDataset(tokens, seq_len=seq_len)
    train_loader, _ = create_dataloaders(
        train_dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    for x_batch, y_batch in train_loader:
        assert x_batch.shape == (batch_size, seq_len)
        assert y_batch.shape == (batch_size, seq_len)
        assert x_batch.dtype == torch.int64
        assert y_batch.dtype == torch.int64
        break


def test_end_to_end_pipeline(tmp_path):
    corpus = "Building a GPT-style LLM from scratch in Python and PyTorch is fun! " * 50
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(corpus)

    seq_len = 16
    batch_size = 4

    train_loader, val_loader = prepare_gpt_dataloaders(
        files_or_text=corpus,
        tokenizer=tokenizer,
        output_dir=tmp_path / "data_cache",
        seq_len=seq_len,
        batch_size=batch_size,
        val_ratio=0.2,
    )

    assert train_loader is not None
    assert val_loader is not None

    for x, y in train_loader:
        assert x.shape == (batch_size, seq_len)
        assert y.shape == (batch_size, seq_len)
        break

    for x_val, y_val in val_loader:
        assert x_val.shape == (batch_size, seq_len)
        assert y_val.shape == (batch_size, seq_len)
        break

