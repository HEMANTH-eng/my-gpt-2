from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np

from tokenizer.bpe_tokenizer import BPETokenizer
from utils.logger import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """Preprocesses large raw text datasets into tokenized, memory-mapped binary files."""

    def __init__(self, tokenizer: BPETokenizer) -> None:
        """Initializes preprocessor with a trained tokenizer.

        Args:
            tokenizer: Instance of BPETokenizer.
        """
        self.tokenizer = tokenizer

    def process_and_save(
        self,
        files_or_text: Union[List[str], str],
        output_dir: Union[str, Path],
        val_ratio: float = 0.1,
        chunk_size: int = 1_000_000,
    ) -> Tuple[Path, Path]:
        """Tokenizes text dataset, splits into train/val, and saves memory-mapped binary files.

        Args:
            files_or_text: Single text string, path to a text file, or list of file paths.
            output_dir: Directory where train.bin and val.bin will be saved.
            val_ratio: Proportion of tokens to assign to validation split (0.0 to 1.0).
            chunk_size: Character chunk size for memory-friendly tokenization.

        Returns:
            Tuple of (train_bin_path, val_bin_path).
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if not (0.0 <= val_ratio < 1.0):
            raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {val_ratio}")

        logger.info("Reading text corpus for preprocessing...")
        text_corpus = self.tokenizer._read_corpus(files_or_text)

        logger.info(f"Tokenizing corpus of {len(text_corpus)} characters...")
        all_tokens: List[int] = []

        for i in range(0, len(text_corpus), chunk_size):
            chunk = text_corpus[i : i + chunk_size]
            tokens = self.tokenizer.encode(chunk)
            all_tokens.extend(tokens)

        total_tokens = len(all_tokens)
        logger.info(f"Tokenization completed. Total tokens: {total_tokens}")

        if total_tokens == 0:
            raise ValueError("Tokenized corpus contains 0 tokens. Cannot build dataset.")

        # Determine numpy dtype based on vocabulary size
        vocab_size = len(self.tokenizer.vocab)
        dtype = np.uint16 if vocab_size < 65535 else np.uint32

        # Train / Validation split
        val_size = int(total_tokens * val_ratio)
        train_size = total_tokens - val_size

        train_tokens = np.array(all_tokens[:train_size], dtype=dtype)
        val_tokens = np.array(all_tokens[train_size:], dtype=dtype)

        train_bin_path = out_path / "train.bin"
        val_bin_path = out_path / "val.bin"

        train_tokens.tofile(train_bin_path)
        logger.info(f"Saved {train_size} training tokens to {train_bin_path}")

        if val_size > 0:
            val_tokens.tofile(val_bin_path)
            logger.info(f"Saved {val_size} validation tokens to {val_bin_path}")
        else:
            # Empty val file if ratio is 0
            val_tokens.tofile(val_bin_path)

        return train_bin_path, val_bin_path

    @staticmethod
    def load_memmap(
        bin_path: Union[str, Path],
        dtype: np.dtype = np.uint16,
    ) -> np.memmap:
        """Loads pre-tokenized binary token file using memory-mapping.

        Args:
            bin_path: Path to binary token file (.bin).
            dtype: NumPy data type (np.uint16 or np.uint32).

        Returns:
            np.memmap array pointing to disk-backed token sequence.
        """
        path = Path(bin_path)
        if not path.exists():
            raise FileNotFoundError(f"Binary file not found: {path}")

        return np.memmap(path, dtype=dtype, mode="r")
