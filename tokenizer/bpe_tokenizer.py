import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from tokenizer.base_tokenizer import BaseTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_pair_counts(
    sequences: List[List[int]],
) -> Dict[Tuple[int, int], int]:
    """Counts frequencies of adjacent integer pairs across sequences."""
    counts: Dict[Tuple[int, int], int] = {}
    for seq in sequences:
        for p1, p2 in zip(seq, seq[1:]):
            pair = (p1, p2)
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge_pair_in_sequence(
    seq: List[int], pair: Tuple[int, int], new_id: int
) -> List[int]:
    """Replaces all occurrences of pair in seq with new_id."""
    new_seq: List[int] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            new_seq.append(new_id)
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return new_seq


class BPETokenizer(BaseTokenizer):
    """Byte Pair Encoding (BPE) Tokenizer implemented from scratch.

    Operates on raw UTF-8 bytes to ensure zero Out-Of-Vocabulary (OOV) tokens,
    with full support for special tokens, training on text files, and file persistence.
    """

    def __init__(
        self,
        vocab_size: int = 500,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """Initializes the tokenizer.

        Args:
            vocab_size: Target vocabulary size.
            special_tokens: List of special token strings (e.g., ["<pad>", "<unk>", "<bos>", "<eos>"]).
        """
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.target_vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens: Dict[int, str] = {}
        self._init_base_vocab(special_tokens)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.special_tokens.get("<eos>", None)

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.special_tokens.get("<bos>", None)

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.special_tokens.get("<pad>", None)

    @property
    def unk_token_id(self) -> Optional[int]:
        return self.special_tokens.get("<unk>", None)

    def _init_base_vocab(self, special_tokens: Optional[List[str]] = None) -> None:
        """Initializes 256 base UTF-8 byte tokens and optional special tokens."""
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.vocab = {}
        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

        # 1. Base byte vocabulary (0..255)
        for i in range(256):
            self.vocab[i] = bytes([i])

        # 2. Special tokens starting at ID 256
        current_id = 256
        for st in special_tokens:
            if st not in self.special_tokens:
                self.special_tokens[st] = current_id
                self.inverse_special_tokens[current_id] = st
                self.vocab[current_id] = st.encode("utf-8")
                current_id += 1

    def _read_corpus(self, files_or_text: Union[List[str], str]) -> str:
        """Helper to load text from a string, a file path, or a list of file paths."""
        if isinstance(files_or_text, str):
            path = Path(files_or_text)
            if path.exists() and path.is_file():
                logger.info(f"Reading training text from file: {path}")
                return path.read_text(encoding="utf-8")
            return files_or_text
        elif isinstance(files_or_text, list):
            texts = []
            for item in files_or_text:
                path = Path(item)
                if path.exists() and path.is_file():
                    logger.info(f"Reading training text from file: {path}")
                    texts.append(path.read_text(encoding="utf-8"))
                else:
                    texts.append(str(item))
            return "\n".join(texts)
        else:
            raise ValueError("Input files_or_text must be a string or list of file paths/strings.")

    def train(
        self,
        files_or_text: Union[List[str], str],
        vocab_size: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """Trains the BPE tokenizer on text string or list of text files.

        Args:
            files_or_text: String text or list of file paths / text strings.
            vocab_size: Optional override for target vocabulary size.
            special_tokens: Optional list of special token strings.
        """
        if vocab_size is not None:
            self.target_vocab_size = vocab_size

        self._init_base_vocab(special_tokens)
        corpus = self._read_corpus(files_or_text)

        if not corpus:
            logger.warning("Empty training corpus provided. Skipping training.")
            return

        min_required_vocab = len(self.vocab)
        if self.target_vocab_size < min_required_vocab:
            raise ValueError(
                f"vocab_size ({self.target_vocab_size}) must be at least {min_required_vocab} "
                f"to accommodate base byte vocabulary and special tokens."
            )

        num_merges = self.target_vocab_size - min_required_vocab
        logger.info(
            f"Starting BPE training. Current vocab size: {len(self.vocab)}, "
            f"Target vocab size: {self.target_vocab_size}, Planned merges: {num_merges}"
        )

        # Split corpus to protect special tokens from being merged into standard BPE pairs
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(s) for s in sorted(self.special_tokens.keys(), key=len, reverse=True)) + ")"
            chunks = re.split(pattern, corpus)
        else:
            chunks = [corpus]

        sequences: List[List[int]] = []
        for chunk in chunks:
            if chunk in self.special_tokens or not chunk:
                continue
            sequences.append(list(chunk.encode("utf-8")))

        for i in range(num_merges):
            pair_counts = _get_pair_counts(sequences)
            if not pair_counts:
                logger.info(f"No further pair candidates to merge at iteration {i}.")
                break

            best_pair = max(pair_counts, key=pair_counts.get) # type: ignore
            if pair_counts[best_pair] < 1:
                break

            new_id = len(self.vocab)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            sequences = [_merge_pair_in_sequence(seq, best_pair, new_id) for seq in sequences]

            if (i + 1) % 50 == 0 or (i + 1) == num_merges:
                logger.info(
                    f"Merge {i + 1}/{num_merges}: Merged pair {best_pair} -> token {new_id} "
                    f"(freq: {pair_counts[best_pair]})"
                )

        logger.info(f"BPE training completed. Final vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """Encodes text into a sequence of token IDs.

        Args:
            text: Input raw text string.

        Returns:
            List of integer token IDs.
        """
        if not text:
            return []

        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(s) for s in sorted(self.special_tokens.keys(), key=len, reverse=True)) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        encoded_ids: List[int] = []

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                encoded_ids.append(self.special_tokens[part])
            else:
                chunk_ids = list(part.encode("utf-8"))
                while len(chunk_ids) >= 2:
                    pairs = [(chunk_ids[i], chunk_ids[i + 1]) for i in range(len(chunk_ids) - 1)]
                    candidate_pairs = [p for p in pairs if p in self.merges]
                    if not candidate_pairs:
                        break
                    best_pair = min(candidate_pairs, key=lambda p: self.merges[p])
                    chunk_ids = _merge_pair_in_sequence(chunk_ids, best_pair, self.merges[best_pair])
                encoded_ids.extend(chunk_ids)

        return encoded_ids

    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs back into text string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded UTF-8 string.
        """
        if not ids:
            return ""

        byte_parts: List[bytes] = []

        for token_id in ids:
            if token_id in self.inverse_special_tokens:
                byte_parts.append(self.inverse_special_tokens[token_id].encode("utf-8"))
            elif token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
            elif "<unk>" in self.special_tokens:
                byte_parts.append(b"<unk>")
            else:
                byte_parts.append(b"?")

        raw_bytes = b"".join(byte_parts)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, directory: Union[str, Path]) -> None:
        """Saves vocabulary and merge rules to JSON files in the specified directory.

        Args:
            directory: Directory path where tokenizer config will be saved.
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        vocab_data = {
            "vocab": {str(token_id): b_val.hex() for token_id, b_val in self.vocab.items()},
            "special_tokens": self.special_tokens,
            "target_vocab_size": self.target_vocab_size,
        }

        merges_data = [
            {"p1": p1, "p2": p2, "id": merged_id}
            for (p1, p2), merged_id in self.merges.items()
        ]

        vocab_file = dir_path / "vocab.json"
        merges_file = dir_path / "merges.json"

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)

        with open(merges_file, "w", encoding="utf-8") as f:
            json.dump(merges_data, f, indent=2)

        logger.info(f"Saved vocabulary to {vocab_file} and merges to {merges_file}")

    def load(self, directory: Union[str, Path]) -> None:
        """Loads vocabulary and merge rules from JSON files in the specified directory.

        Args:
            directory: Directory path containing vocab.json and merges.json.
        """
        dir_path = Path(directory)
        vocab_file = dir_path / "vocab.json"
        merges_file = dir_path / "merges.json"

        if not vocab_file.exists() or not merges_file.exists():
            raise FileNotFoundError(
                f"Tokenizer files missing in {dir_path}. Ensure vocab.json and merges.json exist."
            )

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        with open(merges_file, "r", encoding="utf-8") as f:
            merges_data = json.load(f)

        self.target_vocab_size = vocab_data.get("target_vocab_size", 500)
        self.special_tokens = vocab_data.get("special_tokens", {})
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}

        self.vocab = {}
        for token_id_str, hex_val in vocab_data["vocab"].items():
            self.vocab[int(token_id_str)] = bytes.fromhex(hex_val)

        self.merges = {}
        for item in merges_data:
            self.merges[(item["p1"], item["p2"])] = item["id"]

        logger.info(f"Loaded tokenizer from {dir_path}. Vocab size: {len(self.vocab)}, Merges: {len(self.merges)}")

