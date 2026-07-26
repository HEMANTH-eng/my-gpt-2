from abc import ABC, abstractmethod
from typing import List, Optional, Union
from pathlib import Path


class BaseTokenizer(ABC):
    """Abstract base class defining the contract for tokenizers."""

    @abstractmethod
    def train(
        self,
        files_or_text: Union[List[str], str],
        vocab_size: int,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """Trains the tokenizer on input text or list of text files.

        Args:
            files_or_text: A single text string, path to a text file, or list of file paths.
            vocab_size: Target vocabulary size.
            special_tokens: Optional list of special token strings (e.g. ["<pad>", "<unk>", "<bos>", "<eos>"]).
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encodes input text into a list of token IDs.

        Args:
            text: Raw input string.

        Returns:
            List of integer token IDs.
        """
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs back into a text string.

        Args:
            ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        pass

    @abstractmethod
    def save(self, directory: Union[str, Path]) -> None:
        """Saves the vocabulary and tokenizer state to a directory.

        Args:
            directory: Path to directory where tokenizer state will be saved.
        """
        pass

    @abstractmethod
    def load(self, directory: Union[str, Path]) -> None:
        """Loads the vocabulary and tokenizer state from a directory.

        Args:
            directory: Path to directory containing tokenizer state files.
        """
        pass

