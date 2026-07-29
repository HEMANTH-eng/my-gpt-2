"""Multi-format Dataset Ingestion & Preprocessing Pipeline for Novexa AI."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetProcessor:
    """Ingest, clean, deduplicate, and format text corpora for LLM pretraining and SFT."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("./data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """Sanitizes raw text: removes HTML tags, normalizes whitespace, strips control chars."""
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize non-printable control characters except newline and tab
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        # Normalize multiple spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Normalize multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """Deduplicates text entries based on exact hash matching."""
        seen = set()
        deduped = []
        for t in texts:
            cleaned = t.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                deduped.append(cleaned)
        logger.info(f"Deduplicated dataset: {len(texts)} -> {len(deduped)} entries")
        return deduped

    def ingest_file(self, file_path: Union[str, Path]) -> List[str]:
        """Ingests multi-format file (.json, .jsonl, .txt, .md, .csv) into clean text strings."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()
        extracted_texts: List[str] = []

        if ext == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if isinstance(data, dict):
                            text = data.get("text") or data.get("content") or data.get("response") or str(data)
                        else:
                            text = str(data)
                        extracted_texts.append(self.clean_text(text))

        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text = item.get("text") if isinstance(item, dict) else str(item)
                        extracted_texts.append(self.clean_text(text))
                elif isinstance(data, dict):
                    extracted_texts.append(self.clean_text(json.dumps(data)))

        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                extracted_texts.append(self.clean_text(content))

        elif ext == ".csv":
            import csv
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    extracted_texts.append(self.clean_text(" ".join(row)))
        else:
            logger.warning(f"Unsupported file format extension '{ext}' for {file_path}")

        return self.remove_duplicates(extracted_texts)

    def get_dataset_stats(self, texts: List[str], tokenizer: Optional[Any] = None) -> Dict[str, Any]:
        """Calculates dataset statistics: total characters, word counts, line counts, and optional BPE tokens."""
        total_chars = sum(len(t) for t in texts)
        total_words = sum(len(t.split()) for t in texts)
        total_tokens = 0

        if tokenizer is not None and hasattr(tokenizer, "encode"):
            total_tokens = sum(len(tokenizer.encode(t)) for t in texts)

        stats = {
            "num_samples": len(texts),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_words_per_sample": round(total_words / max(len(texts), 1), 2),
            "total_bpe_tokens": total_tokens,
        }
        logger.info(f"Dataset Stats: {stats}")
        return stats
