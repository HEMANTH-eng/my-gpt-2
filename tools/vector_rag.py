"""Vector Retrieval-Augmented Generation (RAG) Engine for Novexa AI."""

import math
from typing import Any, Dict, List, Optional
import torch

from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Splits raw text documents into fixed-size overlapping text chunks."""

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source_id: str = "doc") -> List[Dict[str, Any]]:
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_str = " ".join(chunk_words)
            chunks.append({
                "id": f"{source_id}_chunk_{len(chunks)}",
                "text": chunk_str,
                "source": source_id,
                "start_word": i,
                "end_word": i + len(chunk_words),
            })
            if i + self.chunk_size >= len(words):
                break

        logger.info(f"Chunked document '{source_id}' into {len(chunks)} text chunks")
        return chunks


class SimpleVectorRAG:
    """Lightweight in-memory vector similarity index for Document RAG grounding."""

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[torch.Tensor] = None

    def _dummy_embed(self, text: str) -> torch.Tensor:
        """Generates deterministic pseudo-embedding vector for text query."""
        vec = torch.zeros(self.embedding_dim)
        for i, char in enumerate(text[:64]):
            vec[i % self.embedding_dim] += ord(char) / 255.0
        norm = vec.norm(p=2)
        if norm > 0:
            vec = vec / norm
        return vec

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Embeds and indexes document chunks."""
        if not chunks:
            return

        new_vecs = [self._dummy_embed(c["text"]) for c in chunks]
        new_tensor = torch.stack(new_vecs)

        if self.embeddings is None:
            self.embeddings = new_tensor
        else:
            self.embeddings = torch.cat([self.embeddings, new_tensor], dim=0)

        self.chunks.extend(chunks)
        logger.info(f"Indexed {len(chunks)} chunks into vector RAG store. Total: {len(self.chunks)}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs Cosine Similarity vector search against indexed document chunks."""
        if not self.chunks or self.embeddings is None:
            return []

        q_vec = self._dummy_embed(query).unsqueeze(0)  # (1, d)
        # Cosine Similarity dot product: (1, d) @ (N, d)^T -> (1, N)
        scores = torch.mm(q_vec, self.embeddings.T).squeeze(0)  # (N,)

        top_k = min(top_k, len(self.chunks))
        top_scores, top_indices = torch.topk(scores, k=top_k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            item = dict(self.chunks[idx])
            item["score"] = round(float(score), 4)
            results.append(item)

        return results
