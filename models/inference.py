from typing import Generator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gpt import GPT
from tokenizer.bpe_tokenizer import BPETokenizer
from utils.logger import get_logger

logger = get_logger(__name__)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    greedy: bool = False,
    unprintable_token_ids: Optional[List[int]] = None,
) -> torch.Tensor:
    """Samples next token ID from 2D logits tensor of shape (batch_size, vocab_size).

    Supports:
        - Greedy decoding (argmax)
        - Temperature scaling
        - Top-k truncation
        - Top-p (Nucleus) cumulative probability sampling
        - Unprintable byte token suppression
    """
    logits = logits.clone()
    if unprintable_token_ids:
        logits[:, unprintable_token_ids] = -float("Inf")

    if greedy or temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Apply Temperature scaling
    logits = logits / max(temperature, 1e-5)

    # Apply Top-k truncation
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

    # Apply Top-p (Nucleus) sampling
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right so the first token above top_p is kept
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        for b in range(logits.size(0)):
            indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
            logits[b, indices_to_remove] = -float("Inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class GPTGenerator:
    """Inference and decoding engine for GPT models supporting advanced sampling, beam search, and streaming."""

    def __init__(
        self,
        model: GPT,
        tokenizer: Optional[BPETokenizer] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes GPTGenerator.

        Args:
            model: Trained GPT PyTorch model.
            tokenizer: Optional BPETokenizer for text prompt encoding/decoding.
            device: Map device ('cuda', 'cpu').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.unprintable_token_ids: List[int] = []

        if self.tokenizer and hasattr(self.tokenizer, "vocab"):
            for token_id, b_val in self.tokenizer.vocab.items():
                try:
                    s = b_val.decode("utf-8")
                    if not all(c.isprintable() or c in "\n\r\t" for c in s):
                        self.unprintable_token_ids.append(token_id)
                except Exception:
                    self.unprintable_token_ids.append(token_id)

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = False,
        num_beams: int = 1,
        eos_token_id: Optional[int] = None,
    ) -> Union[str, torch.Tensor]:
        """Generates text or token IDs given an initial prompt.

        Args:
            prompt: String text or 2D torch.Tensor of prompt token IDs (batch_size, seq_len).
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k truncation threshold.
            top_p: Top-p nucleus sampling threshold.
            greedy: If True, uses deterministic greedy decoding (argmax).
            num_beams: Number of beams for beam search decoding (if > 1).
            eos_token_id: End of sequence token ID to halt generation early.

        Returns:
            Generated text string (if prompt was string and tokenizer is available) or 2D Tensor of token IDs.
        """
        is_string_prompt = isinstance(prompt, str)
        if is_string_prompt:
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be provided to generator when supplying string prompt.")
            prompt_ids = self.tokenizer.encode(prompt)
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        else:
            idx = prompt.to(self.device)

        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        if num_beams > 1:
            generated_idx = self._beam_search(
                idx=idx,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                eos_token_id=eos_token_id,
            )
        else:
            generated_idx = self._sample_generate(
                idx=idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                eos_token_id=eos_token_id,
            )

        if is_string_prompt and self.tokenizer is not None:
            return self.tokenizer.decode(generated_idx[0].tolist())
        return generated_idx

    def _sample_generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        greedy: bool,
        eos_token_id: Optional[int],
    ) -> torch.Tensor:
        """Helper executing sample-based autoregressive generation loop."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.model.config.max_seq_len else idx[:, -self.model.config.max_seq_len :]
            logits, _ = self.model(idx_cond)
            logits_last = logits[:, -1, :]

            next_token = sample_next_token(
                logits=logits_last,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                unprintable_token_ids=self.unprintable_token_ids,
            )

            idx = torch.cat((idx, next_token), dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return idx

    def _beam_search(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        eos_token_id: Optional[int],
    ) -> torch.Tensor:
        """Executes Beam Search decoding over num_beams parallel hypothesis sequences."""
        batch_size = idx.size(0)
        if batch_size != 1:
            raise ValueError("Beam search currently supports batch_size=1.")

        beams = [(idx, 0.0)]

        for _ in range(max_new_tokens):
            all_candidates = []
            for seq, score in beams:
                if eos_token_id is not None and seq[0, -1].item() == eos_token_id:
                    all_candidates.append((seq, score))
                    continue

                seq_cond = seq if seq.size(1) <= self.model.config.max_seq_len else seq[:, -self.model.config.max_seq_len :]
                logits, _ = self.model(seq_cond)
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

                top_log_probs, top_indices = torch.topk(log_probs, num_beams, dim=-1)

                for i in range(num_beams):
                    token_id = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                    token_score = top_log_probs[0, i].item()

                    new_seq = torch.cat((seq, token_id), dim=1)
                    new_score = score + token_score
                    all_candidates.append((new_seq, new_score))

            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:num_beams]

        best_seq, _ = beams[0]
        return best_seq

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = False,
        eos_token_id: Optional[int] = None,
    ) -> Generator[Union[str, int], None, None]:
        """Streaming generator yielding tokens or decoded text chunks in real time as generated.

        Yields:
            Decoded string piece if prompt was a string, or integer token ID if prompt was a Tensor.
        """
        is_string_prompt = isinstance(prompt, str)
        if is_string_prompt:
            if self.tokenizer is None:
                raise ValueError("Tokenizer must be provided to generator when supplying string prompt.")
            prompt_ids = self.tokenizer.encode(prompt)
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        else:
            idx = prompt.to(self.device)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.model.config.max_seq_len else idx[:, -self.model.config.max_seq_len :]
            logits, _ = self.model(idx_cond)
            logits_last = logits[:, -1, :]

            next_token = sample_next_token(
                logits=logits_last,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                unprintable_token_ids=self.unprintable_token_ids,
            )

            idx = torch.cat((idx, next_token), dim=1)
            token_id = next_token.item()

            if is_string_prompt and self.tokenizer is not None:
                yield self.tokenizer.decode([token_id])
            else:
                yield token_id

            if eos_token_id is not None and token_id == eos_token_id:
                break
