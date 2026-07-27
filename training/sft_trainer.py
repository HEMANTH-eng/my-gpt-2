from pathlib import Path
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.train_config import TrainConfig
from dataset.curator import SFTDataset, SFTExample, load_domain_dataset
from models.gpt import GPT
from training.checkpoint import CheckpointManager
from training.optimizer import CosineWarmupScheduler, configure_optimizer
from utils.logger import get_logger

logger = get_logger("sft_trainer")


class SFTTrainer:
    """Supervised Fine-Tuning (SFT) Engine with Response-Only Loss Masking."""

    def __init__(
        self,
        model: GPT,
        tokenizer: Any,
        config: Optional[TrainConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes SFTTrainer.

        Args:
            model: PyTorch GPT model instance.
            tokenizer: BPETokenizer instance.
            config: TrainConfig dataclass.
            device: Target device ('cuda', 'cpu').
        """
        self.config = config or TrainConfig.micro_config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # SFT Hyperparameters
        self.config.learning_rate = 5e-5  # Low learning rate for fine-tuning
        self.config.min_lr = 5e-6
        self.config.warmup_steps = 5

        self.optimizer = configure_optimizer(self.model, self.config)
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            learning_rate=self.config.learning_rate,
            min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps,
            decay_steps=50,
        )
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

    def fine_tune_domain(
        self,
        domain: str = "coding",
        custom_examples: Optional[List[SFTExample]] = None,
        epochs: int = 3,
        batch_size: int = 2,
    ) -> float:
        """Fine-tunes the model for a specific domain (coding, medical, legal, math).

        Args:
            domain: Domain name preset.
            custom_examples: Optional list of SFTExample instances.
            epochs: Number of fine-tuning epochs.
            batch_size: DataLoader batch size.

        Returns:
            Final validation/training loss over assistant response tokens.
        """
        examples = custom_examples or load_domain_dataset(domain)
        logger.info(f"Starting Supervised Fine-Tuning for domain '{domain}' with {len(examples)} examples...")

        sft_dataset = SFTDataset(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_len=self.model.config.max_seq_len,
            ignore_index=-100,
        )

        dataloader = DataLoader(sft_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        global_step = 0
        best_loss = float("inf")

        for epoch in range(epochs):
            total_loss = 0.0
            total_batches = 0

            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                lr = self.scheduler.step(global_step)
                self.optimizer.zero_grad(set_to_none=True)

                logits, _ = self.model(x)

                # Response-Only Loss Masking via ignore_index=-100
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-100,
                )

                loss.backward()
                if self.config.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1
                global_step += 1

            avg_loss = total_loss / max(1, total_batches)
            logger.info(f"[SFT Domain: {domain}] Epoch {epoch + 1}/{epochs} - Response Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.checkpoint_manager.save_checkpoint(
                    filepath=f"sft_{domain}_best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    step=global_step,
                    val_loss=best_loss,
                    is_best=True,
                )

        logger.info(f"Completed SFT for domain '{domain}'. Best Response Loss: {best_loss:.4f}")
        return best_loss
