import math
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.train_config import TrainConfig
from models.gpt import GPT
from training.checkpoint import CheckpointManager
from training.optimizer import CosineWarmupScheduler, configure_optimizer
from utils.logger import get_logger

logger = get_logger(__name__)

# Optional TensorBoard SummaryWriter
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # Fallback if tensorboard is not installed


class Trainer:
    """Production-quality training pipeline for GPT language models."""

    def __init__(
        self,
        model: GPT,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes Trainer.

        Args:
            model: PyTorch GPT model instance.
            train_loader: DataLoader for training dataset.
            val_loader: Optional DataLoader for validation dataset.
            config: TrainConfig hyperparameter dataclass.
            device: Training target device ('cuda', 'cpu').
        """
        self.config = config if config is not None else TrainConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 1. Optimizer & LR Scheduler
        self.optimizer = configure_optimizer(self.model, self.config)
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            learning_rate=self.config.learning_rate,
            min_lr=self.config.min_lr,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.decay_steps,
        )

        # 2. Mixed Precision GradScaler
        self.use_amp = self.config.use_amp and (self.device.startswith("cuda") or self.device == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # 3. Checkpoint Manager
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

        # 4. TensorBoard Logging
        self.writer = None
        if SummaryWriter is not None and self.config.log_dir:
            log_path = Path(self.config.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_path))

        # 5. Tracking State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def evaluate(self) -> Tuple[float, float]:
        """Runs validation evaluation loop.

        Returns:
            Tuple of (average_val_loss, val_perplexity).
        """
        if self.val_loader is None or len(self.val_loader) == 0:
            return 0.0, 1.0

        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                device_type = "cuda" if self.device.startswith("cuda") else "cpu"
                with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                    _, loss = self.model(x, targets=y)

                total_loss += loss.item()
                total_batches += 1

        avg_val_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_val_loss, 20.0))  # Cap to avoid math overflow

        self.model.train()
        return avg_val_loss, perplexity

    def train_epoch(self) -> float:
        """Executes training over a single epoch.

        Returns:
            Average training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Update Learning Rate via Scheduler
            lr = self.scheduler.step(self.global_step)

            # Zero Gradients
            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP autocast
            device_type = "cuda" if self.device.startswith("cuda") else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                _, loss = self.model(x, targets=y)

            # Backward pass with GradScaler
            self.scaler.scale(loss).backward()

            # Unscale and clip gradients
            if self.config.grad_clip > 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # Optimizer Step & Scaler Update
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_batches += 1
            self.global_step += 1

            # Log to TensorBoard
            if self.writer is not None and self.global_step % 10 == 0:
                self.writer.add_scalar("Train/Loss", loss.item(), self.global_step)
                self.writer.add_scalar("Train/LearningRate", lr, self.global_step)

            # Evaluation Interval
            if self.val_loader is not None and self.global_step % self.config.eval_interval == 0:
                val_loss, perplexity = self.evaluate()
                logger.info(
                    f"[Step {self.global_step}] Val Loss: {val_loss:.4f} | "
                    f"Perplexity: {perplexity:.2f}"
                )
                if self.writer is not None:
                    self.writer.add_scalar("Val/Loss", val_loss, self.global_step)
                    self.writer.add_scalar("Val/Perplexity", perplexity, self.global_step)

                # Track best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Save Checkpoint
                if is_best or self.global_step % self.config.save_interval == 0:
                    self.checkpoint_manager.save_checkpoint(
                        filepath=f"step_{self.global_step}.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                        epoch=self.current_epoch,
                        step=self.global_step,
                        val_loss=val_loss,
                        config=self.config,
                        is_best=is_best,
                    )

                # Early Stopping Check
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered! Validation loss did not improve "
                        f"for {self.patience_counter} evaluation cycles."
                    )
                    return total_loss / max(1, total_batches)

        return total_loss / max(1, total_batches)

    def fit(self) -> None:
        """Main entrypoint running complete training loop across max_epochs."""
        logger.info(
            f"Starting training on device '{self.device}' for {self.config.max_epochs} epochs. "
            f"AMP: {self.use_amp}"
        )

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            logger.info(f"--- Epoch {epoch + 1}/{self.config.max_epochs} ---")

            avg_train_loss = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} completed. Average Train Loss: {avg_train_loss:.4f}")

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info("Stopping fit loop due to early stopping.")
                break

        if self.writer is not None:
            self.writer.close()

    def resume_from_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Restores model, optimizer, scaler, epoch, and step from a checkpoint file.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
        """
        epoch, step, val_loss = self.checkpoint_manager.load_checkpoint(
            filepath=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            device=self.device,
        )
        self.current_epoch = epoch
        self.global_step = step
        if val_loss is not None:
            self.best_val_loss = val_loss
        logger.info(f"Resumed training state from step {step}, epoch {epoch}.")

