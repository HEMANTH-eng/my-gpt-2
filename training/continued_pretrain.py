from pathlib import Path
from typing import Optional, Union
import torch

from config.train_config import TrainConfig
from dataset.dataloader import create_dataloaders
from dataset.dataset import GPTDataset
from models.gpt import GPT
from training.checkpoint import CheckpointManager
from training.trainer import Trainer
from utils.logger import get_logger

logger = get_logger("continued_pretrain")


class ContinuedPretrainer:
    """Pipeline for continuing model pretraining on new domain text streams."""

    def __init__(
        self,
        model: GPT,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[TrainConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initializes ContinuedPretrainer.

        Args:
            model: PyTorch GPT model instance.
            checkpoint_path: Path to existing .pt checkpoint file to resume from.
            config: TrainConfig dataclass.
            device: Training target device ('cuda', 'cpu').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config or TrainConfig.micro_config()
        self.checkpoint_manager = CheckpointManager(self.config.checkpoint_dir)

        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Restoring base model weights from checkpoint: {checkpoint_path}")
            self.checkpoint_manager.load_checkpoint(
                filepath=checkpoint_path,
                model=self.model,
                device=self.device,
            )

    def train_on_corpus(
        self,
        token_ids: torch.Tensor,
        seq_len: int = 64,
        batch_size: int = 4,
        epochs: int = 2,
    ) -> float:
        """Runs continued pretraining on a new 1D array of token IDs.

        Args:
            token_ids: 1D Tensor of token IDs.
            seq_len: Context window sequence length.
            batch_size: DataLoader batch size.
            epochs: Number of continued pretraining epochs.

        Returns:
            Average training loss.
        """
        dataset = GPTDataset(token_ids.numpy() if isinstance(token_ids, torch.Tensor) else token_ids, seq_len=seq_len)
        train_loader, _ = create_dataloaders(dataset, batch_size=batch_size)

        # Configure Continued Pretraining Hyperparameters
        self.config.max_epochs = epochs
        self.config.learning_rate = 1e-4  # Lower learning rate to avoid catastrophic forgetting
        self.config.min_lr = 1e-5
        self.config.warmup_steps = 5

        trainer = Trainer(
            model=self.model,
            train_loader=train_loader,
            config=self.config,
            device=self.device,
        )

        logger.info(f"Starting continued pretraining for {epochs} epochs...")
        trainer.fit()
        return trainer.best_val_loss
