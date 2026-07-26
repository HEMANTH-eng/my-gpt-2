from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer

from config.train_config import TrainConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages saving, loading, and tracking model checkpoints."""

    def __init__(self, checkpoint_dir: Union[str, Path] = "checkpoints") -> None:
        """Initializes CheckpointManager.

        Args:
            checkpoint_dir: Directory path for checkpoint files.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scaler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        val_loss: Optional[float] = None,
        config: Optional[TrainConfig] = None,
        is_best: bool = False,
    ) -> Path:
        """Saves complete model and optimizer training state dict to disk.

        Args:
            filepath: Target file path or filename within checkpoint_dir.
            model: PyTorch model instance.
            optimizer: Optional PyTorch optimizer instance.
            scaler: Optional PyTorch GradScaler instance.
            epoch: Current epoch index.
            step: Current step index.
            val_loss: Validation loss metric.
            config: TrainConfig instance.
            is_best: If True, also updates best.pt checkpoint.

        Returns:
            Path to saved checkpoint file.
        """
        save_path = Path(filepath)
        if not save_path.is_absolute():
            save_path = self.checkpoint_dir / save_path

        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scaler is not None and hasattr(scaler, "state_dict"):
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        if config is not None:
            checkpoint["config"] = config

        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path} (epoch={epoch}, step={step}, val_loss={val_loss})")

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Updated best checkpoint at {best_path}")

        # Always maintain latest.pt
        latest_path = self.checkpoint_dir / "latest.pt"
        if save_path != latest_path:
            torch.save(checkpoint, latest_path)

        return save_path

    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scaler: Optional[Any] = None,
        device: str = "cpu",
    ) -> Tuple[int, int, Optional[float]]:
        """Loads model and optimizer state from checkpoint.

        Args:
            filepath: Path to checkpoint file.
            model: PyTorch model instance.
            optimizer: Optional PyTorch optimizer instance.
            scaler: Optional PyTorch GradScaler instance.
            device: Map storage location ('cpu' or 'cuda').

        Returns:
            Tuple of (epoch, step, val_loss).
        """
        load_path = Path(filepath)
        if not load_path.is_absolute():
            load_path = self.checkpoint_dir / load_path

        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        logger.info(f"Loading checkpoint from {load_path}...")
        checkpoint = torch.load(load_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scaler is not None and "scaler_state_dict" in checkpoint and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)
        val_loss = checkpoint.get("val_loss", None)

        logger.info(f"Loaded checkpoint successfully (epoch={epoch}, step={step}, val_loss={val_loss})")
        return epoch, step, val_loss

