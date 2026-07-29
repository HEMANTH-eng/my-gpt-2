"""Model Manager for checkpoint discovery, versioning, safe loading, and configuration validation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from config.model_config import GPTConfig
from models.gpt import GPT
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """Manages discovery, versioning, verification, and safe loading of GPT model checkpoints."""

    def __init__(self, checkpoints_dir: Union[str, Path] = "./checkpoints") -> None:
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def list_checkpoints(self) -> List[Path]:
        """Discovers all available .pt model checkpoint files in priority order."""
        candidates = []
        best_pt = self.checkpoints_dir / "best.pt"
        if best_pt.exists():
            candidates.append(best_pt)

        for p in sorted(self.checkpoints_dir.glob("*.pt"), reverse=True):
            if p not in candidates:
                candidates.append(p)

        return candidates

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Returns the highest priority model checkpoint file if it exists."""
        checkpoints = self.list_checkpoints()
        return checkpoints[0] if checkpoints else None

    def load_safe(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[GPTConfig] = None,
        device: str = "cpu",
    ) -> Tuple[GPT, GPTConfig, Dict[str, Any]]:
        """Safely loads a model checkpoint without silent random model initialization.

        Raises:
            FileNotFoundError: If no valid model checkpoint exists and no config is explicitly supplied.
        """
        target_path = Path(checkpoint_path) if checkpoint_path else self.get_latest_checkpoint()

        if target_path is None or not target_path.exists():
            if config is None:
                raise FileNotFoundError(
                    f"No trained model checkpoint found in '{self.checkpoints_dir}'. "
                    "Cannot safely initialize model without a trained weight checkpoint or explicit configuration."
                )
            logger.warning(
                f"No model checkpoint found at '{target_path}'. Initializing model from scratch using config."
            )
            model = GPT(config).to(device)
            return model, config, {}

        logger.info(f"Safely loading model weights from checkpoint: '{target_path}'")
        try:
            checkpoint = torch.load(target_path, map_location=device, weights_only=False)
        except Exception as e:
            logger.error(f"Failed to load checkpoint at '{target_path}': {e}")
            raise RuntimeError(f"Corrupt or invalid PyTorch checkpoint at '{target_path}': {e}")

        # Extract config metadata if present
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            saved_config = checkpoint["config"]
            if isinstance(saved_config, GPTConfig):
                config = saved_config
            elif isinstance(saved_config, dict):
                config = GPTConfig(**saved_config)

        if config is None:
            config = GPTConfig.gpt_micro()

        model = GPT(config).to(device)

        # Load weights state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        meta = {
            "checkpoint_path": str(target_path),
            "step": checkpoint.get("step", 0) if isinstance(checkpoint, dict) else 0,
            "epoch": checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0,
            "val_loss": checkpoint.get("val_loss", None) if isinstance(checkpoint, dict) else None,
        }

        logger.info(f"Model successfully loaded on '{device}'. Step: {meta['step']}, Val Loss: {meta['val_loss']}")
        return model, config, meta
