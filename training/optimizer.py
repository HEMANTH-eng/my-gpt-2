import math
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW

from config.train_config import TrainConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def configure_optimizer(model: nn.Module, config: TrainConfig) -> AdamW:
    """Configures AdamW optimizer with selective weight decay.

    Splits model parameters into two groups:
        1. 2D weight matrices (linear layers, embedding tables) -> Weight decay applied.
        2. 1D bias vectors and LayerNorm parameters -> Weight decay set to 0.0.

    Handles tied parameter tensors (e.g., token embedding and LM head) robustly.

    Args:
        model: PyTorch model instance (GPT).
        config: TrainConfig dataclass.

    Returns:
        Configured torch.optim.AdamW optimizer.
    """
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1D tensors (biases, LayerNorm gains/biases) do not receive weight decay
        if param.ndim < 2 or name.endswith("bias") or "ln" in name or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    optim_groups = [
        {
            "params": decay,
            "weight_decay": config.weight_decay,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ]

    logger.info(
        f"Configured AdamW: {len(decay)} decayed parameter tensors, "
        f"{len(no_decay)} non-decayed parameter tensors."
    )

    return AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )



class CosineWarmupScheduler:
    """Cosine Learning Rate Scheduler with Linear Warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        min_lr: float,
        warmup_steps: int,
        decay_steps: int,
    ) -> None:
        """Initializes CosineWarmupScheduler.

        Args:
            optimizer: PyTorch optimizer instance.
            learning_rate: Peak learning rate (max_lr).
            min_lr: Floor learning rate (min_lr).
            warmup_steps: Number of linear warmup steps.
            decay_steps: Total decay step target.
        """
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def get_lr(self, step: int) -> float:
        """Computes learning rate for the given training step.

        Args:
            step: Current step index.

        Returns:
            Calculated float learning rate.
        """
        # 1. Linear warmup
        if step < self.warmup_steps:
            return self.min_lr + (self.learning_rate - self.min_lr) * (step / max(1, self.warmup_steps))

        # 2. Beyond decay steps -> floor min_lr
        if step > self.decay_steps:
            return self.min_lr

        # 3. Cosine decay phase
        decay_ratio = (step - self.warmup_steps) / max(1, self.decay_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def step(self, step: int) -> float:
        """Updates the learning rate across all optimizer parameter groups.

        Args:
            step: Current training step index.

        Returns:
            Updated learning rate.
        """
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

