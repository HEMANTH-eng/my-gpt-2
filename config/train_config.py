from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    """Training Hyperparameter and Execution System Configuration.

    Attributes:
        learning_rate: Peak learning rate for AdamW optimizer.
        min_lr: Minimum learning rate floor after decay.
        weight_decay: L2 regularization coefficient for 2D weight parameters.
        beta1: AdamW beta1 coefficient.
        beta2: AdamW beta2 coefficient.
        grad_clip: Maximum norm for gradient clipping (0.0 disables clipping).
        max_epochs: Maximum training epochs to run.
        batch_size: Batch size per device.
        warmup_steps: Number of linear warmup steps.
        decay_steps: Total number of cosine decay steps.
        use_amp: Whether to enable Automatic Mixed Precision (AMP).
        checkpoint_dir: Directory path for saving model checkpoints.
        log_dir: Directory path for TensorBoard event logs.
        eval_interval: Number of steps between validation evaluations.
        save_interval: Number of steps between checkpoint saves.
        early_stopping_patience: Number of un-improved evaluations before early stopping.
    """

    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    max_epochs: int = 10
    batch_size: int = 16
    warmup_steps: int = 100
    decay_steps: int = 1000

    use_amp: bool = True
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    eval_interval: int = 50
    save_interval: int = 200
    early_stopping_patience: int = 5

    @classmethod
    def micro_config(cls) -> "TrainConfig":
        """Fast training configuration for quick local testing and unit tests."""
        return cls(
            learning_rate=1e-3,
            min_lr=1e-4,
            max_epochs=2,
            batch_size=4,
            warmup_steps=10,
            decay_steps=50,
            use_amp=False,
            eval_interval=5,
            save_interval=10,
            early_stopping_patience=2,
        )

