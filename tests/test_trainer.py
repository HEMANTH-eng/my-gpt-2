import numpy as np
import pytest
import torch

from config.model_config import GPTConfig
from config.train_config import TrainConfig
from dataset.dataloader import create_dataloaders
from dataset.dataset import GPTDataset
from models.gpt import GPT
from training.checkpoint import CheckpointManager
from training.optimizer import CosineWarmupScheduler, configure_optimizer
from training.trainer import Trainer


def test_optimizer_decay_splitting():
    config = GPTConfig.gpt_micro()
    model = GPT(config)
    train_config = TrainConfig(weight_decay=0.1)

    optimizer = configure_optimizer(model, train_config)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_cosine_warmup_scheduler():
    config = GPTConfig.gpt_micro()
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        learning_rate=1e-3,
        min_lr=1e-4,
        warmup_steps=10,
        decay_steps=50,
    )

    # Step 0 -> min_lr
    lr_0 = scheduler.get_lr(0)
    assert abs(lr_0 - 1e-4) < 1e-6

    # Step 10 -> peak learning rate
    lr_10 = scheduler.get_lr(10)
    assert abs(lr_10 - 1e-3) < 1e-6

    # Step 50 -> min_lr
    lr_50 = scheduler.get_lr(50)
    assert abs(lr_50 - 1e-4) < 1e-6


def test_checkpoint_manager_save_and_load(tmp_path):
    config = GPTConfig.gpt_micro()
    model1 = GPT(config)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    manager = CheckpointManager(checkpoint_dir=tmp_path)
    save_path = manager.save_checkpoint(
        filepath="model_step_10.pt",
        model=model1,
        optimizer=optimizer1,
        epoch=1,
        step=10,
        val_loss=2.5,
        is_best=True,
    )

    assert save_path.exists()
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "latest.pt").exists()

    model2 = GPT(config)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

    epoch, step, val_loss = manager.load_checkpoint(
        filepath=save_path,
        model=model2,
        optimizer=optimizer2,
    )

    assert epoch == 1
    assert step == 10
    assert val_loss == 2.5

    # Verify parameters match identically
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_trainer_fit_single_epoch(tmp_path):
    config = GPTConfig.gpt_micro(vocab_size=100)
    model = GPT(config)

    train_config = TrainConfig.micro_config()
    train_config.checkpoint_dir = str(tmp_path / "checkpoints")
    train_config.log_dir = str(tmp_path / "runs")

    tokens = np.random.randint(0, 100, size=200, dtype=np.int64)
    dataset = GPTDataset(tokens, seq_len=16)
    train_loader, val_loader = create_dataloaders(
        train_dataset=dataset,
        val_dataset=dataset,
        batch_size=4,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device="cpu",
    )

    avg_loss = trainer.train_epoch()
    assert avg_loss > 0.0
    assert trainer.global_step > 0


def test_trainer_early_stopping(tmp_path):
    config = GPTConfig.gpt_micro(vocab_size=100)
    model = GPT(config)

    train_config = TrainConfig.micro_config()
    train_config.early_stopping_patience = 2
    train_config.eval_interval = 1
    train_config.checkpoint_dir = str(tmp_path / "checkpoints")
    train_config.log_dir = str(tmp_path / "runs")

    tokens = np.random.randint(0, 100, size=150, dtype=np.int64)
    dataset = GPTDataset(tokens, seq_len=16)
    train_loader, val_loader = create_dataloaders(
        train_dataset=dataset,
        val_dataset=dataset,
        batch_size=4,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device="cpu",
    )

    # Manually simulate non-improving validation loss
    trainer.best_val_loss = 0.01
    trainer.patience_counter = 0

    trainer.train_epoch()
    assert trainer.patience_counter >= train_config.early_stopping_patience


def test_resume_training(tmp_path):
    config = GPTConfig.gpt_micro(vocab_size=100)
    model1 = GPT(config)

    train_config = TrainConfig.micro_config()
    train_config.checkpoint_dir = str(tmp_path / "checkpoints")
    train_config.log_dir = str(tmp_path / "runs")

    tokens = np.random.randint(0, 100, size=200, dtype=np.int64)
    dataset = GPTDataset(tokens, seq_len=16)
    train_loader, _ = create_dataloaders(dataset, batch_size=4)

    trainer1 = Trainer(
        model=model1,
        train_loader=train_loader,
        config=train_config,
        device="cpu",
    )
    trainer1.train_epoch()
    ckpt_path = trainer1.checkpoint_manager.save_checkpoint(
        filepath="resume_checkpoint.pt",
        model=trainer1.model,
        optimizer=trainer1.optimizer,
        scaler=trainer1.scaler,
        epoch=1,
        step=15,
        val_loss=1.8,
    )

    model2 = GPT(config)
    trainer2 = Trainer(
        model=model2,
        train_loader=train_loader,
        config=train_config,
        device="cpu",
    )
    trainer2.resume_from_checkpoint(ckpt_path)

    assert trainer2.current_epoch == 1
    assert trainer2.global_step == 15
    assert trainer2.best_val_loss == 1.8

