"""
PINNACLE — Training loop.
CPU / MPS compatible. AdamW + cosine annealing (matching draft.tex Table 4).
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Dict, List

from pinnacle.utils import logger, count_parameters


class PINNACLETrainer:
    """
    Training loop for the PINNACLE model.

    Features:
        - AdamW optimizer with cosine annealing LR schedule
        - Mixed precision (disabled for CPU, optional for MPS)
        - EMA model tracking
        - Checkpoint save/resume
        - Early stopping
        - Training/validation metrics logging
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        log_dir: str = "outputs/logs",
        checkpoint_dir: str = "outputs/checkpoints",
        patience: int = 12,
        ema_decay: float = 0.999,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.ema_decay = ema_decay

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizer (draft.tex Table 4)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # EMA model
        self.ema_model = self._create_ema_model()

        # Tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }

        logger.info(f"Trainer initialised: {count_parameters(model):,} params, device={device}")

    def _create_ema_model(self):
        """Create exponential moving average copy of the model."""
        import copy
        ema = copy.deepcopy(self.model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    def _update_ema(self):
        """Update EMA model weights and BatchNorm buffers."""
        # EMA-smooth the parameters (weights, biases)
        for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1 - self.ema_decay)
        # Copy buffers directly (BatchNorm running_mean/var are already running averages)
        for ema_buf, model_buf in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buf.copy_(model_buf)

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (raman, scalogram, labels) in enumerate(train_loader):
            raman = raman.to(self.device)
            scalogram = scalogram.to(self.device)
            labels = labels.to(self.device)

            # Forward
            logits, alpha, beta = self.model(raman, scalogram)
            loss = self.criterion(logits, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # Update EMA
            self._update_ema()

            # Metrics
            total_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Log progress
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = 100.0 * correct / total
                logger.info(
                    f"  Epoch {epoch:03d} | Batch {batch_idx+1:04d}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}%"
                )

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def evaluate(
        self, val_loader: DataLoader, use_ema: bool = False
    ) -> Dict[str, float]:
        """Evaluate model on validation/test set."""
        model = self.ema_model if use_ema else self.model
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for raman, scalogram, labels in val_loader:
            raman = raman.to(self.device)
            scalogram = scalogram.to(self.device)
            labels = labels.to(self.device)

            logits, _, _ = model(raman, scalogram)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "predictions": np.array(all_preds),
            "labels": np.array(all_labels),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict:
        """
        Full training loop.

        Returns:
            Dictionary with best metrics and history.
        """
        logger.info("=" * 70)
        logger.info("🚀 Starting Training")
        logger.info(f"   Epochs: {self.epochs}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Train batches: {len(train_loader)}")
        logger.info(f"   Val batches: {len(val_loader)}")
        logger.info("=" * 70)

        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.evaluate(val_loader)
            val_ema_metrics = self.evaluate(val_loader, use_ema=True)

            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["lr"].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Log epoch summary
            logger.info("=" * 70)
            logger.info(
                f"📊 Epoch {epoch:03d}/{self.epochs} ({epoch_time:.1f}s)"
            )
            logger.info(f"   Train — Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            logger.info(f"   Val   — Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            logger.info(f"   Val EMA — Acc: {val_ema_metrics['accuracy']:.2f}%")
            logger.info(f"   LR: {current_lr:.6f}")
            logger.info("=" * 70)

            # Check for improvement
            best_current = max(val_metrics["accuracy"], val_ema_metrics["accuracy"])
            if best_current > self.best_val_acc:
                self.best_val_acc = best_current
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save best checkpoint
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"   ✨ New best: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1

            # Save periodic checkpoint
            self._save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"⏹️  Early stopping at epoch {epoch} (patience={self.patience})")
                break

        total_time = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"🎉 Training Complete!")
        logger.info(f"   Total time: {total_time/3600:.2f} hours")
        logger.info(f"   Best val acc: {self.best_val_acc:.2f}% (epoch {self.best_epoch})")
        logger.info("=" * 70)

        # Final test evaluation
        results = {
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "total_time_hours": total_time / 3600,
            "history": self.history,
        }

        if test_loader is not None:
            # Load best model
            self._load_best_checkpoint()
            test_metrics = self.evaluate(test_loader)
            test_ema_metrics = self.evaluate(test_loader, use_ema=True)

            logger.info("=" * 70)
            logger.info("🧪 Final Test Evaluation")
            logger.info(f"   Test Acc: {test_metrics['accuracy']:.2f}%")
            logger.info(f"   Test EMA: {test_ema_metrics['accuracy']:.2f}%")
            logger.info("=" * 70)

            results["test_acc"] = test_metrics["accuracy"]
            results["test_ema_acc"] = test_ema_metrics["accuracy"]
            results["test_predictions"] = test_metrics["predictions"]
            results["test_labels"] = test_metrics["labels"]

        return results

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "ema_state": self.ema_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }

        # Always save last checkpoint
        last_path = os.path.join(self.checkpoint_dir, "checkpoint_last.pth")
        torch.save(state, last_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)

    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_path):
            state = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state"])
            self.ema_model.load_state_dict(state["ema_state"])
            logger.info(f"  ✅ Loaded best checkpoint (epoch {state['epoch']})")

    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.ema_model.load_state_dict(state["ema_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        self.best_val_acc = state.get("best_val_acc", 0.0)
        self.history = state.get("history", self.history)

        start_epoch = state["epoch"] + 1
        logger.info(f"  ✅ Resumed from epoch {start_epoch}")
        return start_epoch
