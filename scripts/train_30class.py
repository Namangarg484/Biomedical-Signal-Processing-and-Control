#!/usr/bin/env python3
"""
PINNACLE — 30-Class Taxonomy Training Pipeline.

End-to-end training on the extended 30-species Raman benchmark.
Splits are pre-defined (reference/finetune/test) with zero data leakage.

Usage:
    python scripts/train_30class.py
    python scripts/train_30class.py --config configs/30class.yaml
    python scripts/train_30class.py --no-fusion   # spectral-only ablation
"""

import argparse
import os
import sys
import time
import yaml

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from pinnacle.utils import set_seed, get_device, logger, count_parameters
from pinnacle.model import PINNACLE
from pinnacle.trainer import PINNACLETrainer
from pinnacle.dataset_30class import load_30class_data, create_30class_dataloaders
from pinnacle.evaluate import evaluate_model
from pinnacle.visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_perclass_metrics,
)


# 30 species names (from Bacteria-ID benchmark)
SPECIES_NAMES_30 = [
    f"Species_{i}" for i in range(30)
]


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="PINNACLE 30-Class Training")
    parser.add_argument("--config", default="configs/30class.yaml", help="Config file")
    parser.add_argument("--no-fusion", action="store_true", help="Spectral-only (no CWT)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # ---- Load config ----
    cfg = load_config(args.config)
    seed = cfg["training"]["seed"]
    set_seed(seed)

    device = get_device()

    # ---- Banners ----
    use_fusion = cfg["model"]["use_fusion"] and not args.no_fusion
    num_classes = cfg["data"]["num_classes"]
    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["lr"]

    logger.info("=" * 70)
    logger.info("🚀 PINNACLE — 30-Class Taxonomy Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Config:      {args.config}")
    logger.info(f"  Classes:     {num_classes}")
    logger.info(f"  Epochs:      {epochs}")
    logger.info(f"  Batch size:  {batch_size}")
    logger.info(f"  LR:          {lr}")
    logger.info(f"  Device:      {device}")
    logger.info(f"  Fusion:      {use_fusion}")
    logger.info("=" * 70)

    # ---- Load data ----
    data = load_30class_data(cfg["data"]["dir"])

    # Verify class count matches config
    actual_classes = data["num_classes"]
    if actual_classes != num_classes:
        logger.warning(
            f"⚠️ Config says {num_classes} classes, data has {actual_classes}. "
            f"Using {actual_classes}."
        )
        num_classes = actual_classes

    # ---- Check wavelets ----
    if use_fusion and data.get("X_train_wav") is None:
        logger.error(
            "❌ Wavelet data not found! Generate it first:\n"
            "   python scripts/generate_wavelets_30class.py"
        )
        sys.exit(1)

    # ---- Create dataloaders ----
    train_loader, val_loader, test_loader = create_30class_dataloaders(
        data,
        batch_size=batch_size,
        num_workers=0,
        use_augmentation=True,
    )

    # ---- Build model ----
    model = PINNACLE(
        num_classes=num_classes,
        embed_dim=cfg["model"]["embed_dim"],
        dropout=cfg["model"]["dropout"],
        use_fusion=use_fusion,
    )

    logger.info(f"  Total parameters: {count_parameters(model):,}")

    # ---- Trainer ----
    ckpt_dir = cfg["output"]["checkpoint_dir"]
    log_dir = cfg["output"]["log_dir"]
    fig_dir = cfg["output"]["figure_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    trainer = PINNACLETrainer(
        model=model,
        device=device,
        lr=lr,
        weight_decay=cfg["training"]["weight_decay"],
        epochs=epochs,
        log_dir=log_dir,
        checkpoint_dir=ckpt_dir,
        patience=cfg["training"]["patience"],
        ema_decay=cfg["training"]["ema_decay"],
    )

    # ---- Resume ----
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        trainer.scheduler.load_state_dict(ckpt["scheduler_state"])
        trainer.best_val_acc = ckpt.get("best_val_acc", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"  Resumed from epoch {start_epoch}, best_val_acc={trainer.best_val_acc:.2f}%")

    # ---- Training loop ----
    logger.info("\n" + "=" * 70)
    logger.info("📊 Starting training...")
    logger.info("=" * 70)

    total_start = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)

        # Validate (standard model)
        val_metrics = trainer.evaluate(val_loader, use_ema=False)

        # Validate (EMA model)
        val_ema_metrics = trainer.evaluate(val_loader, use_ema=True)

        # Step scheduler
        trainer.scheduler.step()

        # Track history
        trainer.history["train_loss"].append(train_metrics["loss"])
        trainer.history["train_acc"].append(train_metrics["accuracy"])
        trainer.history["val_loss"].append(val_metrics["loss"])
        trainer.history["val_acc"].append(val_metrics["accuracy"])
        trainer.history["lr"].append(trainer.optimizer.param_groups[0]["lr"])

        # Best model checkpoint
        is_best = val_metrics["accuracy"] > trainer.best_val_acc
        if is_best:
            trainer.best_val_acc = val_metrics["accuracy"]
            trainer.best_epoch = epoch
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:03d}/{epochs} [{epoch_time:.1f}s] | "
            f"Train — Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val — Acc: {val_metrics['accuracy']:.2f}% | "
            f"Val EMA — Acc: {val_ema_metrics['accuracy']:.2f}% | "
            f"LR: {trainer.optimizer.param_groups[0]['lr']:.6f} | "
            f"{'★ BEST' if is_best else f'patience {trainer.patience_counter}/{trainer.patience}'}"
        )

        # Save checkpoints
        ckpt_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "ema_model_state": trainer.ema_model.state_dict(),
            "optimizer_state": trainer.optimizer.state_dict(),
            "scheduler_state": trainer.scheduler.state_dict(),
            "best_val_acc": trainer.best_val_acc,
            "history": trainer.history,
            "num_classes": num_classes,
            "config": cfg,
        }

        if is_best:
            torch.save(ckpt_data, os.path.join(ckpt_dir, "best_model.pth"))

        torch.save(ckpt_data, os.path.join(ckpt_dir, "checkpoint_last.pth"))

        # Early stopping
        if trainer.patience_counter >= trainer.patience:
            logger.info(f"\n⏹️  Early stopping at epoch {epoch} (patience={trainer.patience})")
            break

    total_time = time.time() - total_start
    logger.info(f"\n✅ Training complete in {total_time / 60:.1f} minutes")
    logger.info(f"   Best val accuracy: {trainer.best_val_acc:.2f}% at epoch {trainer.best_epoch}")

    # ---- Final evaluation on test set ----
    logger.info("\n" + "=" * 70)
    logger.info("🧪 Final Evaluation on Test Set")
    logger.info("=" * 70)

    # Load best model
    best_ckpt = torch.load(
        os.path.join(ckpt_dir, "best_model.pth"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state"])

    results = evaluate_model(model, test_loader, device, species_names=SPECIES_NAMES_30)

    # ---- Generate figures ----
    logger.info("\n📈 Generating figures...")

    try:
        history = trainer.history
        plot_training_curves(history, os.path.join(fig_dir, "fig_training_curves"))
        plot_confusion_matrix(
            results["confusion_matrix"], SPECIES_NAMES_30,
            os.path.join(fig_dir, "fig_confusion_matrix"),
        )
        plot_perclass_metrics(
            results["per_class"], os.path.join(fig_dir, "fig_perclass_metrics")
        )
        logger.info("  ✅ Figures saved")
    except Exception as e:
        logger.warning(f"  ⚠️ Figure generation failed: {e}")

    # ---- Final summary ----
    logger.info("\n" + "=" * 70)
    logger.info("🏁 FINAL SUMMARY — 30-Class Taxonomy")
    logger.info("=" * 70)
    logger.info(f"  Classes:          {num_classes}")
    logger.info(f"  Train samples:    {len(data['X_train']):,}")
    logger.info(f"  Val samples:      {len(data['X_val']):,}")
    logger.info(f"  Test samples:     {len(data['X_test']):,}")
    logger.info(f"  Best val acc:     {trainer.best_val_acc:.2f}%")
    logger.info(f"  Test accuracy:    {results['accuracy']:.2f}%")
    logger.info(f"  95% CI:           [{results['ci_95'][0]:.1f}, {results['ci_95'][1]:.1f}]%")
    logger.info(f"  Parameters:       {count_parameters(model):,}")
    logger.info(f"  Training time:    {total_time / 60:.1f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
