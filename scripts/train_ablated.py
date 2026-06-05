#!/usr/bin/env python3
"""
Preprocessing Ablation — Training Script
=========================================
Trains PINNACLE on the ablated dataset (no ALS / no Savitzky-Golay preprocessing)
and reports the final test accuracy with 95 % Wilson CI.

Identical hyperparameters to the main pipeline:
  - 30 epochs, batch size 32, AdamW, CosineAnnealingLR
  - MPS / CPU device (no AMP)
  - Checkpoints saved to  outputs_ablated/checkpoints/
  - Logs saved to         outputs_ablated/logs/

Usage:
  python scripts/train_ablated.py

Prerequisite:
  python scripts/generate_ablated_data.py   # generates data/ablated/
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch

from pinnacle.utils import set_seed, get_device, logger
from pinnacle.model import PINNACLE
from pinnacle.dataset import load_data, create_dataloaders
from pinnacle.trainer import PINNACLETrainer
from pinnacle.evaluate import evaluate_model
from pinnacle.visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_perclass_metrics,
)
from sklearn.metrics import confusion_matrix as compute_cm

# ---------------------------------------------------------------------------
# Configuration — identical to default.yaml except for data/output paths
# ---------------------------------------------------------------------------
CONFIG = {
    # Architecture (must match draft.tex / main pipeline exactly)
    "model": "pinnacle",
    "embed_dim": 128,
    "num_classes": 5,
    "dropout": 0.3,
    "use_fusion": True,

    # Data — point to ablated dataset
    "data_dir": os.path.join(PROJECT_ROOT, "data", "ablated"),
    "seq_len": 1000,
    "img_size": 224,

    # Training (matching default.yaml exactly)
    "epochs": 30,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "cosine",

    # Augmentation & regularisation
    "use_augmentation": True,
    "ema_decay": 0.999,
    "grad_clip": 5.0,
    "early_stopping_patience": 12,

    # Device
    "device": "auto",
    "use_amp": False,
    "num_workers": 0,

    # Reproducibility
    "seed": 42,

    # Output paths — separate from main run to avoid overwriting
    "log_dir":        os.path.join(PROJECT_ROOT, "outputs_ablated", "logs"),
    "checkpoint_dir": os.path.join(PROJECT_ROOT, "outputs_ablated", "checkpoints"),
    "figure_dir":     os.path.join(PROJECT_ROOT, "outputs_ablated", "figures"),
}


def main():
    # Check ablated data exists
    ablated_dir = CONFIG["data_dir"]
    required = ["X_2018_proc.npy", "X_2019_proc.npy",
                "X_2018_wavelet.npy", "X_2019_wavelet.npy",
                "y_2018clinical.npy", "y_2019clinical.npy"]
    missing = [f for f in required if not os.path.exists(os.path.join(ablated_dir, f))]
    if missing:
        logger.error(
            f"Ablated data not found in {ablated_dir}.\n"
            f"Missing files: {missing}\n"
            "Run:  python scripts/generate_ablated_data.py  first."
        )
        sys.exit(1)

    # Create output dirs
    for d in [CONFIG["log_dir"], CONFIG["checkpoint_dir"], CONFIG["figure_dir"]]:
        os.makedirs(d, exist_ok=True)

    set_seed(CONFIG["seed"])
    device = get_device(CONFIG.get("device", "auto"))

    logger.info("=" * 70)
    logger.info("PINNACLE — Preprocessing Ablation Training (no ALS / no SG)")
    logger.info("=" * 70)
    logger.info(f"  Data dir   : {CONFIG['data_dir']}")
    logger.info(f"  Epochs     : {CONFIG['epochs']}")
    logger.info(f"  Batch size : {CONFIG['batch_size']}")
    logger.info(f"  LR         : {CONFIG['lr']}")
    logger.info(f"  Device     : {device}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Load ablated data
    # ------------------------------------------------------------------
    data = load_data(CONFIG["data_dir"])

    train_loader, val_loader, test_loader, y_train = create_dataloaders(
        data,
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"],
        num_workers=CONFIG.get("num_workers", 0),
        use_augmentation=CONFIG.get("use_augmentation", True),
    )

    # ------------------------------------------------------------------
    # Model — identical architecture, no modifications
    # ------------------------------------------------------------------
    model = PINNACLE(
        num_classes=CONFIG["num_classes"],
        embed_dim=CONFIG["embed_dim"],
        dropout=CONFIG.get("dropout", 0.3),
        use_fusion=CONFIG.get("use_fusion", True),
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = PINNACLETrainer(
        model=model,
        device=device,
        lr=CONFIG["lr"],
        weight_decay=CONFIG.get("weight_decay", 1e-4),
        epochs=CONFIG["epochs"],
        log_dir=CONFIG["log_dir"],
        checkpoint_dir=CONFIG["checkpoint_dir"],
        patience=CONFIG.get("early_stopping_patience", 12),
        ema_decay=CONFIG.get("ema_decay", 0.999),
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    results = trainer.train(train_loader, val_loader, test_loader)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    figure_dir = CONFIG["figure_dir"]
    if "history" in results:
        plot_training_curves(
            results["history"],
            output_path=os.path.join(figure_dir, "ablated_training_curves.png"),
        )
    if "test_predictions" in results and "test_labels" in results:
        cm = compute_cm(results["test_labels"], results["test_predictions"])
        plot_confusion_matrix(
            cm,
            output_path=os.path.join(figure_dir, "ablated_confusion_matrix.png"),
        )

    # ------------------------------------------------------------------
    # Full evaluation with Wilson CI
    # ------------------------------------------------------------------
    eval_results = evaluate_model(model, test_loader, device)
    plot_perclass_metrics(
        eval_results["per_class"],
        output_path=os.path.join(figure_dir, "ablated_perclass_metrics.png"),
    )

    # Compute Wilson CI
    n = eval_results.get("n_samples", 1250)
    acc = eval_results["accuracy"] / 100.0
    z = 1.96
    centre = (acc + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = (z / (1 + z**2 / n)) * ((acc * (1 - acc) / n + z**2 / (4 * n**2)) ** 0.5)
    ci_lo = max(0.0, (centre - margin) * 100)
    ci_hi = min(100.0, (centre + margin) * 100)

    logger.info("=" * 70)
    logger.info("ABLATION RESULT (no ALS / no Savitzky-Golay preprocessing)")
    logger.info("=" * 70)
    logger.info(
        f"  Test Accuracy : {eval_results['accuracy']:.2f}%  "
        f"[95% CI: {ci_lo:.1f}–{ci_hi:.1f}%]"
    )
    logger.info(f"  n = {n}")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Per-class F1 scores:")
    for cls_name, metrics in eval_results["per_class"].items():
        logger.info(f"    {cls_name:20s}  F1={metrics['f1']*100:.1f}%")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
