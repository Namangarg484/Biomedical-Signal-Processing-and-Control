#!/usr/bin/env python3
"""
Train the PINNACLE model.

Usage:
    python scripts/train.py                        # Default config
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --epochs 10 --batch-size 16   # Override
    python scripts/train.py --no-fusion            # Raman-only ablation
"""

import argparse
import os
import sys
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def main():
    parser = argparse.ArgumentParser(description="Train PINNACLE")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--no-fusion", action="store_true", help="Disable fusion (Raman-only)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    if args.device is not None:
        config["device"] = args.device
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.no_fusion:
        config["use_fusion"] = False
    if args.seed is not None:
        config["seed"] = args.seed

    # Setup
    set_seed(config["seed"])
    device = get_device(config.get("device", "auto"))

    logger.info("=" * 70)
    logger.info("🚀 PINNACLE — Training Pipeline")
    logger.info("=" * 70)
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Epochs: {config['epochs']}")
    logger.info(f"  Batch size: {config['batch_size']}")
    logger.info(f"  LR: {config['lr']}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Fusion: {config.get('use_fusion', True)}")
    logger.info("=" * 70)

    # Load data
    data = load_data(config["data_dir"])

    # Create dataloaders
    train_loader, val_loader, test_loader, y_train = create_dataloaders(
        data,
        batch_size=config["batch_size"],
        seed=config["seed"],
        num_workers=config.get("num_workers", 0),
        use_augmentation=config.get("use_augmentation", True),
    )

    # Create model
    model = PINNACLE(
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"],
        dropout=config.get("dropout", 0.3),
        use_fusion=config.get("use_fusion", True),
    )

    # Create trainer
    trainer = PINNACLETrainer(
        model=model,
        device=device,
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
        epochs=config["epochs"],
        log_dir=config.get("log_dir", "outputs/logs"),
        checkpoint_dir=config.get("checkpoint_dir", "outputs/checkpoints"),
        patience=config.get("early_stopping_patience", 12),
        ema_decay=config.get("ema_decay", 0.999),
    )

    # Resume if requested
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # Train
    results = trainer.train(train_loader, val_loader, test_loader)

    # Generate figures
    figure_dir = config.get("figure_dir", "outputs/figures")

    if "history" in results:
        plot_training_curves(
            results["history"],
            output_path=os.path.join(figure_dir, "fig4_training_curves.png"),
        )

    if "test_predictions" in results and "test_labels" in results:
        from sklearn.metrics import confusion_matrix as compute_cm
        cm = compute_cm(results["test_labels"], results["test_predictions"])
        plot_confusion_matrix(
            cm,
            output_path=os.path.join(figure_dir, "fig5_confusion_matrix.png"),
        )

    # Full evaluation
    if test_loader is not None:
        eval_results = evaluate_model(model, test_loader, device)
        plot_perclass_metrics(
            eval_results["per_class"],
            output_path=os.path.join(figure_dir, "fig6_perclass_metrics.png"),
        )

    # Save config used for this run
    run_config_path = os.path.join(config.get("log_dir", "outputs/logs"), "run_config.yaml")
    with open(run_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("✅ Training pipeline complete!")


if __name__ == "__main__":
    main()
