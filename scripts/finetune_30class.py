#!/usr/bin/env python3
"""
PINNACLE — Phase 2: Domain Adaptation Fine-Tuning.

Loads the pre-trained backbone from Phase 1 (reference set) and fine-tunes
on the target-domain finetune set to bridge the instrument gap.

Protocol:
    1. Load pre-trained weights from Phase 1
    2. Freeze backbone for N epochs (train classifier head only)
    3. Unfreeze and fine-tune end-to-end with low LR
    4. Evaluate on held-out test set

Usage:
    python scripts/finetune_30class.py
    python scripts/finetune_30class.py --config configs/finetune.yaml
"""

import argparse
import os
import sys
import time
import yaml
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from pinnacle.utils import set_seed, get_device, logger, count_parameters
from pinnacle.model import PINNACLE
from pinnacle.dataset import PINNACLEDataset, RamanAugmentation
from pinnacle.evaluate import evaluate_model
from pinnacle.visualize import plot_training_curves


SPECIES_NAMES_30 = [f"Species_{i}" for i in range(30)]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_finetune_data(data_dir: str, val_fraction: float = 0.2, seed: int = 42):
    """
    Load finetune and test data.
    Split finetune into train/val for monitoring.
    """
    logger.info(f"Loading finetune data from {data_dir}...")

    X_ft = np.load(os.path.join(data_dir, "X_finetune.npy"))
    y_ft = np.load(os.path.join(data_dir, "y_finetune.npy")).astype(np.int64)
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)

    logger.info(f"  Finetune: {X_ft.shape}, {len(np.unique(y_ft))} classes")
    logger.info(f"  Test:     {X_test.shape}, {len(np.unique(y_test))} classes")

    # Stratified split of finetune → ft_train / ft_val
    indices = np.arange(len(X_ft))
    idx_train, idx_val = train_test_split(
        indices, test_size=val_fraction, random_state=seed, stratify=y_ft,
    )

    result = {
        "X_train": X_ft[idx_train],
        "y_train": y_ft[idx_train],
        "X_val": X_ft[idx_val],
        "y_val": y_ft[idx_val],
        "X_test": X_test,
        "y_test": y_test,
    }

    logger.info(f"  FT Train: {result['X_train'].shape} ({len(idx_train)} samples)")
    logger.info(f"  FT Val:   {result['X_val'].shape} ({len(idx_val)} samples)")

    # Load wavelets
    ft_wav_path = os.path.join(data_dir, "X_finetune_wavelet.npy")
    test_wav_path = os.path.join(data_dir, "X_test_wavelet.npy")

    if os.path.exists(ft_wav_path):
        X_ft_wav = np.load(ft_wav_path, mmap_mode="r")
        result["X_train_wav"] = np.array(X_ft_wav[idx_train])
        result["X_val_wav"] = np.array(X_ft_wav[idx_val])
        logger.info(f"  FT Train wavelets: {result['X_train_wav'].shape}")
    else:
        result["X_train_wav"] = None
        result["X_val_wav"] = None
        logger.warning("  Finetune wavelets not found!")

    if os.path.exists(test_wav_path):
        result["X_test_wav"] = np.load(test_wav_path, mmap_mode="r")
        logger.info(f"  Test wavelets: {result['X_test_wav'].shape}")
    else:
        result["X_test_wav"] = None
        logger.warning("  Test wavelets not found!")

    # Leakage check
    train_hashes = set(map(lambda x: hash(x.tobytes()), result["X_train"]))
    val_hashes = set(map(lambda x: hash(x.tobytes()), result["X_val"]))
    test_hashes = set(map(lambda x: hash(x.tobytes()), result["X_test"][:500]))
    tv_leak = len(train_hashes & val_hashes)
    tt_leak = len(train_hashes & test_hashes)
    if tv_leak + tt_leak == 0:
        logger.info("  ✅ No data leakage detected")
    else:
        logger.warning(f"  ⚠️ LEAKAGE: train∩val={tv_leak}, train∩test={tt_leak}")

    return result


def freeze_backbone(model):
    """Freeze all parameters except the classifier head."""
    frozen = 0
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
            frozen += 1
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"  🧊 Backbone frozen: {frozen} params frozen, {trainable} trainable (classifier only)")


def unfreeze_all(model):
    """Unfreeze all parameters for end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"  🔥 All parameters unfrozen: {trainable} trainable")


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=5.0):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for raman, scalogram, labels in loader:
        raman = raman.to(device)
        scalogram = scalogram.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _, _ = model(raman, scalogram)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on val/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for raman, scalogram, labels in loader:
        raman = raman.to(device)
        scalogram = scalogram.to(device)
        labels = labels.to(device)

        logits, _, _ = model(raman, scalogram)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description="PINNACLE Phase 2 Fine-Tuning")
    parser.add_argument("--config", default="configs/finetune.yaml", help="Config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ft_cfg = cfg["finetuning"]
    set_seed(ft_cfg["seed"])
    device = get_device()

    # ---- Banner ----
    logger.info("=" * 70)
    logger.info("🔧 PINNACLE — Phase 2: Domain Adaptation Fine-Tuning")
    logger.info("=" * 70)
    logger.info(f"  Pre-trained:       {cfg['pretrained']['checkpoint']}")
    logger.info(f"  Epochs:            {ft_cfg['epochs']}")
    logger.info(f"  Batch size:        {ft_cfg['batch_size']}")
    logger.info(f"  LR:                {ft_cfg['lr']}")
    logger.info(f"  Freeze epochs:     {ft_cfg['freeze_backbone_epochs']}")
    logger.info(f"  Device:            {device}")
    logger.info("=" * 70)

    # ---- Load data ----
    data = load_finetune_data(
        cfg["data"]["dir"],
        val_fraction=ft_cfg["val_fraction"],
        seed=ft_cfg["seed"],
    )

    # ---- Create DataLoaders ----
    raman_aug = RamanAugmentation(noise_std=0.01, shift_range=5, scale_range=0.05, probability=0.5)

    train_ds = PINNACLEDataset(
        data["X_train"], data["y_train"], data.get("X_train_wav"),
        transform_raman=raman_aug, split="train",
    )
    val_ds = PINNACLEDataset(
        data["X_val"], data["y_val"], data.get("X_val_wav"), split="val",
    )
    test_ds = PINNACLEDataset(
        data["X_test"], data["y_test"], data.get("X_test_wav"), split="test",
    )

    train_loader = DataLoader(
        train_ds, batch_size=ft_cfg["batch_size"], shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=ft_cfg["batch_size"], shuffle=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=ft_cfg["batch_size"], shuffle=False,
        num_workers=0, pin_memory=False,
    )

    logger.info(f"  DataLoaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches")

    # ---- Build model and load pre-trained weights ----
    model = PINNACLE(
        num_classes=cfg["data"]["num_classes"],
        embed_dim=cfg["model"]["embed_dim"],
        dropout=cfg["model"]["dropout"],
        use_fusion=cfg["model"]["use_fusion"],
    )

    ckpt_path = cfg["pretrained"]["checkpoint"]
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"  ✅ Loaded pre-trained weights from {ckpt_path}")
        logger.info(f"     Phase 1 best val acc: {ckpt.get('best_val_acc', 'N/A')}")
    else:
        logger.error(f"  ❌ Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model = model.to(device)

    # ---- Output dirs ----
    ckpt_dir = cfg["output"]["checkpoint_dir"]
    log_dir = cfg["output"]["log_dir"]
    fig_dir = cfg["output"]["figure_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Phase 2a: Freeze backbone, train classifier only ----
    freeze_epochs = ft_cfg["freeze_backbone_epochs"]
    criterion = nn.CrossEntropyLoss()

    # History tracking
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    total_start = time.time()

    for epoch in range(ft_cfg["epochs"]):
        epoch_start = time.time()

        # ---- Freeze/unfreeze logic ----
        if epoch == 0:
            freeze_backbone(model)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=ft_cfg["lr"] * 10,  # Higher LR for classifier-only phase
                weight_decay=ft_cfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=freeze_epochs,
            )
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Phase 2a: Classifier-only training (epochs 0–{freeze_epochs - 1})")
            logger.info(f"{'=' * 70}")

        if epoch == freeze_epochs:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=ft_cfg["lr"],
                weight_decay=ft_cfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=ft_cfg["epochs"] - freeze_epochs,
            )
            patience_counter = 0  # Reset patience for Phase 2b
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Phase 2b: End-to-end fine-tuning (epochs {freeze_epochs}–{ft_cfg['epochs'] - 1})")
            logger.info(f"{'=' * 70}")

        # ---- Train ----
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)

        # ---- Validate ----
        val_metrics = evaluate(model, val_loader, criterion, device)

        # ---- Scheduler step ----
        scheduler.step()

        # ---- History ----
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # ---- Best model tracking ----
        is_best = val_metrics["accuracy"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val_acc": best_val_acc,
                "config": cfg,
            }, os.path.join(ckpt_dir, "best_finetuned.pth"))
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start
        phase = "2a" if epoch < freeze_epochs else "2b"
        pat_max = ft_cfg["patience"]
        status = "★ BEST" if is_best else f"patience {patience_counter}/{pat_max}"

        logger.info(
            f"Epoch {epoch:03d}/{ft_cfg['epochs']} [{epoch_time:.1f}s] Phase {phase} | "
            f"Train — Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val — Acc: {val_metrics['accuracy']:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | {status}"
        )

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        }, os.path.join(ckpt_dir, "checkpoint_last.pth"))

        # Early stopping (only after unfreeze phase)
        if epoch >= freeze_epochs and patience_counter >= ft_cfg["patience"]:
            logger.info(f"\n⏹️  Early stopping at epoch {epoch}")
            break

    total_time = time.time() - total_start
    logger.info(f"\n✅ Fine-tuning complete in {total_time / 60:.1f} minutes")
    logger.info(f"   Best val accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    # ---- Final evaluation on test set ----
    logger.info("\n" + "=" * 70)
    logger.info("🧪 Final Evaluation on Test Set (after fine-tuning)")
    logger.info("=" * 70)

    best_ckpt = torch.load(
        os.path.join(ckpt_dir, "best_finetuned.pth"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state"])

    results = evaluate_model(model, test_loader, device, species_names=SPECIES_NAMES_30)

    # ---- Quick test without fine-tuning for comparison ----
    logger.info("\n" + "=" * 70)
    logger.info("📊 Comparison: Pre-trained (Phase 1) vs Fine-tuned (Phase 2)")
    logger.info("=" * 70)

    # Reload Phase 1 weights
    phase1_ckpt = torch.load(
        cfg["pretrained"]["checkpoint"],
        map_location=device, weights_only=False,
    )
    model.load_state_dict(phase1_ckpt["model_state"])
    phase1_test = evaluate(model, test_loader, criterion, device)

    # Reload Phase 2 weights for final numbers
    model.load_state_dict(best_ckpt["model_state"])
    phase2_test = evaluate(model, test_loader, criterion, device)

    logger.info(f"  Phase 1 (pre-trained only):  Test Acc = {phase1_test['accuracy']:.2f}%")
    logger.info(f"  Phase 2 (after fine-tuning):  Test Acc = {phase2_test['accuracy']:.2f}%")
    logger.info(f"  Improvement:                 +{phase2_test['accuracy'] - phase1_test['accuracy']:.2f}%")

    # ---- Generate figures ----
    try:
        plot_training_curves(history, os.path.join(fig_dir, "fig_finetune_curves"))
        logger.info("  ✅ Figures saved")
    except Exception as e:
        logger.warning(f"  ⚠️ Figure generation failed: {e}")

    # ---- Final summary ----
    logger.info("\n" + "=" * 70)
    logger.info("🏁 FINAL SUMMARY — Phase 2 Fine-Tuning")
    logger.info("=" * 70)
    logger.info(f"  Classes:          {cfg['data']['num_classes']}")
    logger.info(f"  FT Train:         {len(data['X_train']):,}")
    logger.info(f"  FT Val:           {len(data['X_val']):,}")
    logger.info(f"  Test:             {len(data['X_test']):,}")
    logger.info(f"  Best FT val acc:  {best_val_acc:.2f}%")
    logger.info(f"  Test accuracy:    {results['accuracy']:.2f}%")
    logger.info(f"  95% CI:           [{results['ci_95'][0]:.1f}, {results['ci_95'][1]:.1f}]%")
    logger.info(f"  Phase 1 test acc: {phase1_test['accuracy']:.2f}%")
    logger.info(f"  Gain from FT:     +{results['accuracy'] - phase1_test['accuracy']:.2f}%")
    logger.info(f"  Training time:    {total_time / 60:.1f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
