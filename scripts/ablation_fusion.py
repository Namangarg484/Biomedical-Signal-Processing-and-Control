#!/usr/bin/env python3
"""
PINNACLE — Ablation Study: Spectral-Only vs Full Fusion.

Runs Phase 1 + Phase 2 with use_fusion=False, then compares
against the existing fusion results to quantify SeparationCross contribution.

Usage:
    python scripts/ablation_fusion.py
"""

import os
import sys
import time
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from pinnacle.utils import set_seed, get_device, logger, count_parameters
from pinnacle.model import PINNACLE
from pinnacle.dataset import PINNACLEDataset, RamanAugmentation
from pinnacle.evaluate import evaluate_model


SPECIES_NAMES_30 = [f"Species_{i}" for i in range(30)]


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=5.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for raman, scalogram, labels in loader:
        raman, scalogram, labels = raman.to(device), scalogram.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(raman, scalogram)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for raman, scalogram, labels in loader:
        raman, scalogram, labels = raman.to(device), scalogram.to(device), labels.to(device)
        logits, _, _ = model(raman, scalogram)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    set_seed(42)
    device = get_device()
    data_dir = "New data"

    logger.info("=" * 70)
    logger.info("🔬 ABLATION: Spectral-Only (no fusion) vs PINNACLE (with fusion)")
    logger.info("=" * 70)

    # ==================================================================
    # Phase 1: Pre-train spectral-only on reference
    # ==================================================================
    logger.info("\n📦 Loading reference data...")
    X_ref = np.load(os.path.join(data_dir, "X_reference.npy"))
    y_ref = np.load(os.path.join(data_dir, "y_reference.npy")).astype(np.int64)

    indices = np.arange(len(X_ref))
    idx_train, idx_val = train_test_split(indices, test_size=0.1, random_state=42, stratify=y_ref)

    # No wavelets needed for spectral-only
    train_ds = PINNACLEDataset(X_ref[idx_train], y_ref[idx_train], None,
                                transform_raman=RamanAugmentation(), split="train")
    val_ds = PINNACLEDataset(X_ref[idx_val], y_ref[idx_val], None, split="val")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    logger.info(f"  Train: {len(idx_train)}, Val: {len(idx_val)}")

    # Build spectral-only model
    model = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, use_fusion=False).to(device)
    logger.info(f"  Spectral-only params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_acc = 0.0
    patience = 0
    best_state = None
    start = time.time()

    logger.info("\n🚀 Phase 1: Spectral-only pre-training (50 epochs max)...")
    for epoch in range(50):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        is_best = v_acc > best_val_acc
        if is_best:
            best_val_acc = v_acc
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if epoch % 5 == 0 or is_best:
            status = "★ BEST" if is_best else f"p={patience}/15"
            logger.info(f"  Epoch {epoch:03d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")

        if patience >= 15:
            logger.info(f"  ⏹️ Early stopping at epoch {epoch}")
            break

    p1_time = time.time() - start
    model.load_state_dict(best_state)
    logger.info(f"  ✅ Phase 1 complete: best val = {best_val_acc:.2f}% ({p1_time/60:.1f} min)")

    # Save checkpoint
    os.makedirs("outputs_30class_nofusion/checkpoints", exist_ok=True)
    torch.save({"model_state": best_state, "best_val_acc": best_val_acc},
               "outputs_30class_nofusion/checkpoints/best_model.pth")

    # Quick test (before fine-tuning)
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)
    test_ds = PINNACLEDataset(X_test, y_test, None, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    _, p1_test_acc = eval_epoch(model, test_loader, criterion, device)
    logger.info(f"  Phase 1 test acc (no FT): {p1_test_acc:.2f}%")

    # ==================================================================
    # Phase 2: Fine-tune spectral-only on finetune set
    # ==================================================================
    logger.info("\n🔧 Phase 2: Fine-tuning spectral-only on target domain...")
    X_ft = np.load(os.path.join(data_dir, "X_finetune.npy"))
    y_ft = np.load(os.path.join(data_dir, "y_finetune.npy")).astype(np.int64)

    ft_idx = np.arange(len(X_ft))
    ft_train, ft_val = train_test_split(ft_idx, test_size=0.2, random_state=42, stratify=y_ft)

    ft_train_ds = PINNACLEDataset(X_ft[ft_train], y_ft[ft_train], None,
                                   transform_raman=RamanAugmentation(), split="train")
    ft_val_ds = PINNACLEDataset(X_ft[ft_val], y_ft[ft_val], None, split="val")

    ft_train_loader = DataLoader(ft_train_ds, batch_size=16, shuffle=True, drop_last=True)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=16, shuffle=False)

    # Freeze backbone first
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=0.0005, weight_decay=0.001)

    # Phase 2a: classifier only
    for epoch in range(5):
        t_loss, t_acc = train_epoch(model, ft_train_loader, optimizer, criterion, device)

    # Phase 2b: unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    best_ft_acc = 0.0
    patience = 0
    best_ft_state = None

    for epoch in range(25):
        t_loss, t_acc = train_epoch(model, ft_train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, ft_val_loader, criterion, device)
        scheduler.step()

        if v_acc > best_ft_acc:
            best_ft_acc = v_acc
            patience = 0
            best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if epoch % 5 == 0:
            logger.info(f"  FT Epoch {epoch:03d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}%")

        if patience >= 10:
            break

    model.load_state_dict(best_ft_state)
    logger.info(f"  ✅ Phase 2 complete: best FT val = {best_ft_acc:.2f}%")

    # Final test
    results = evaluate_model(model, test_loader, device, species_names=SPECIES_NAMES_30)

    # ==================================================================
    # COMPARISON TABLE
    # ==================================================================
    # Load fusion results
    fusion_p1_val = 91.52
    fusion_p1_test = 45.43
    fusion_p2_test = 73.63

    logger.info("\n" + "=" * 70)
    logger.info("📊 ABLATION RESULTS: Spectral-Only vs PINNACLE (Fusion)")
    logger.info("=" * 70)
    logger.info(f"  {'Metric':<30} {'Spectral-Only':>15} {'PINNACLE (Fusion)':>18} {'Δ Fusion':>10}")
    logger.info(f"  {'-'*73}")
    logger.info(f"  {'Phase 1 Val Acc':<30} {best_val_acc:>14.2f}% {fusion_p1_val:>17.2f}% {fusion_p1_val - best_val_acc:>+9.2f}%")
    logger.info(f"  {'Phase 1 Test (no FT)':<30} {p1_test_acc:>14.2f}% {fusion_p1_test:>17.2f}% {fusion_p1_test - p1_test_acc:>+9.2f}%")
    logger.info(f"  {'Phase 2 FT Val':<30} {best_ft_acc:>14.2f}% {'82.33':>17}% {82.33 - best_ft_acc:>+9.2f}%")
    logger.info(f"  {'Phase 2 Test (final)':<30} {results['accuracy']:>14.2f}% {fusion_p2_test:>17.2f}% {fusion_p2_test - results['accuracy']:>+9.2f}%")
    logger.info(f"  {'Parameters':<30} {count_parameters(model):>14,} {'303,998':>18}")
    logger.info("=" * 70)

    if fusion_p2_test > results["accuracy"]:
        delta = fusion_p2_test - results["accuracy"]
        logger.info(f"  ✅ SeparationCross fusion contributes +{delta:.2f}% to final test accuracy")
    else:
        delta = results["accuracy"] - fusion_p2_test
        logger.info(f"  ⚠️ Spectral-only is +{delta:.2f}% better — fusion may not help on this task")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
