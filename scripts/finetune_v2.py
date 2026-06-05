#!/usr/bin/env python3
"""
PINNACLE — Optimized Phase 2 Fine-Tuning for Fusion Model.

Strategy:
  Phase 2a: Freeze BOTH encoders, train only fusion + classifier (10 epochs)
  Phase 2b: Unfreeze scalogram branch + fusion, keep spectral frozen (20 epochs)
  Phase 2c: Unfreeze all for final polish (20 epochs, very low LR)

This gives the fusion module time to adapt before the backbone gets updated.
"""

import os, sys, time, yaml, numpy as np
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


def set_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad = value


def main():
    set_seed(42)
    device = get_device()
    data_dir = "New data"

    logger.info("=" * 70)
    logger.info("PINNACLE — Optimized Phase 2 Fine-Tuning (3-stage)")
    logger.info("=" * 70)

    # ---- Load data ----
    X_ft = np.load(os.path.join(data_dir, "X_finetune.npy"))
    y_ft = np.load(os.path.join(data_dir, "y_finetune.npy")).astype(np.int64)
    X_ft_wav = np.load(os.path.join(data_dir, "X_finetune_wavelet.npy"), mmap_mode="r")
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)
    X_test_wav = np.load(os.path.join(data_dir, "X_test_wavelet.npy"), mmap_mode="r")

    ft_idx = np.arange(len(X_ft))
    ft_train, ft_val = train_test_split(ft_idx, test_size=0.2, random_state=42, stratify=y_ft)

    aug = RamanAugmentation(noise_std=0.01, shift_range=5, scale_range=0.05, probability=0.5)
    train_ds = PINNACLEDataset(X_ft[ft_train], y_ft[ft_train], np.array(X_ft_wav[ft_train]),
                                transform_raman=aug, split="train")
    val_ds = PINNACLEDataset(X_ft[ft_val], y_ft[ft_val], np.array(X_ft_wav[ft_val]), split="val")
    test_ds = PINNACLEDataset(X_test, y_test, X_test_wav, split="test")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    logger.info(f"  FT Train: {len(ft_train)}, FT Val: {len(ft_val)}, Test: {len(X_test)}")

    # ---- Load pre-trained model ----
    model = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, use_fusion=True).to(device)
    ckpt = torch.load("outputs_30class/checkpoints/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"  Loaded Phase 1 weights (val={ckpt['best_val_acc']:.2f}%)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for regularization

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    total_start = time.time()

    # ==================================================================
    # Phase 2a: Freeze BOTH encoders → train fusion + classifier only
    # ==================================================================
    set_requires_grad(model.spectral_branch, False)
    set_requires_grad(model.scalogram_branch, False)
    # fusion + classifier remain trainable

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nPhase 2a: Fusion + Classifier only ({trainable:,} trainable params)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0005, weight_decay=0.001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(10):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        status = "BEST" if v_acc >= best_val_acc else ""
        logger.info(f"  2a Epoch {epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f} {status}")

    # ==================================================================
    # Phase 2b: Unfreeze scalogram + fusion, keep spectral frozen
    # ==================================================================
    set_requires_grad(model.scalogram_branch, True)
    # spectral stays frozen — it's the weaker branch

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nPhase 2b: Scalogram + Fusion + Classifier ({trainable:,} trainable params)")

    optimizer = torch.optim.AdamW([
        {"params": model.scalogram_branch.parameters(), "lr": 0.00002},
        {"params": model.fusion.parameters(), "lr": 0.0001},
        {"params": model.classifier.parameters(), "lr": 0.0001},
    ], weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    patience = 0

    for epoch in range(20):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = 10 + epoch
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        status = "BEST" if patience == 0 else f"p={patience}"
        logger.info(f"  2b Epoch {epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")

        if patience >= 10:
            logger.info("  Early stopping Phase 2b")
            break

    # ==================================================================
    # Phase 2c: Unfreeze all for final polish
    # ==================================================================
    model.load_state_dict(best_state)  # restart from best
    set_requires_grad(model.spectral_branch, True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nPhase 2c: Full model ({trainable:,} trainable params)")

    optimizer = torch.optim.AdamW([
        {"params": model.spectral_branch.parameters(), "lr": 0.000005},   # Very low
        {"params": model.scalogram_branch.parameters(), "lr": 0.000005},  # Very low
        {"params": model.fusion.parameters(), "lr": 0.00002},
        {"params": model.classifier.parameters(), "lr": 0.00002},
    ], weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    patience = 0

    for epoch in range(20):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = 30 + epoch
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        status = "BEST" if patience == 0 else f"p={patience}"
        logger.info(f"  2c Epoch {epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")

        if patience >= 10:
            logger.info("  Early stopping Phase 2c")
            break

    total_time = time.time() - total_start

    # ---- Final eval ----
    model.load_state_dict(best_state)
    logger.info(f"\nBest val: {best_val_acc:.2f}% at epoch {best_epoch}")

    # Save
    os.makedirs("outputs_30class_finetune_v2/checkpoints", exist_ok=True)
    torch.save({"model_state": best_state, "best_val_acc": best_val_acc, "epoch": best_epoch},
               "outputs_30class_finetune_v2/checkpoints/best_finetuned.pth")

    logger.info("\n" + "=" * 70)
    logger.info("Final Test Evaluation")
    logger.info("=" * 70)
    results = evaluate_model(model, test_loader, device, species_names=SPECIES_NAMES_30)

    # ---- Comparison ----
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE 3-WAY COMPARISON (updated)")
    logger.info("=" * 70)
    logger.info(f"  {'Model':<25} {'Val':>8} {'Test':>8}")
    logger.info(f"  {'-'*43}")
    logger.info(f"  {'Spectral-only':<25} {'60.30%':>8} {'41.07%':>8}")
    logger.info(f"  {'Scalogram-only':<25} {'92.33%':>8} {'76.87%':>8}")
    logger.info(f"  {'PINNACLE v1 FT':<25} {'82.33%':>8} {'73.63%':>8}")
    logger.info(f"  {'PINNACLE v2 FT (new)':<25} {best_val_acc:>7.2f}% {results['accuracy']:>7.2f}%")
    logger.info(f"  Training time: {total_time/60:.1f} min")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
