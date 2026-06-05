#!/usr/bin/env python3
"""
PINNACLE — Ablation: Scalogram-Only (CWT branch only, no raw spectra).

Trains using ONLY the 2D ScalogramBranch through Phase 1 + Phase 2,
then compares against spectral-only and full fusion results.

Usage:
    python scripts/ablation_scalogram_only.py
"""

import os
import sys
import time
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
    logger.info("ABLATION: Scalogram-Only (CWT 2D branch, no raw Raman spectra)")
    logger.info("=" * 70)

    # ==================================================================
    # Phase 1: Pre-train scalogram-only on reference
    # ==================================================================
    logger.info("\nLoading reference data...")
    X_ref = np.load(os.path.join(data_dir, "X_reference.npy"))
    y_ref = np.load(os.path.join(data_dir, "y_reference.npy")).astype(np.int64)
    X_ref_wav = np.load(os.path.join(data_dir, "X_reference_wavelet.npy"), mmap_mode="r")

    indices = np.arange(len(X_ref))
    idx_train, idx_val = train_test_split(indices, test_size=0.1, random_state=42, stratify=y_ref)

    # Need wavelets for scalogram-only
    X_train_wav = np.array(X_ref_wav[idx_train])
    X_val_wav = np.array(X_ref_wav[idx_val])

    train_ds = PINNACLEDataset(X_ref[idx_train], y_ref[idx_train], X_train_wav,
                                transform_raman=RamanAugmentation(), split="train")
    val_ds = PINNACLEDataset(X_ref[idx_val], y_ref[idx_val], X_val_wav, split="val")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    logger.info(f"  Train: {len(idx_train)}, Val: {len(idx_val)}")
    logger.info(f"  Train wavelets: {X_train_wav.shape}")

    # Build scalogram-only model
    model = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, mode="scalogram_only").to(device)
    logger.info(f"  Scalogram-only params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_acc = 0.0
    patience = 0
    best_state = None
    start = time.time()

    logger.info("\nPhase 1: Scalogram-only pre-training (50 epochs max)...")
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
            status = "BEST" if is_best else f"p={patience}/15"
            logger.info(f"  Epoch {epoch:03d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")

        if patience >= 15:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    p1_time = time.time() - start
    model.load_state_dict(best_state)
    logger.info(f"  Phase 1 complete: best val = {best_val_acc:.2f}% ({p1_time/60:.1f} min)")

    # Save checkpoint
    os.makedirs("outputs_30class_scalogram_only/checkpoints", exist_ok=True)
    torch.save({"model_state": best_state, "best_val_acc": best_val_acc},
               "outputs_30class_scalogram_only/checkpoints/best_model.pth")

    # Quick test (before fine-tuning)
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)
    X_test_wav = np.load(os.path.join(data_dir, "X_test_wavelet.npy"), mmap_mode="r")
    test_ds = PINNACLEDataset(X_test, y_test, X_test_wav, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    _, p1_test_acc = eval_epoch(model, test_loader, criterion, device)
    logger.info(f"  Phase 1 test acc (no FT): {p1_test_acc:.2f}%")

    # ==================================================================
    # Phase 2: Fine-tune scalogram-only on finetune set
    # ==================================================================
    logger.info("\nPhase 2: Fine-tuning scalogram-only on target domain...")
    X_ft = np.load(os.path.join(data_dir, "X_finetune.npy"))
    y_ft = np.load(os.path.join(data_dir, "y_finetune.npy")).astype(np.int64)
    X_ft_wav = np.load(os.path.join(data_dir, "X_finetune_wavelet.npy"), mmap_mode="r")

    ft_idx = np.arange(len(X_ft))
    ft_train, ft_val = train_test_split(ft_idx, test_size=0.2, random_state=42, stratify=y_ft)

    ft_train_ds = PINNACLEDataset(X_ft[ft_train], y_ft[ft_train], np.array(X_ft_wav[ft_train]),
                                   transform_raman=RamanAugmentation(), split="train")
    ft_val_ds = PINNACLEDataset(X_ft[ft_val], y_ft[ft_val], np.array(X_ft_wav[ft_val]), split="val")

    ft_train_loader = DataLoader(ft_train_ds, batch_size=16, shuffle=True, drop_last=True)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=16, shuffle=False)

    # Phase 2a: freeze backbone
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=0.0005, weight_decay=0.001)
    for epoch in range(5):
        t_loss, t_acc = train_epoch(model, ft_train_loader, optimizer, criterion, device)

    # Phase 2b: unfreeze
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
    logger.info(f"  Phase 2 complete: best FT val = {best_ft_acc:.2f}%")

    # Final test
    results = evaluate_model(model, test_loader, device, species_names=SPECIES_NAMES_30)

    # ==================================================================
    # FULL 3-WAY COMPARISON
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE ABLATION: 3-Way Comparison")
    logger.info("=" * 70)
    logger.info(f"  {'Metric':<25} {'Spectral':>10} {'Scalogram':>10} {'PINNACLE':>10} ")
    logger.info(f"  {'-'*57}")
    logger.info(f"  {'Phase 1 Val':<25} {'60.30%':>10} {best_val_acc:>9.2f}% {'91.52%':>10}")
    logger.info(f"  {'Phase 1 Test (no FT)':<25} {'21.33%':>10} {p1_test_acc:>9.2f}% {'45.43%':>10}")
    logger.info(f"  {'Phase 2 FT Val':<25} {'48.50%':>10} {best_ft_acc:>9.2f}% {'82.33%':>10}")
    logger.info(f"  {'Phase 2 Test (final)':<25} {'41.07%':>10} {results['accuracy']:>9.2f}% {'73.63%':>10}")
    logger.info(f"  {'Parameters':<25} {'205,054':>10} {count_parameters(model):>9,} {'303,998':>10}")
    logger.info("=" * 70)

    # Synergy check
    spec_acc = 41.07
    scal_acc = results['accuracy']
    fuse_acc = 73.63
    avg_branches = (spec_acc + scal_acc) / 2
    synergy = fuse_acc - avg_branches

    logger.info(f"  Spectral-only test:    {spec_acc:.2f}%")
    logger.info(f"  Scalogram-only test:   {scal_acc:.2f}%")
    logger.info(f"  Average of branches:   {avg_branches:.2f}%")
    logger.info(f"  PINNACLE (fusion):     {fuse_acc:.2f}%")
    logger.info(f"  Fusion synergy:        +{synergy:.2f}% above average")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
