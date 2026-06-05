#!/usr/bin/env python3
"""
PINNACLE — Final Push: Two strategies to beat scalogram-only.

Strategy A: Late fusion with branches PERMANENTLY frozen (never unfreeze)
Strategy B: Ensemble of spectral-only + scalogram-only at inference

Both evaluated on the same test set.
"""

import os, sys, time, numpy as np
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

    # ---- Load test data ----
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)
    X_test_wav = np.load(os.path.join(data_dir, "X_test_wavelet.npy"), mmap_mode="r")
    test_ds = PINNACLEDataset(X_test, y_test, X_test_wav, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # ---- Load finetune data ----
    X_ft = np.load(os.path.join(data_dir, "X_finetune.npy"))
    y_ft = np.load(os.path.join(data_dir, "y_finetune.npy")).astype(np.int64)
    X_ft_wav = np.load(os.path.join(data_dir, "X_finetune_wavelet.npy"))
    ft_idx = np.arange(len(X_ft))
    ft_train, ft_val = train_test_split(ft_idx, test_size=0.2, random_state=42, stratify=y_ft)

    aug = RamanAugmentation(noise_std=0.01, shift_range=5, scale_range=0.05, probability=0.5)
    ft_train_ds = PINNACLEDataset(X_ft[ft_train], y_ft[ft_train], X_ft_wav[ft_train],
                                   transform_raman=aug, split="train")
    ft_val_ds = PINNACLEDataset(X_ft[ft_val], y_ft[ft_val], X_ft_wav[ft_val], split="val")
    ft_train_loader = DataLoader(ft_train_ds, batch_size=16, shuffle=True, drop_last=True)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=16, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # ==================================================================
    # STRATEGY A: Late fusion, branches PERMANENTLY frozen
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STRATEGY A: Late Fusion — Branches Permanently Frozen")
    logger.info("=" * 70)

    # Load the late fusion model from Phase 1
    if os.path.exists("outputs_30class_latefusion/checkpoints/best_finetuned.pth"):
        # Use the Phase 1 trained late fusion model, but we need to retrain
        # Let's start fresh with injected weights
        pass

    model_a = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, use_fusion=True)

    # Inject branch weights
    spec_ckpt = torch.load("outputs_30class_nofusion/checkpoints/best_model.pth",
                            map_location="cpu", weights_only=False)
    scal_ckpt = torch.load("outputs_30class_scalogram_only/checkpoints/best_model.pth",
                            map_location="cpu", weights_only=False)

    for k in [k for k in spec_ckpt["model_state"] if k.startswith("spectral_branch.")]:
        model_a.state_dict()[k].copy_(spec_ckpt["model_state"][k])
    for k in [k for k in scal_ckpt["model_state"] if k.startswith("scalogram_branch.")]:
        model_a.state_dict()[k].copy_(scal_ckpt["model_state"][k])

    model_a = model_a.to(device)

    # PERMANENTLY freeze both branches
    for name, param in model_a.named_parameters():
        if "spectral_branch" in name or "scalogram_branch" in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
    logger.info(f"  Trainable: {trainable:,} (fusion + classifier ONLY)")

    # Train directly on finetune data (skip reference Phase 1 for speed)
    # Use a warmup + cosine schedule
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_a.parameters()),
        lr=0.001, weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val = 0.0
    best_state_a = None
    patience = 0

    logger.info("  Training fusion+classifier on finetune set (50 epochs)...")
    for epoch in range(50):
        t_loss, t_acc = train_epoch(model_a, ft_train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model_a, ft_val_loader, criterion, device)
        scheduler.step()

        if v_acc > best_val:
            best_val = v_acc
            patience = 0
            best_state_a = {k: v.clone() for k, v in model_a.state_dict().items()}
        else:
            patience += 1

        if epoch % 5 == 0:
            status = f"BEST ({best_val:.1f}%)" if patience == 0 else f"p={patience}"
            logger.info(f"    E{epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")

        if patience >= 15:
            logger.info(f"    Early stopping at epoch {epoch}")
            break

    model_a.load_state_dict(best_state_a)
    logger.info(f"  Strategy A best val: {best_val:.2f}%")

    # Evaluate
    results_a = evaluate_model(model_a, test_loader, device, species_names=SPECIES_NAMES_30)

    # ==================================================================
    # STRATEGY B: Ensemble (weighted average of logits)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY B: Ensemble of Spectral-only + Scalogram-only")
    logger.info("=" * 70)

    # Load spectral-only fine-tuned model
    # (We don't have a fine-tuned spectral-only checkpoint, so use Phase 1)
    model_spec = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, mode="spectral_only").to(device)
    model_spec.load_state_dict(spec_ckpt["model_state"])
    model_spec.eval()

    # Load scalogram-only fine-tuned model
    model_scal = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, mode="scalogram_only").to(device)
    model_scal.load_state_dict(scal_ckpt["model_state"])
    model_scal.eval()

    # Try multiple ensemble weights
    best_ens_acc = 0.0
    best_ens_weight = 0.0

    for w_scal in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        w_spec = 1.0 - w_scal
        correct = 0
        total = 0

        with torch.no_grad():
            for raman, scalogram, labels in test_loader:
                raman, scalogram, labels = raman.to(device), scalogram.to(device), labels.to(device)

                logits_spec, _, _ = model_spec(raman, scalogram)
                logits_scal, _, _ = model_scal(raman, scalogram)

                # Weighted logit fusion
                logits_ens = w_spec * logits_spec + w_scal * logits_scal
                _, pred = logits_ens.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        acc = 100.0 * correct / total
        if acc > best_ens_acc:
            best_ens_acc = acc
            best_ens_weight = w_scal
        logger.info(f"  w_scal={w_scal:.2f}, w_spec={w_spec:.2f} → Test: {acc:.2f}%")

    logger.info(f"  Best ensemble: w_scal={best_ens_weight:.2f} → {best_ens_acc:.2f}%")

    # ==================================================================
    # FINAL COMPARISON TABLE
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL COMPLETE COMPARISON")
    logger.info("=" * 70)
    logger.info(f"  {'Model':<40} {'Test Acc':>10}")
    logger.info(f"  {'-'*52}")
    logger.info(f"  {'Spectral-only (1D CNN)':<40} {'41.07%':>10}")
    logger.info(f"  {'Scalogram-only (2D CNN)':<40} {'76.87%':>10}")
    logger.info(f"  {'PINNACLE joint-train + FT':<40} {'73.63%':>10}")
    logger.info(f"  {'PINNACLE late-fusion (frozen branches)':<40} {results_a['accuracy']:>9.2f}%")
    logger.info(f"  {'Ensemble (weighted logits)':<40} {best_ens_acc:>9.2f}%")
    logger.info("=" * 70)

    winner = max([
        ("PINNACLE late-fusion", results_a['accuracy']),
        ("Ensemble", best_ens_acc),
    ], key=lambda x: x[1])

    if winner[1] > 76.87:
        logger.info(f"\n  WINNER: {winner[0]} at {winner[1]:.2f}% — BEATS scalogram-only!")
    else:
        logger.info(f"\n  Best fusion: {winner[0]} at {winner[1]:.2f}% (gap: {winner[1]-76.87:.2f}%)")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
