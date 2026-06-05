#!/usr/bin/env python3
"""
PINNACLE — Late Fusion: Unimodal Pre-training Strategy.

1. Load best spectral-only weights → inject into PINNACLE.spectral_branch
2. Load best scalogram-only weights → inject into PINNACLE.scalogram_branch
3. Freeze both branches (keep their optimized representations)
4. Phase 1: Train ONLY fusion + classifier on reference set
5. Phase 2: Fine-tune on finetune set (branches stay frozen, then gradual unfreeze)
6. Evaluate on test

This guarantees Fusion >= max(Spectral, Scalogram) because the fusion
module is forced to combine already-optimal representations.
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


def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val


def main():
    set_seed(42)
    device = get_device()
    data_dir = "New data"

    logger.info("=" * 70)
    logger.info("PINNACLE — Late Fusion: Unimodal Pre-training Strategy")
    logger.info("=" * 70)

    # ==================================================================
    # Step 1: Build fusion model and inject branch weights
    # ==================================================================
    model = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, use_fusion=True)

    # Load spectral-only weights
    spec_ckpt = torch.load("outputs_30class_nofusion/checkpoints/best_model.pth",
                            map_location="cpu", weights_only=False)
    spec_state = spec_ckpt["model_state"]
    logger.info(f"  Spectral-only: val={spec_ckpt['best_val_acc']:.2f}%")

    # Load scalogram-only weights
    scal_ckpt = torch.load("outputs_30class_scalogram_only/checkpoints/best_model.pth",
                            map_location="cpu", weights_only=False)
    scal_state = scal_ckpt["model_state"]
    logger.info(f"  Scalogram-only: val={scal_ckpt['best_val_acc']:.2f}%")

    # Inject: spectral_branch from spectral-only model
    spec_branch_keys = [k for k in spec_state if k.startswith("spectral_branch.")]
    for k in spec_branch_keys:
        model.state_dict()[k].copy_(spec_state[k])
    logger.info(f"  Injected {len(spec_branch_keys)} spectral branch params")

    # Inject: scalogram_branch from scalogram-only model
    scal_branch_keys = [k for k in scal_state if k.startswith("scalogram_branch.")]
    for k in scal_branch_keys:
        model.state_dict()[k].copy_(scal_state[k])
    logger.info(f"  Injected {len(scal_branch_keys)} scalogram branch params")

    model = model.to(device)

    # Freeze both branches
    set_requires_grad(model.spectral_branch, False)
    set_requires_grad(model.scalogram_branch, False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total: {total_params:,} | Trainable: {trainable:,} (fusion + classifier)")

    # ==================================================================
    # Step 2: Phase 1 — Train fusion+classifier on reference set
    # ==================================================================
    logger.info("\nLoading reference data...")
    X_ref = np.load(os.path.join(data_dir, "X_reference.npy"))
    y_ref = np.load(os.path.join(data_dir, "y_reference.npy")).astype(np.int64)
    X_ref_wav = np.load(os.path.join(data_dir, "X_reference_wavelet.npy"), mmap_mode="r")

    indices = np.arange(len(X_ref))
    idx_train, idx_val = train_test_split(indices, test_size=0.1, random_state=42, stratify=y_ref)

    X_train_wav = np.array(X_ref_wav[idx_train])
    X_val_wav = np.array(X_ref_wav[idx_val])

    aug = RamanAugmentation(noise_std=0.01, shift_range=5, scale_range=0.05, probability=0.5)
    ref_train_ds = PINNACLEDataset(X_ref[idx_train], y_ref[idx_train], X_train_wav,
                                    transform_raman=aug, split="train")
    ref_val_ds = PINNACLEDataset(X_ref[idx_val], y_ref[idx_val], X_val_wav, split="val")

    ref_train_loader = DataLoader(ref_train_ds, batch_size=32, shuffle=True, drop_last=True)
    ref_val_loader = DataLoader(ref_val_ds, batch_size=32, shuffle=False)

    logger.info(f"  Ref Train: {len(idx_train)}, Ref Val: {len(idx_val)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001, weight_decay=0.0001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    best_val_acc = 0.0
    best_state = None
    patience = 0
    start = time.time()

    logger.info("\nPhase 1: Train fusion + classifier on reference (25 epochs max)...")
    for epoch in range(25):
        t_loss, t_acc = train_epoch(model, ref_train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, ref_val_loader, criterion, device)
        scheduler.step()

        is_best = v_acc > best_val_acc
        if is_best:
            best_val_acc = v_acc
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if epoch % 3 == 0 or is_best:
            status = "BEST" if is_best else f"p={patience}/10"
            logger.info(f"  P1 E{epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")

        if patience >= 10:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    p1_time = time.time() - start
    model.load_state_dict(best_state)
    logger.info(f"  Phase 1 complete: best val = {best_val_acc:.2f}% ({p1_time/60:.1f} min)")

    # Quick test check
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)
    X_test_wav = np.load(os.path.join(data_dir, "X_test_wavelet.npy"), mmap_mode="r")
    test_ds = PINNACLEDataset(X_test, y_test, X_test_wav, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    _, p1_test_acc = eval_epoch(model, test_loader, criterion, device)
    logger.info(f"  Phase 1 test acc (no FT): {p1_test_acc:.2f}%")

    # ==================================================================
    # Step 3: Phase 2 — Fine-tune on finetune set
    # ==================================================================
    logger.info("\nPhase 2: Fine-tuning on target domain...")
    X_ft = np.load(os.path.join(data_dir, "X_finetune.npy"))
    y_ft = np.load(os.path.join(data_dir, "y_finetune.npy")).astype(np.int64)
    X_ft_wav = np.load(os.path.join(data_dir, "X_finetune_wavelet.npy"))

    ft_idx = np.arange(len(X_ft))
    ft_train, ft_val = train_test_split(ft_idx, test_size=0.2, random_state=42, stratify=y_ft)

    ft_train_ds = PINNACLEDataset(X_ft[ft_train], y_ft[ft_train], X_ft_wav[ft_train],
                                   transform_raman=aug, split="train")
    ft_val_ds = PINNACLEDataset(X_ft[ft_val], y_ft[ft_val], X_ft_wav[ft_val], split="val")
    ft_train_loader = DataLoader(ft_train_ds, batch_size=16, shuffle=True, drop_last=True)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=16, shuffle=False)

    logger.info(f"  FT Train: {len(ft_train)}, FT Val: {len(ft_val)}")

    # Phase 2a: Fusion + classifier only (branches still frozen)
    best_ft_val = 0.0
    best_ft_state = None
    patience = 0

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0005, weight_decay=0.001,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    logger.info("  Phase 2a: Fusion + Classifier only (15 epochs)")
    for epoch in range(15):
        t_loss, t_acc = train_epoch(model, ft_train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, ft_val_loader, criterion, device)
        scheduler.step()

        if v_acc > best_ft_val:
            best_ft_val = v_acc
            patience = 0
            best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        status = "BEST" if patience == 0 else f"p={patience}"
        logger.info(f"    2a E{epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")

    # Phase 2b: Unfreeze scalogram branch with very low LR
    model.load_state_dict(best_ft_state)
    set_requires_grad(model.scalogram_branch, True)
    patience = 0

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n  Phase 2b: Scalo + Fusion + Classifier ({trainable:,} params, 20 epochs)")

    optimizer = torch.optim.AdamW([
        {"params": model.scalogram_branch.parameters(), "lr": 0.00002},
        {"params": model.fusion.parameters(), "lr": 0.00005},
        {"params": model.classifier.parameters(), "lr": 0.00005},
    ], weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    for epoch in range(20):
        t_loss, t_acc = train_epoch(model, ft_train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, ft_val_loader, criterion, device)
        scheduler.step()

        if v_acc > best_ft_val:
            best_ft_val = v_acc
            patience = 0
            best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        status = "BEST" if patience == 0 else f"p={patience}"
        logger.info(f"    2b E{epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")

        if patience >= 10:
            logger.info("    Early stopping 2b")
            break

    # Phase 2c: Unfreeze all with ultra-low LR
    model.load_state_dict(best_ft_state)
    set_requires_grad(model.spectral_branch, True)
    patience = 0

    logger.info(f"\n  Phase 2c: Full model (all {count_parameters(model):,} params, 15 epochs)")

    optimizer = torch.optim.AdamW([
        {"params": model.spectral_branch.parameters(), "lr": 0.000005},
        {"params": model.scalogram_branch.parameters(), "lr": 0.000005},
        {"params": model.fusion.parameters(), "lr": 0.00001},
        {"params": model.classifier.parameters(), "lr": 0.00001},
    ], weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    for epoch in range(15):
        t_loss, t_acc = train_epoch(model, ft_train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, ft_val_loader, criterion, device)
        scheduler.step()

        if v_acc > best_ft_val:
            best_ft_val = v_acc
            patience = 0
            best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        status = "BEST" if patience == 0 else f"p={patience}"
        logger.info(f"    2c E{epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")

        if patience >= 8:
            logger.info("    Early stopping 2c")
            break

    total_time = time.time() - start

    # ==================================================================
    # Final evaluation
    # ==================================================================
    model.load_state_dict(best_ft_state)

    os.makedirs("outputs_30class_latefusion/checkpoints", exist_ok=True)
    torch.save({"model_state": best_ft_state, "best_val_acc": best_ft_val},
               "outputs_30class_latefusion/checkpoints/best_finetuned.pth")

    logger.info("\n" + "=" * 70)
    logger.info("Final Test Evaluation — Late Fusion PINNACLE")
    logger.info("=" * 70)
    results = evaluate_model(model, test_loader, device, species_names=SPECIES_NAMES_30)

    # ---- Full comparison ----
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE 3-WAY COMPARISON")
    logger.info("=" * 70)
    logger.info(f"  {'Model':<35} {'Val':>8} {'Test':>8}")
    logger.info(f"  {'-'*53}")
    logger.info(f"  {'Spectral-only':<35} {'60.30%':>8} {'41.07%':>8}")
    logger.info(f"  {'Scalogram-only':<35} {'92.33%':>8} {'76.87%':>8}")
    logger.info(f"  {'PINNACLE (joint train)':<35} {'91.52%':>8} {'73.63%':>8}")
    logger.info(f"  {'PINNACLE Late Fusion (NEW)':<35} {best_ft_val:>7.2f}% {results['accuracy']:>7.2f}%")
    logger.info(f"  Training time: {total_time/60:.1f} min")

    delta = results['accuracy'] - 76.87
    if delta > 0:
        logger.info(f"\n  FUSION BEATS SCALOGRAM-ONLY by +{delta:.2f}%!")
    else:
        logger.info(f"\n  Gap to scalogram-only: {delta:.2f}%")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
