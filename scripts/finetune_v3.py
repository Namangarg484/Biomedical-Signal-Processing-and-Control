#!/usr/bin/env python3
"""
PINNACLE — Phase 2 v3: Mixed-Domain Fine-Tuning.

Strategy: Instead of fine-tuning on just 2,400 target-domain samples,
mix in a subsample of reference data (data replay) to give the fusion
model enough data to leverage its extra capacity.

This prevents catastrophic forgetting and gives the larger fusion model
a statistical advantage over the simpler scalogram-only model.

Protocol:
  1. Load 2400 finetune train + 600 finetune val
  2. Add 6000 reference samples (10%) as "replay" data with lower weight
  3. Train with domain-weighted loss (finetune samples weighted higher)
  4. 3-stage unfreeze: classifier → fusion+scalo → all
"""

import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from pinnacle.utils import set_seed, get_device, logger, count_parameters
from pinnacle.model import PINNACLE
from pinnacle.dataset import PINNACLEDataset, RamanAugmentation
from pinnacle.evaluate import evaluate_model

SPECIES_NAMES_30 = [f"Species_{i}" for i in range(30)]


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=5.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (raman, scalogram, labels) in enumerate(loader):
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
    logger.info("PINNACLE — Phase 2 v3: Mixed-Domain Fine-Tuning")
    logger.info("=" * 70)

    # ---- Load finetune data ----
    X_ft = np.load(os.path.join(data_dir, "X_finetune.npy"))
    y_ft = np.load(os.path.join(data_dir, "y_finetune.npy")).astype(np.int64)
    X_ft_wav = np.load(os.path.join(data_dir, "X_finetune_wavelet.npy"))

    ft_idx = np.arange(len(X_ft))
    ft_train_idx, ft_val_idx = train_test_split(ft_idx, test_size=0.2, random_state=42, stratify=y_ft)

    # ---- Load reference subset for replay ----
    X_ref = np.load(os.path.join(data_dir, "X_reference.npy"))
    y_ref = np.load(os.path.join(data_dir, "y_reference.npy")).astype(np.int64)
    X_ref_wav = np.load(os.path.join(data_dir, "X_reference_wavelet.npy"), mmap_mode="r")

    # Stratified subsample: 200 per class = 6000 total (10% of reference)
    ref_sub_idx = []
    for c in range(30):
        class_idx = np.where(y_ref == c)[0]
        chosen = np.random.RandomState(42).choice(class_idx, size=200, replace=False)
        ref_sub_idx.extend(chosen)
    ref_sub_idx = np.array(ref_sub_idx)
    np.random.RandomState(42).shuffle(ref_sub_idx)

    X_ref_sub = X_ref[ref_sub_idx]
    y_ref_sub = y_ref[ref_sub_idx].astype(np.int64)
    X_ref_sub_wav = np.array(X_ref_wav[ref_sub_idx])

    logger.info(f"  FT Train: {len(ft_train_idx)}, FT Val: {len(ft_val_idx)}")
    logger.info(f"  Reference replay: {len(ref_sub_idx)} (200/class)")

    # ---- Load test data ----
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)
    X_test_wav = np.load(os.path.join(data_dir, "X_test_wavelet.npy"), mmap_mode="r")

    # ---- Create datasets ----
    aug = RamanAugmentation(noise_std=0.01, shift_range=5, scale_range=0.05, probability=0.5)

    ft_train_ds = PINNACLEDataset(X_ft[ft_train_idx], y_ft[ft_train_idx],
                                   np.array(X_ft_wav[ft_train_idx]),
                                   transform_raman=aug, split="train")
    ref_replay_ds = PINNACLEDataset(X_ref_sub, y_ref_sub, X_ref_sub_wav,
                                     transform_raman=aug, split="train")

    # Combine: finetune + reference replay
    combined_ds = ConcatDataset([ft_train_ds, ref_replay_ds])

    # Weighted sampler: finetune samples get 3x weight (prioritize domain adaptation)
    weights = [3.0] * len(ft_train_ds) + [1.0] * len(ref_replay_ds)
    sampler = WeightedRandomSampler(weights, num_samples=len(combined_ds), replacement=True)

    train_loader = DataLoader(combined_ds, batch_size=32, sampler=sampler, drop_last=True)
    val_loader = DataLoader(
        PINNACLEDataset(X_ft[ft_val_idx], y_ft[ft_val_idx],
                        np.array(X_ft_wav[ft_val_idx]), split="val"),
        batch_size=32, shuffle=False)
    test_loader = DataLoader(
        PINNACLEDataset(X_test, y_test, X_test_wav, split="test"),
        batch_size=32, shuffle=False)

    logger.info(f"  Combined train: {len(combined_ds)} samples (FT 3x weighted)")
    logger.info(f"  Train batches/epoch: {len(train_loader)}")

    # ---- Load model ----
    model = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, use_fusion=True).to(device)
    ckpt = torch.load("outputs_30class/checkpoints/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"  Loaded Phase 1 weights (val={ckpt['best_val_acc']:.2f}%)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0
    best_state = None
    best_epoch = -1
    total_start = time.time()

    # ==================================================================
    # Phase 2a: Freeze both encoders (10 epochs)
    # ==================================================================
    set_requires_grad(model.spectral_branch, False)
    set_requires_grad(model.scalogram_branch, False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nPhase 2a: Fusion + Classifier ({trainable:,} params) — 10 epochs")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0003, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(10):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        logger.info(f"  2a E{epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | "
                     f"{'BEST' if v_acc >= best_val_acc else ''}")

    # ==================================================================
    # Phase 2b: Unfreeze scalogram + fusion (25 epochs)
    # ==================================================================
    model.load_state_dict(best_state)
    set_requires_grad(model.scalogram_branch, True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nPhase 2b: Scalo + Fusion + Classifier ({trainable:,} params) — 25 epochs")

    optimizer = torch.optim.AdamW([
        {"params": model.scalogram_branch.parameters(), "lr": 0.00003},
        {"params": model.fusion.parameters(), "lr": 0.00008},
        {"params": model.classifier.parameters(), "lr": 0.00008},
    ], weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    patience = 0

    for epoch in range(25):
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
        logger.info(f"  2b E{epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")
        if patience >= 12:
            logger.info("  Early stopping 2b")
            break

    # ==================================================================
    # Phase 2c: Unfreeze all (15 epochs)
    # ==================================================================
    model.load_state_dict(best_state)
    set_requires_grad(model.spectral_branch, True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nPhase 2c: Full model ({trainable:,} params) — 15 epochs")

    optimizer = torch.optim.AdamW([
        {"params": model.spectral_branch.parameters(), "lr": 0.00001},
        {"params": model.scalogram_branch.parameters(), "lr": 0.00001},
        {"params": model.fusion.parameters(), "lr": 0.00003},
        {"params": model.classifier.parameters(), "lr": 0.00003},
    ], weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    patience = 0

    for epoch in range(15):
        t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = 35 + epoch
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        status = "BEST" if patience == 0 else f"p={patience}"
        logger.info(f"  2c E{epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")
        if patience >= 8:
            logger.info("  Early stopping 2c")
            break

    total_time = time.time() - total_start

    # ---- Final eval ----
    model.load_state_dict(best_state)
    os.makedirs("outputs_30class_finetune_v3/checkpoints", exist_ok=True)
    torch.save({"model_state": best_state, "best_val_acc": best_val_acc},
               "outputs_30class_finetune_v3/checkpoints/best_finetuned.pth")

    logger.info("\n" + "=" * 70)
    logger.info("Final Test Evaluation")
    logger.info("=" * 70)
    results = evaluate_model(model, test_loader, device, species_names=SPECIES_NAMES_30)

    # ---- Comparison ----
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE COMPARISON")
    logger.info("=" * 70)
    logger.info(f"  {'Model':<30} {'Val':>8} {'Test':>8}")
    logger.info(f"  {'-'*48}")
    logger.info(f"  {'Spectral-only':<30} {'60.30%':>8} {'41.07%':>8}")
    logger.info(f"  {'Scalogram-only':<30} {'92.33%':>8} {'76.87%':>8}")
    logger.info(f"  {'PINNACLE v1 FT':<30} {'82.33%':>8} {'73.63%':>8}")
    logger.info(f"  {'PINNACLE v3 mixed-domain':<30} {best_val_acc:>7.2f}% {results['accuracy']:>7.2f}%")
    logger.info(f"  Training time: {total_time/60:.1f} min")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
