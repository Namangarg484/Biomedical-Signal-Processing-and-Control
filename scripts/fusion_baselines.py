#!/usr/bin/env python3
"""
PINNACLE — McNemar test for 30-class results + FiLM / Gated Fusion baselines.

1. McNemar's test: PINNACLE vs scalogram-only (30-species)
2. FiLM fusion baseline (Feature-wise Linear Modulation)
3. Simple gated fusion baseline (additive gating without cross-attention)

Usage:
    python scripts/fusion_baselines.py
"""

import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

from pinnacle.utils import set_seed, get_device, logger, count_parameters
from pinnacle.model import SpectralBranch, ScalogramBranch
from pinnacle.dataset import PINNACLEDataset, RamanAugmentation
from pinnacle.evaluate import evaluate_model

SPECIES_NAMES_30 = [f"Species_{i}" for i in range(30)]


# ======================================================================
# Alternative Fusion Modules
# ======================================================================

class FiLMFusion(nn.Module):
    """Feature-wise Linear Modulation (FiLM).
    
    The spectral branch modulates the scalogram embedding via learned
    scale (gamma) and shift (beta) parameters:
        z_fused = gamma(z_s) * z_w + beta(z_s)
    
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning
    Layer", AAAI 2018.
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.gamma_net = nn.Linear(embed_dim, embed_dim)
        self.beta_net = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, z_s, z_w):
        gamma = self.gamma_net(z_s)  # (B, D)
        beta = self.beta_net(z_s)    # (B, D)
        z_modulated = gamma * z_w + beta  # (B, D)
        z_fused = torch.cat([z_modulated, z_w], dim=1)  # (B, 2D)
        return z_fused, gamma, beta


class GatedFusion(nn.Module):
    """Simple additive gated fusion (no cross-attention).
    
    Each branch gets a learned gate, then outputs are concatenated:
        z_fused = [sigma(g_s) * z_s ; sigma(g_w) * z_w]
    """
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.gate_s = nn.Linear(embed_dim * 2, embed_dim)
        self.gate_w = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(self, z_s, z_w):
        z_cat = torch.cat([z_s, z_w], dim=1)  # (B, 2D)
        g_s = torch.sigmoid(self.gate_s(z_cat))  # (B, D)
        g_w = torch.sigmoid(self.gate_w(z_cat))  # (B, D)
        z_fused = torch.cat([g_s * z_s, g_w * z_w], dim=1)  # (B, 2D)
        return z_fused, g_s, g_w


class ConcatFusion(nn.Module):
    """Naive concatenation baseline (no gating, no attention)."""
    def __init__(self, embed_dim: int = 128):
        super().__init__()
    
    def forward(self, z_s, z_w):
        z_fused = torch.cat([z_s, z_w], dim=1)  # (B, 2D)
        return z_fused, None, None


# ======================================================================
# Generic dual-branch model with pluggable fusion
# ======================================================================

class DualBranchModel(nn.Module):
    def __init__(self, num_classes, embed_dim, dropout, fusion_module):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels=1, embed_dim=embed_dim)
        self.scalogram_branch = ScalogramBranch(in_channels=3, embed_dim=embed_dim)
        self.fusion = fusion_module
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
    
    def forward(self, raman, scalogram=None):
        z_s = self.spectral_branch(raman)
        z_w = self.scalogram_branch(scalogram)
        # Defensive pooling: ensure (B, D) even if branch returns unpooled
        if z_s.dim() > 2:
            z_s = z_s.view(z_s.size(0), z_s.size(1), -1).mean(dim=-1)
        if z_w.dim() > 2:
            z_w = z_w.view(z_w.size(0), z_w.size(1), -1).mean(dim=-1)
        z_fused, a, b = self.fusion(z_s, z_w)
        logits = self.classifier(z_fused)
        return logits, a, b


# ======================================================================
# Training helpers
# ======================================================================

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


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for raman, scalogram, labels in loader:
        raman, scalogram, labels = raman.to(device), scalogram.to(device), labels.to(device)
        logits, _, _ = model(raman, scalogram)
        _, pred = logits.max(1)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def train_full_pipeline(model, ref_train_loader, ref_val_loader,
                        ft_train_loader, ft_val_loader,
                        device, name, max_p1_epochs=50, max_p2_epochs=30):
    """Phase 1 + Phase 2 training pipeline."""
    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Pre-train on reference
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_p1_epochs)
    
    best_val, best_state, patience = 0.0, None, 0
    start = time.time()
    
    logger.info(f"\n  [{name}] Phase 1: Pre-training ({max_p1_epochs} epochs max)...")
    for epoch in range(max_p1_epochs):
        t_loss, t_acc = train_epoch(model, ref_train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, ref_val_loader, criterion, device)
        scheduler.step()
        
        if v_acc > best_val:
            best_val = v_acc
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        
        if epoch % 10 == 0 or patience == 0:
            status = "BEST" if patience == 0 else f"p={patience}"
            logger.info(f"    P1 E{epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")
        
        if patience >= 15:
            logger.info(f"    Early stopping at epoch {epoch}")
            break
    
    p1_time = time.time() - start
    model.load_state_dict(best_state)
    logger.info(f"  [{name}] Phase 1: {best_val:.2f}% val ({p1_time/60:.1f} min)")
    
    # Phase 2: Fine-tune
    # Phase 2a: freeze backbone
    for n, p in model.named_parameters():
        if "classifier" not in n and "fusion" not in n:
            p.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0005, weight_decay=0.001)
    for epoch in range(5):
        train_epoch(model, ft_train_loader, optimizer, criterion, device)
    
    # Phase 2b: unfreeze all
    for p in model.parameters():
        p.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_p2_epochs)
    
    best_ft_val, best_ft_state, patience = 0.0, None, 0
    for epoch in range(max_p2_epochs):
        t_loss, t_acc = train_epoch(model, ft_train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, ft_val_loader, criterion, device)
        scheduler.step()
        
        if v_acc > best_ft_val:
            best_ft_val = v_acc
            patience = 0
            best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        
        if epoch % 10 == 0:
            logger.info(f"    FT E{epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}%")
        
        if patience >= 10:
            break
    
    model.load_state_dict(best_ft_state)
    logger.info(f"  [{name}] Phase 2: {best_ft_val:.2f}% FT val")
    
    return model, best_val, best_ft_val


def mcnemar_test(preds_a, preds_b, labels):
    """McNemar's test: are models A and B significantly different?"""
    from scipy.stats import chi2
    
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    
    # b: A correct, B wrong
    b = np.sum(correct_a & ~correct_b)
    # c: A wrong, B correct
    c = np.sum(~correct_a & correct_b)
    
    # McNemar statistic with continuity correction
    if b + c == 0:
        return 1.0, b, c
    
    chi2_stat = (max(0, abs(b - c) - 1))**2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    
    return p_value, b, c


def main():
    set_seed(42)
    device = get_device()
    data_dir = "New data"

    logger.info("=" * 70)
    logger.info("PINNACLE — McNemar Test + Fusion Baseline Comparison")
    logger.info("=" * 70)

    # ---- Load data ----
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

    X_ft = np.load(os.path.join(data_dir, "X_finetune.npy"))
    y_ft = np.load(os.path.join(data_dir, "y_finetune.npy")).astype(np.int64)
    X_ft_wav = np.load(os.path.join(data_dir, "X_finetune_wavelet.npy"))
    ft_idx = np.arange(len(X_ft))
    ft_train, ft_val = train_test_split(ft_idx, test_size=0.2, random_state=123, stratify=y_ft)

    ft_train_ds = PINNACLEDataset(X_ft[ft_train], y_ft[ft_train], X_ft_wav[ft_train],
                                   transform_raman=aug, split="train")
    ft_val_ds = PINNACLEDataset(X_ft[ft_val], y_ft[ft_val], X_ft_wav[ft_val], split="val")
    ft_train_loader = DataLoader(ft_train_ds, batch_size=16, shuffle=True, drop_last=True)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=16, shuffle=False)

    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)
    X_test_wav = np.load(os.path.join(data_dir, "X_test_wavelet.npy"), mmap_mode="r")
    test_ds = PINNACLEDataset(X_test, y_test, X_test_wav, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    logger.info(f"  Ref: {len(idx_train)} train / {len(idx_val)} val")
    logger.info(f"  FT: {len(ft_train)} train / {len(ft_val)} val")
    logger.info(f"  Test: {len(X_test)}")

    # ==================================================================
    # Part 1: McNemar test — PINNACLE vs Scalogram-only
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 1: McNemar Test — PINNACLE vs Scalogram-only (30-species)")
    logger.info("=" * 70)

    from pinnacle.model import PINNACLE

    # Load PINNACLE (fine-tuned)
    model_pin = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, use_fusion=True).to(device)
    pin_ckpt = torch.load("outputs_30class_finetune_v2b/checkpoints/best_finetuned.pth",
                           map_location=device, weights_only=False)
    model_pin.load_state_dict(pin_ckpt["model_state"])
    preds_pin, labels = get_predictions(model_pin, test_loader, device)
    acc_pin = 100.0 * np.mean(preds_pin == labels)
    logger.info(f"  PINNACLE test acc: {acc_pin:.2f}%")

    # Load Scalogram-only — need to get fine-tuned predictions
    # Re-run scalogram-only with same seed to get predictions
    model_scal = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, mode="scalogram_only").to(device)
    scal_ckpt = torch.load("outputs_30class_scalogram_only/checkpoints/best_model.pth",
                            map_location=device, weights_only=False)
    model_scal.load_state_dict(scal_ckpt["model_state"])

    # Fine-tune scalogram-only with same protocol
    criterion = nn.CrossEntropyLoss()
    for n, p in model_scal.named_parameters():
        if "classifier" not in n:
            p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_scal.parameters()),
                             lr=0.0005, weight_decay=0.001)
    for ep in range(5):
        train_epoch(model_scal, ft_train_loader, opt, criterion, device)
    for p in model_scal.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model_scal.parameters(), lr=0.00005, weight_decay=0.001)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    best_sv, best_ss = 0.0, None
    for ep in range(25):
        train_epoch(model_scal, ft_train_loader, opt, criterion, device)
        _, v_acc = eval_epoch(model_scal, ft_val_loader, criterion, device)
        sch.step()
        if v_acc > best_sv:
            best_sv = v_acc
            best_ss = {k: v.clone() for k, v in model_scal.state_dict().items()}
    model_scal.load_state_dict(best_ss)

    preds_scal, _ = get_predictions(model_scal, test_loader, device)
    acc_scal = 100.0 * np.mean(preds_scal == labels)
    logger.info(f"  Scalogram-only test acc: {acc_scal:.2f}%")

    # McNemar test
    p_val, b, c = mcnemar_test(preds_pin, preds_scal, labels)
    logger.info(f"  McNemar: b={b} (PINNACLE✓,Scal✗), c={c} (PINNACLE✗,Scal✓)")
    logger.info(f"  McNemar p-value: {p_val:.4f}")
    if p_val < 0.05:
        logger.info(f"  ✅ SIGNIFICANT at α=0.05!")
    else:
        logger.info(f"  ⚠️ Not significant at α=0.05 (report as comparable)")

    # ==================================================================
    # Part 2: Fusion Baselines — FiLM, Gated, Concat
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: Fusion Baseline Comparison (30-species)")
    logger.info("=" * 70)

    results = {}
    fusion_configs = {
        "Concat (no fusion)": ConcatFusion(128),
        "Gated Fusion": GatedFusion(128),
        "FiLM": FiLMFusion(128),
    }

    for name, fusion_module in fusion_configs.items():
        logger.info(f"\n  Training {name}...")
        model = DualBranchModel(
            num_classes=30, embed_dim=128, dropout=0.3,
            fusion_module=fusion_module
        ).to(device)
        
        params = count_parameters(model)
        logger.info(f"  {name}: {params:,} params")

        model, p1_val, ft_val = train_full_pipeline(
            model, ref_train_loader, ref_val_loader,
            ft_train_loader, ft_val_loader,
            device, name, max_p1_epochs=50, max_p2_epochs=30
        )

        preds, _ = get_predictions(model, test_loader, device)
        test_acc = 100.0 * np.mean(preds == labels)
        
        # McNemar vs PINNACLE
        p_val_vs_pin, _, _ = mcnemar_test(preds_pin, preds, labels)
        
        results[name] = {
            "p1_val": p1_val, "ft_val": ft_val,
            "test": test_acc, "params": params,
            "p_mcnemar": p_val_vs_pin
        }
        logger.info(f"  [{name}] Test: {test_acc:.2f}% | McNemar vs PINNACLE: p={p_val_vs_pin:.4f}")

    # ==================================================================
    # Final Table
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE FUSION COMPARISON (30-species, cross-instrument)")
    logger.info("=" * 70)
    logger.info(f"  {'Method':<30} {'P1 Val':>8} {'Test':>8} {'Params':>8} {'p vs PIN':>10}")
    logger.info(f"  {'-'*66}")
    logger.info(f"  {'Spectral-only':<30} {'60.30%':>8} {'41.07%':>8} {'205K':>8} {'—':>10}")
    logger.info(f"  {'Scalogram-only':<30} {'92.33%':>8} {acc_scal:>7.2f}% {'205K':>8} {p_val:>9.4f}")

    for name, r in results.items():
        sig = "*" if r["p_mcnemar"] < 0.05 else ""
        logger.info(f"  {name:<30} {r['p1_val']:>7.2f}% {r['test']:>7.2f}% {r['params']:>7,} "
                     f" {r['p_mcnemar']:>9.4f}{sig}")

    logger.info(f"  {'PINNACLE (SeparationCross)':<30} {'92.20%':>8} {acc_pin:>7.2f}% {'304K':>8} {'ref':>10}")
    logger.info("=" * 70)
    logger.info("  * = significant at α=0.05 (McNemar)")


if __name__ == "__main__":
    main()
