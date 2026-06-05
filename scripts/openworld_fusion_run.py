#!/usr/bin/env python3
"""
Experiment C — Open-world 30-species fusion-method comparison (fully backed).

Replaces the partially-hardcoded tab:openworld. Every row is computed on the
SAME 3,000-sample held-out test set with the SAME two-phase (pre-train ->
domain-adaptation fine-tune) protocol, and every McNemar p-value is paired
against PINNACLE.

Rows:
  - Spectral-only   : trained from scratch (PINNACLE mode=spectral_only)
                      [was a hardcoded literal 41.07% in fusion_baselines.py]
  - Scalogram-only  : load pretrained ckpt, fine-tune, evaluate
  - Concat / Gated / FiLM : trained from scratch (two-phase)
  - PINNACLE (SeparationCross) : load fine-tuned ckpt, evaluate  (reference)

Outputs -> outputs/openworld_fusion/results.json
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from pinnacle.utils import set_seed, get_device, logger, count_parameters
import pinnacle.model as _pinnacle_model


# ----------------------------------------------------------------------
# Legacy SpectralBranch — EXACT architecture used to train the 30-class
# checkpoints (verified: reproduces the cached 77.07% test accuracy and
# the 303,999-parameter count to the digit). pinnacle/model.py was later
# refactored to a 4-conv residual stem; the 30-class checkpoints predate
# that change, so we restore the original branch here to load them and to
# train every fusion baseline under an identical, consistent backbone.
#   features = [Conv1d(1,64,7,s1,p3) BN ReLU
#               Conv1d(64,128,5,s1,p2) BN ReLU
#               Conv1d(128,128,3,s1,p1) BN ReLU]   (stride 1 throughout)
# ----------------------------------------------------------------------
class LegacySpectralBranch(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim), nn.ReLU(inplace=True),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return F.adaptive_avg_pool1d(self.features(x), 1).squeeze(-1)


# Restore the legacy backbone everywhere BEFORE any model is constructed.
_pinnacle_model.SpectralBranch = LegacySpectralBranch
from pinnacle.model import PINNACLE
from pinnacle.dataset import PINNACLEDataset, RamanAugmentation

import scripts.fusion_baselines as _fusion_baselines
_fusion_baselines.SpectralBranch = LegacySpectralBranch

from scripts.fusion_baselines import (
    DualBranchModel, ConcatFusion, GatedFusion, FiLMFusion,
    train_full_pipeline, train_epoch, eval_epoch, get_predictions, mcnemar_test,
)

DATA_DIR = "New data"
OUT_DIR = "outputs/openworld_fusion"
# Canonical 30-class PINNACLE (reviewer Morlet-default run): reproduces the
# paper's 77.07% test / 92.42% P1-val exactly with the legacy backbone.
PIN_CKPT = "outputs_reviewer/PINNACLE-Morlet_default/p2_done.pth"
PIN_P1_CKPT = "outputs_reviewer/PINNACLE-Morlet_default/p1_done.pth"
SCAL_CKPT = "outputs_30class_scalogram_only/checkpoints/best_model.pth"


def build_loaders():
    X_ref = np.load(os.path.join(DATA_DIR, "X_reference.npy"))
    y_ref = np.load(os.path.join(DATA_DIR, "y_reference.npy")).astype(np.int64)
    X_ref_wav = np.load(os.path.join(DATA_DIR, "X_reference_wavelet.npy"), mmap_mode="r")
    idx = np.arange(len(X_ref))
    idx_tr, idx_va = train_test_split(idx, test_size=0.1, random_state=42, stratify=y_ref)

    aug = RamanAugmentation(noise_std=0.01, shift_range=5, scale_range=0.05, probability=0.5)
    ref_tr = PINNACLEDataset(X_ref[idx_tr], y_ref[idx_tr], np.array(X_ref_wav[idx_tr]),
                             transform_raman=aug, split="train")
    ref_va = PINNACLEDataset(X_ref[idx_va], y_ref[idx_va], np.array(X_ref_wav[idx_va]),
                             split="val")
    ref_tr_l = DataLoader(ref_tr, batch_size=32, shuffle=True, drop_last=True)
    ref_va_l = DataLoader(ref_va, batch_size=32, shuffle=False)

    X_ft = np.load(os.path.join(DATA_DIR, "X_finetune.npy"))
    y_ft = np.load(os.path.join(DATA_DIR, "y_finetune.npy")).astype(np.int64)
    X_ft_wav = np.load(os.path.join(DATA_DIR, "X_finetune_wavelet.npy"))
    fi = np.arange(len(X_ft))
    ft_tr, ft_va = train_test_split(fi, test_size=0.2, random_state=123, stratify=y_ft)
    ft_tr_ds = PINNACLEDataset(X_ft[ft_tr], y_ft[ft_tr], X_ft_wav[ft_tr],
                               transform_raman=aug, split="train")
    ft_va_ds = PINNACLEDataset(X_ft[ft_va], y_ft[ft_va], X_ft_wav[ft_va], split="val")
    ft_tr_l = DataLoader(ft_tr_ds, batch_size=16, shuffle=True, drop_last=True)
    ft_va_l = DataLoader(ft_va_ds, batch_size=16, shuffle=False)

    X_te = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_te = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(np.int64)
    X_te_wav = np.load(os.path.join(DATA_DIR, "X_test_wavelet.npy"), mmap_mode="r")
    te_ds = PINNACLEDataset(X_te, y_te, np.array(X_te_wav), split="test")
    te_l = DataLoader(te_ds, batch_size=32, shuffle=False)
    return ref_tr_l, ref_va_l, ft_tr_l, ft_va_l, te_l


def finetune_existing(model, ft_tr_l, ft_va_l, device, epochs=25):
    """Two-phase fine-tune protocol for a pre-trained single-branch model."""
    criterion = nn.CrossEntropyLoss()
    for n, p in model.named_parameters():
        if "classifier" not in n:
            p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=5e-4, weight_decay=1e-3)
    for _ in range(5):
        train_epoch(model, ft_tr_l, opt, criterion, device)
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_v, best_s = 0.0, None
    for _ in range(epochs):
        train_epoch(model, ft_tr_l, opt, criterion, device)
        _, v = eval_epoch(model, ft_va_l, criterion, device)
        sch.step()
        if v > best_v:
            best_v, best_s = v, {k: t.clone() for k, t in model.state_dict().items()}
    if best_s is not None:
        model.load_state_dict(best_s)
    return model, best_v


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(42)
    device = get_device()
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("EXPERIMENT C — 30-species fusion comparison (fully backed)")
    logger.info("=" * 70)

    ref_tr_l, ref_va_l, ft_tr_l, ft_va_l, te_l = build_loaders()
    results = {}

    # ---- PINNACLE (reference): load fine-tuned ckpt --------------------
    pin = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, use_fusion=True).to(device)
    pin.load_state_dict(torch.load(PIN_CKPT, map_location=device,
                                   weights_only=False)["model_state"])
    preds_pin, labels = get_predictions(pin, te_l, device)
    acc_pin = 100.0 * np.mean(preds_pin == labels)
    _p1ck = torch.load(PIN_P1_CKPT, map_location="cpu", weights_only=False)
    pin_p1 = _p1ck.get("best_val", _p1ck.get("best_val_acc"))
    results["PINNACLE (SeparationCross)"] = {
        "p1_val": pin_p1, "test": acc_pin, "params": count_parameters(pin),
        "p_mcnemar": None,
    }
    logger.info(f"  PINNACLE: P1={pin_p1:.2f}% Test={acc_pin:.2f}% "
                f"params={count_parameters(pin):,}")

    # ---- Scalogram-only: load + fine-tune ------------------------------
    scal = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, mode="scalogram_only").to(device)
    scal_ck = torch.load(SCAL_CKPT, map_location=device, weights_only=False)
    scal.load_state_dict(scal_ck["model_state"], strict=False)
    scal, _ = finetune_existing(scal, ft_tr_l, ft_va_l, device)
    preds_scal, _ = get_predictions(scal, te_l, device)
    acc_scal = 100.0 * np.mean(preds_scal == labels)
    p_scal, _, _ = mcnemar_test(preds_pin, preds_scal, labels)
    results["Scalogram-only (2D CNN)"] = {
        "p1_val": scal_ck.get("best_val_acc"), "test": acc_scal,
        "params": count_parameters(scal), "p_mcnemar": p_scal,
    }
    logger.info(f"  Scalogram-only: Test={acc_scal:.2f}% p={p_scal:.4f}")

    # ---- Spectral-only: train from scratch (was hardcoded) -------------
    spec = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, mode="spectral_only").to(device)
    spec, spec_p1, _ = train_full_pipeline(
        spec, ref_tr_l, ref_va_l, ft_tr_l, ft_va_l, device,
        "Spectral-only", max_p1_epochs=50, max_p2_epochs=30)
    preds_spec, _ = get_predictions(spec, te_l, device)
    acc_spec = 100.0 * np.mean(preds_spec == labels)
    p_spec, _, _ = mcnemar_test(preds_pin, preds_spec, labels)
    results["Spectral-only (1D CNN)"] = {
        "p1_val": spec_p1, "test": acc_spec,
        "params": count_parameters(spec), "p_mcnemar": p_spec,
    }
    logger.info(f"  Spectral-only: P1={spec_p1:.2f}% Test={acc_spec:.2f}% p={p_spec:.4f}")

    # ---- Fusion variants: Concat / Gated / FiLM ------------------------
    for name, fusion in [("Concat (no fusion)", ConcatFusion(128)),
                         ("Gated Fusion", GatedFusion(128)),
                         ("FiLM", FiLMFusion(128))]:
        model = DualBranchModel(num_classes=30, embed_dim=128, dropout=0.3,
                                fusion_module=fusion).to(device)
        model, p1, _ = train_full_pipeline(
            model, ref_tr_l, ref_va_l, ft_tr_l, ft_va_l, device,
            name, max_p1_epochs=50, max_p2_epochs=30)
        preds, _ = get_predictions(model, te_l, device)
        acc = 100.0 * np.mean(preds == labels)
        p, _, _ = mcnemar_test(preds_pin, preds, labels)
        results[name] = {"p1_val": p1, "test": acc,
                         "params": count_parameters(model), "p_mcnemar": p}
        logger.info(f"  {name}: P1={p1:.2f}% Test={acc:.2f}% p={p:.4f}")

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("=" * 70)
    logger.info(f"  Saved -> {OUT_DIR}/results.json  ({(time.time()-t0)/60:.1f} min)")
    for k, v in results.items():
        logger.info(f"  {k:<28} P1={v['p1_val']} Test={v['test']:.2f}% "
                    f"params={v['params']:,} p={v['p_mcnemar']}")


if __name__ == "__main__":
    main()
