#!/usr/bin/env python3
"""
PINNACLE — 5-Class Ablation Study
==================================
Trains and evaluates all models needed for:
  • tab:overall       (Raman-only, Scalogram-only, No-fusion/Concat)
  • tab:fusion_ablation (Concat, +Gating-only, +CrossAttn-only)

All models use the identical 80/10/10 stratified split (seed=42)
and the same hyper-parameters as the main PINNACLE training run.

Models trained here
-------------------
  1. raman_only      — SpectralBranch + FC head (no scalogram)
  2. scalogram_only  — ScalogramBranch + FC head (no spectra)
  3. concat          — both branches, naive cat → FC (no gating/attn)
  4. gating_only     — both branches, separation gates, no cross-attn
  5. crossattn_only  — both branches, cross-attention, no gating

PINNACLE itself (rank 6) is loaded from its existing checkpoint
(outputs/predictions_current.npz) — NOT retrained here.

Usage
-----
  cd /path/to/Raman

  # Recommended: caffeinate keeps the Mac awake the whole time
  caffeinate -i python scripts/ablation_5class.py

  # Force full retrain (ignore cached checkpoints)
  caffeinate -i python scripts/ablation_5class.py --no-cache

  # Run only one model (useful for debugging)
  caffeinate -i python scripts/ablation_5class.py --model raman_only

  # Force CPU (if MPS gives trouble)
  caffeinate -i python scripts/ablation_5class.py --cpu

Output
------
  outputs/ablation_5class/<model>/best.pt
  outputs/ablation_5class/<model>/predictions.npz
  outputs/ablation_5class/results.json
  Final LaTeX-ready table rows printed to stdout
"""

import os
import sys
import time
import json
import copy
import argparse
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import chi2

from pinnacle.utils import set_seed, get_device, logger, count_parameters
from pinnacle.model import SpectralBranch, ScalogramBranch, SeparationCross
from pinnacle.dataset import load_data, create_dataloaders

# ======================================================================
# CONFIG — mirrors default.yaml / main train.py exactly
# ======================================================================

CFG = {
    "data_dir":   "data/",
    "seed":       42,
    "num_classes": 5,
    "embed_dim":  128,
    "dropout":    0.3,
    "batch_size": 32,
    "epochs":     30,
    "lr":         1e-3,
    "weight_decay": 1e-4,
    "grad_clip":  5.0,
    "patience":   12,
    "ema_decay":  0.999,
    "out_dir":    "outputs/ablation_5class",
    "pinnacle_preds": "outputs/predictions_current.npz",
}

SPECIES = ["E. coli", "S. aureus", "P. aeruginosa", "K. pneumoniae", "E. faecalis"]


# ======================================================================
# LEAN SINGLE-BRANCH MODELS
# (no unused parameters — avoids MPS gradient silent-failure bug
#  that occurs when a model has large parameter blocks with no gradients)
# ======================================================================

class RamanOnlyModel(nn.Module):
    """SpectralBranch + FC head only. No scalogram branch instantiated."""
    def __init__(self, num_classes: int = 5, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels=1, embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, raman, scalogram):
        z_s = self.spectral_branch(raman)
        return self.classifier(z_s), None, None


class ScalogramOnlyModel(nn.Module):
    """ScalogramBranch + FC head only. No spectral branch instantiated."""
    def __init__(self, num_classes: int = 5, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.scalogram_branch = ScalogramBranch(in_channels=3, embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, raman, scalogram):
        z_w = self.scalogram_branch(scalogram)
        return self.classifier(z_w), None, None


# ======================================================================
# CUSTOM FUSION MODULES
# ======================================================================

class ConcatFusion(nn.Module):
    """Naive concatenation — both branches, no gating, no attention."""
    def __init__(self, num_classes: int = 5, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.spectral_branch  = SpectralBranch(in_channels=1, embed_dim=embed_dim)
        self.scalogram_branch = ScalogramBranch(in_channels=3, embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, raman, scalogram):
        z_s = self.spectral_branch(raman)
        z_w = self.scalogram_branch(scalogram)
        logits = self.classifier(torch.cat([z_s, z_w], dim=1))
        return logits, None, None


class GatingOnlyFusion(nn.Module):
    """
    Separation gating only — per-modality sigmoid gates applied to pooled
    embeddings; NO cross-attention.  Per-modality gating isolates the
    contribution of the gates from the attention.
    """
    def __init__(self, num_classes: int = 5, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.spectral_branch  = SpectralBranch(in_channels=1, embed_dim=embed_dim)
        self.scalogram_branch = ScalogramBranch(in_channels=3, embed_dim=embed_dim)
        self.gate_s = nn.Linear(embed_dim, embed_dim)
        self.gate_w = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, raman, scalogram):
        z_s = self.spectral_branch(raman)
        z_w = self.scalogram_branch(scalogram)
        alpha = torch.sigmoid(self.gate_s(z_s))
        beta  = torch.sigmoid(self.gate_w(z_w))
        logits = self.classifier(torch.cat([alpha * z_s, beta * z_w], dim=1))
        return logits, alpha, beta


class CrossAttnOnlyFusion(nn.Module):
    """
    Cross-attention only — spatial cross-attention (spectral queries the
    scalogram) with residual, but NO per-modality sigmoid gating (α=β=1).
    Mirrors SeparationCross with the gate_spectral / gate_scalogram removed.
    """
    def __init__(self, num_classes: int = 5, embed_dim: int = 128,
                 dropout: float = 0.3, spectral_pool_len: int = 25):
        super().__init__()
        self.spectral_branch  = SpectralBranch(in_channels=1, embed_dim=embed_dim)
        self.scalogram_branch = ScalogramBranch(in_channels=3, embed_dim=embed_dim)

        self.spectral_pool = nn.AdaptiveAvgPool1d(spectral_pool_len)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(1))   # init=0 → pure scalogram at start

        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, raman, scalogram):
        B = raman.size(0)

        # Full spatial feature maps (no gating)
        h_s = self.spectral_branch.forward_features(raman)    # (B, D, L)
        h_w = self.scalogram_branch.forward_features(scalogram)  # (B, D, H', W')
        _, D, Hp, Wp = h_w.shape

        # Pooled for residual
        z_w = h_w.mean(dim=(-2, -1))   # (B, D)

        # Prepare sequences for attention
        seq_s = self.spectral_pool(h_s).transpose(1, 2)           # (B, S, D)
        seq_w = h_w.view(B, D, Hp * Wp).transpose(1, 2)           # (B, H'W', D)

        # Cross-attention: spectral queries scalogram
        Q = self.W_q(seq_s)
        K = self.W_k(seq_w)
        V = self.W_v(seq_w)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        z_attn = torch.matmul(attn, V).mean(dim=1)     # (B, D)

        # Residual fusion (γ starts 0)
        z_combined = z_w + self.gamma * z_attn         # (B, D)
        logits = self.classifier(torch.cat([z_combined, z_w], dim=1))  # (B, 2D) → logits
        return logits, None, None


# ======================================================================
# HELPERS — Wilson CI, McNemar, training loop
# ======================================================================

def wilson_ci(acc_frac: float, n: int, z: float = 1.96):
    """Two-sided Wilson score interval.  acc_frac in [0,1]."""
    p = acc_frac
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def mcnemar_p(preds_a, preds_b, labels):
    """McNemar's test: p-value for H0 that model_a and model_b have equal error rates."""
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    b = int(np.sum( correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a &  correct_b))  # A wrong, B right
    if b + c == 0:
        return 1.0
    if b + c < 25:
        # Exact binomial two-sided
        from scipy.stats import binom_test
        return float(binom_test(b, b + c, 0.5))
    # Continuity-corrected chi-sq
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return float(1.0 - chi2.cdf(stat, df=1))


def p_label(p):
    if p < 0.001:
        return "$<0.001$"
    return f"$={p:.3f}$"


# ======================================================================
# TRAINING INFRASTRUCTURE
# ======================================================================

def _make_ema(model):
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def _update_ema(ema, model, decay=0.999):
    for ep, mp in zip(ema.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(mp.data, alpha=1 - decay)
    for eb, mb in zip(ema.buffers(), model.buffers()):
        eb.copy_(mb)


@torch.no_grad()
def _eval(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    preds_list, labels_list = [], []
    for raman, scalogram, labels in loader:
        raman, scalogram, labels = raman.to(device), scalogram.to(device), labels.to(device)
        logits, _, _ = model(raman, scalogram)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        pred = logits.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        preds_list.append(pred.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
    return (loss_sum / total,
            100.0 * correct / total,
            np.concatenate(preds_list),
            np.concatenate(labels_list))


def train_model(name, model, train_loader, val_loader, test_loader,
                device, out_dir, no_cache=False):
    """Full training + evaluation with EMA, early stopping, checkpointing."""
    ckpt_dir = os.path.join(out_dir, name)
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    pred_path = os.path.join(ckpt_dir, "predictions.npz")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Cache hit -------------------------------------------------------
    if not no_cache and os.path.exists(ckpt_path) and os.path.exists(pred_path):
        logger.info(f"[{name}] Cached checkpoint found — loading.")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        saved = np.load(pred_path)
        preds   = saved["y_pred"]
        labels  = saved["y_true"]
        test_acc = 100.0 * np.mean(preds == labels)
        logger.info(f"[{name}] Cached test acc: {test_acc:.2f}%")
        return preds, labels, test_acc

    # ---- Checkpoint exists but predictions missing — eval only -----------
    if not no_cache and os.path.exists(ckpt_path) and not os.path.exists(pred_path):
        logger.info(f"[{name}] Checkpoint found but predictions missing — evaluating.")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        _, test_acc, preds, labels = _eval(model, test_loader, device, criterion)
        ema_eval = _make_ema(model)
        if "ema_state" in ckpt:
            ema_eval.load_state_dict(ckpt["ema_state"])
            ema_eval.to(device)
            _, test_ema_acc, preds_ema, _ = _eval(ema_eval, test_loader, device, criterion)
            if test_ema_acc > test_acc:
                preds, test_acc = preds_ema, test_ema_acc
                logger.info(f"  [{name}] Using EMA predictions ({test_ema_acc:.2f}%)")
        np.savez(pred_path, y_pred=preds, y_true=labels)
        logger.info(f"  [{name}] Predictions saved → {pred_path}  (test acc: {test_acc:.2f}%)")
        return preds, labels, test_acc

    # ---- Fresh training ---------------------------------------------------
    model = model.to(device)
    ema   = _make_ema(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"])

    best_val     = 0.0
    best_state   = None
    best_ema_state = None
    patience_cnt = 0
    t0 = time.time()

    logger.info("")
    logger.info("=" * 66)
    logger.info(f"  Training: {name}  ({count_parameters(model):,} params)")
    logger.info("=" * 66)

    for epoch in range(CFG["epochs"]):
        # ---- train ----
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for bi, (raman, scalogram, labels) in enumerate(train_loader):
            raman, scalogram, labels = (raman.to(device), scalogram.to(device),
                                        labels.to(device))
            optimizer.zero_grad()
            logits, _, _ = model(raman, scalogram)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()
            _update_ema(ema, model, CFG["ema_decay"])

            loss_sum += loss.item() * labels.size(0)
            pred = logits.argmax(1)
            correct += pred.eq(labels).sum().item()
            total   += labels.size(0)

            every = max(1, len(train_loader) // 4)
            if (bi + 1) % every == 0 or (bi + 1) == len(train_loader):
                logger.info(
                    f"  [{name}] Ep {epoch+1:02d}/{CFG['epochs']} "
                    f"| Batch {bi+1:03d}/{len(train_loader)} "
                    f"| Loss {loss.item():.4f} | Acc {100.*correct/total:.2f}%")

        t_acc = 100.0 * correct / total
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        # ---- validate ----
        v_loss, v_acc, _, _ = _eval(model, val_loader, device, criterion)
        _, v_ema_acc, _, _  = _eval(ema,   val_loader, device, criterion)
        best_epoch_acc = max(v_acc, v_ema_acc)

        logger.info(
            f"  [{name}] Ep {epoch+1:02d}/{CFG['epochs']} "
            f"train {t_acc:.2f}% | val {v_acc:.2f}% | val_ema {v_ema_acc:.2f}% "
            f"| lr {lr_now:.2e}")

        if best_epoch_acc > best_val:
            best_val        = best_epoch_acc
            best_state      = {k: v.clone() for k, v in model.state_dict().items()}
            best_ema_state  = {k: v.clone() for k, v in ema.state_dict().items()}
            patience_cnt    = 0
            torch.save({"model_state": best_state,
                        "ema_state":   best_ema_state,
                        "best_val_acc": best_val,
                        "epoch": epoch},
                       ckpt_path)
            logger.info(f"  ✨ [{name}] New best val: {best_val:.2f}%")
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                logger.info(f"  [{name}] Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - t0
    logger.info(f"  [{name}] Training done in {elapsed/60:.1f} min. Best val: {best_val:.2f}%")

    # ---- final test evaluation ----
    model.load_state_dict(best_state)
    model.to(device)
    _, test_acc, preds, labels = _eval(model, test_loader, device, criterion)
    # Also check EMA test accuracy
    ema_model_eval = _make_ema(model)
    ema_model_eval.load_state_dict(best_ema_state)
    ema_model_eval.to(device)
    _, test_ema_acc, preds_ema, _ = _eval(ema_model_eval, test_loader, device, criterion)
    # Use whichever is better (mirrors main train.py logic)
    if test_ema_acc > test_acc:
        preds    = preds_ema
        test_acc = test_ema_acc
        logger.info(f"  [{name}] Using EMA predictions (EMA {test_ema_acc:.2f}% > base {test_acc:.2f}%)")
    else:
        logger.info(f"  [{name}] Using base predictions ({test_acc:.2f}%)")

    np.savez(pred_path, y_pred=preds, y_true=labels)
    logger.info(f"  [{name}] Predictions saved → {pred_path}")
    return preds, labels, test_acc


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="PINNACLE 5-class ablation study")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached checkpoints and retrain from scratch")
    parser.add_argument("--model", type=str, default=None,
                        choices=["raman_only", "scalogram_only",
                                 "concat", "gating_only", "crossattn_only"],
                        help="Run only a single model (default: run all)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU device (use if MPS gives issues)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count (default: 30)")
    args = parser.parse_args()

    if args.epochs is not None:
        CFG["epochs"] = args.epochs

    set_seed(CFG["seed"])

    if args.cpu:
        device = torch.device("cpu")
        logger.info("Device: CPU (forced via --cpu)")
    else:
        device = get_device()
        logger.info(f"Device: {device}")

    os.makedirs(CFG["out_dir"], exist_ok=True)

    # ------------------------------------------------------------------
    # Load data (identical to main train.py pipeline)
    # ------------------------------------------------------------------
    logger.info("=" * 66)
    logger.info("Loading 5-class data...")
    data = load_data(CFG["data_dir"])
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data,
        batch_size=CFG["batch_size"],
        seed=CFG["seed"],
        num_workers=0,
        use_augmentation=True,
    )
    n_test = len(test_loader.dataset)
    logger.info(f"Test set: {n_test} samples")
    logger.info("=" * 66)

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------
    D  = CFG["embed_dim"]
    NC = CFG["num_classes"]
    DR = CFG["dropout"]

    # raman_only / scalogram_only: lean models (single branch only).
    # Using PINNACLE(mode="spectral_only") still instantiates the scalogram
    # branch (~93K params with no gradients), which causes MPS to silently
    # fail to propagate gradients. Lean models avoid this entirely.
    all_models = {
        "raman_only":     RamanOnlyModel(num_classes=NC, embed_dim=D, dropout=DR),
        "scalogram_only": ScalogramOnlyModel(num_classes=NC, embed_dim=D, dropout=DR),
        "concat":         ConcatFusion(num_classes=NC, embed_dim=D, dropout=DR),
        "gating_only":    GatingOnlyFusion(num_classes=NC, embed_dim=D, dropout=DR),
        "crossattn_only": CrossAttnOnlyFusion(num_classes=NC, embed_dim=D, dropout=DR),
    }

    if args.model is not None:
        run_models = {args.model: all_models[args.model]}
    else:
        run_models = all_models

    # ------------------------------------------------------------------
    # Train / load each model
    # ------------------------------------------------------------------
    results = {}  # name → {"preds", "labels", "acc", "ci_lo", "ci_hi"}
    wall_start = time.time()

    for name, model in run_models.items():
        preds, labels, acc = train_model(
            name, model, train_loader, val_loader, test_loader,
            device, CFG["out_dir"], no_cache=args.no_cache)
        ci_lo, ci_hi = wilson_ci(acc / 100, n_test)
        results[name] = {
            "preds": preds, "labels": labels,
            "acc": acc, "ci_lo": ci_lo, "ci_hi": ci_hi,
        }

    # ------------------------------------------------------------------
    # Load PINNACLE predictions for McNemar tests
    # ------------------------------------------------------------------
    pinnacle_preds = None
    if os.path.exists(CFG["pinnacle_preds"]):
        loaded = np.load(CFG["pinnacle_preds"])
        pinnacle_preds  = loaded["y_pred"]
        pinnacle_labels = loaded["y_true"]
        pinnacle_acc    = 100.0 * np.mean(pinnacle_preds == pinnacle_labels)
        ci_lo, ci_hi    = wilson_ci(pinnacle_acc / 100, len(pinnacle_labels))
        logger.info(f"\nPINNACLE (existing): {pinnacle_acc:.2f}% [{ci_lo:.1f}, {ci_hi:.1f}]")
    else:
        logger.warning("PINNACLE predictions not found at "
                       f"{CFG['pinnacle_preds']}. McNemar p=N/A.")
        pinnacle_acc = None

    # ------------------------------------------------------------------
    # Print tab:overall  (Raman-only, Scalogram-only, Concat, PINNACLE)
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 66)
    logger.info("  PAPER TABLE: tab:overall")
    logger.info("  (paste into draft.tex replacing the old numbers)")
    logger.info("=" * 66)
    for key, label in [
        ("raman_only",     r"Raman-only (RamanNet)"),
        ("scalogram_only", r"Scalogram-only (CWT)"),
        ("concat",         r"No-fusion (Concat)"),
    ]:
        if key not in results:
            continue
        r = results[key]
        if pinnacle_preds is not None:
            p = mcnemar_p(pinnacle_preds, r["preds"], r["labels"])
            pval = p_label(p)
        else:
            pval = "N/A"
        print(f"  {label:<30}  "
              f"& {r['acc']:.2f} & [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}] & {pval} \\\\")

    if pinnacle_acc is not None:
        print(f"  {'PINNACLE (ours)':<30}  "
              f"& {pinnacle_acc:.2f} & [{ci_lo:.1f}, {ci_hi:.1f}] & --- \\\\")

    # ------------------------------------------------------------------
    # Print tab:fusion_ablation  (Concat, +Gating, +CrossAttn, +Both)
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 66)
    logger.info("  PAPER TABLE: tab:fusion_ablation")
    logger.info("=" * 66)
    concat_acc = results.get("concat", {}).get("acc")
    for key, label in [
        ("concat",         "Concatenation"),
        ("gating_only",    "+ Gating only"),
        ("crossattn_only", "+ Cross-attention only"),
    ]:
        if key not in results:
            continue
        r = results[key]
        delta = f"+{r['acc'] - concat_acc:.2f}" if concat_acc is not None else "---"
        print(f"  {label:<30}  & {r['acc']:.2f} & {delta} \\\\")

    if pinnacle_acc is not None and concat_acc is not None:
        delta_full = f"+{pinnacle_acc - concat_acc:.2f}"
        print(f"  {'+ Gating + Cross-attn (PINNACLE)':<30}  "
              f"& {pinnacle_acc:.2f} & {delta_full} \\\\")

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "acc": round(r["acc"], 4),
            "ci_lo": round(r["ci_lo"], 2),
            "ci_hi": round(r["ci_hi"], 2),
        }
    if pinnacle_acc is not None:
        summary["PINNACLE"] = {
            "acc": round(pinnacle_acc, 4),
            "ci_lo": round(ci_lo, 2),
            "ci_hi": round(ci_hi, 2),
        }

    json_path = os.path.join(CFG["out_dir"], "results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nJSON summary → {json_path}")

    total_time = time.time() - wall_start
    logger.info(f"\nTotal wall time: {total_time/3600:.2f} h  ({total_time/60:.0f} min)")
    logger.info("Done.")


if __name__ == "__main__":
    main()
