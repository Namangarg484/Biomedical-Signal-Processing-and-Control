#!/usr/bin/env python3
"""
5-Class extra baselines for Table 4 (tab:sota) — reviewer request.

Adds two models to the 5-class head-to-head comparison:
  1. Transformer-1D  — 1D spectral tokens → multi-layer Transformer encoder
  2. FiLM (5-class)  — feature-wise linear modulation dual-stream fusion

Uses the identical 80/10/10 split, preprocessing, and hyperparameters as
sota_baselines.py so results are directly comparable.

Usage:
    cd /path/to/Raman
    python scripts/5class_extra_baselines.py
    python scripts/5class_extra_baselines.py --no-cache   # force re-train
"""

import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import chi2

from pinnacle.utils import set_seed, get_device, logger, count_parameters
from pinnacle.model import SpectralBranch, ScalogramBranch
from pinnacle.dataset import PINNACLEDataset, RamanAugmentation

# -------------------------------------------------------------------------
# Config — must match sota_baselines.py / PINNACLE paper Table 3
# -------------------------------------------------------------------------
CFG = {
    "data_dir":               "data/",
    "seed":                   42,
    "num_classes":            5,
    "embed_dim":              128,
    "dropout":                0.3,
    "batch_size":             32,
    "epochs":                 30,
    "lr":                     1e-3,
    "weight_decay":           1e-4,
    "grad_clip":              5.0,
    "early_stopping_patience": 12,
    "test_size":              0.10,
    "val_size":               0.10,
    "num_workers":            0,
    "out_dir":                "outputs/sota_baselines",
}

# =========================================================================
# Model 1 — Transformer-1D
# =========================================================================

class Transformer1D(nn.Module):
    """
    CNN-stem + Transformer encoder for Raman spectra.

    Architecture:
      Stem (local feature extraction with inductive bias):
        Conv1d(1→64, k=7, s=2) → BN → ReLU          [halves L to 500]
        Conv1d(64→128, k=5, s=2) → BN → ReLU         [halves L to 250]
        AdaptiveAvgPool1d(25)                          [→ 25 tokens × 128]

      Transformer:
        Learnable positional embedding
        4× TransformerEncoderLayer (d_model=128, nhead=8, ffn=512, dropout)
        Global avg pool over tokens → LayerNorm → Dropout → FC(num_classes)

    The CNN stem provides the local inductive bias needed to extract
    meaningful spectral tokens; the Transformer then models long-range
    band interactions. Global avg pool is used instead of CLS for
    more stable early-training gradient flow.
    ~537K parameters.
    """
    def __init__(self, embed_dim=128, num_heads=8, num_layers=4,
                 ffn_dim=512, num_classes=5, dropout=0.3):
        super().__init__()
        # CNN stem — gives tokens local spectral context
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, embed_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )
        n_tokens = 25
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)  # fixed 25 tokens regardless of input length

        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, raman, scalogram=None):
        if raman.dim() == 2:
            raman = raman.unsqueeze(1)          # (B, 1, L)
        # Per-spectrum z-score normalization keeps activations in a stable
        # range and avoids train/val mismatch from data amplitude scaling.
        raman = (raman - raman.mean(dim=-1, keepdim=True)) / (raman.std(dim=-1, keepdim=True) + 1e-6)
        x = self.stem(raman)                    # (B, 128, ~250)
        x = self.pool(x)                        # (B, 128, 25)
        x = x.transpose(1, 2)                   # (B, 25, 128)
        x = x + self.pos_embed                  # add positional embedding
        x = self.transformer(x)                 # (B, 25, 128)
        # Global avg pool over all tokens — more stable gradient than CLS
        out = self.norm(x.mean(dim=1))          # (B, 128)
        return self.head(self.dropout(out)), None, None


# =========================================================================
# Model 2 — FiLM 5-class (dual-stream)
# =========================================================================

class FiLMFusion5(nn.Module):
    """FiLM: spectral embedding modulates scalogram embedding."""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.gamma_net = nn.Linear(embed_dim, embed_dim)
        self.beta_net  = nn.Linear(embed_dim, embed_dim)

    def forward(self, z_s, z_w):
        gamma = self.gamma_net(z_s)
        beta  = self.beta_net(z_s)
        return torch.cat([gamma * z_w + beta, z_w], dim=1), gamma, beta


class FiLMModel5(nn.Module):
    """Dual-stream CNN + FiLM fusion for the 5-class benchmark."""
    def __init__(self, num_classes=5, embed_dim=128, dropout=0.3):
        super().__init__()
        self.spectral_branch  = SpectralBranch(in_channels=1, embed_dim=embed_dim)
        self.scalogram_branch = ScalogramBranch(in_channels=3, embed_dim=embed_dim)
        self.fusion = FiLMFusion5(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, raman, scalogram):
        z_s = self.spectral_branch(raman)
        z_w = self.scalogram_branch(scalogram)
        z_fused, g, b = self.fusion(z_s, z_w)
        return self.classifier(z_fused), g, b


# =========================================================================
# Data loading (mirrors sota_baselines.py exactly)
# =========================================================================

def load_5class_data(data_dir, seed, test_size, val_size):
    X_2018 = np.load(os.path.join(data_dir, "X_2018_proc.npy")).astype(np.float32)
    X_2019 = np.load(os.path.join(data_dir, "X_2019_proc.npy")).astype(np.float32)
    y_2018 = np.load(os.path.join(data_dir, "y_2018clinical.npy")).astype(np.int64)
    y_2019 = np.load(os.path.join(data_dir, "y_2019clinical.npy")).astype(np.int64)

    X_all = np.concatenate([X_2018, X_2019], axis=0)
    y_all = np.concatenate([y_2018, y_2019], axis=0)

    wav_2018 = os.path.join(data_dir, "X_2018_wavelet.npy")
    wav_2019 = os.path.join(data_dir, "X_2019_wavelet.npy")
    W_all = None
    if os.path.exists(wav_2018) and os.path.exists(wav_2019):
        W_all = np.concatenate([
            np.load(wav_2018).astype(np.float32),
            np.load(wav_2019).astype(np.float32)
        ], axis=0)

    idx = np.arange(len(X_all))
    idx_tr, idx_tmp = train_test_split(
        idx, test_size=test_size + val_size, random_state=seed, stratify=y_all)
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=test_size / (test_size + val_size),
        random_state=seed, stratify=y_all[idx_tmp])

    splits = {}
    for name, ix in [("train", idx_tr), ("val", idx_val), ("test", idx_te)]:
        splits[f"X_{name}"] = X_all[ix]
        splits[f"y_{name}"] = y_all[ix]
        splits[f"W_{name}"] = W_all[ix] if W_all is not None else None
    logger.info(f"Split: train={len(idx_tr)}, val={len(idx_val)}, test={len(idx_te)}")
    return splits, W_all is not None


def resize_scalograms(W, size=128):
    """Resize scalogram array (N, 3, H, W) to (N, 3, size, size) in numpy."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(W)              # (N, 3, H, W) float32
    out = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return out.numpy()


def make_loaders(splits, batch_size, num_workers, need_scalo=True, scalo_size=None):
    # NOTE:
    # X_2018_proc / X_2019_proc are already low-amplitude processed spectra
    # (std is typically ~1e-3). The default RamanAugmentation(noise_std=0.02)
    # can overwhelm this signal and collapse training for lightweight
    # transformer-style models. Keep augmentation off for these controlled
    # 5-class baseline runs so train/val/test distributions remain aligned.
    aug = None
    loaders = {}
    for split in ("train", "val", "test"):
        W = splits[f"W_{split}"] if need_scalo else None
        # Resize ALL splits consistently (not just train), so BN stats match
        if W is not None and scalo_size is not None:
            W = resize_scalograms(W, size=scalo_size)
        ds = PINNACLEDataset(
            splits[f"X_{split}"], splits[f"y_{split}"],
            X_scalogram=W,
            transform_raman=(aug if split == "train" else None),
            split=split,
        )
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=False)
    return loaders


# =========================================================================
# Training loop
# =========================================================================

def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    loss_sum = correct = total = 0
    for raman, scalogram, labels in loader:
        raman, scalogram, labels = raman.to(device), scalogram.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(raman, scalogram)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total   += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for raman, scalogram, labels in loader:
        raman, scalogram, labels = raman.to(device), scalogram.to(device), labels.to(device)
        logits, _, _ = model(raman, scalogram)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total   += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def get_preds(model, loader, device):
    model.eval()
    preds, labs = [], []
    for raman, scalogram, labels in loader:
        raman, scalogram = raman.to(device), scalogram.to(device)
        preds.append(model(raman, scalogram)[0].argmax(1).cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(labs)


def wilson_ci(acc_frac, n, z=1.96):
    p = acc_frac
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * ((p * (1 - p) + z**2 / (4 * n)) / n) ** 0.5 / d
    return (c - m) * 100, (c + m) * 100


def mcnemar_p(preds_a, preds_b, labels):
    ca, cb = preds_a == labels, preds_b == labels
    n01, n10 = np.sum(ca & ~cb), np.sum(~ca & cb)
    if n01 + n10 == 0:
        return 1.0
    return float(1 - chi2.cdf((abs(n01 - n10) - 1) ** 2 / (n01 + n10), df=1))


def mps_sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def infer_latency(model, device, seq_len=1000, n_warmup=10, n_runs=50):
    r = torch.randn(1, seq_len).to(device)
    s = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(r, s)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            mps_sync(device); t0 = time.time()
            model(r, s)
            mps_sync(device)
            times.append(time.time() - t0)
    return float(np.median(times)) * 1000


def run_one(name, model, loaders, device, cfg, no_cache):
    out_dir = os.path.join(cfg["out_dir"], name)
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "predictions.npz")
    ckpt_path = os.path.join(out_dir, "best.pt")

    if not no_cache and os.path.exists(pred_path):
        logger.info(f"  [{name}] loading cached results")
        d = np.load(pred_path)
        return dict(accuracy=float(d["accuracy"]), ci_lo=float(d["ci_lo"]),
                    ci_hi=float(d["ci_hi"]), predictions=d["predictions"],
                    labels=d["labels"], num_params=int(d["num_params"]),
                    infer_ms=float(d["infer_ms"]))

    model = model.to(device)
    n_params = count_parameters(model)
    logger.info(f"\n{'='*60}\n  Training {name}  ({n_params:,} params)\n{'='*60}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    crit  = nn.CrossEntropyLoss()

    best_val, patience_cnt = 0.0, 0
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        _, tr_acc = train_epoch(model, loaders["train"], opt, crit, device, cfg["grad_clip"])
        _, vl_acc = eval_epoch(model, loaders["val"],   crit, device)
        sched.step()
        logger.info(f"  Epoch {ep:3d} | Train {tr_acc:.2f}%  Val {vl_acc:.2f}%  ({time.time()-t0:.1f}s)")
        if vl_acc > best_val:
            best_val, patience_cnt = vl_acc, 0
            torch.save({"epoch": ep, "state_dict": model.state_dict()}, ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["early_stopping_patience"]:
                logger.info(f"  Early stopping at epoch {ep}.")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    preds, labs = get_preds(model, loaders["test"], device)
    acc = 100.0 * np.mean(preds == labs)
    lo, hi = wilson_ci(np.mean(preds == labs), len(labs))
    inf_ms = infer_latency(model, device)

    np.savez(pred_path, predictions=preds, labels=labs,
             accuracy=acc, ci_lo=lo, ci_hi=hi,
             num_params=n_params, infer_ms=inf_ms)
    logger.info(f"  {name}: acc={acc:.2f}%  CI=[{lo:.1f}, {hi:.1f}]  "
                f"params={n_params:,}  latency={inf_ms:.1f}ms")
    return dict(accuracy=acc, ci_lo=lo, ci_hi=hi, predictions=preds,
                labels=labs, num_params=n_params, infer_ms=inf_ms)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    set_seed(CFG["seed"])
    device = get_device()
    logger.info(f"Device: {device}")

    splits, has_scalo = load_5class_data(
        CFG["data_dir"], CFG["seed"], CFG["test_size"], CFG["val_size"])

    # Transformer-1D needs only Raman (unimodal), FiLM needs both
    loaders_raman = make_loaders(splits, CFG["batch_size"], CFG["num_workers"], need_scalo=False)
    loaders_both  = make_loaders(splits, CFG["batch_size"], CFG["num_workers"],
                                 need_scalo=has_scalo, scalo_size=128)  # 128×128 speeds up FiLM 4×

    # ---- Transformer-1D (CNN-stem + Transformer) ----
    # Use lower LR (3e-4): Transformers diverge at 1e-3
    t1d_cfg = {**CFG, "lr": 3e-4}
    t1d = Transformer1D(
        embed_dim=CFG["embed_dim"], num_heads=8, num_layers=4, ffn_dim=512,
        num_classes=CFG["num_classes"], dropout=CFG["dropout"]
    )
    res_t1d = run_one("transformer_1d", t1d, loaders_raman, device, t1d_cfg, args.no_cache)

    # ---- FiLM 5-class ----
    film = FiLMModel5(num_classes=CFG["num_classes"],
                      embed_dim=CFG["embed_dim"], dropout=CFG["dropout"])
    res_film = run_one("film_5class", film, loaders_both, device, CFG, args.no_cache)

    # Load PINNACLE predictions for McNemar (support both key formats)
    pin_preds, pin_labs = None, None
    pinnacle_candidates = [
        "outputs/checkpoints/test_predictions.npz",  # keys: predictions, labels
        "outputs/predictions_current.npz",           # keys: y_pred, y_true
    ]
    for pinnacle_pred_path in pinnacle_candidates:
        if not os.path.exists(pinnacle_pred_path):
            continue
        pin_data = np.load(pinnacle_pred_path)
        if "predictions" in pin_data.files and "labels" in pin_data.files:
            pin_preds, pin_labs = pin_data["predictions"], pin_data["labels"]
            logger.info(f"Using PINNACLE predictions from {pinnacle_pred_path}")
            break
        if "y_pred" in pin_data.files and "y_true" in pin_data.files:
            pin_preds, pin_labs = pin_data["y_pred"], pin_data["y_true"]
            logger.info(f"Using PINNACLE predictions from {pinnacle_pred_path}")
            break

    if pin_preds is None:
        logger.warning("PINNACLE test predictions not found; skipping McNemar p-values.")

    logger.info("\n" + "="*70)
    logger.info("RESULTS (add to Table 4 / tab:sota in draft.tex)")
    logger.info("="*70)
    pvals = {}
    for key, name, res in [
        ("transformer_1d", "Transformer-1D (spectral only)", res_t1d),
        ("film_5class", "FiLM (5-class)", res_film),
    ]:
        p_val = ""
        if pin_preds is not None:
            p = mcnemar_p(res["predictions"], pin_preds, res["labels"])
            pvals[key] = p
            p_val = f"{p:.3f}"
        logger.info(
            f"  {name:35s}  acc={res['accuracy']:.2f}%  "
            f"CI=[{res['ci_lo']:.1f},{res['ci_hi']:.1f}]  "
            f"params={res['num_params']:,}  "
            f"latency={res['infer_ms']:.1f}ms  "
            f"p={p_val}"
        )

    # Save JSON summary for draft.tex update
    summary = {
        "transformer_1d": {k: v for k, v in res_t1d.items() if k != "predictions" and k != "labels"},
        "film_5class":    {k: v for k, v in res_film.items() if k != "predictions" and k != "labels"},
    }
    if "transformer_1d" in pvals:
        summary["transformer_1d"]["p_vs_pinnacle"] = float(pvals["transformer_1d"])
    if "film_5class" in pvals:
        summary["film_5class"]["p_vs_pinnacle"] = float(pvals["film_5class"])
    out_json = os.path.join(CFG["out_dir"], "extra_5class_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved → {out_json}")


if __name__ == "__main__":
    main()
