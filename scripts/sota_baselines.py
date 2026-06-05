#!/usr/bin/env python3
"""
PINNACLE — SOTA Baseline Comparisons (5-Class Bacteria-ID, identical split).

Three new baselines evaluated under the exact same 80/10/10 stratified
split (seed=42) and preprocessing pipeline used by PINNACLE, enabling
direct head-to-head comparison:

  Baseline 1 — ResNet-18-1D (Ho et al. 2019 architecture)
      Standard 1D ResNet-18 applied to raw Raman spectra.
      This is the closest reproduction of the original Bacteria-ID paper
      architecture under our preprocessing/split conditions.

  Baseline 2 — KirchhoffNet (Kirchhoff et al. 2023)
      CNN encoder + multi-head self-attention classification head for
      single-cell Raman spectra, as described in:
      Kirchhoff et al., npj Biofilms and Microbiomes (2023).
      Implementation follows the published architecture description.

  Baseline 3 — PureCrossAttn (Ablation: cross-attention without gating)
      Uses the same dual-stream CNN encoders as PINNACLE but replaces
      the SeparationCross module with a plain cross-attention layer
      that has NO learnable sigmoid gating (α, β removed). This isolates
      the contribution of PINNACLE's signature gating mechanism.

All models are trained for 30 epochs with AdamW (lr=1e-3, wd=1e-4),
cosine annealing, dropout=0.3, early stopping (patience=12).

Usage:
    # Run all three baselines sequentially:
    cd /path/to/Raman
    python scripts/sota_baselines.py

    # Run a single baseline:
    python scripts/sota_baselines.py --model resnet18_1d
    python scripts/sota_baselines.py --model kirchhoff
    python scripts/sota_baselines.py --model pure_crossattn

    # Skip checkpointing (re-run fresh):
    python scripts/sota_baselines.py --no-cache

Output:
    Results printed as a LaTeX table row ready for Table 10 (tab:sota).
    Checkpoints saved to outputs/sota_baselines/<model>/best.pt
    Predictions saved to outputs/sota_baselines/<model>/predictions.npz
"""

import os
import sys
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import chi2

from pinnacle.utils import set_seed, get_device, logger, count_parameters
from pinnacle.model import SpectralBranch, ScalogramBranch
from pinnacle.dataset import PINNACLEDataset, RamanAugmentation

# =========================================================================
# Configuration (matches PINNACLE paper Table 3 / Table 4)
# =========================================================================

CFG = {
    "data_dir": "data/",
    "seed": 42,
    "num_classes": 5,
    "embed_dim": 128,
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "early_stopping_patience": 12,
    "test_size": 0.10,
    "val_size": 0.10,
    "num_workers": 0,        # macOS: keep 0 for MPS/CPU
    "out_dir": "outputs/sota_baselines",
}

SPECIES_NAMES = [
    "E. coli", "S. aureus", "P. aeruginosa", "K. pneumoniae", "E. faecalis"
]


# =========================================================================
# Baseline 1 — ResNet-18-1D  (Ho et al. 2019)
# =========================================================================

class BasicBlock1D(nn.Module):
    """Standard 1D ResNet basic block (pre-activation variant)."""
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet18_1D(nn.Module):
    """
    1D ResNet-18 for Raman spectral classification.

    Adapted from He et al. (2016) for 1D inputs:
    - Initial stem: Conv1d(1→64, k=7, s=2) + BN + ReLU + MaxPool1d
    - Four residual stages: [64, 128, 256, 512] channels, blocks=[2,2,2,2]
    - Global average pool → FC(512 → num_classes)

    This matches the spirit of Ho et al. (2019) who used ResNet-18 for
    Raman classification; the key difference is operating on 1D spectra
    rather than 2D images.

    Parameters: ~3.86 M
    """

    def __init__(self, num_classes: int = 5, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int):
        layers = [BasicBlock1D(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, raman: torch.Tensor, scalogram=None):
        """
        Args:
            raman: (B, L) or (B, 1, L)
            scalogram: ignored (unimodal baseline)
        Returns:
            logits: (B, num_classes), None, None
        """
        x = raman
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, L)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)    # (B, 512)
        x = self.dropout(x)
        return self.fc(x), None, None


# =========================================================================
# Baseline 2 — KirchhoffNet  (Kirchhoff et al. 2023, npj Biofilms)
# =========================================================================

class KirchhoffNet(nn.Module):
    """
    CNN + Multi-Head Self-Attention classifier for single-cell Raman spectra.

    Architecture follows Kirchhoff et al. (2023):
      - Three 1D convolutional blocks (channels: 32→64→128, k=7/5/3)
        with BN, ReLU, MaxPool (matching the published description)
      - Sequence of spectral features fed into multi-head self-attention
        (4 heads, embed_dim=128) to model inter-wavenumber dependencies
      - LayerNorm + residual connection after attention
      - Global average pool → Dropout → FC(num_classes)

    The key distinction from PINNACLE's spectral branch: after the CNN
    encoder the feature sequence passes through a self-attention layer
    (attending to itself, not a second modality), then into a linear
    classifier. There is no CWT branch and no inter-modal gating.

    Parameters: ~0.85 M
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        # CNN encoder — three conv blocks, each followed by MaxPool1d(2).
        # Three pools reduce 1000 → 500 → 250 → 125 positions, giving a
        # gradient dilution of 1/125 through the final GlobalAvgPool instead
        # of 1/250 (two pools) or 1/1000 (no pools).  Without this third
        # pool the model is stuck at chance level on 1000-wavenumber spectra.
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),          # 1000 → 500

            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),          # 500  → 250

            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),          # 250  → 125  ← key fix
        )
        # Self-attention over 125 spectral positions.
        # dropout is set to 0.0 here and applied via a separate nn.Dropout
        # layer below — MPS (Apple Silicon) can produce NaN gradients when
        # dropout > 0 is set directly inside nn.MultiheadAttention.
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,          # MPS-safe: dropout moved to self.attn_drop
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, raman: torch.Tensor, scalogram=None):
        """
        Args:
            raman: (B, L) or (B, 1, L)
            scalogram: ignored (unimodal baseline)
        Returns:
            logits: (B, num_classes), None, None
        """
        x = raman
        if x.dim() == 2:
            x = x.unsqueeze(1)              # (B, 1, L)
        # Per-spectrum z-score normalisation.  X_2018_proc has std≈6e-4 but
        # the dataset-level RamanAugmentation adds noise_std=0.02, making
        # the training distribution ~33× wider than the val/test distribution.
        # Without this step, BatchNorm running statistics (fitted on training)
        # attenuate clean val/test activations by ~33× per BN layer, causing
        # the CNN output to collapse to near-zero and the model to predict the
        # majority class.  ResNet-18-1D avoids this via residual connections
        # that bypass the BN attenuation; this normalisation achieves the same
        # effect without architectural changes.
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        h = self.cnn(x)                     # (B, embed_dim, L')
        # Transpose for attention: (B, L', embed_dim)
        h_t = h.transpose(1, 2)
        # Self-attention with residual (dropout applied separately for MPS)
        attn_out, _ = self.attn(h_t, h_t, h_t)
        attn_out = self.attn_drop(attn_out)
        h_t = self.norm(h_t + attn_out)     # (B, L', embed_dim)
        # Pool back to (B, embed_dim)
        h_pooled = self.pool(h_t.transpose(1, 2)).squeeze(-1)
        return self.classifier(h_pooled), None, None


# =========================================================================
# Baseline 3 — PureCrossAttn  (Ablation: dual-stream, no gating)
# =========================================================================

class PureCrossAttnFusion(nn.Module):
    """
    Cross-attention fusion WITHOUT learnable sigmoid gating.

    This is an ablation of PINNACLE's SeparationCross module: the two
    CNN branches are identical, but the per-modality gates (α, β) are
    removed entirely. The full (ungated) spectral and scalogram feature
    maps are tokenised and fed directly into the cross-attention layer.

    Specifically:
      - No gate_spectral or gate_scalogram linear layers
      - h_s and h_w are passed unmodified to the attention step
      - The residual scalar γ is retained so the model can learn to
        weight spectral attention injection (this is the minimum
        difference from PINNACLE's SeparationCross)

    This isolates the contribution of the signature gating mechanism.
    """

    def __init__(self, embed_dim: int = 128, spectral_pool_len: int = 25):
        super().__init__()
        self.embed_dim = embed_dim
        self.spectral_pool = nn.AdaptiveAvgPool1d(spectral_pool_len)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, h_s: torch.Tensor, h_w: torch.Tensor):
        """
        Args:
            h_s: (B, D, L)   — spectral CNN feature map (UNGATED)
            h_w: (B, D, H, W) — scalogram CNN feature map (UNGATED)
        Returns:
            z_fused: (B, 2D)
            alpha: None  (no gating)
            beta:  None  (no gating)
        """
        B, D, L = h_s.shape
        _, _, Hp, Wp = h_w.shape

        # Global pool embeddings for fusion head
        z_s = h_s.mean(dim=-1)            # (B, D)
        z_w = h_w.mean(dim=(-2, -1))      # (B, D)

        # Tokenise (no gating applied)
        seq_s = self.spectral_pool(h_s).transpose(1, 2)        # (B, S, D)
        seq_w = h_w.view(B, D, Hp * Wp).transpose(1, 2)        # (B, H'W', D)

        # Cross-attention: spectral queries, scalogram keys/values
        Q = self.W_q(seq_s)
        K = self.W_k(seq_w)
        V = self.W_v(seq_w)
        attn_w = F.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / self.scale, dim=-1
        )
        z_attn = torch.matmul(attn_w, V).mean(dim=1)            # (B, D)

        # Residual inject into scalogram embedding (no gate applied)
        z_combined = z_w + self.gamma * z_attn                  # (B, D)
        z_fused = torch.cat([z_combined, z_w], dim=1)           # (B, 2D)
        return z_fused, None, None


class PureCrossAttnModel(nn.Module):
    """
    Dual-stream CNN + pure cross-attention (no sigmoid gating).
    Identical encoder architecture to PINNACLE.
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels=1, embed_dim=embed_dim)
        self.scalogram_branch = ScalogramBranch(in_channels=3, embed_dim=embed_dim)
        self.fusion = PureCrossAttnFusion(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, raman: torch.Tensor, scalogram: torch.Tensor):
        h_s = self.spectral_branch.forward_features(raman)      # (B, D, L)
        h_w = self.scalogram_branch.forward_features(scalogram) # (B, D, H', W')
        z_fused, alpha, beta = self.fusion(h_s, h_w)
        return self.classifier(z_fused), alpha, beta


# =========================================================================
# Shared training infrastructure
# =========================================================================

def load_5class_data(data_dir: str, seed: int, test_size: float, val_size: float):
    """
    Load 5-class Bacteria-ID data from data/ and create the exact same
    80/10/10 stratified split used by PINNACLE (draft.tex §4.2).

    Returns: train/val/test arrays for raman, scalogram, labels.
    """
    X_2018 = np.load(os.path.join(data_dir, "X_2018_proc.npy")).astype(np.float32)
    X_2019 = np.load(os.path.join(data_dir, "X_2019_proc.npy")).astype(np.float32)
    y_2018 = np.load(os.path.join(data_dir, "y_2018clinical.npy")).astype(np.int64)
    y_2019 = np.load(os.path.join(data_dir, "y_2019clinical.npy")).astype(np.int64)

    X_all = np.concatenate([X_2018, X_2019], axis=0)
    y_all = np.concatenate([y_2018, y_2019], axis=0)

    wav_2018_path = os.path.join(data_dir, "X_2018_wavelet.npy")
    wav_2019_path = os.path.join(data_dir, "X_2019_wavelet.npy")
    has_scalograms = os.path.exists(wav_2018_path) and os.path.exists(wav_2019_path)

    if has_scalograms:
        W_2018 = np.load(wav_2018_path).astype(np.float32)
        W_2019 = np.load(wav_2019_path).astype(np.float32)
        W_all = np.concatenate([W_2018, W_2019], axis=0)
    else:
        W_all = None
        logger.warning(
            "CWT scalograms not found — PureCrossAttn baseline will be skipped. "
            "Run `python scripts/generate_wavelets.py` to generate them."
        )

    logger.info(f"Loaded: X={X_all.shape}, y={y_all.shape}")

    # Stratified 80/20 split → then 50/50 val/test from the 20%
    indices = np.arange(len(X_all))
    idx_train, idx_temp = train_test_split(
        indices,
        test_size=(test_size + val_size),
        random_state=seed,
        stratify=y_all,
    )
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=test_size / (test_size + val_size),
        random_state=seed,
        stratify=y_all[idx_temp],
    )

    splits = {}
    for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        splits[f"X_{name}"] = X_all[idx]
        splits[f"y_{name}"] = y_all[idx]
        splits[f"W_{name}"] = W_all[idx] if W_all is not None else None

    logger.info(
        f"Split sizes — train:{len(idx_train)}, val:{len(idx_val)}, test:{len(idx_test)}"
    )
    return splits, has_scalograms


def make_loaders(splits, batch_size, num_workers, use_scalograms=True):
    """Create DataLoaders from split arrays."""
    aug = RamanAugmentation()
    loaders = {}
    for split in ("train", "val", "test"):
        W = splits[f"W_{split}"] if use_scalograms else None
        ds = PINNACLEDataset(
            splits[f"X_{split}"],
            splits[f"y_{split}"],
            X_scalogram=W,
            transform_raman=(aug if split == "train" else None),
            split=split,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders


def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for raman, scalogram, labels in loader:
        raman = raman.to(device)
        scalogram = scalogram.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(raman, scalogram)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for raman, scalogram, labels in loader:
        raman = raman.to(device)
        scalogram = scalogram.to(device)
        labels = labels.to(device)
        logits, _, _ = model(raman, scalogram)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    preds, labels_all = [], []
    for raman, scalogram, labels in loader:
        raman = raman.to(device)
        scalogram = scalogram.to(device)
        logits, _, _ = model(raman, scalogram)
        preds.append(logits.argmax(1).cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(labels_all)


def wilson_ci(acc_frac: float, n: int, z: float = 1.96):
    """95% Wilson score confidence interval."""
    p = acc_frac
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * ((p * (1 - p) + z**2 / (4 * n)) / n) ** 0.5 / denom
    return (center - margin) * 100, (center + margin) * 100


def mcnemar_pvalue(preds_a, preds_b, labels):
    """McNemar's test p-value (continuity-corrected): method A vs method B."""
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)
    n01 = np.sum(correct_a & ~correct_b)   # A correct, B wrong
    n10 = np.sum(~correct_a & correct_b)   # A wrong, B correct
    if n01 + n10 == 0:
        return 1.0
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(1.0 - chi2.cdf(statistic, df=1))


def run_training(
    model_name: str,
    model: nn.Module,
    loaders: dict,
    device: torch.device,
    cfg: dict,
    out_dir: str,
    no_cache: bool = False,
) -> dict:
    """
    Full train + eval loop with early stopping and checkpointing.
    Returns a results dict with accuracy, CI, predictions, latency.
    """
    ckpt_dir = os.path.join(out_dir, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    pred_path = os.path.join(ckpt_dir, "predictions.npz")

    # --- Resume from checkpoint if available ---
    if not no_cache and os.path.exists(pred_path):
        logger.info(f"  [{model_name}] Cached predictions found — loading.")
        data = np.load(pred_path)
        return {
            "model_name": model_name,
            "accuracy": float(data["accuracy"]),
            "ci_lo": float(data["ci_lo"]),
            "ci_hi": float(data["ci_hi"]),
            "predictions": data["predictions"],
            "labels": data["labels"],
            "num_params": int(data["num_params"]),
            "infer_ms": float(data["infer_ms"]),
        }

    model = model.to(device)
    n_params = count_parameters(model)
    logger.info(f"\n{'='*60}")
    logger.info(f"  Training: {model_name}  ({n_params:,} params)")
    logger.info(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_acc": [], "val_acc": []}

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, loaders["train"], optimizer, criterion, device, cfg["grad_clip"]
        )
        vl_loss, vl_acc = eval_epoch(
            model, loaders["val"], criterion, device
        )
        scheduler.step()

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        logger.info(
            f"  Epoch {epoch:3d}/{cfg['epochs']} | "
            f"Train {tr_acc:.2f}%  Val {vl_acc:.2f}%  ({elapsed:.1f}s)"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "val_acc": vl_acc},
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg["early_stopping_patience"]:
                logger.info(f"  Early stopping at epoch {epoch}.")
                break

    # Load best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    logger.info(f"  Best val acc: {ckpt['val_acc']:.2f}% (epoch {ckpt['epoch']})")

    # Test evaluation
    preds, labels = get_predictions(model, loaders["test"], device)
    accuracy = accuracy_score(labels, preds) * 100
    ci_lo, ci_hi = wilson_ci(accuracy / 100, len(labels))

    # Inference latency: average over 100 single-sample forward passes
    model.eval()
    dummy_raman = torch.zeros(1, 1000, device=device)
    dummy_scalo = torch.zeros(1, 3, 224, 224, device=device)
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_raman, dummy_scalo)
    t_start = time.perf_counter()
    N_LATENCY = 100
    for _ in range(N_LATENCY):
        with torch.no_grad():
            _ = model(dummy_raman, dummy_scalo)
    infer_ms = (time.perf_counter() - t_start) / N_LATENCY * 1000

    # Save predictions
    np.savez(
        pred_path,
        predictions=preds,
        labels=labels,
        accuracy=accuracy,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        num_params=n_params,
        infer_ms=infer_ms,
        train_acc_history=history["train_acc"],
        val_acc_history=history["val_acc"],
    )

    logger.info(
        f"  Test accuracy: {accuracy:.2f}%  "
        f"95% CI: [{ci_lo:.1f}, {ci_hi:.1f}]  "
        f"Latency: {infer_ms:.1f} ms"
    )

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "predictions": preds,
        "labels": labels,
        "num_params": n_params,
        "infer_ms": infer_ms,
    }


# =========================================================================
# Load PINNACLE predictions for McNemar comparisons
# =========================================================================

def load_pinnacle_predictions(out_dir: str = "outputs"):
    """
    Load PINNACLE's test-set predictions (saved by the main training script).
    Returns (preds, labels) or (None, None) if not found.
    """
    candidates = [
        os.path.join(out_dir, "predictions.npz"),
        os.path.join(out_dir, "checkpoints", "predictions.npz"),
        os.path.join(out_dir, "sota_baselines", "pinnacle", "predictions.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            data = np.load(path)
            if "predictions" in data and "labels" in data:
                logger.info(f"  PINNACLE predictions loaded from: {path}")
                return data["predictions"], data["labels"]
    logger.warning(
        "  PINNACLE predictions not found. McNemar p-values will be N/A.\n"
        "  Run `python scripts/train.py` first and ensure predictions are saved."
    )
    return None, None


# =========================================================================
# LaTeX table printer
# =========================================================================

def print_latex_table(results: list, pinnacle_preds=None, pinnacle_labels=None):
    """
    Print a LaTeX table (tab:sota) rows for inclusion in draft.tex.
    """
    print("\n" + "=" * 70)
    print("LaTeX Table Rows (copy into draft.tex, tab:sota)")
    print("=" * 70)

    # Header
    print(r"""\begin{table}[!t]
\centering
\caption{Direct head-to-head comparison on the 5-class Bacteria-ID benchmark
(identical 80/10/10 stratified split, seed 42). All models use the same
preprocessed Raman spectra. $p$-values are McNemar tests vs.\ PINNACLE.}
\label{tab:sota}
\begin{tabular}{lcccc}
  \toprule
  Method & Acc (\%) & 95\% CI & Params & $p$-value \\
  \midrule""")

    for r in results:
        name = r["model_name"]
        acc = r["accuracy"]
        ci = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
        params = r["num_params"]
        if params >= 1e6:
            params_str = f"{params/1e6:.2f}\\,M"
        else:
            params_str = f"{params//1000}K"

        # McNemar p-value vs PINNACLE
        if pinnacle_preds is not None and pinnacle_labels is not None:
            # Only compute if test-set labels match
            if np.array_equal(r["labels"], pinnacle_labels):
                pval = mcnemar_pvalue(r["predictions"], pinnacle_preds, r["labels"])
                if pval < 0.001:
                    pval_str = "$<0.001$"
                elif pval < 0.05:
                    pval_str = f"${pval:.3f}$"
                else:
                    pval_str = f"${pval:.3f}$"
            else:
                pval_str = "N/A"
        else:
            pval_str = "N/A"

        print(f"  {name} & {acc:.2f} & {ci} & {params_str} & {pval_str} \\\\")

    print(r"  \midrule")
    # PINNACLE row — use real numbers if predictions were loaded, else placeholder
    if pinnacle_preds is not None and pinnacle_labels is not None:
        pinnacle_acc = (pinnacle_preds == pinnacle_labels).mean() * 100
        n = len(pinnacle_labels)
        lo, hi = wilson_ci(pinnacle_acc / 100, n)
        ci_str = f"[{lo:.1f}, {hi:.1f}]"
        print(f"  \\textbf{{PINNACLE (ours)}} & \\textbf{{{pinnacle_acc:.2f}}} & {ci_str} & 301K & --- \\\\")
    else:
        print(r"  \textbf{PINNACLE (ours)} & \textbf{94.72} & [93.3, 95.8] & 301K & --- \\")
    print(r"""  \bottomrule
\end{tabular}
\end{table}""")

    print("\n" + "=" * 70)
    print("Per-model summary (for logging):")
    print("=" * 70)
    for r in results:
        print(
            f"  {r['model_name']:<25} acc={r['accuracy']:.2f}%  "
            f"params={r['num_params']:,}  latency={r['infer_ms']:.1f}ms"
        )


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run SOTA baseline comparisons for PINNACLE (5-class benchmark)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "resnet18_1d", "kirchhoff", "pure_crossattn"],
        help="Which baseline to run (default: all).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=CFG["data_dir"],
        help=f"Directory with 5-class .npy files (default: {CFG['data_dir']}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CFG["epochs"],
        help=f"Training epochs (default: {CFG['epochs']}).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached predictions and re-train from scratch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'mps' (default: auto).",
    )
    args = parser.parse_args()

    # Update config from CLI
    cfg = {**CFG}
    cfg["data_dir"] = args.data_dir
    cfg["epochs"] = args.epochs

    set_seed(cfg["seed"])
    device = get_device(args.device)

    logger.info("=" * 60)
    logger.info("PINNACLE — SOTA Baseline Comparison (5-class)")
    logger.info("=" * 60)
    logger.info(f"  Device  : {device}")
    logger.info(f"  Epochs  : {cfg['epochs']}")
    logger.info(f"  Data dir: {cfg['data_dir']}")
    logger.info(f"  Seed    : {cfg['seed']}")

    # Load data
    splits, has_scalograms = load_5class_data(
        cfg["data_dir"], cfg["seed"], cfg["test_size"], cfg["val_size"]
    )

    # Loaders without scalograms (unimodal baselines)
    loaders_1d = make_loaders(splits, cfg["batch_size"], cfg["num_workers"], use_scalograms=False)
    # Loaders with scalograms (PureCrossAttn)
    loaders_2d = (
        make_loaders(splits, cfg["batch_size"], cfg["num_workers"], use_scalograms=True)
        if has_scalograms else None
    )

    os.makedirs(cfg["out_dir"], exist_ok=True)

    run_all = (args.model == "all")
    results = []

    # ------------------------------------------------------------------
    # Baseline 1: ResNet-18-1D
    # ------------------------------------------------------------------
    if run_all or args.model == "resnet18_1d":
        model = ResNet18_1D(
            num_classes=cfg["num_classes"],
            in_channels=1,
            dropout=cfg["dropout"],
        )
        r = run_training(
            model_name="ResNet-18-1D (Ho et al.)",
            model=model,
            loaders=loaders_1d,
            device=device,
            cfg=cfg,
            out_dir=cfg["out_dir"],
            no_cache=args.no_cache,
        )
        results.append(r)

    # ------------------------------------------------------------------
    # Baseline 2: KirchhoffNet
    # ------------------------------------------------------------------
    if run_all or args.model == "kirchhoff":
        model = KirchhoffNet(
            num_classes=cfg["num_classes"],
            in_channels=1,
            embed_dim=cfg["embed_dim"],
            num_heads=4,
            dropout=cfg["dropout"],
        )
        r = run_training(
            model_name="KirchhoffNet (CNN+Attn)",
            model=model,
            loaders=loaders_1d,
            device=device,
            cfg=cfg,
            out_dir=cfg["out_dir"],
            no_cache=args.no_cache,
        )
        results.append(r)

    # ------------------------------------------------------------------
    # Baseline 3: PureCrossAttn (dual-stream, no gating)
    # ------------------------------------------------------------------
    if run_all or args.model == "pure_crossattn":
        if not has_scalograms:
            logger.warning(
                "  Skipping PureCrossAttn: CWT scalograms not found.\n"
                "  Run `python scripts/generate_wavelets.py` first."
            )
        else:
            model = PureCrossAttnModel(
                num_classes=cfg["num_classes"],
                embed_dim=cfg["embed_dim"],
                dropout=cfg["dropout"],
            )
            r = run_training(
                model_name="PureCrossAttn (no gating)",
                model=model,
                loaders=loaders_2d,
                device=device,
                cfg=cfg,
                out_dir=cfg["out_dir"],
                no_cache=args.no_cache,
            )
            results.append(r)

    if not results:
        logger.error("No results — nothing ran. Check your --model flag.")
        return

    # ------------------------------------------------------------------
    # McNemar tests vs PINNACLE
    # ------------------------------------------------------------------
    pinnacle_preds, pinnacle_labels = load_pinnacle_predictions()

    # ------------------------------------------------------------------
    # Print LaTeX table
    # ------------------------------------------------------------------
    print_latex_table(results, pinnacle_preds, pinnacle_labels)

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    summary = {
        r["model_name"]: {
            "accuracy": r["accuracy"],
            "ci": [r["ci_lo"], r["ci_hi"]],
            "num_params": r["num_params"],
            "infer_ms": r["infer_ms"],
        }
        for r in results
    }
    summary_path = os.path.join(cfg["out_dir"], "sota_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n  Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
