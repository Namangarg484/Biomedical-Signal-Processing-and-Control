#!/usr/bin/env python3
"""
PINNACLE — All Three Reviewer Experiments in One Script.

Experiment 1: ResNet-1D baseline (modern deep 1D architecture)
Experiment 2: Wavelet ablation (Morlet vs Mexican Hat vs Bump)
Experiment 3: Cross-attention directionality (flip Q/K/V)

All experiments use the 30-class dataset with Phase 1 + Phase 2 protocol.
Results are printed as LaTeX table rows ready for the manuscript.

Usage:
    python scripts/reviewer_experiments.py                # Run all 3
    python scripts/reviewer_experiments.py --exp resnet    # Only ResNet-1D
    python scripts/reviewer_experiments.py --exp wavelet   # Only wavelet ablation
    python scripts/reviewer_experiments.py --exp flip      # Only attention flip
"""

import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from pinnacle.utils import set_seed, get_device, logger, count_parameters
from pinnacle.model import SpectralBranch, ScalogramBranch, PINNACLE
from pinnacle.dataset import PINNACLEDataset, RamanAugmentation
from pinnacle.wavelet import spectrum_to_scalogram


# ======================================================================
# SHARED: Training infrastructure
# ======================================================================

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=5.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, batch in enumerate(loader):
        raman, scalogram, labels = batch
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
        if batch_idx % 500 == 0 and batch_idx > 0:
            logger.info(f"    Batch {batch_idx}/{len(loader)} | Acc: {100.*correct/total:.1f}%")
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


# (full_pipeline with checkpointing is defined below, after all model classes)


# ======================================================================
# EXPERIMENT 1: ResNet-1D Baseline
# ======================================================================

class BasicBlock1D(nn.Module):
    """1D ResNet basic block."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class ResNet1D(nn.Module):
    """ResNet-18 adapted for 1D Raman spectra."""

    def __init__(self, num_classes=30, in_channels=1):
        super().__init__()
        self.in_ch = 64

        # Initial conv with large kernel for 1D spectral data
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=11, stride=2, padding=5, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)

        # ResNet-18 layers: [2, 2, 2, 2]
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = [BasicBlock1D(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x, scalogram=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits, None, None  # Match PINNACLE interface


# ======================================================================
# EXPERIMENT 3: Flipped Cross-Attention SeparationCross
# ======================================================================

class SeparationCrossFlipped(nn.Module):
    """SeparationCross with FLIPPED spatial cross-attention direction.
    
    Original:  Q=spectral (S positions),  K/V=scalogram (H'W' positions)
    Flipped:   Q=scalogram (H'W' positions), K/V=spectral (S positions)
    """

    def __init__(self, embed_dim=128, spectral_pool_len=25):
        super().__init__()
        self.embed_dim = embed_dim
        self.gate_spectral = nn.Linear(embed_dim, embed_dim)
        self.gate_scalogram = nn.Linear(embed_dim, embed_dim)
        self.spectral_pool = nn.AdaptiveAvgPool1d(spectral_pool_len)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, h_s, h_w):
        B, D, L = h_s.shape
        _, _, Hp, Wp = h_w.shape

        z_s = h_s.mean(dim=-1)
        z_w = h_w.mean(dim=(-2, -1))

        alpha = torch.sigmoid(self.gate_spectral(z_s))
        beta = torch.sigmoid(self.gate_scalogram(z_w))

        h_s_gated = alpha.unsqueeze(-1) * h_s
        h_w_gated = beta.unsqueeze(-1).unsqueeze(-1) * h_w

        seq_s = self.spectral_pool(h_s_gated).transpose(1, 2)  # (B, S, D)
        seq_w = h_w_gated.view(B, D, Hp * Wp).transpose(1, 2)  # (B, H'W', D)

        # FLIPPED: Q=scalogram, K/V=spectral
        Q = self.W_q(seq_w)    # (B, H'W', D) — scalogram queries
        K = self.W_k(seq_s)    # (B, S, D)    — spectral keys
        V = self.W_v(seq_s)    # (B, S, D)    — spectral values

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H'W', S)
        attn = F.softmax(attn, dim=-1)
        z_attn = torch.matmul(attn, V)  # (B, H'W', D)
        z_attn = z_attn.mean(dim=1)     # (B, D)

        z_beta = beta * z_w
        z_combined = z_beta + self.gamma * z_attn
        z_fused = torch.cat([z_combined, z_beta], dim=1)
        return z_fused, alpha, beta


class PINNACLEFlipped(nn.Module):
    """PINNACLE with flipped cross-attention direction."""

    def __init__(self, num_classes=30, embed_dim=128, dropout=0.3):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels=1, embed_dim=embed_dim)
        self.scalogram_branch = ScalogramBranch(in_channels=3, embed_dim=embed_dim)
        self.fusion = SeparationCrossFlipped(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        logger.info(f"PINNACLEFlipped: params={count_parameters(self):,}")

    def forward(self, raman, scalogram=None):
        h_s = self.spectral_branch.forward_features(raman)       # (B, D, L)
        h_w = self.scalogram_branch.forward_features(scalogram)  # (B, D, H', W')
        z_fused, alpha, beta = self.fusion(h_s, h_w)
        logits = self.classifier(z_fused)
        return logits, alpha, beta


# ======================================================================
# EXPERIMENT 4: 1D Transformer Baseline
# ======================================================================

class Transformer1D(nn.Module):
    """1D Vision Transformer for Raman spectra (spectral-only baseline).
    
    Splits the 1000-point spectrum into patches, projects them,
    adds positional embeddings, and runs through Transformer encoder layers.
    Uses a CLS token for classification.
    """

    def __init__(self, num_classes=30, seq_len=1000, patch_size=20,
                 embed_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        assert seq_len % patch_size == 0, "seq_len must be divisible by patch_size"
        self.num_patches = seq_len // patch_size
        self.patch_size = patch_size

        # Patch embedding: each patch of size patch_size → embed_dim
        self.patch_embed = nn.Linear(patch_size, embed_dim)

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier
        self.fc = nn.Linear(embed_dim, num_classes)

        logger.info(f"Transformer1D: patches={self.num_patches}, "
                    f"embed={embed_dim}, heads={num_heads}, "
                    f"layers={num_layers}, params={count_parameters(self):,}")

    def forward(self, x, scalogram=None):
        B = x.size(0)
        # x: (B, 1000) → (B, num_patches, patch_size)
        if x.dim() == 3:
            x = x.squeeze(1)
        x = x.view(B, self.num_patches, self.patch_size)

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)          # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer encoder
        x = self.encoder(x)
        x = self.norm(x)

        # CLS token output → classification
        cls_out = x[:, 0]  # (B, embed_dim)
        logits = self.fc(cls_out)
        return logits, None, None  # Match PINNACLE interface


# ======================================================================
# EXPERIMENT 5: Shared Gating (vs Independent Gating ablation)
# ======================================================================

class SeparationCrossSharedGating(nn.Module):
    """SeparationCross with SHARED gating instead of independent gates.
    
    Instead of independent W_alpha(z_s) and W_beta(z_w), this version
    concatenates [z_s; z_w] and computes a joint gate that is shared
    across both modalities.
    """

    def __init__(self, embed_dim=128, spectral_pool_len=25):
        super().__init__()
        self.embed_dim = embed_dim

        # Shared gate: takes concatenated [z_s; z_w] → single gate
        self.gate_shared = nn.Linear(2 * embed_dim, embed_dim)

        # Pool spectral sequence
        self.spectral_pool = nn.AdaptiveAvgPool1d(spectral_pool_len)

        # Cross-attention projections (same as original)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, h_s, h_w):
        B, D, L = h_s.shape
        _, _, Hp, Wp = h_w.shape

        z_s = h_s.mean(dim=-1)
        z_w = h_w.mean(dim=(-2, -1))

        # SHARED gate: same gate applied to both modalities
        z_cat = torch.cat([z_s, z_w], dim=1)                 # (B, 2D)
        shared_gate = torch.sigmoid(self.gate_shared(z_cat))  # (B, D)

        # Apply SAME gate to both branches
        h_s_gated = shared_gate.unsqueeze(-1) * h_s
        h_w_gated = shared_gate.unsqueeze(-1).unsqueeze(-1) * h_w

        seq_s = self.spectral_pool(h_s_gated).transpose(1, 2)
        seq_w = h_w_gated.view(B, D, Hp * Wp).transpose(1, 2)

        Q = self.W_q(seq_s)
        K = self.W_k(seq_w)
        V = self.W_v(seq_w)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        z_attn = torch.matmul(attn, V).mean(dim=1)

        z_beta = shared_gate * z_w
        z_combined = z_beta + self.gamma * z_attn
        z_fused = torch.cat([z_combined, z_beta], dim=1)
        return z_fused, shared_gate, shared_gate  # alpha=beta=shared


class PINNACLESharedGating(nn.Module):
    """PINNACLE with shared gating (ablation: proves independent gating matters)."""

    def __init__(self, num_classes=30, embed_dim=128, dropout=0.3):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels=1, embed_dim=embed_dim)
        self.scalogram_branch = ScalogramBranch(in_channels=3, embed_dim=embed_dim)
        self.fusion = SeparationCrossSharedGating(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        logger.info(f"PINNACLESharedGating: params={count_parameters(self):,}")

    def forward(self, raman, scalogram=None):
        h_s = self.spectral_branch.forward_features(raman)
        h_w = self.scalogram_branch.forward_features(scalogram)
        z_fused, alpha, beta = self.fusion(h_s, h_w)
        logits = self.classifier(z_fused)
        return logits, alpha, beta


# ======================================================================
# EXPERIMENT 7: Bidirectional Attention
# ======================================================================

class SeparationCrossBidirectional(nn.Module):
    """Bidirectional cross-attention: both directions fused together.
    
    Runs Q=spec→KV=scalo AND Q=scalo→KV=spec, averages the attended
    outputs, and concatenates with the gated scalogram embedding.
    """

    def __init__(self, embed_dim=128, spectral_pool_len=25):
        super().__init__()
        self.embed_dim = embed_dim

        self.gate_spectral = nn.Linear(embed_dim, embed_dim)
        self.gate_scalogram = nn.Linear(embed_dim, embed_dim)
        self.spectral_pool = nn.AdaptiveAvgPool1d(spectral_pool_len)

        # Direction 1: Q=spec, KV=scalo
        self.W_q1 = nn.Linear(embed_dim, embed_dim)
        self.W_k1 = nn.Linear(embed_dim, embed_dim)
        self.W_v1 = nn.Linear(embed_dim, embed_dim)

        # Direction 2: Q=scalo, KV=spec
        self.W_q2 = nn.Linear(embed_dim, embed_dim)
        self.W_k2 = nn.Linear(embed_dim, embed_dim)
        self.W_v2 = nn.Linear(embed_dim, embed_dim)

        self.scale = embed_dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, h_s, h_w):
        B, D, L = h_s.shape
        _, _, Hp, Wp = h_w.shape

        z_s = h_s.mean(dim=-1)
        z_w = h_w.mean(dim=(-2, -1))

        alpha = torch.sigmoid(self.gate_spectral(z_s))
        beta = torch.sigmoid(self.gate_scalogram(z_w))

        h_s_gated = alpha.unsqueeze(-1) * h_s
        h_w_gated = beta.unsqueeze(-1).unsqueeze(-1) * h_w

        seq_s = self.spectral_pool(h_s_gated).transpose(1, 2)      # (B, S, D)
        seq_w = h_w_gated.view(B, D, Hp * Wp).transpose(1, 2)      # (B, H'W', D)

        # Direction 1: spec queries scalo
        Q1 = self.W_q1(seq_s)
        K1 = self.W_k1(seq_w)
        V1 = self.W_v1(seq_w)
        attn1 = F.softmax(torch.matmul(Q1, K1.transpose(-2, -1)) / self.scale, dim=-1)
        z_attn1 = torch.matmul(attn1, V1).mean(dim=1)              # (B, D)

        # Direction 2: scalo queries spec
        Q2 = self.W_q2(seq_w)
        K2 = self.W_k2(seq_s)
        V2 = self.W_v2(seq_s)
        attn2 = F.softmax(torch.matmul(Q2, K2.transpose(-2, -1)) / self.scale, dim=-1)
        z_attn2 = torch.matmul(attn2, V2).mean(dim=1)              # (B, D)

        # Combine both directions
        z_attn_bi = (z_attn1 + z_attn2) / 2

        z_beta = beta * z_w
        z_combined = z_beta + self.gamma * z_attn_bi
        z_fused = torch.cat([z_combined, z_beta], dim=1)
        return z_fused, alpha, beta


class PINNACLEBidirectional(nn.Module):
    """PINNACLE with bidirectional cross-attention."""

    def __init__(self, num_classes=30, embed_dim=128, dropout=0.3):
        super().__init__()
        self.spectral_branch = SpectralBranch(in_channels=1, embed_dim=embed_dim)
        self.scalogram_branch = ScalogramBranch(in_channels=3, embed_dim=embed_dim)
        self.fusion = SeparationCrossBidirectional(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
        logger.info(f"PINNACLEBidirectional: params={count_parameters(self):,}")

    def forward(self, raman, scalogram=None):
        h_s = self.spectral_branch.forward_features(raman)
        h_w = self.scalogram_branch.forward_features(scalogram)
        z_fused, alpha, beta = self.fusion(h_s, h_w)
        logits = self.classifier(z_fused)
        return logits, alpha, beta


# ======================================================================
# DATA LOADING (shared by all experiments)
# ======================================================================

def load_all_data(data_dir="New data", wavelet_suffix=""):
    """Load 30-class data. wavelet_suffix="" for Morlet, "_mexh", "_bump"."""
    X_ref = np.load(os.path.join(data_dir, "X_reference.npy"))
    y_ref = np.load(os.path.join(data_dir, "y_reference.npy")).astype(np.int64)

    if wavelet_suffix == "_stft":
        wav_name = "X_reference_stft.npy"
    elif wavelet_suffix == "_224":
        wav_name = "X_reference_wavelet_224.npy"
    else:
        wav_name = f"X_reference_wavelet{wavelet_suffix}.npy"
    wav_path = os.path.join(data_dir, wav_name)
    if os.path.exists(wav_path):
        X_ref_wav = np.load(wav_path, mmap_mode="r")
    else:
        X_ref_wav = None
        logger.warning(f"  Wavelet file not found: {wav_path}")

    indices = np.arange(len(X_ref))
    idx_train, idx_val = train_test_split(indices, test_size=0.1, random_state=42, stratify=y_ref)

    aug = RamanAugmentation(noise_std=0.01, shift_range=5, scale_range=0.05, probability=0.5)

    if X_ref_wav is not None:
        X_train_wav = np.array(X_ref_wav[idx_train])
        X_val_wav = np.array(X_ref_wav[idx_val])
    else:
        X_train_wav = X_val_wav = None

    ref_train_ds = PINNACLEDataset(X_ref[idx_train], y_ref[idx_train], X_train_wav,
                                    transform_raman=aug, split="train")
    ref_val_ds = PINNACLEDataset(X_ref[idx_val], y_ref[idx_val], X_val_wav, split="val")
    ref_train_loader = DataLoader(ref_train_ds, batch_size=32, shuffle=True, drop_last=True)
    ref_val_loader = DataLoader(ref_val_ds, batch_size=32, shuffle=False)

    # Fine-tune data
    X_ft = np.load(os.path.join(data_dir, "X_finetune.npy"))
    y_ft = np.load(os.path.join(data_dir, "y_finetune.npy")).astype(np.int64)

    if wavelet_suffix == "_stft":
        ft_wav_name = "X_finetune_stft.npy"
    elif wavelet_suffix == "_224":
        ft_wav_name = "X_finetune_wavelet_224.npy"
    else:
        ft_wav_name = f"X_finetune_wavelet{wavelet_suffix}.npy"
    ft_wav_path = os.path.join(data_dir, ft_wav_name)
    if os.path.exists(ft_wav_path):
        X_ft_wav = np.load(ft_wav_path)
    else:
        X_ft_wav = None

    ft_idx = np.arange(len(X_ft))
    ft_train, ft_val = train_test_split(ft_idx, test_size=0.2, random_state=123, stratify=y_ft)

    ft_train_ds = PINNACLEDataset(X_ft[ft_train], y_ft[ft_train],
                                   X_ft_wav[ft_train] if X_ft_wav is not None else None,
                                   transform_raman=aug, split="train")
    ft_val_ds = PINNACLEDataset(X_ft[ft_val], y_ft[ft_val],
                                 X_ft_wav[ft_val] if X_ft_wav is not None else None, split="val")
    ft_train_loader = DataLoader(ft_train_ds, batch_size=16, shuffle=True, drop_last=True)
    ft_val_loader = DataLoader(ft_val_ds, batch_size=16, shuffle=False)

    # Test data
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)

    if wavelet_suffix == "_stft":
        test_wav_name = "X_test_stft.npy"
    elif wavelet_suffix == "_224":
        test_wav_name = "X_test_wavelet_224.npy"
    else:
        test_wav_name = f"X_test_wavelet{wavelet_suffix}.npy"
    test_wav_path = os.path.join(data_dir, test_wav_name)
    if os.path.exists(test_wav_path):
        X_test_wav = np.load(test_wav_path)
    else:
        X_test_wav = None

    test_ds = PINNACLEDataset(X_test, y_test, X_test_wav, split="test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    return ref_train_loader, ref_val_loader, ft_train_loader, ft_val_loader, test_loader


# ======================================================================
# EXPERIMENT 2: Wavelet Generation for Mexican Hat and Bump
# ======================================================================

def generate_alt_wavelets(data_dir="New data"):
    """Generate Mexican Hat and Bump scalograms if they don't exist."""
    from tqdm import tqdm

    for wavelet_name, suffix in [("mexh", "_mexh"), ("gaus1", "_bump")]:
        for split, x_file in [("reference", "X_reference.npy"),
                               ("finetune", "X_finetune.npy"),
                               ("test", "X_test.npy")]:
            out_name = x_file.replace(".npy", f"_wavelet{suffix}.npy")
            out_path = os.path.join(data_dir, out_name)

            if os.path.exists(out_path):
                logger.info(f"  ⏭️  {out_name} exists, skipping")
                continue

            logger.info(f"  Generating {out_name} ({wavelet_name})...")
            X = np.load(os.path.join(data_dir, x_file))
            N = X.shape[0]
            # NOTE: 128×128 resolution is intentional for the 30-class open-world
            # reviewer experiments (vs. 224×224 in the original 5-class manuscript).
            # This must be explicitly stated in the manuscript (Section 3.2.2 footnote)
            # to avoid reviewer flags about tensor shape discrepancies.
            result = np.zeros((N, 3, 128, 128), dtype=np.float32)

            for i in tqdm(range(N), desc=f"{split}/{wavelet_name}"):
                result[i] = spectrum_to_scalogram(X[i], img_size=128,
                                                   n_scales=256, wavelet=wavelet_name)

            np.save(out_path, result)
            size_gb = result.nbytes / (1024**3)
            logger.info(f"  ✅ Saved {out_name} ({size_gb:.2f} GB)")


def generate_scale_wavelets(data_dir="New data"):
    """Generate Morlet scalograms with different scale counts (128, 512)."""
    from tqdm import tqdm

    for n_scales in [128, 512]:
        suffix = f"_s{n_scales}"
        for split, x_file in [("reference", "X_reference.npy"),
                               ("finetune", "X_finetune.npy"),
                               ("test", "X_test.npy")]:
            out_name = x_file.replace(".npy", f"_wavelet{suffix}.npy")
            out_path = os.path.join(data_dir, out_name)

            if os.path.exists(out_path):
                logger.info(f"  ⏭️  {out_name} exists, skipping")
                continue

            logger.info(f"  Generating {out_name} (cmor1.5-1.0, {n_scales} scales)...")
            X = np.load(os.path.join(data_dir, x_file))
            # Same 128×128 resolution as generate_alt_wavelets — see note there.
            N = X.shape[0]
            result = np.zeros((N, 3, 128, 128), dtype=np.float32)


            for i in tqdm(range(N), desc=f"{split}/s{n_scales}"):
                result[i] = spectrum_to_scalogram(X[i], img_size=128,
                                                   n_scales=n_scales)

            np.save(out_path, result)
            size_gb = result.nbytes / (1024**3)
            logger.info(f"  ✅ Saved {out_name} ({size_gb:.2f} GB)")


def full_pipeline(model, ref_train_loader, ref_val_loader,
                  ft_train_loader, ft_val_loader, test_loader,
                  device, name, max_p1=50, max_p2=30):
    """Phase 1 pre-train + Phase 2 fine-tune + Test evaluation.
    
    Checkpoints are saved to outputs_reviewer/{name}/ so training
    can resume after crashes.
    """
    import json as _json

    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    ckpt_dir = os.path.join("outputs_reviewer", safe_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    # ---- Check if Phase 1 already completed ----
    p1_done_path = os.path.join(ckpt_dir, "p1_done.pth")
    p1_ckpt_path = os.path.join(ckpt_dir, "p1_latest.pth")

    if os.path.exists(p1_done_path):
        logger.info(f"\n  [{name}] Phase 1: LOADING completed checkpoint...")
        p1_ckpt = torch.load(p1_done_path, map_location=device, weights_only=False)
        model.load_state_dict(p1_ckpt["model_state"])
        best_val = p1_ckpt["best_val"]
        logger.info(f"  [{name}] Phase 1 loaded: {best_val:.2f}% val (skipped)")
    else:
        # ---- Phase 1: Pre-train on reference (with resume) ----
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_p1)

        best_val, best_state, patience = 0.0, None, 0
        start_epoch = 0

        # Resume from latest checkpoint if exists
        if os.path.exists(p1_ckpt_path):
            logger.info(f"  [{name}] Resuming Phase 1 from checkpoint...")
            ckpt = torch.load(p1_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            best_val = ckpt["best_val"]
            best_state = ckpt["best_state"]
            patience = ckpt["patience"]
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"  [{name}] Resumed at epoch {start_epoch}, best_val={best_val:.2f}%")

        t0 = time.time()
        logger.info(f"\n  [{name}] Phase 1: Pre-training ({max_p1} epochs max)...")

        for epoch in range(start_epoch, max_p1):
            t_loss, t_acc = train_epoch(model, ref_train_loader, optimizer, criterion, device)
            v_loss, v_acc = eval_epoch(model, ref_val_loader, criterion, device)
            scheduler.step()

            if v_acc > best_val:
                best_val = v_acc
                patience = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1

            if epoch % 5 == 0 or patience == 0:
                status = "BEST" if patience == 0 else f"p={patience}"
                logger.info(f"    P1 E{epoch:02d} | Train: {t_acc:.1f}% | Val: {v_acc:.1f}% | {status}")

            # Save checkpoint every 5 epochs
            if epoch % 5 == 0 or patience == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val": best_val,
                    "best_state": best_state,
                    "patience": patience,
                }, p1_ckpt_path)

            if patience >= 15:
                logger.info(f"    Early stopping at epoch {epoch}")
                break

        p1_time = (time.time() - t0) / 60
        model.load_state_dict(best_state)
        logger.info(f"  [{name}] Phase 1 done: {best_val:.2f}% val ({p1_time:.1f} min)")

        # Save completed Phase 1
        torch.save({
            "model_state": best_state,
            "best_val": best_val,
        }, p1_done_path)

    # ---- Check if Phase 2 already completed ----
    p2_done_path = os.path.join(ckpt_dir, "p2_done.pth")

    if os.path.exists(p2_done_path):
        logger.info(f"  [{name}] Phase 2: LOADING completed checkpoint...")
        p2_ckpt = torch.load(p2_done_path, map_location=device, weights_only=False)
        model.load_state_dict(p2_ckpt["model_state"])
        best_ft_val = p2_ckpt["best_ft_val"]
        logger.info(f"  [{name}] Phase 2 loaded: {best_ft_val:.2f}% FT val (skipped)")
    else:
        # ---- Phase 2: Fine-tune ----
        # Phase 2a: freeze backbone (5 epochs)
        for n, p in model.named_parameters():
            if "classifier" not in n and "fusion" not in n and "fc" not in n:
                p.requires_grad = False
        opt_ft = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.0005, weight_decay=0.001)
        for ep in range(5):
            train_epoch(model, ft_train_loader, opt_ft, criterion, device)

        # Phase 2b: unfreeze all
        for p in model.parameters():
            p.requires_grad = True
        opt_ft = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.001)
        sch_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=max_p2)

        best_ft_val, best_ft_state, patience = 0.0, None, 0
        for ep in range(max_p2):
            train_epoch(model, ft_train_loader, opt_ft, criterion, device)
            _, v_acc = eval_epoch(model, ft_val_loader, criterion, device)
            sch_ft.step()
            if v_acc > best_ft_val:
                best_ft_val = v_acc
                patience = 0
                best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if ep % 10 == 0:
                logger.info(f"    FT E{ep:02d} | Val: {v_acc:.1f}%")
            if patience >= 10:
                break

        model.load_state_dict(best_ft_state)
        logger.info(f"  [{name}] Phase 2 done: {best_ft_val:.2f}% FT val")

        # Save completed Phase 2
        torch.save({
            "model_state": best_ft_state,
            "best_ft_val": best_ft_val,
        }, p2_done_path)

    # ---- Test ----
    preds, labels = get_predictions(model, test_loader, device)
    test_acc = 100.0 * np.mean(preds == labels)
    params = count_parameters(model)

    # Inference time — synchronized so GPU/MPS ops complete before t1 is recorded.
    # Warmup eliminates JIT/kernel-launch spikes from the first forward pass.
    model.eval()
    sample_r = torch.randn(1, 1000).to(device)
    sample_s = torch.randn(1, 3, 128, 128).to(device)

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    # Warmup: 10 passes so kernel/JIT init costs are excluded from timings
    with torch.no_grad():
        for _ in range(10):
            model(sample_r, sample_s)
    _sync()

    times = []
    for _ in range(50):
        _sync()                          # wait: ensure previous ops are done
        t0 = time.time()
        with torch.no_grad():
            model(sample_r, sample_s)
        _sync()                          # wait: ensure THIS op is done before t1
        times.append(time.time() - t0)
    inf_ms = np.median(times) * 1000

    logger.info(f"  [{name}] Test: {test_acc:.2f}% | Params: {params:,} | Inf: {inf_ms:.1f}ms")
    return test_acc, best_val, best_ft_val, params, inf_ms


# ======================================================================
# EXPERIMENT RESULT CACHE
# ======================================================================

RESULTS_CACHE = os.path.join("outputs_reviewer", "results_cache.json")


def load_cached_results():
    """Load previously completed experiment results."""
    import json
    if os.path.exists(RESULTS_CACHE):
        with open(RESULTS_CACHE, "r") as f:
            return json.load(f)
    return {}


def save_result(key, result):
    """Save a single experiment result to cache."""
    import json
    cache = load_cached_results()
    cache[key] = result
    os.makedirs(os.path.dirname(RESULTS_CACHE), exist_ok=True)
    with open(RESULTS_CACHE, "w") as f:
        json.dump(cache, f, indent=2)
    logger.info(f"  💾 Result cached: {key}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="PINNACLE Reviewer Experiments")
    parser.add_argument("--exp", choices=[
        "all", "resnet", "wavelet", "flip",
        "transformer", "sharedgate", "scales", "bidir",
        "stft", "res224",
        "new",  # run only the 4 new experiments
    ], default="all", help="Which experiment to run")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached results exist")
    args = parser.parse_args()

    # Define which experiments each mode runs
    run_new = args.exp == "new"

    set_seed(42)
    device = get_device()

    results = load_cached_results()

    logger.info("=" * 70)
    logger.info("PINNACLE — Reviewer Experiments")
    logger.info("=" * 70)

    if results and not args.force:
        logger.info(f"  📂 Loaded {len(results)} cached results: {list(results.keys())}")

    # ==================================================================
    # EXPERIMENT 1: ResNet-1D
    # ==================================================================
    if args.exp in ("all", "resnet"):
        exp_key = "ResNet-1D"
        if exp_key in results and not args.force:
            logger.info(f"\n  ⏭️  {exp_key} already completed (test={results[exp_key]['test']:.2f}%), skipping")
        else:
            logger.info("\n" + "=" * 70)
            logger.info("EXP 1: ResNet-1D Baseline (modern deep 1D architecture)")
            logger.info("=" * 70)

            loaders = load_all_data()
            ref_train, ref_val, ft_train, ft_val, test_ld = loaders

            model = ResNet1D(num_classes=30, in_channels=1).to(device)
            logger.info(f"  ResNet-1D: {count_parameters(model):,} params")

            test_acc, p1_val, ft_val_acc, params, inf_ms = full_pipeline(
                model, ref_train, ref_val, ft_train, ft_val, test_ld,
                device, "ResNet-1D", max_p1=50, max_p2=30
            )
            results[exp_key] = {
                "test": test_acc, "p1_val": p1_val, "params": params, "inf_ms": inf_ms
            }
            save_result(exp_key, results[exp_key])

    # ==================================================================
    # EXPERIMENT 2: Wavelet Ablation
    # ==================================================================
    if args.exp in ("all", "wavelet"):
        logger.info("\n" + "=" * 70)
        logger.info("EXP 2: Wavelet Preprocessing Ablation (Morlet vs MexHat vs Bump)")
        logger.info("=" * 70)

        logger.info("  Step 1: Generating alternative wavelet scalograms...")
        generate_alt_wavelets()

        for wav_label, wav_suffix in [("Morlet (default)", ""),
                                       ("Mexican Hat", "_mexh"),
                                       ("Bump (Gauss1)", "_bump")]:
            exp_key = f"Wavelet: {wav_label}"
            if exp_key in results and not args.force:
                logger.info(f"\n  ⏭️  {exp_key} already completed (test={results[exp_key]['test']:.2f}%), skipping")
                continue

            logger.info(f"\n  Training with {wav_label} wavelet...")
            set_seed(42)

            loaders = load_all_data(wavelet_suffix=wav_suffix)
            ref_train, ref_val, ft_train, ft_val, test_ld = loaders

            model = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3,
                             use_fusion=True).to(device)

            test_acc, p1_val, ft_val_acc, params, inf_ms = full_pipeline(
                model, ref_train, ref_val, ft_train, ft_val, test_ld,
                device, f"PINNACLE-{wav_label}", max_p1=50, max_p2=30
            )
            results[exp_key] = {
                "test": test_acc, "p1_val": p1_val, "params": params, "inf_ms": inf_ms
            }
            save_result(exp_key, results[exp_key])

    # ==================================================================
    # EXPERIMENT 3: Attention Directionality
    # ==================================================================
    if args.exp in ("all", "flip"):
        exp_key = "Flipped (Q=scalo, KV=spec)"
        if exp_key in results and not args.force:
            logger.info(f"\n  ⏭️  {exp_key} already completed (test={results[exp_key]['test']:.2f}%), skipping")
        else:
            logger.info("\n" + "=" * 70)
            logger.info("EXP 3: Attention Directionality (Flip Q/K/V)")
            logger.info("=" * 70)

            set_seed(42)
            loaders = load_all_data()
            ref_train, ref_val, ft_train, ft_val, test_ld = loaders

            model = PINNACLEFlipped(num_classes=30, embed_dim=128, dropout=0.3).to(device)

            test_acc, p1_val, ft_val_acc, params, inf_ms = full_pipeline(
                model, ref_train, ref_val, ft_train, ft_val, test_ld,
                device, "PINNACLE-Flipped", max_p1=50, max_p2=30
            )
            results[exp_key] = {
                "test": test_acc, "p1_val": p1_val, "params": params, "inf_ms": inf_ms
            }
            save_result(exp_key, results[exp_key])

    # ==================================================================
    # EXPERIMENT 4: Transformer-1D Baseline
    # ==================================================================
    if args.exp in ("all", "transformer", "new") or run_new:
        exp_key = "Transformer-1D"
        if exp_key in results and not args.force:
            logger.info(f"\n  ⏭️  {exp_key} already completed (test={results[exp_key]['test']:.2f}%), skipping")
        else:
            logger.info("\n" + "=" * 70)
            logger.info("EXP 4: Transformer-1D Baseline (spectral-only ViT)")
            logger.info("=" * 70)

            set_seed(42)
            loaders = load_all_data()
            ref_train, ref_val, ft_train, ft_val, test_ld = loaders

            model = Transformer1D(
                num_classes=30, seq_len=1000, patch_size=20,
                embed_dim=128, num_heads=4, num_layers=4, dropout=0.1
            ).to(device)

            test_acc, p1_val, ft_val_acc, params, inf_ms = full_pipeline(
                model, ref_train, ref_val, ft_train, ft_val, test_ld,
                device, "Transformer-1D", max_p1=50, max_p2=30
            )
            results[exp_key] = {
                "test": test_acc, "p1_val": p1_val, "params": params, "inf_ms": inf_ms
            }
            save_result(exp_key, results[exp_key])

    # ==================================================================
    # EXPERIMENT 5: Shared vs Independent Gating
    # ==================================================================
    if args.exp in ("all", "sharedgate", "new") or run_new:
        exp_key = "Shared Gating"
        if exp_key in results and not args.force:
            logger.info(f"\n  ⏭️  {exp_key} already completed (test={results[exp_key]['test']:.2f}%), skipping")
        else:
            logger.info("\n" + "=" * 70)
            logger.info("EXP 5: Shared Gating Ablation (joint gate vs independent)")
            logger.info("=" * 70)

            set_seed(42)
            loaders = load_all_data()
            ref_train, ref_val, ft_train, ft_val, test_ld = loaders

            model = PINNACLESharedGating(num_classes=30, embed_dim=128, dropout=0.3).to(device)

            test_acc, p1_val, ft_val_acc, params, inf_ms = full_pipeline(
                model, ref_train, ref_val, ft_train, ft_val, test_ld,
                device, "PINNACLE-SharedGating", max_p1=50, max_p2=30
            )
            results[exp_key] = {
                "test": test_acc, "p1_val": p1_val, "params": params, "inf_ms": inf_ms
            }
            save_result(exp_key, results[exp_key])

    # ==================================================================
    # EXPERIMENT 6: Wavelet Scale Range Ablation
    # ==================================================================
    if args.exp in ("all", "scales", "new") or run_new:
        logger.info("\n" + "=" * 70)
        logger.info("EXP 6: Wavelet Scale Range Ablation (128 vs 256 vs 512)")
        logger.info("=" * 70)

        logger.info("  Step 1: Generating scale-variant wavelets...")
        generate_scale_wavelets()

        for n_scales, wav_suffix in [(128, "_s128"), (256, ""), (512, "_s512")]:
            exp_key = f"Scales: {n_scales}"
            if exp_key in results and not args.force:
                logger.info(f"\n  ⏭️  {exp_key} already completed (test={results[exp_key]['test']:.2f}%), skipping")
                continue

            if n_scales == 256:
                logger.info(f"\n  Scales=256 is same as Wavelet:Morlet — reusing result...")
                morlet_key = "Wavelet: Morlet (default)"
                if morlet_key in results:
                    results[exp_key] = results[morlet_key].copy()
                    save_result(exp_key, results[exp_key])
                    continue

            logger.info(f"\n  Training with {n_scales} scales...")
            set_seed(42)

            loaders = load_all_data(wavelet_suffix=wav_suffix)
            ref_train, ref_val, ft_train, ft_val, test_ld = loaders

            model = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3,
                             use_fusion=True).to(device)

            test_acc, p1_val, ft_val_acc, params, inf_ms = full_pipeline(
                model, ref_train, ref_val, ft_train, ft_val, test_ld,
                device, f"PINNACLE-Scales{n_scales}", max_p1=50, max_p2=30
            )
            results[exp_key] = {
                "test": test_acc, "p1_val": p1_val, "params": params, "inf_ms": inf_ms
            }
            save_result(exp_key, results[exp_key])

    # ==================================================================
    # EXPERIMENT 7: Bidirectional Attention
    # ==================================================================
    if args.exp in ("all", "bidir", "new") or run_new:
        exp_key = "Bidirectional"
        if exp_key in results and not args.force:
            logger.info(f"\n  ⏭️  {exp_key} already completed (test={results[exp_key]['test']:.2f}%), skipping")
        else:
            logger.info("\n" + "=" * 70)
            logger.info("EXP 7: Bidirectional Cross-Attention")
            logger.info("=" * 70)

            set_seed(42)
            loaders = load_all_data()
            ref_train, ref_val, ft_train, ft_val, test_ld = loaders

            model = PINNACLEBidirectional(num_classes=30, embed_dim=128, dropout=0.3).to(device)

            test_acc, p1_val, ft_val_acc, params, inf_ms = full_pipeline(
                model, ref_train, ref_val, ft_train, ft_val, test_ld,
                device, "PINNACLE-Bidirectional", max_p1=50, max_p2=30
            )
            results[exp_key] = {
                "test": test_acc, "p1_val": p1_val, "params": params, "inf_ms": inf_ms
            }
            save_result(exp_key, results[exp_key])

    # ==================================================================
    # EXPERIMENT 8: STFT Ablation
    # ==================================================================
    if args.exp in ("all", "stft", "new") or run_new:
        exp_key = "2D Representation: STFT"
        if exp_key in results and not args.force:
            logger.info(f"\n  ⏭️  {exp_key} already completed (test={results[exp_key]['test']:.2f}%), skipping")
        else:
            logger.info("\n" + "=" * 70)
            logger.info("EXP 8: 2D Representation Ablation (STFT vs CWT)")
            logger.info("=" * 70)

            set_seed(42)
            loaders = load_all_data(wavelet_suffix="_stft")
            ref_train, ref_val, ft_train, ft_val, test_ld = loaders

            # Use standard PINNACLE
            model = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, use_fusion=True).to(device)

            test_acc, p1_val, ft_val_acc, params, inf_ms = full_pipeline(
                model, ref_train, ref_val, ft_train, ft_val, test_ld,
                device, "PINNACLE-STFT", max_p1=50, max_p2=30
            )
            results[exp_key] = {
                "test": test_acc, "p1_val": p1_val, "params": params, "inf_ms": inf_ms
            }
            save_result(exp_key, results[exp_key])

    # ==================================================================
    # EXPERIMENT 9: Resolution Ablation (224x224)
    # ==================================================================
    if args.exp in ("all", "res224", "new") or run_new:
        exp_key = "Resolution: 224x224"
        if exp_key in results and not args.force:
            logger.info(f"\n  ⏭️  {exp_key} already completed (test={results[exp_key]['test']:.2f}%), skipping")
        else:
            logger.info("\n" + "=" * 70)
            logger.info("EXP 9: Resolution Ablation (224x224 vs 128x128)")
            logger.info("=" * 70)

            set_seed(42)
            loaders = load_all_data(wavelet_suffix="_224")
            ref_train, ref_val, ft_train, ft_val, test_ld = loaders

            # Use standard PINNACLE
            model = PINNACLE(num_classes=30, embed_dim=128, dropout=0.3, use_fusion=True).to(device)

            test_acc, p1_val, ft_val_acc, params, inf_ms = full_pipeline(
                model, ref_train, ref_val, ft_train, ft_val, test_ld,
                device, "PINNACLE-Res224", max_p1=50, max_p2=30
            )
            results[exp_key] = {
                "test": test_acc, "p1_val": p1_val, "params": params, "inf_ms": inf_ms
            }
            save_result(exp_key, results[exp_key])

    # ==================================================================
    # FINAL RESULTS TABLE
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS — All Reviewer Experiments")
    logger.info("=" * 70)
    logger.info(f"  {'Experiment':<35} {'P1 Val':>8} {'Test':>8} {'Params':>10} {'Inf(ms)':>8}")
    logger.info(f"  {'-'*71}")

    # Reference: PINNACLE baseline
    logger.info(f"  {'PINNACLE (SeparationCross) [ref]':<35} {'92.20%':>8} {'78.00%':>8} {'304K':>10} {'—':>8}")

    for name, r in results.items():
        logger.info(f"  {name:<35} {r['p1_val']:>7.2f}% {r['test']:>7.2f}% {r['params']:>10,} {r['inf_ms']:>7.1f}")

    logger.info("=" * 70)
    logger.info("Done! Copy these numbers into draft.tex tables.")


if __name__ == "__main__":
    main()

