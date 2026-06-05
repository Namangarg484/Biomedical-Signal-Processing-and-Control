#!/usr/bin/env python3
"""
PINNACLE — Robustness and Gating Baselines (5-Class Bacteria-ID, identical split)

Two additional baselines completing the ablation suite:

  Baseline 4 — Raman-only + Heavy Augmentation
      SpectralBranch + FC classifier with aggressive in-loop augmentation:
      Gaussian noise, per-sample spectral shift, linear baseline drift.
      This is the strongest possible unimodal + augmentation baseline.
      Goal: show that PINNACLE's dual-stream CWT fusion beats data
      augmentation alone.

  Baseline 5 — Gated Multimodal Fusion (GMF, shared gating)
      Standard GMF: a single gate vector g is computed from
      concat([z_s, z_w]).  Identical SpectralBranch + ScalogramBranch
      encoders as PINNACLE; only the fusion module differs.
      Goal: prove the "Separation Principle" — PINNACLE's per-modality
      gates (α from z_s alone, β from z_w alone) outperform shared gating.

  Asymmetric Noise Test  (Section 4.4.2 — Separation Principle)
      Gaussian noise (default 20 dB SNR) is injected ONLY into the
      Raman input; the CWT scalogram is passed clean.
      In GMF, a corrupted z_s taints the shared gate g, which then
      misweights the clean z_w contribution.
      In PINNACLE, β = σ(W_β · z_w) is computed from z_w alone and is
      unaffected by spectral noise — demonstrating robustness via
      separation.

Usage:
    # Run both baselines sequentially:
    cd /path/to/Raman
    python scripts/robustness_baselines.py

    # Run one baseline:
    python scripts/robustness_baselines.py --model aug_raman
    python scripts/robustness_baselines.py --model gmf

    # Re-train ignoring cached checkpoints:
    python scripts/robustness_baselines.py --no-cache

    # Custom noise level for asymmetric test:
    python scripts/robustness_baselines.py --snr-db 15

Output:
    Clean-accuracy table printed to stdout.
    Asymmetric noise results printed at the end for paper Section 4.4.2.
    Checkpoints → outputs/robustness_baselines/<model>/best.pt
    Predictions → outputs/robustness_baselines/<model>/predictions.npz
    JSON summary → outputs/robustness_baselines/robustness_results.json
"""

import os
import sys
import json
import argparse

# Add project root (one level up) AND scripts/ itself to sys.path so that
# both `pinnacle.*` and `sota_baselines` (in the same scripts/ dir) are
# importable without packaging either as a namespace package.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pinnacle.utils import set_seed, get_device, logger, count_parameters
from pinnacle.model import SpectralBranch, ScalogramBranch, PINNACLE

# Reuse the shared infrastructure from sota_baselines.py (same file, scripts/)
from sota_baselines import (
    CFG,
    load_5class_data,
    make_loaders,
    run_training,
    load_pinnacle_predictions,
    wilson_ci,
    mcnemar_pvalue,
)


# =========================================================================
# Baseline 4 — Raman-only + Heavy Augmentation
# =========================================================================

class HeavyRamanAugModel(nn.Module):
    """
    Standalone Raman CNN classifier with aggressive in-loop augmentation.

    IMPORTANT — why we do NOT reuse SpectralBranch directly:
    SpectralBranch has no intermediate striding (all conv layers use
    padding='same').  When trained standalone the global pool collapses
    L=1000 positions to 1, giving a gradient dilution factor of 1/1000
    through the pool — 31× worse than ResNet-18-1D (1/32).  In PINNACLE
    the same branch works because cross-attention pools it to 25 positions
    first (1/25 dilution).  For a standalone classifier we add one
    MaxPool1d(2) per conv block (3 total), reducing 1000→125 (1/125
    dilution) before the final pool.

    Augmentation (training only):
      1. Gaussian noise  std=noise_std  (default 0.05)
      2. Per-sample spectral shift  ±max_shift  (vectorised gather, no loop)
      3. Random linear baseline drift  ±0.03 amplitude

    Parameters: ~0.28 M
    """

    def __init__(
        self,
        num_classes: int = 5,
        embed_dim: int = 128,
        dropout: float = 0.3,
        noise_std: float = 0.05,
        max_shift: int = 10,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.max_shift = max_shift

        # Downsampling CNN backbone (1000 → 500 → 250 → 125 via MaxPool)
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),   # 1000 → 500

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),   # 500 → 250

            nn.Conv1d(128, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),   # 250 → 125  ← gradient dilution now 1/125
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    # ------------------------------------------------------------------
    # Internal augmentation helpers
    # ------------------------------------------------------------------

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorised heavy augmentation applied to a batch of Raman spectra.

        Uses torch.gather for the shift step (no Python for loop) to avoid
        potential MPS autograd issues with per-sample tensor indexing.

        Args:
            x: (B, L) float tensor on device
        Returns:
            x_aug: (B, L) augmented tensor (same device)
        """
        B, L = x.shape
        device = x.device

        # Step 1: Additive Gaussian noise
        x = x + torch.randn_like(x) * self.noise_std

        # Step 2: Per-sample spectral shift via batched gather (circular roll).
        # Positive shift = right-shift; negative = left-shift.
        # Circular boundary effect is negligible: |Δ| ≤ 10 / 1000 = 1%.
        shifts = torch.randint(
            -self.max_shift, self.max_shift + 1, (B,), device=device
        )                                               # (B,)
        pos = (
            torch.arange(L, device=device).unsqueeze(0)  # (1, L)
            - shifts.unsqueeze(1)                         # (B, 1)
        ) % L                                             # (B, L)
        x = x.gather(1, pos)                             # (B, L)

        # Step 3: Random linear baseline drift
        t = torch.linspace(0.0, 1.0, L, device=device)           # (L,)
        drift_amp = (torch.rand(B, device=device) - 0.5) * 0.06  # (B,)
        x = x + drift_amp.unsqueeze(1) * t.unsqueeze(0)          # (B, L)

        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, raman: torch.Tensor, scalogram=None):
        """
        Args:
            raman:     (B, L) preprocessed Raman spectrum
            scalogram: any — ignored (unimodal baseline)
        Returns:
            logits: (B, num_classes), None, None
        """
        x = raman
        if x.dim() == 2:
            x = x.unsqueeze(1)                      # (B, 1, L)
        # Per-spectrum z-score normalisation BEFORE augmentation.
        # X_2018_proc std≈6e-4 << noise_std=0.05 would be 83× signal without
        # this step, completely destroying the training signal.  After
        # normalisation std=1, so noise_std=0.05 is correctly 5% of signal.
        # Also fixes the train/val BN mismatch (see KirchhoffNet.forward).
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        if self.training:
            x = self._augment(x.squeeze(1)).unsqueeze(1)  # augment at unit scale

        h = self.features(x)                        # (B, embed_dim, 125)
        z = self.pool(h).squeeze(-1)                # (B, embed_dim)
        return self.classifier(z), None, None


# =========================================================================
# Baseline 5 — Gated Multimodal Fusion  (GMF, shared gating)
# =========================================================================

class SharedGateFusion(nn.Module):
    """
    Standard Gated Multimodal Fusion with a single shared gate vector.

    Gate computation:
        g = σ( W_g · concat([z_s, z_w]) + b_g )   ∈ ℝ^D

    Fused representation (same output dim 2D as PINNACLE for fair comparison):
        z_fused = concat( g ⊙ z_s ,  (1-g) ⊙ z_w )   ∈ ℝ^{2D}

    Key weakness (the "Separation Principle"):
    Because g is computed jointly from z_s and z_w, corrupting z_s
    (e.g. spectral noise) directly corrupts g and in turn degrades the
    scalogram contribution — even when the scalogram is perfectly clean.
    PINNACLE avoids this by using decoupled gates:
        α = σ(W_α · z_s)   (spectral gate — unaffected by scalogram noise)
        β = σ(W_β · z_w)   (scalogram gate — unaffected by spectral noise)
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        # Gate projects 2D → D; the shared input means spectral noise
        # directly contaminates the gate signal for the scalogram branch.
        self.gate = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, z_s: torch.Tensor, z_w: torch.Tensor):
        """
        Args:
            z_s: (B, D) pooled spectral embedding
            z_w: (B, D) pooled scalogram embedding
        Returns:
            z_fused: (B, 2D)
            g:   (B, D) shared gate (returned as 'alpha' for API compat.)
            1-g: (B, D)            (returned as 'beta'  for API compat.)
        """
        g = torch.sigmoid(
            self.gate(torch.cat([z_s, z_w], dim=-1))           # (B, D)
        )
        z_fused = torch.cat([g * z_s, (1.0 - g) * z_w], dim=-1)  # (B, 2D)
        return z_fused, g, (1.0 - g)


class GMFModel(nn.Module):
    """
    Dual-stream CNN + Gated Multimodal Fusion (shared gate).

    Identical SpectralBranch + ScalogramBranch encoders as PINNACLE.
    The only difference vs PINNACLE is the fusion module:
        PINNACLE  → SeparationCross  (per-modality gates α, β + cross-attn)
        GMFModel  → SharedGateFusion (single shared gate g, no cross-attn)

    This is the correct ablation to isolate the "Separation Principle":
    same data, same encoders, same classifier head — only the gating
    strategy changes.

    Parameters: ~0.56 M
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
        self.fusion = SharedGateFusion(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, raman: torch.Tensor, scalogram: torch.Tensor):
        """
        Args:
            raman:     (B, L) Raman spectrum
            scalogram: (B, 3, H, W) CWT scalogram
        Returns:
            logits: (B, num_classes),  g (alpha),  1-g (beta)

        NOTE: GMF operates on POOLED embeddings (z_s, z_w), not unpooled
        feature maps.  SharedGateFusion.forward() therefore receives
        (B, D) vectors, not spatial tensors — this is intentional.
        Cross-attention over spatial positions is a PINNACLE-specific
        component and is deliberately absent here.
        """
        z_s = self.spectral_branch(raman)          # (B, D) — pooled via branch.forward()
        z_w = self.scalogram_branch(scalogram)     # (B, D) — pooled via branch.forward()
        z_fused, alpha, beta = self.fusion(z_s, z_w)
        return self.classifier(z_fused), alpha, beta


# =========================================================================
# Asymmetric noise evaluation
# =========================================================================

@torch.no_grad()
def eval_asymmetric_noise(
    model: nn.Module,
    test_loader,
    device: torch.device,
    snr_db: float = 20.0,
):
    """
    Evaluate model accuracy when Gaussian noise is injected ONLY into the
    Raman (1D spectral) input.  The CWT scalogram is passed clean.

    Noise is scaled per-sample to achieve the requested SNR:
        SNR (dB) = 10 log10( P_signal / P_noise )
        noise_std_i = sqrt( mean(x_i^2) / 10^(snr_db / 10) )

    This models a realistic degradation scenario (e.g. fluorescence
    interference, shot noise, detector drift) that affects the 1D
    spectrum but leaves the CWT representation intact.

    Expected GMF behaviour: accuracy drops significantly because the
    corrupted z_s taints the shared gate g.
    Expected PINNACLE behaviour: more robust because β = σ(W_β · z_w)
    is computed from the clean scalogram alone and remains unaffected.

    Args:
        model:       trained model with forward(raman, scalogram) API
        test_loader: DataLoader for the test split (must have scalograms)
        device:      torch device
        snr_db:      noise level in dB (default 20 dB ≈ ×10 amplitude ratio)

    Returns:
        accuracy (float, %),
        predictions (np.ndarray, shape [N]),
        labels     (np.ndarray, shape [N])
    """
    model.eval()
    preds_list, labels_list = [], []

    for raman, scalogram, labels in test_loader:
        raman = raman.to(device)
        scalogram = scalogram.to(device)

        # Per-sample noise level matched to individual signal power
        signal_power = raman.pow(2).mean(dim=-1, keepdim=True)       # (B, 1)
        noise_std = torch.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
        raman_noisy = raman + torch.randn_like(raman) * noise_std     # (B, L)

        logits, _, _ = model(raman_noisy, scalogram)
        preds_list.append(logits.argmax(1).cpu().numpy())
        labels_list.append(labels.numpy())

    preds = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)
    acc = (preds == labels).mean() * 100.0
    return acc, preds, labels


# =========================================================================
# Helper: load trained PINNACLE for the noise test
# =========================================================================

def load_pinnacle_for_noise_test(device: torch.device, cfg: dict) -> nn.Module:
    """
    Instantiate PINNACLE (fusion mode) and load weights from the standard
    checkpoint location produced by scripts/train.py.

    Returns the model (on `device`).  If no checkpoint is found, returns
    the randomly-initialised model and logs a warning — results will not
    be meaningful for the paper in that case.
    """
    ckpt_candidates = [
        os.path.join("outputs", "checkpoints", "best_model.pth"),
        os.path.join("outputs", "checkpoints", "best.pt"),
        os.path.join("outputs", "best.pt"),
        os.path.join("outputs_30class", "checkpoints", "best.pt"),
    ]
    model = PINNACLE(
        num_classes=cfg["num_classes"],
        embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"],
        mode="fusion",
    ).to(device)

    for path in ckpt_candidates:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(
                    f"  PINNACLE ckpt: {len(missing)} missing keys "
                    f"(first 3: {missing[:3]})"
                )
            logger.info(f"  PINNACLE loaded from: {path}")
            return model

    logger.warning(
        "  PINNACLE checkpoint not found — asymmetric noise results "
        "will not be meaningful.  Run `python scripts/train.py` first."
    )
    return model


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "PINNACLE — Robustness baselines: "
            "Raman-only + HeavyAug (Baseline 4) and GMF shared-gating "
            "(Baseline 5), plus Asymmetric Noise Test."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "aug_raman", "gmf"],
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
        "--snr-db",
        type=float,
        default=20.0,
        help="SNR in dB for the asymmetric noise test (default: 20 dB).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'mps' (default: auto).",
    )
    args = parser.parse_args()

    cfg = {**CFG}
    cfg["data_dir"] = args.data_dir
    cfg["epochs"] = args.epochs

    set_seed(cfg["seed"])
    device = get_device(args.device)

    logger.info("=" * 60)
    logger.info("PINNACLE — Robustness Baselines")
    logger.info("=" * 60)
    logger.info(f"  Device   : {device}")
    logger.info(f"  Epochs   : {cfg['epochs']}")
    logger.info(f"  Data dir : {cfg['data_dir']}")
    logger.info(f"  Seed     : {cfg['seed']}")
    logger.info(f"  SNR (asymmetric noise test): {args.snr_db} dB")

    # ------------------------------------------------------------------
    # Load data — identical 80/10/10 split as PINNACLE paper §4.2
    # ------------------------------------------------------------------
    splits, has_scalograms = load_5class_data(
        cfg["data_dir"], cfg["seed"], cfg["test_size"], cfg["val_size"]
    )

    # Loaders without scalograms (for HeavyRamanAugModel — unimodal)
    loaders_1d = make_loaders(
        splits, cfg["batch_size"], cfg["num_workers"], use_scalograms=False
    )
    # Loaders with scalograms (for GMFModel and asymmetric noise test)
    loaders_2d = (
        make_loaders(splits, cfg["batch_size"], cfg["num_workers"], use_scalograms=True)
        if has_scalograms
        else None
    )

    out_dir = os.path.join("outputs", "robustness_baselines")
    os.makedirs(out_dir, exist_ok=True)

    run_all = args.model == "all"
    results = []

    # ------------------------------------------------------------------
    # Baseline 4: Raman-only + Heavy Augmentation
    # ------------------------------------------------------------------
    if run_all or args.model == "aug_raman":
        model = HeavyRamanAugModel(
            num_classes=cfg["num_classes"],
            embed_dim=cfg["embed_dim"],
            dropout=cfg["dropout"],
        )
        r = run_training(
            model_name="aug_raman_only",           # filesystem-safe name
            model=model,
            loaders=loaders_1d,
            device=device,
            cfg=cfg,
            out_dir=out_dir,
            no_cache=args.no_cache,
        )
        r["display_name"] = "Raman-only + HeavyAug"
        results.append(r)

    # ------------------------------------------------------------------
    # Baseline 5: GMF — Gated Multimodal Fusion (shared gating)
    # ------------------------------------------------------------------
    if run_all or args.model == "gmf":
        if not has_scalograms:
            logger.warning(
                "  Skipping GMF: CWT scalograms not found in "
                f"{cfg['data_dir']}.\n"
                "  Run `python scripts/generate_wavelets.py` to generate them."
            )
        else:
            model = GMFModel(
                num_classes=cfg["num_classes"],
                embed_dim=cfg["embed_dim"],
                dropout=cfg["dropout"],
            )
            r = run_training(
                model_name="gmf_shared_gating",    # filesystem-safe name
                model=model,
                loaders=loaders_2d,
                device=device,
                cfg=cfg,
                out_dir=out_dir,
                no_cache=args.no_cache,
            )
            r["display_name"] = "GMF (shared gating)"
            results.append(r)

    if not results:
        logger.error("No results — nothing ran. Check your --model flag.")
        return

    # ------------------------------------------------------------------
    # Print clean-accuracy summary
    # ------------------------------------------------------------------
    pinnacle_preds, pinnacle_labels = load_pinnacle_predictions()

    print("\n" + "=" * 60)
    print("Robustness Baselines — Clean Accuracy")
    print("=" * 60)
    for r in results:
        name = r.get("display_name", r["model_name"])
        ci = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
        pval_str = "N/A"
        if (
            pinnacle_preds is not None
            and np.array_equal(r["labels"], pinnacle_labels)
        ):
            pval = mcnemar_pvalue(r["predictions"], pinnacle_preds, r["labels"])
            pval_str = f"{pval:.4f}"
        print(
            f"  {name:<35}  {r['accuracy']:.2f}%  "
            f"CI: {ci}  params: {r['num_params']:,}  "
            f"McNemar p={pval_str}"
        )

    # ------------------------------------------------------------------
    # Asymmetric noise test — Separation Principle (Section 4.4.2)
    # ------------------------------------------------------------------
    if has_scalograms and loaders_2d is not None:
        print("\n" + "=" * 60)
        print(
            f"Asymmetric Noise Test  "
            f"(SNR = {args.snr_db} dB, noise in Raman branch only)"
        )
        print("=" * 60)

        # --- Test GMF under asymmetric noise ---
        gmf_ckpt_path = os.path.join(out_dir, "gmf_shared_gating", "best.pt")
        gmf_noisy_acc = None

        if os.path.exists(gmf_ckpt_path):
            gmf_noise_model = GMFModel(
                num_classes=cfg["num_classes"],
                embed_dim=cfg["embed_dim"],
                dropout=cfg["dropout"],
            ).to(device)
            ckpt = torch.load(gmf_ckpt_path, map_location=device)
            gmf_noise_model.load_state_dict(ckpt["state_dict"])

            gmf_noisy_acc, _, _ = eval_asymmetric_noise(
                gmf_noise_model, loaders_2d["test"], device, snr_db=args.snr_db
            )
            gmf_clean = next(
                (r["accuracy"] for r in results if "gmf" in r["model_name"]),
                None,
            )
            if gmf_clean is not None:
                drop = gmf_clean - gmf_noisy_acc
                print(
                    f"  GMF (shared gating):       "
                    f"clean={gmf_clean:.2f}%  "
                    f"noisy={gmf_noisy_acc:.2f}%  "
                    f"drop={drop:.2f} pp"
                )
            else:
                print(f"  GMF (shared gating):       noisy={gmf_noisy_acc:.2f}%")
        else:
            logger.warning(
                f"  GMF checkpoint not found at {gmf_ckpt_path}.\n"
                "  Run with --model gmf (or --model all) first."
            )

        # --- Test PINNACLE under asymmetric noise ---
        pinnacle_noise_model = load_pinnacle_for_noise_test(device, cfg)
        pinn_noisy_acc, _, _ = eval_asymmetric_noise(
            pinnacle_noise_model, loaders_2d["test"], device, snr_db=args.snr_db
        )
        print(f"  PINNACLE (SeparationCross): noisy={pinn_noisy_acc:.2f}%")

        if gmf_noisy_acc is not None:
            gap = pinn_noisy_acc - gmf_noisy_acc
            print(f"\n  Robustness advantage (PINNACLE − GMF): {gap:+.2f} pp")

        print(
            "\n  → Copy the 'clean' and 'noisy' numbers above into\n"
            "    the 'Validation of the Separation Principle' paragraph\n"
            "    in Section 4.4.2 of draft.tex."
        )

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    summary = {}
    for r in results:
        key = r.get("display_name", r["model_name"])
        summary[key] = {
            "accuracy": r["accuracy"],
            "ci": [r["ci_lo"], r["ci_hi"]],
            "num_params": r["num_params"],
            "infer_ms": r["infer_ms"],
        }

    summary_path = os.path.join(out_dir, "robustness_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n  Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
