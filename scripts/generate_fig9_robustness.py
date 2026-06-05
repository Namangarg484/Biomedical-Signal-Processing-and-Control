#!/usr/bin/env python3
"""
Generate Figure 9: Robustness under noise and spectral occlusion.

Panel A — Noise Robustness (SNR sweep)
    Compares PINNACLE (full fusion) against the trained Raman-only baseline
    (HeavyRamanAugModel, 82.6% clean) at SNR = clean, 40, 30, 20 dB.
    Gaussian noise is added per-sample to the 1D Raman input only.
    PINNACLE's CWT scalogram inputs are pre-computed and unaffected,
    demonstrating the SeparationCross principle: the scalogram gate β is
    independent of Raman noise.

Panel B — Sliding-window Occlusion
    A 50 cm⁻¹ window of the Raman spectrum is zeroed and accuracy is
    measured vs. window-centre wavenumber.  PINNACLE vs. Raman-only.
    Shows which spectral regions are causally necessary and whether
    PINNACLE's CWT branch compensates for Raman occlusion.

Both evaluated on the same 1,250-sample test set (seed=42).
"""

import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_SCRIPTS)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SCRIPTS)

from pinnacle.utils import set_seed, get_device, logger
from pinnacle.model import PINNACLE
from pinnacle.dataset import load_data, create_dataloaders
from robustness_baselines import HeavyRamanAugModel   # trained Raman-only model

# ── Config ────────────────────────────────────────────────────────────────────
SEED           = 42
DATA_DIR       = "data"
CKPT_PINN      = "outputs/checkpoints/best_model.pth"
CKPT_RAMAN     = "outputs/robustness_baselines/aug_raman_only/best.pt"
OUT_PNG        = "outputs/figures/fig10_robustness.png"
OUT_PDF        = "outputs/figures/fig10_robustness.pdf"
FIG_DIR        = "figures"
SNR_LEVELS     = [None, 40, 30, 20]
SNR_LABELS     = ["Clean", "40 dB", "30 dB", "20 dB"]
OCCLUSION_W    = 50    # cm⁻¹ window width
OCCLUSION_STEP = 40    # cm⁻¹ step between window centres

COLORS = {"PINNACLE": "#1f77b4", "Raman-only": "#ff7f0e"}

BANDS = [
    (785,  875,  "Nucleic\nacids",  "#e74c3c"),
    (1000, 1100, "Lipids\nC–C",     "#27ae60"),
    (1200, 1350, "Amide III",       "#2980b9"),
    (1600, 1700, "Amide I",         "#e67e22"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def add_noise_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    sig_pwr = x.pow(2).mean(dim=1, keepdim=True)
    std     = torch.sqrt(sig_pwr / (10.0 ** (snr_db / 10.0)))
    return x + torch.randn_like(x) * std


@torch.no_grad()
def eval_model(model, loader, device,
               snr_db=None, occ_centre=None, wavenumbers=None,
               is_raman_only=False):
    """Evaluate model accuracy under optional noise/occlusion."""
    correct, total = 0, 0
    for batch in loader:
        if len(batch) == 3:
            raman, scalogram, labels = batch
        else:
            raman, labels = batch; scalogram = None

        raman  = raman.to(device)
        labels = labels.to(device)
        if scalogram is not None:
            scalogram = scalogram.to(device)

        # Apply noise to Raman
        if snr_db is not None:
            raman = add_noise_snr(raman, snr_db)

        # Apply occlusion to Raman (use tensor multiply to stay MPS-safe)
        if occ_centre is not None and wavenumbers is not None:
            lo, hi = occ_centre - OCCLUSION_W/2, occ_centre + OCCLUSION_W/2
            mask_np = (wavenumbers >= lo) & (wavenumbers <= hi)
            if mask_np.any():
                keep = torch.from_numpy((~mask_np).astype(np.float32)).to(device)
                raman = raman * keep  # zero out occluded region

        if is_raman_only:
            # HeavyRamanAugModel returns (logits, None, None)
            out = model(raman)
            logits = out[0] if isinstance(out, tuple) else out
        else:
            # PINNACLE returns (logits, alpha, beta)
            logits, _, _ = model(raman, scalogram)

        correct += (logits.argmax(1) == labels).sum().item()
        total   += labels.size(0)

    # Free MPS cache to prevent memory accumulation across loop iterations
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    return 100.0 * correct / total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    device = get_device()
    logger.info("=" * 65)
    logger.info("PINNACLE — Fig 9: Noise Robustness + Occlusion")
    logger.info("=" * 65)

    # Load data ----------------------------------------------------------------
    data = load_data(DATA_DIR, remap=True)
    _, _, test_loader, _ = create_dataloaders(
        data, batch_size=256, seed=SEED, num_workers=0, use_augmentation=False
    )
    wavenumbers = data["wavenumbers"].astype(np.float32)

    # Load PINNACLE ------------------------------------------------------------
    ckpt_p = torch.load(CKPT_PINN, map_location="cpu", weights_only=False)
    pinn   = PINNACLE(num_classes=5, embed_dim=128, dropout=0.3, mode="fusion")
    pinn.load_state_dict(ckpt_p["model_state"])
    pinn   = pinn.to(device).eval()
    logger.info(f"  PINNACLE loaded — epoch {ckpt_p.get('epoch','?')}, "
                f"best val {ckpt_p.get('best_val_acc','?'):.2f}%")

    # Load Raman-only (HeavyRamanAugModel) ------------------------------------
    ckpt_r  = torch.load(CKPT_RAMAN, map_location="cpu", weights_only=False)
    raman_m = HeavyRamanAugModel(num_classes=5, embed_dim=128, dropout=0.3)
    raman_m.load_state_dict(ckpt_r["state_dict"])
    raman_m = raman_m.to(device).eval()
    logger.info(f"  Raman-only loaded — epoch {ckpt_r.get('epoch','?')}, "
                f"val acc {ckpt_r.get('val_acc','?'):.2f}%")

    # ── Panel A: SNR sweep ────────────────────────────────────────────────────
    logger.info("  SNR sweep …")
    snr_pinn, snr_raman = [], []
    for snr in SNR_LEVELS:
        tag = "clean" if snr is None else f"{snr} dB"
        logger.info(f"    {tag}")
        snr_pinn.append(eval_model(pinn,   test_loader, device, snr_db=snr))
        snr_raman.append(eval_model(raman_m, test_loader, device, snr_db=snr,
                                    is_raman_only=True))
    logger.info(f"  PINNACLE  : {[f'{v:.1f}%' for v in snr_pinn]}")
    logger.info(f"  Raman-only: {[f'{v:.1f}%' for v in snr_raman]}")

    # ── Panel B: Occlusion sweep ──────────────────────────────────────────────
    logger.info("  Occlusion sweep …")
    wn_min, wn_max = float(wavenumbers.min()), float(wavenumbers.max())
    centres = np.arange(wn_min + OCCLUSION_W/2,
                        wn_max - OCCLUSION_W/2,
                        OCCLUSION_STEP, dtype=np.float32)

    clean_pinn  = snr_pinn[0]
    clean_raman = snr_raman[0]
    occ_pinn, occ_raman = [], []
    pbar = tqdm(centres, desc="  Occlusion", unit="win", ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for c in pbar:
        p_acc = eval_model(pinn,   test_loader, device,
                           occ_centre=float(c), wavenumbers=wavenumbers)
        r_acc = eval_model(raman_m, test_loader, device,
                           occ_centre=float(c), wavenumbers=wavenumbers,
                           is_raman_only=True)
        occ_pinn.append(p_acc)
        occ_raman.append(r_acc)
        pbar.set_postfix(PINNACLE=f"{p_acc:.1f}%", Raman=f"{r_acc:.1f}%",
                         wn=f"{c:.0f}")
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    drop_pinn  = clean_pinn  - np.array(occ_pinn)
    drop_raman = clean_raman - np.array(occ_raman)
    best_p  = centres[drop_pinn.argmax()]
    best_r  = centres[drop_raman.argmax()]
    logger.info(f"  Max drop PINNACLE  : {drop_pinn.max():.1f} pp at {best_p:.0f} cm⁻¹")
    logger.info(f"  Max drop Raman-only: {drop_raman.max():.1f} pp at {best_r:.0f} cm⁻¹")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    # Panel A
    x = np.arange(len(SNR_LEVELS))
    for vals, name, ls in [(snr_pinn, "PINNACLE", "-"), (snr_raman, "Raman-only", "--")]:
        ax1.plot(x, vals, marker="o", lw=2.2, ms=8,
                 color=COLORS[name], linestyle=ls, label=name)
        ax1.annotate(f"{vals[-1]:.1f}%", (x[-1], vals[-1]),
                     textcoords="offset points", xytext=(8, -2),
                     fontsize=8, color=COLORS[name])

    ax1.set_xticks(x); ax1.set_xticklabels(SNR_LABELS, fontsize=9)
    ax1.set_xlabel("Gaussian noise level (Raman input only)", fontsize=10)
    ax1.set_ylabel("Test accuracy (%)", fontsize=10)
    ax1.set_title("(a)  Noise Robustness (SNR sweep)", fontsize=11, fontweight="bold")
    ylo = max(0, min(min(snr_pinn), min(snr_raman)) - 6)
    yhi = min(100, max(max(snr_pinn), max(snr_raman)) + 5)
    ax1.set_ylim(ylo, yhi)
    ax1.grid(axis="y", alpha=0.35, linestyle="--")
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel B
    ref_y = max(drop_raman.max(), drop_pinn.max())
    for drop, name, ls in [(drop_pinn, "PINNACLE", "-"), (drop_raman, "Raman-only", "--")]:
        ax2.fill_between(centres, drop, alpha=0.15, color=COLORS[name])
        ax2.plot(centres, drop, lw=2.2, color=COLORS[name],
                 linestyle=ls, label=name)

    for lo, hi, bname, bcol in BANDS:
        if lo >= wn_min and hi <= wn_max:
            ax2.axvspan(lo, hi, alpha=0.08, color=bcol, zorder=1)
            ax2.text((lo+hi)/2, ref_y * 1.05, bname,
                     ha="center", va="bottom", fontsize=7.5,
                     color=bcol, clip_on=True)

    ax2.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax2.set_xlabel("Occluded window centre (cm⁻¹)", fontsize=10)
    ax2.set_ylabel("Accuracy drop vs. clean (pp)", fontsize=10)
    ax2.set_title("(b)  Sliding-window Occlusion (50 cm⁻¹ window)",
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(bottom=max(-1, -ref_y * 0.15))
    ax2.grid(axis="y", alpha=0.35, linestyle="--")
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.invert_xaxis()

    fig.suptitle(
        "Figure 10 \u2014 Robustness: Gaussian noise (a) and spectral occlusion (b)",
        fontsize=11, y=1.01)
    fig.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    import shutil
    shutil.copy(OUT_PNG, os.path.join(FIG_DIR, "fig10_robustness.png"))
    logger.info(f"  Saved: {OUT_PNG}  \u2192  {FIG_DIR}/fig10_robustness.png")

    # ── Summary for paper text ────────────────────────────────────────────────
    logger.info("")
    logger.info("=== Numbers for paper text ===")
    for label, p, r in zip(SNR_LABELS, snr_pinn, snr_raman):
        logger.info(f"  {label:8s}: PINNACLE={p:.1f}%  Raman-only={r:.1f}%  "
                    f"diff={p-r:.1f} pp")
    logger.info("")
    amide1_mask = (centres >= 1600) & (centres <= 1700)
    logger.info(f"  Amide-I (1600-1700 cm⁻¹) avg drop — "
                f"PINNACLE: {drop_pinn[amide1_mask].mean():.1f} pp  "
                f"Raman-only: {drop_raman[amide1_mask].mean():.1f} pp")
    logger.info(f"  Max drop — PINNACLE: {drop_pinn.max():.1f} pp @{best_p:.0f}  "
                f"Raman-only: {drop_raman.max():.1f} pp @{best_r:.0f}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
