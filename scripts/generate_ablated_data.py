#!/usr/bin/env python3
"""
Preprocessing Ablation — Data Generation Script
================================================
Generates the "ablated" dataset by:
  - Loading the clinical (raw) Raman spectra from 'New data/'
  - Applying ONLY per-spectrum min-max normalisation (no ALS, no Savitzky-Golay)
  - Generating 2D CWT scalograms with the exact same PyWavelets parameters as the
    main pipeline (morl wavelet, 256 log-spaced scales, 224x224, Viridis RGB)
  - Saving results into data/ablated/ so that scripts/train.py can consume them
    unchanged via --data-dir data/ablated/

Output files (in data/ablated/):
  X_2018_proc.npy          — ablated 1D spectra (2018, 10000 x 1000)
  X_2019_proc.npy          — ablated 1D spectra (2019,  2500 x 1000)
  X_2018_wavelet.npy       — ablated CWT scalograms (2018, 10000 x 3 x 224 x 224)
  X_2019_wavelet.npy       — ablated CWT scalograms (2019,  2500 x 3 x 224 x 224)
  y_2018clinical.npy       — symlink to original labels
  y_2019clinical.npy       — symlink to original labels
  wavenumbers.npy          — symlink to original wavenumbers

Usage:
  python scripts/generate_ablated_data.py
  python scripts/generate_ablated_data.py --raw-dir "New data" --out-dir data/ablated
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2
import pywt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def minmax_per_spectrum(X: np.ndarray) -> np.ndarray:
    """Apply per-spectrum min-max normalisation → [0, 1].  Vectorised."""
    X = X.astype(np.float32)
    xmin = X.min(axis=1, keepdims=True)
    xmax = X.max(axis=1, keepdims=True)
    denom = np.where((xmax - xmin) > 1e-12, xmax - xmin, 1.0)
    return (X - xmin) / denom


def spectrum_to_scalogram(
    spectrum: np.ndarray,
    img_size: int = 224,
    n_scales: int = 256,
    wavelet: str = "morl",
) -> np.ndarray:
    """Convert one 1-D spectrum to a (3, H, W) Viridis-coloured CWT scalogram.

    Parameters match the main pipeline exactly (draft.tex §3.2.2):
      - wavelet  : 'morl' (real Morlet)
      - n_scales : 256 log-spaced scales from 1 to 256
      - img_size : 224 (bilinear resize via cv2.INTER_LINEAR)
      - colormap : cv2.COLORMAP_VIRIDIS
    """
    # 256 log-spaced scales: 2^((i-1)*log2(256)/255), i = 1..256
    scales = np.logspace(np.log2(1), np.log2(n_scales), num=n_scales, base=2)

    cwt_matrix, _ = pywt.cwt(spectrum.astype(np.float64), scales, wavelet)
    power = np.abs(cwt_matrix) ** 2

    # Normalise to [0, 1]
    p_min, p_max = power.min(), power.max()
    power_norm = (power - p_min) / (p_max - p_min + 1e-12)

    # Resize to (img_size, img_size) via bilinear interpolation
    scalogram_gray = cv2.resize(
        power_norm.astype(np.float32),
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR,
    )

    # Viridis colormap → 3-channel RGB
    scalogram_uint8 = (scalogram_gray * 255).astype(np.uint8)
    scalogram_bgr = cv2.applyColorMap(scalogram_uint8, cv2.COLORMAP_VIRIDIS)
    scalogram_rgb = cv2.cvtColor(scalogram_bgr, cv2.COLOR_BGR2RGB)

    # (H, W, 3) → (3, H, W) float32 in [0, 1]
    return (scalogram_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)


def generate_scalograms(
    X: np.ndarray,
    out_path: str,
    img_size: int = 224,
    n_scales: int = 256,
    wavelet: str = "morl",
    chunk_size: int = 200,
    desc: str = "Scalograms",
) -> np.ndarray:
    """Generate CWT scalograms for all spectra in X, writing to a memmap file."""
    N = X.shape[0]
    mem_gb = N * 3 * img_size * img_size * 4 / (1024 ** 3)
    print(f"  Generating {N} scalograms → {out_path}  (~{mem_gb:.2f} GB)")

    arr = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(N, 3, img_size, img_size),
    )

    for start in tqdm(range(0, N, chunk_size), desc=desc):
        end = min(start + chunk_size, N)
        for i in range(start, end):
            arr[i] = spectrum_to_scalogram(
                X[i], img_size=img_size, n_scales=n_scales, wavelet=wavelet
            )
        arr.flush()  # write chunk to disk

    return arr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate ablated (no ALS/SG) dataset")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="New data",
        help="Directory containing X_2018clinical.npy and X_2019clinical.npy",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/ablated",
        help="Output directory (will be created if absent)",
    )
    parser.add_argument(
        "--img-size", type=int, default=224, help="CWT image size (default 224)"
    )
    parser.add_argument(
        "--n-scales", type=int, default=256, help="Number of CWT scales (default 256)"
    )
    parser.add_argument(
        "--wavelet", type=str, default="morl", help="PyWavelets wavelet (default 'morl')"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Spectra per processing chunk (tune to fit RAM)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    raw_dir = os.path.join(PROJECT_ROOT, args.raw_dir)
    out_dir = os.path.join(PROJECT_ROOT, args.out_dir)
    data_dir = os.path.join(PROJECT_ROOT, "data")

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("PINNACLE — Preprocessing Ablation: Data Generation")
    print("=" * 70)
    print(f"  Raw input : {raw_dir}")
    print(f"  Output    : {out_dir}")
    print(f"  Wavelet   : {args.wavelet}, {args.n_scales} scales, {args.img_size}×{args.img_size}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load clinical (raw) spectra — min-max normalised, no ALS/SG
    # ------------------------------------------------------------------
    print("\n[1/4] Loading raw clinical spectra...")
    X_2018_raw = np.load(os.path.join(raw_dir, "X_2018clinical.npy"))
    X_2019_raw = np.load(os.path.join(raw_dir, "X_2019clinical.npy"))
    print(f"  X_2018_raw : {X_2018_raw.shape}  range [{X_2018_raw.min():.4f}, {X_2018_raw.max():.4f}]")
    print(f"  X_2019_raw : {X_2019_raw.shape}  range [{X_2019_raw.min():.4f}, {X_2019_raw.max():.4f}]")

    # ------------------------------------------------------------------
    # 2. Apply ONLY min-max normalisation per spectrum (no ALS, no SG)
    # ------------------------------------------------------------------
    print("\n[2/4] Applying per-spectrum min-max normalisation (no ALS, no SG)...")
    X_2018_ablated = minmax_per_spectrum(X_2018_raw)
    X_2019_ablated = minmax_per_spectrum(X_2019_raw)
    print(f"  X_2018_ablated range [{X_2018_ablated.min():.4f}, {X_2018_ablated.max():.4f}]")
    print(f"  X_2019_ablated range [{X_2019_ablated.min():.4f}, {X_2019_ablated.max():.4f}]")

    # Save ablated 1-D spectra (named _proc.npy so train.py finds them)
    np.save(os.path.join(out_dir, "X_2018_proc.npy"), X_2018_ablated)
    np.save(os.path.join(out_dir, "X_2019_proc.npy"), X_2019_ablated)
    print(f"  Saved X_2018_proc.npy  ({os.path.getsize(os.path.join(out_dir,'X_2018_proc.npy'))/1e6:.1f} MB)")
    print(f"  Saved X_2019_proc.npy  ({os.path.getsize(os.path.join(out_dir,'X_2019_proc.npy'))/1e6:.1f} MB)")

    # ------------------------------------------------------------------
    # 3. Generate CWT scalograms
    # ------------------------------------------------------------------
    print("\n[3/4] Generating CWT scalograms...")
    t0 = time.time()

    generate_scalograms(
        X_2018_ablated,
        out_path=os.path.join(out_dir, "X_2018_wavelet.npy"),
        img_size=args.img_size,
        n_scales=args.n_scales,
        wavelet=args.wavelet,
        chunk_size=args.chunk_size,
        desc="2018 scalograms",
    )
    generate_scalograms(
        X_2019_ablated,
        out_path=os.path.join(out_dir, "X_2019_wavelet.npy"),
        img_size=args.img_size,
        n_scales=args.n_scales,
        wavelet=args.wavelet,
        chunk_size=args.chunk_size,
        desc="2019 scalograms",
    )

    elapsed = time.time() - t0
    print(f"  Scalogram generation complete in {elapsed/60:.1f} min")

    # ------------------------------------------------------------------
    # 4. Symlink labels and wavenumbers so load_data() finds everything
    # ------------------------------------------------------------------
    print("\n[4/4] Creating symlinks for labels and wavenumbers...")
    symlink_files = ["y_2018clinical.npy", "y_2019clinical.npy", "wavenumbers.npy"]
    for fname in symlink_files:
        src = os.path.realpath(os.path.join(data_dir, fname))
        dst = os.path.join(out_dir, fname)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        print(f"  {fname} → {src}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅  Ablated dataset ready in:", out_dir)
    print("    Files written:")
    for f in sorted(os.listdir(out_dir)):
        fp = os.path.join(out_dir, f)
        size = os.path.getsize(os.path.realpath(fp)) / 1e9
        print(f"      {f:35s}  {size:.2f} GB")
    print()
    print("  To train on ablated data:")
    print("    python scripts/train_ablated.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
