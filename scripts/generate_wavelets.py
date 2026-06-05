#!/usr/bin/env python3
"""
Generate CWT scalogram datasets from preprocessed Raman spectra.
Generates X_2018_wavelet.npy and/or X_2019_wavelet.npy.

Usage:
    python scripts/generate_wavelets.py                    # Generate all missing
    python scripts/generate_wavelets.py --dataset 2018     # Only 2018
    python scripts/generate_wavelets.py --dataset 2019     # Only 2019
"""

import argparse
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnacle.wavelet import generate_wavelet_dataset
from pinnacle.utils import logger


def main():
    parser = argparse.ArgumentParser(description="Generate CWT scalograms")
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["2018", "2019", "all"],
        help="Which dataset to process",
    )
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--n-scales", type=int, default=256)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--wavelet", type=str, default="mexh", choices=["mexh", "morl"])
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    datasets = []
    if args.dataset in ("2018", "all"):
        datasets.append(("2018", "X_2018_proc.npy", "X_2018_wavelet.npy"))
    if args.dataset in ("2019", "all"):
        datasets.append(("2019", "X_2019_proc.npy", "X_2019_wavelet.npy"))

    for name, proc_file, wav_file in datasets:
        proc_path = os.path.join(args.data_dir, proc_file)
        wav_path = os.path.join(args.data_dir, wav_file)

        if not os.path.exists(proc_path):
            logger.error(f"❌ {proc_path} not found. Run preprocessing first.")
            continue

        if os.path.exists(wav_path) and not args.force:
            size_gb = os.path.getsize(wav_path) / (1024**3)
            logger.info(f"⏭️  {wav_file} already exists ({size_gb:.2f} GB). Use --force to regenerate.")
            continue

        logger.info(f"🔬 Generating wavelets for {name}...")
        X_proc = np.load(proc_path)

        generate_wavelet_dataset(
            X_proc,
            img_size=args.img_size,
            n_scales=args.n_scales,
            wavelet=args.wavelet,
            chunk_size=args.chunk_size,
            output_path=wav_path,
        )

    logger.info("✅ Wavelet generation complete!")


if __name__ == "__main__":
    main()
