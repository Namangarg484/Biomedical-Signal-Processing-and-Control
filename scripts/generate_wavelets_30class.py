#!/usr/bin/env python3
"""
PINNACLE — Wavelet generation for the 30-class taxonomy dataset.

Generates CWT scalograms for reference, finetune, and test splits.
Memory-efficient: processes in chunks and saves incrementally.

Usage:
    python scripts/generate_wavelets_30class.py
    python scripts/generate_wavelets_30class.py --force         # Regenerate all
    python scripts/generate_wavelets_30class.py --split test    # Single split only
"""

import argparse
import os
import sys
import time

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pinnacle.wavelet import generate_wavelet_dataset
from pinnacle.utils import logger


SPLITS = {
    "reference": ("X_reference.npy", "X_reference_wavelet.npy"),
    "finetune":  ("X_finetune.npy",  "X_finetune_wavelet.npy"),
    "test":      ("X_test.npy",      "X_test_wavelet.npy"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate CWT scalograms for the 30-class dataset"
    )
    parser.add_argument("--data-dir", default="New data", help="Data directory")
    parser.add_argument("--img-size", type=int, default=128,
                        help="Scalogram image size (default: 128 to save disk)")
    parser.add_argument("--n-scales", type=int, default=256, help="Number of CWT scales")
    parser.add_argument("--wavelet", default="morl", help="Wavelet type (morl, mexh)")
    parser.add_argument("--chunk-size", type=int, default=200, help="Processing chunk size")
    parser.add_argument("--force", action="store_true", help="Regenerate even if exists")
    parser.add_argument("--split", choices=["reference", "finetune", "test", "all"],
                        default="all", help="Which split to generate")
    args = parser.parse_args()

    splits_to_process = list(SPLITS.keys()) if args.split == "all" else [args.split]

    total_start = time.time()

    for split_name in splits_to_process:
        input_file, output_file = SPLITS[split_name]
        input_path = os.path.join(args.data_dir, input_file)
        output_path = os.path.join(args.data_dir, output_file)

        if not os.path.exists(input_path):
            logger.error(f"❌ Input file not found: {input_path}")
            continue

        # Check if output already exists
        if os.path.exists(output_path) and not args.force:
            size_gb = os.path.getsize(output_path) / (1024 ** 3)
            logger.info(f"⏭️  {output_file} already exists ({size_gb:.2f} GB). Use --force to regenerate.")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Generating wavelets for split: {split_name}")
        logger.info(f"  Input:  {input_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Size:   {args.img_size}×{args.img_size}")
        logger.info(f"{'=' * 60}")

        # Load spectra
        X = np.load(input_path)
        logger.info(f"  Loaded {X.shape[0]} spectra, shape={X.shape}")

        # Estimate output size
        est_gb = X.shape[0] * 3 * args.img_size * args.img_size * 4 / (1024 ** 3)
        logger.info(f"  Estimated output size: {est_gb:.2f} GB")

        split_start = time.time()

        generate_wavelet_dataset(
            X_proc=X,
            img_size=args.img_size,
            n_scales=args.n_scales,
            wavelet=args.wavelet,
            chunk_size=args.chunk_size,
            output_path=output_path,
        )

        elapsed = time.time() - split_start
        logger.info(f"  ⏱️  {split_name} completed in {elapsed / 60:.1f} minutes")

    total_elapsed = time.time() - total_start
    logger.info(f"\n✅ All wavelet generation complete! Total time: {total_elapsed / 60:.1f} minutes")

    # Summary
    logger.info("\nGenerated files:")
    for split_name in splits_to_process:
        _, output_file = SPLITS[split_name]
        output_path = os.path.join(args.data_dir, output_file)
        if os.path.exists(output_path):
            size_gb = os.path.getsize(output_path) / (1024 ** 3)
            logger.info(f"  {output_file}: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
