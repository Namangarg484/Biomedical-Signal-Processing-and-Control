#!/usr/bin/env python3
"""
PINNACLE — 224x224 Wavelet generation for the 30-class taxonomy dataset.

Generates CWT scalograms for reference, finetune, and test splits.
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
    "reference": ("X_reference.npy", "X_reference_wavelet_224.npy"),
    "finetune":  ("X_finetune.npy",  "X_finetune_wavelet_224.npy"),
    "test":      ("X_test.npy",      "X_test_wavelet_224.npy"),
}

def main():
    parser = argparse.ArgumentParser(description="Generate 224x224 CWT scalograms")
    parser.add_argument("--data-dir", default="New data", help="Data directory")
    parser.add_argument("--img-size", type=int, default=224, help="Scalogram image size")
    parser.add_argument("--n-scales", type=int, default=256, help="Number of CWT scales")
    parser.add_argument("--wavelet", default="morl", help="Wavelet type")
    parser.add_argument("--chunk-size", type=int, default=100, help="Processing chunk size")
    parser.add_argument("--force", action="store_true", help="Regenerate even if exists")
    parser.add_argument("--split", choices=["reference", "finetune", "test", "all"], default="all", help="Which split to generate")
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

        X = np.load(input_path)
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

if __name__ == "__main__":
    main()
