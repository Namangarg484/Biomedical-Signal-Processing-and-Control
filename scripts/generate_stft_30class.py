#!/usr/bin/env python3
"""
PINNACLE — STFT generation for the 30-class taxonomy dataset.

Generates STFT spectrograms for reference, finetune, and test splits.
Memory-efficient: processes in chunks and saves incrementally.
"""

import argparse
import os
import sys
import time

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from scipy.signal import stft
from tqdm import tqdm

from pinnacle.utils import logger


SPLITS = {
    "reference": ("X_reference.npy", "X_reference_stft.npy"),
    "finetune":  ("X_finetune.npy",  "X_finetune_stft.npy"),
    "test":      ("X_test.npy",      "X_test_stft.npy"),
}


def spectrum_to_stft(spectrum: np.ndarray, img_size: int = 128) -> np.ndarray:
    """
    Convert a 1D preprocessed Raman spectrum to a 3-channel STFT spectrogram image.
    """
    # Compute STFT
    # nperseg controls the vertical (frequency) resolution. 
    # Since spectrum length is 1000, 64 is a reasonable window size.
    f, t, Zxx = stft(spectrum, nperseg=64, noverlap=60)
    
    # Compute power
    power = np.abs(Zxx) ** 2

    # Normalise to [0, 1]
    p_min, p_max = power.min(), power.max()
    if p_max > p_min:
        power_norm = (power - p_min) / (p_max - p_min)
    else:
        power_norm = np.zeros_like(power)

    # Resize to target size (128x128)
    spectrogram_gray = cv2.resize(
        power_norm.astype(np.float32),
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR,
    )

    # Apply Viridis colourmap → 3-channel RGB
    spectrogram_uint8 = (spectrogram_gray * 255).astype(np.uint8)
    spectrogram_rgb = cv2.applyColorMap(spectrogram_uint8, cv2.COLORMAP_VIRIDIS)
    spectrogram_rgb = cv2.cvtColor(spectrogram_rgb, cv2.COLOR_BGR2RGB)

    # Convert to (3, H, W) float32 in [0, 1]
    spectrogram = spectrogram_rgb.astype(np.float32) / 255.0
    spectrogram = np.transpose(spectrogram, (2, 0, 1))  # HWC → CHW

    return spectrogram


def generate_stft_dataset(
    X_proc: np.ndarray,
    img_size: int = 128,
    chunk_size: int = 200,
    output_path: str = None,
) -> np.ndarray:
    N = X_proc.shape[0]
    logger.info(f"Generating {N} STFT spectrograms ({img_size}×{img_size})...")

    X_stft = np.zeros((N, 3, img_size, img_size), dtype=np.float32)

    for start in tqdm(range(0, N, chunk_size), desc="Generating STFT"):
        end = min(start + chunk_size, N)
        for i in range(start, end):
            X_stft[i] = spectrum_to_stft(X_proc[i], img_size=img_size)

    if output_path is not None:
        np.save(output_path, X_stft)
        file_size_gb = X_stft.nbytes / (1024 ** 3)
        logger.info(f"  💾 Saved to {output_path} ({file_size_gb:.2f} GB)")

    return X_stft


def main():
    parser = argparse.ArgumentParser(description="Generate STFT spectrograms for the 30-class dataset")
    parser.add_argument("--data-dir", default="New data", help="Data directory")
    parser.add_argument("--img-size", type=int, default=128, help="Spectrogram image size")
    parser.add_argument("--chunk-size", type=int, default=200, help="Processing chunk size")
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
        logger.info(f"Generating STFT for split: {split_name}")
        logger.info(f"  Input:  {input_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"{'=' * 60}")

        X = np.load(input_path)
        split_start = time.time()
        
        generate_stft_dataset(
            X_proc=X,
            img_size=args.img_size,
            chunk_size=args.chunk_size,
            output_path=output_path,
        )

        elapsed = time.time() - split_start
        logger.info(f"  ⏱️  {split_name} completed in {elapsed / 60:.1f} minutes")

    total_elapsed = time.time() - total_start
    logger.info(f"\n✅ All STFT generation complete! Total time: {total_elapsed / 60:.1f} minutes")

if __name__ == "__main__":
    main()
