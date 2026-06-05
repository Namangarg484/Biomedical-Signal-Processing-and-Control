"""
PINNACLE — CWT Scalogram generation.
Converts 1D preprocessed Raman spectra to 2D scale-wavenumber scalograms.

Paper (draft.tex §3.2): Complex Morlet CWT, 256 scales, 224×224 Viridis RGB.
"""

import numpy as np
import cv2
import pywt
from tqdm import tqdm
import os
from typing import Optional

from pinnacle.utils import logger


def spectrum_to_scalogram(
    spectrum: np.ndarray,
    img_size: int = 224,
    n_scales: int = 256,
    wavelet: str = "morl",
) -> np.ndarray:
    """
    Convert a 1D preprocessed Raman spectrum to a 3-channel scalogram image.

    Args:
        spectrum: 1D preprocessed Raman spectrum (seq_len,)
        img_size: Output image size (default 224)
        n_scales: Number of CWT scales (default 256, matching paper)
        wavelet: PyWavelets wavelet name — 'morl' (Morlet), 'mexh' (Ricker/Mexican Hat)

    Returns:
        3-channel RGB scalogram image (3, img_size, img_size) float32 in [0,1]
    """
    # Generate scales (log-spaced for multi-resolution coverage)
    scales = np.logspace(np.log2(1), np.log2(n_scales), num=n_scales, base=2)

    # Compute CWT using PyWavelets
    cwt_matrix, _ = pywt.cwt(spectrum, scales, wavelet)

    # Compute power scalogram
    power = np.abs(cwt_matrix) ** 2

    # Normalise to [0, 1]
    p_min, p_max = power.min(), power.max()
    if p_max > p_min:
        power_norm = (power - p_min) / (p_max - p_min)
    else:
        power_norm = np.zeros_like(power)

    # Resize to target size
    scalogram_gray = cv2.resize(
        power_norm.astype(np.float32),
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR,
    )

    # Apply Viridis colourmap → 3-channel RGB
    scalogram_uint8 = (scalogram_gray * 255).astype(np.uint8)
    scalogram_rgb = cv2.applyColorMap(scalogram_uint8, cv2.COLORMAP_VIRIDIS)
    scalogram_rgb = cv2.cvtColor(scalogram_rgb, cv2.COLOR_BGR2RGB)

    # Convert to (3, H, W) float32 in [0, 1]
    scalogram = scalogram_rgb.astype(np.float32) / 255.0
    scalogram = np.transpose(scalogram, (2, 0, 1))  # HWC → CHW

    return scalogram


def generate_wavelet_dataset(
    X_proc: np.ndarray,
    img_size: int = 224,
    n_scales: int = 256,
    wavelet: str = "morlet",
    chunk_size: int = 100,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate CWT scalograms for an entire dataset.

    Memory-efficient: processes in chunks to avoid OOM on MacBook.

    Args:
        X_proc: Preprocessed spectra (N, seq_len)
        img_size: Output image size
        n_scales: Number of CWT scales
        wavelet: Wavelet type
        chunk_size: Process this many spectra at a time
        output_path: If provided, save result to this .npy path

    Returns:
        Scalogram array (N, 3, img_size, img_size) float32
    """
    N = X_proc.shape[0]
    logger.info(
        f"Generating {N} scalograms ({img_size}×{img_size}, {n_scales} scales, {wavelet})..."
    )

    # Estimate memory: N × 3 × 224 × 224 × 4 bytes
    mem_gb = N * 3 * img_size * img_size * 4 / (1024 ** 3)
    logger.info(f"  Estimated output size: {mem_gb:.2f} GB")

    if output_path is not None:
        X_wavelet = np.lib.format.open_memmap(
            output_path, mode='w+', dtype=np.float32, shape=(N, 3, img_size, img_size)
        )
    else:
        X_wavelet = np.zeros((N, 3, img_size, img_size), dtype=np.float32)

    for start in tqdm(range(0, N, chunk_size), desc="Generating scalograms"):
        end = min(start + chunk_size, N)
        for i in range(start, end):
            X_wavelet[i] = spectrum_to_scalogram(
                X_proc[i],
                img_size=img_size,
                n_scales=n_scales,
                wavelet=wavelet,
            )
        # Flush to disk every chunk to prevent memory buildup
        if output_path is not None:
            X_wavelet.flush()

    logger.info(f"  ✅ Generated {N} scalograms, shape: {X_wavelet.shape}")

    # Verify statistics (use chunks to avoid OOM)
    # Using a subset for stats to avoid loading 36GB into RAM
    subset_size = min(N, 1000)
    subset_indices = np.linspace(0, N-1, subset_size, dtype=int)
    subset_data = X_wavelet[subset_indices]
    logger.info(
        f"  Stats (subset): mean={subset_data.mean():.4f}, std={subset_data.std():.4f}, "
        f"range=[{subset_data.min():.4f}, {subset_data.max():.4f}]"
    )

    if output_path is not None:
        file_size_gb = os.path.getsize(output_path) / (1024 ** 3)
        logger.info(f"  💾 Saved to {output_path} ({file_size_gb:.2f} GB)")

    return X_wavelet
