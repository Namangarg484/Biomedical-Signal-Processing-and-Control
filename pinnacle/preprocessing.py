"""
PINNACLE — Preprocessing pipeline for Raman spectra.
Extracted from naman AMR.ipynb Cell 2. Adapted for local execution.

Pipeline: Median filter → ALS baseline correction → Savitzky-Golay → L2 normalisation
"""

import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from pinnacle.utils import logger


# ================================================================
# ALS BASELINE CORRECTION
# ================================================================

def als_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 10,
) -> np.ndarray:
    """
    Asymmetric Least Squares baseline removal.

    Args:
        y: Input spectrum (1D array)
        lam: Smoothness parameter (larger = smoother baseline)
        p: Asymmetry parameter (smaller = more aggressive baseline removal)
        niter: Number of iterations

    Returns:
        Estimated baseline (same shape as y)
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L))
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * (D @ D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


# ================================================================
# SPECTRUM VALIDATION
# ================================================================

def validate_spectrum(s: np.ndarray, name: str = "spectrum") -> bool:
    """Check spectrum for quality issues."""
    if np.isnan(s).any():
        logger.error(f"{name} contains NaN values")
        return False
    if np.isinf(s).any():
        logger.error(f"{name} contains Inf values")
        return False
    if (s == 0).all():
        logger.error(f"{name} is all zeros")
        return False
    return True


# ================================================================
# SINGLE SPECTRUM PREPROCESSING
# ================================================================

def preprocess_spectrum(
    s: np.ndarray,
    params: Dict,
) -> Tuple[np.ndarray, Dict]:
    """
    Full preprocessing pipeline with quality metrics.

    Pipeline matches draft.tex Table 2:
      ALS baseline correction → Min-max normalisation → Savitzky-Golay denoising

    Returns:
        Preprocessed spectrum and quality metrics dict
    """
    s = s.astype(np.float64)
    metrics = {}

    # 1) Spike removal (median filter)
    k = params.get("median_kernel", 3)
    if k > 1:
        s_filtered = medfilt(s, kernel_size=k)
        metrics["spikes_removed"] = int(
            np.abs(s - s_filtered).max() > 0.1 * s.std()
        )
        s = s_filtered

    # 2) Baseline correction (ALS)
    baseline = als_baseline(
        s,
        lam=params.get("als_lambda", 1e5),
        p=params.get("als_p", 0.01),
        niter=params.get("als_niter", 10),
    )
    s_corrected = s - baseline
    metrics["baseline_intensity"] = float(np.mean(baseline))
    s = s_corrected

    # 3) Savitzky-Golay smoothing
    win = params.get("savgol_window", 11)
    poly = params.get("savgol_poly", 2)
    if win >= 3 and win % 2 == 1 and poly < win:
        s = savgol_filter(s, win, poly)

    # 4) Optional Gaussian smoothing
    sigma = params.get("gauss_sigma", 0.0)
    if sigma > 0:
        s = gaussian_filter1d(s, sigma=sigma)

    # 5) Normalisation
    norm_method = params.get("normalization", "minmax")

    if norm_method == "l2":
        norm = np.linalg.norm(s) + 1e-12
        s = s / norm
    elif norm_method == "minmax":
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    elif norm_method == "area":
        area = np.trapz(np.abs(s)) + 1e-12
        s = s / area
    elif norm_method == "standard":
        s = (s - s.mean()) / (s.std() + 1e-12)
    else:
        raise ValueError(f"Unknown normalization: {norm_method}")

    metrics["final_range"] = (float(s.min()), float(s.max()))
    metrics["final_std"] = float(s.std())

    if not validate_spectrum(s, "processed"):
        raise ValueError("Preprocessing produced invalid spectrum")

    return s.astype(np.float32), metrics


# ================================================================
# BATCH PROCESSING
# ================================================================

DEFAULT_PREPROCESS_PARAMS = {
    "median_kernel": 3,
    "als_lambda": 1e5,
    "als_p": 0.01,
    "als_niter": 10,
    "savgol_window": 11,
    "savgol_poly": 2,
    "gauss_sigma": 0.0,
    "normalization": "minmax",
}


def process_dataset(
    X_raw: np.ndarray,
    y_labels: np.ndarray,
    dataset_name: str,
    preprocess_params: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Preprocess an entire dataset of raw Raman spectra.

    Args:
        X_raw: Raw Raman spectra (N, seq_len)
        y_labels: Labels (N,)
        dataset_name: Name for logging (e.g., "2018", "2019")
        preprocess_params: Override default preprocessing parameters

    Returns:
        X_processed (N, seq_len), statistics dict
    """
    if preprocess_params is None:
        preprocess_params = DEFAULT_PREPROCESS_PARAMS.copy()

    N, seq_len = X_raw.shape
    logger.info(f"Processing dataset: {dataset_name} — {N} samples × {seq_len} points")

    X_processed = np.zeros((N, seq_len), dtype=np.float32)
    failed = 0

    for i in tqdm(range(N), desc=f"Preprocessing {dataset_name}"):
        try:
            spectrum_proc, _ = preprocess_spectrum(X_raw[i], preprocess_params)
            X_processed[i] = spectrum_proc
        except Exception as e:
            logger.warning(f"Spectrum {i} failed: {e}")
            X_processed[i] = np.zeros(seq_len, dtype=np.float32)
            failed += 1

    stats = {
        "num_samples": N,
        "seq_len": seq_len,
        "mean_intensity": float(X_processed.mean()),
        "std_intensity": float(X_processed.std()),
        "min_value": float(X_processed.min()),
        "max_value": float(X_processed.max()),
        "num_classes": len(np.unique(y_labels)),
        "class_distribution": {
            int(k): int(v) for k, v in zip(*np.unique(y_labels, return_counts=True))
        },
        "failed_spectra": failed,
    }

    logger.info(
        f"  ✅ {dataset_name}: mean={stats['mean_intensity']:.4f}, "
        f"std={stats['std_intensity']:.4f}, failed={failed}"
    )
    return X_processed, stats
