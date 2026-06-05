#!/usr/bin/env python3
"""
Experiment A — Preprocessing ablation (real, replacing the unbacked 97.04%).

Trains the full PINNACLE fusion model on RAW min-max-normalised spectra
(ALS baseline correction + Savitzky-Golay denoising BYPASSED), using the
identical architecture, split (seed=42, 80/10/10), and training recipe as
the main 5-class model. Reports test accuracy + Wilson CI and a paired
McNemar test against the fully-preprocessed PINNACLE predictions.

Raw spectra:  New data/X_2018clinical.npy, X_2019clinical.npy  (min-max [0,1])
Labels:       remapped [0,2,3,5,6] -> [0..4] (verified per-row identical to
              the processed labels, so the seed=42 split is the same test set).

No synthetic numbers: every figure is computed here and logged.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import torch

from pinnacle.utils import set_seed, get_device, logger
from pinnacle.model import PINNACLE
from pinnacle.dataset import create_dataloaders
from pinnacle.wavelet import generate_wavelet_dataset

# Reuse the exact training recipe used for the main 5-class results
from scripts.ablation_5class import train_model, wilson_ci, mcnemar_p, CFG

RAW_DIR = "New data"
OUT_DIR = "outputs/preproc_ablation"
WAV18 = os.path.join(RAW_DIR, "X_2018clinical_wavelet224.npy")
WAV19 = os.path.join(RAW_DIR, "X_2019clinical_wavelet224.npy")
PROC_PREDS = "outputs/predictions_current.npz"  # processed PINNACLE preds
SEED = 42
LABEL_MAP = {0: 0, 2: 1, 3: 2, 5: 3, 6: 4}


def remap(y):
    return np.array([LABEL_MAP[int(v)] for v in y], dtype=np.int64)


def ensure_raw_wavelets(X, path):
    if os.path.exists(path):
        logger.info(f"  Cached raw wavelets: {path}")
        return np.load(path, mmap_mode="r")
    logger.info(f"  Generating raw-spectrum wavelets -> {path}")
    return generate_wavelet_dataset(
        X, img_size=224, n_scales=256, wavelet="morl",
        chunk_size=100, output_path=path,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    device = get_device()
    logger.info("=" * 70)
    logger.info("EXPERIMENT A — Preprocessing ablation (raw min-max, no ALS/SG)")
    logger.info("=" * 70)

    X18 = np.load(os.path.join(RAW_DIR, "X_2018clinical.npy")).astype(np.float32)
    X19 = np.load(os.path.join(RAW_DIR, "X_2019clinical.npy")).astype(np.float32)
    y18 = remap(np.load(os.path.join(RAW_DIR, "y_2018clinical.npy")))
    y19 = remap(np.load(os.path.join(RAW_DIR, "y_2019clinical.npy")))
    logger.info(f"  Raw spectra: X18 {X18.shape} [{X18.min():.3f},{X18.max():.3f}]"
                f"  X19 {X19.shape}")

    wav18 = ensure_raw_wavelets(X18, WAV18)
    wav19 = ensure_raw_wavelets(X19, WAV19)

    data = {
        "X_2018": X18, "X_2019": X19,
        "y_2018": y18, "y_2019": y19,
        "X_2018_wav": np.asarray(wav18), "X_2019_wav": np.asarray(wav19),
        "wavenumbers": None,
    }

    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data, batch_size=CFG["batch_size"], seed=SEED, num_workers=0,
        use_augmentation=True,
    )
    n_test = len(test_loader.dataset)
    logger.info(f"  Test set: {n_test} samples")

    model = PINNACLE(num_classes=5, embed_dim=CFG["embed_dim"],
                     dropout=CFG["dropout"], mode="fusion")
    preds, labels, test_acc = train_model(
        "preproc_raw", model, train_loader, val_loader, test_loader,
        device, OUT_DIR, no_cache=True,
    )

    acc_frac = test_acc / 100.0
    lo, hi = wilson_ci(acc_frac, n_test)
    logger.info(f"  >>> No-preprocessing test acc: {test_acc:.2f}% "
                f"[{100*lo:.2f}, {100*hi:.2f}]  (n={n_test})")

    # Paired McNemar vs processed PINNACLE on the identical test split
    result = {
        "no_preproc_acc": test_acc, "ci_lo": 100 * lo, "ci_hi": 100 * hi,
        "n_test": int(n_test),
    }
    if os.path.exists(PROC_PREDS):
        z = np.load(PROC_PREDS)
        proc_pred = z["y_pred"]
        proc_true = z["y_true"]
        if np.array_equal(proc_true, labels):
            correct_raw  = (preds     == labels)
            correct_proc = (proc_pred == labels)
            b = int(np.sum( correct_raw & ~correct_proc))
            c = int(np.sum(~correct_raw &  correct_proc))
            p = mcnemar_p(preds, proc_pred, labels)
            proc_acc = 100.0 * np.mean(proc_pred == proc_true)
            result.update({
                "processed_acc": proc_acc,
                "delta_pp": test_acc - proc_acc,
                "mcnemar_p": p, "mcnemar_b": b, "mcnemar_c": c,
            })
            logger.info(f"  Processed PINNACLE acc: {proc_acc:.2f}%")
            logger.info(f"  Delta: {test_acc - proc_acc:+.2f} pp")
            logger.info(f"  McNemar p={p:.4g} (b={b}, c={c})")
        else:
            logger.warning("  Processed test labels DIFFER from raw test labels — "
                           "split mismatch; skipping paired McNemar.")
            result["mcnemar_p"] = None
    else:
        logger.warning(f"  {PROC_PREDS} not found — skipping McNemar.")
        result["mcnemar_p"] = None

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved -> {os.path.join(OUT_DIR, 'results.json')}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
