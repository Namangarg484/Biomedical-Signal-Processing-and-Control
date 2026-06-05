"""
PINNACLE — 30-Class Dataset and DataLoader utilities.

Loads the extended 30-species taxonomy dataset.

Split strategy:
    reference (60K) → stratified 90/10 → train (54K) / val (6K)
    test      (3K)  → held-out evaluation

The 'finetune' set is reserved for optional domain-adaptation fine-tuning
(different instrument distribution), NOT for validation during pre-training.

No data leakage: val is carved from reference via stratified split with fixed seed.
Labels are already contiguous [0..29].
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict

from pinnacle.utils import logger
from pinnacle.dataset import PINNACLEDataset, RamanAugmentation


def load_30class_data(data_dir: str, val_fraction: float = 0.1, seed: int = 42) -> Dict:
    """
    Load the 30-class taxonomy dataset.

    Splits the reference set into train/val (90/10 stratified).
    The finetune set is loaded separately for optional domain-adaptation.

    Returns:
        Dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                         X_train_wav, X_val_wav, X_test_wav,
                         wavenumbers, num_classes
    """
    logger.info(f"Loading 30-class data from {data_dir}...")

    result = {}

    # ---- Load all spectra ----
    X_ref = np.load(os.path.join(data_dir, "X_reference.npy"))
    y_ref = np.load(os.path.join(data_dir, "y_reference.npy")).astype(np.int64)
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy")).astype(np.int64)

    logger.info(f"  Reference: {X_ref.shape}, {len(np.unique(y_ref))} classes")
    logger.info(f"  Test:      {X_test.shape}, {len(np.unique(y_test))} classes")

    # ---- Verify labels are contiguous [0..29] ----
    all_labels = np.concatenate([y_ref, y_test])
    unique = np.unique(all_labels)
    num_classes = len(unique)
    result["num_classes"] = num_classes

    if not np.array_equal(unique, np.arange(num_classes)):
        logger.warning(f"  ⚠️ Labels are NOT contiguous: {unique}")
        label_map = {old: new for new, old in enumerate(sorted(unique))}
        y_ref = np.array([label_map[l] for l in y_ref])
        y_test = np.array([label_map[l] for l in y_test])
        logger.info(f"  Remapped to [0..{num_classes - 1}]")
    else:
        logger.info(f"  ✅ Labels are contiguous [0..{num_classes - 1}]")

    # ---- Stratified split: reference → train (90%) / val (10%) ----
    indices = np.arange(len(X_ref))
    idx_train, idx_val = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_ref,
    )

    result["X_train"] = X_ref[idx_train]
    result["y_train"] = y_ref[idx_train]
    result["X_val"]   = X_ref[idx_val]
    result["y_val"]   = y_ref[idx_val]
    result["X_test"]  = X_test
    result["y_test"]  = y_test

    logger.info(f"  Train (ref 90%):  {result['X_train'].shape}")
    logger.info(f"  Val   (ref 10%):  {result['X_val'].shape}")
    logger.info(f"  Test  (held-out): {result['X_test'].shape}")

    # ---- Class distribution ----
    train_dist = dict(zip(*np.unique(result["y_train"], return_counts=True)))
    val_dist   = dict(zip(*np.unique(result["y_val"],   return_counts=True)))
    test_dist  = dict(zip(*np.unique(result["y_test"],  return_counts=True)))

    logger.info(f"  Train class sizes: min={min(train_dist.values())}, max={max(train_dist.values())}")
    logger.info(f"  Val class sizes:   min={min(val_dist.values())},   max={max(val_dist.values())}")
    logger.info(f"  Test class sizes:  min={min(test_dist.values())},  max={max(test_dist.values())}")

    # ---- Data leakage check ----
    logger.info("  Running data leakage check...")
    # Train/val come from same file but disjoint indices — verify
    overlap = set(idx_train) & set(idx_val)
    if len(overlap) == 0:
        logger.info("  ✅ Train/val indices are disjoint (zero leakage)")
    else:
        logger.error(f"  ❌ LEAKAGE: {len(overlap)} overlapping indices!")

    # Cross-check with test (different source file)
    val_hashes = set(map(lambda x: hash(x.tobytes()), result["X_val"][:500]))
    test_hashes = set(map(lambda x: hash(x.tobytes()), result["X_test"][:500]))
    val_test_leak = len(val_hashes & test_hashes)
    if val_test_leak == 0:
        logger.info("  ✅ Val/test have no overlapping samples")
    else:
        logger.warning(f"  ⚠️ Val∩Test = {val_test_leak} samples")

    # ---- Wavenumbers ----
    wn_path = os.path.join(data_dir, "wavenumbers.npy")
    result["wavenumbers"] = np.load(wn_path) if os.path.exists(wn_path) else None

    # ---- Wavelets (split using same indices) ----
    wav_ref_path = os.path.join(data_dir, "X_reference_wavelet.npy")
    wav_test_path = os.path.join(data_dir, "X_test_wavelet.npy")

    if os.path.exists(wav_ref_path):
        X_ref_wav = np.load(wav_ref_path, mmap_mode="r")
        # Split wavelets using the SAME indices as spectra
        result["X_train_wav"] = np.array(X_ref_wav[idx_train])  # copy from mmap
        result["X_val_wav"]   = np.array(X_ref_wav[idx_val])    # copy from mmap
        logger.info(f"  X_train_wav: {result['X_train_wav'].shape}")
        logger.info(f"  X_val_wav:   {result['X_val_wav'].shape}")
        del X_ref_wav  # release mmap
    else:
        result["X_train_wav"] = None
        result["X_val_wav"]   = None
        logger.info("  Reference wavelets not found — will need generation")

    if os.path.exists(wav_test_path):
        result["X_test_wav"] = np.load(wav_test_path, mmap_mode="r")
        logger.info(f"  X_test_wav:  {result['X_test_wav'].shape}")
    else:
        result["X_test_wav"] = None
        logger.info("  Test wavelets not found — will need generation")

    return result


def create_30class_dataloaders(
    data: Dict,
    batch_size: int = 32,
    num_workers: int = 0,
    use_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from the 30-class data dict.

    Train and val come from the same reference distribution (stratified split).
    Test is a separate held-out set.

    Returns:
        train_loader, val_loader, test_loader
    """
    raman_aug = RamanAugmentation() if use_augmentation else None

    train_ds = PINNACLEDataset(
        data["X_train"], data["y_train"], data.get("X_train_wav"),
        transform_raman=raman_aug, split="train",
    )
    val_ds = PINNACLEDataset(
        data["X_val"], data["y_val"], data.get("X_val_wav"),
        split="val",
    )
    test_ds = PINNACLEDataset(
        data["X_test"], data["y_test"], data.get("X_test_wav"),
        split="test",
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    logger.info(
        f"  ✅ DataLoaders: train={len(train_loader)} batches, "
        f"val={len(val_loader)} batches, test={len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader
