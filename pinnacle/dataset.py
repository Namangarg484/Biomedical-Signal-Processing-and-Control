"""
PINNACLE — Dataset and DataLoader utilities.
Loads preprocessed Raman spectra + CWT scalograms for multimodal classification.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict

from pinnacle.utils import logger


class PINNACLEDataset(Dataset):
    """
    Multimodal dataset: 1D Raman spectra + 2D CWT scalograms + labels.

    Returns:
        raman: (seq_len,) float32
        scalogram: (3, 224, 224) float32
        label: int64
    """

    def __init__(
        self,
        X_raman: np.ndarray,
        y: np.ndarray,
        X_scalogram: Optional[np.ndarray] = None,
        transform_raman=None,
        transform_image=None,
        split: str = "train",
    ):
        self.X_raman = X_raman.astype(np.float32)
        self.y = y.astype(np.int64)
        self.X_scalogram = X_scalogram
        self.transform_raman = transform_raman
        self.transform_image = transform_image
        self.split = split

        logger.info(
            f"  Dataset ({split}): {len(self)} samples, "
            f"{len(np.unique(y))} classes"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        raman = torch.from_numpy(self.X_raman[idx].copy()).float()

        # Scalogram
        if self.X_scalogram is not None:
            scalogram = torch.from_numpy(self.X_scalogram[idx].copy()).float()
        else:
            scalogram = torch.zeros(3, 224, 224)

        label = torch.tensor(self.y[idx]).long()

        # Apply augmentation (train only)
        if self.split == "train":
            if self.transform_raman is not None:
                raman = self.transform_raman(raman)
            if self.transform_image is not None and self.X_scalogram is not None:
                scalogram = self.transform_image(scalogram)

        return raman, scalogram, label


class RamanAugmentation:
    """
    Raman-specific data augmentation.
    Applies: noise, shift, scaling, baseline drift.
    """

    def __init__(
        self,
        noise_std: float = 0.02,
        shift_range: int = 10,
        scale_range: float = 0.1,
        probability: float = 0.7,
    ):
        self.noise_std = noise_std
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.probability = probability

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.probability:
            return spectrum

        spectrum = spectrum.clone()

        # Gaussian noise
        if torch.rand(1).item() > 0.5:
            spectrum = spectrum + torch.randn_like(spectrum) * self.noise_std

        # Random shift (calibration error)
        if torch.rand(1).item() > 0.5:
            import random
            shift = random.randint(-self.shift_range, self.shift_range)
            spectrum = torch.roll(spectrum, shift, dims=0)

        # Random scaling (intensity variation)
        if torch.rand(1).item() > 0.5:
            import random
            scale = 1.0 + random.uniform(-self.scale_range, self.scale_range)
            spectrum = spectrum * scale

        return spectrum


def remap_labels(y: np.ndarray) -> np.ndarray:
    """
    Remap non-contiguous labels (e.g., [0, 2, 3, 5, 6]) to contiguous [0, 1, 2, 3, 4].
    """
    unique_labels = np.unique(y)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_remapped = np.array([label_map[label] for label in y])
    logger.info(f"  Label mapping: {label_map}")
    return y_remapped


def load_data(
    data_dir: str,
    remap: bool = True,
) -> Dict:
    """
    Load all data files from the data directory.

    Returns dict with keys:
        X_2018, X_2019, y_2018, y_2019, X_2018_wav (optional), X_2019_wav, wavenumbers
    """
    logger.info(f"Loading data from {data_dir}...")

    result = {}

    # Processed Raman spectra
    result["X_2018"] = np.load(os.path.join(data_dir, "X_2018_proc.npy"))
    result["X_2019"] = np.load(os.path.join(data_dir, "X_2019_proc.npy"))
    logger.info(f"  X_2018: {result['X_2018'].shape}, X_2019: {result['X_2019'].shape}")

    # Labels
    result["y_2018"] = np.load(os.path.join(data_dir, "y_2018clinical.npy"))
    result["y_2019"] = np.load(os.path.join(data_dir, "y_2019clinical.npy"))

    if remap:
        all_labels = np.concatenate([result["y_2018"], result["y_2019"]])
        unique = np.unique(all_labels)
        if not np.array_equal(unique, np.arange(len(unique))):
            logger.info("  Remapping non-contiguous labels...")
            label_map = {old: new for new, old in enumerate(unique)}
            result["y_2018"] = np.array([label_map[l] for l in result["y_2018"]])
            result["y_2019"] = np.array([label_map[l] for l in result["y_2019"]])
            result["label_map"] = label_map
            logger.info(f"  Label mapping: {label_map}")

    logger.info(
        f"  y_2018 classes: {np.unique(result['y_2018'])}, "
        f"y_2019 classes: {np.unique(result['y_2019'])}"
    )

    # Wavelet scalograms (optional — may be very large)
    wav_2019_path = os.path.join(data_dir, "X_2019_wavelet.npy")
    wav_2018_path = os.path.join(data_dir, "X_2018_wavelet.npy")

    if os.path.exists(wav_2019_path):
        result["X_2019_wav"] = np.load(wav_2019_path)
        logger.info(f"  X_2019_wavelet: {result['X_2019_wav'].shape}")
    else:
        result["X_2019_wav"] = None
        logger.warning("  X_2019_wavelet.npy not found")

    if os.path.exists(wav_2018_path):
        result["X_2018_wav"] = np.load(wav_2018_path)
        logger.info(f"  X_2018_wavelet: {result['X_2018_wav'].shape}")
    else:
        result["X_2018_wav"] = None
        logger.warning("  X_2018_wavelet.npy not found — will need to generate")

    # Wavenumbers (optional)
    wn_path = os.path.join(data_dir, "wavenumbers.npy")
    if os.path.exists(wn_path):
        result["wavenumbers"] = np.load(wn_path)
    else:
        result["wavenumbers"] = None

    return result


def create_dataloaders(
    data: Dict,
    batch_size: int = 32,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    use_augmentation: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Create train/val/test DataLoaders from loaded data.

    Split strategy (matching paper §4.2):
        Combined 2018+2019 → stratified 80/10/10 split.

    Returns:
        train_loader, val_loader, test_loader, y_train
    """
    # Combine datasets
    X_all = np.concatenate([data["X_2018"], data["X_2019"]], axis=0)
    y_all = np.concatenate([data["y_2018"], data["y_2019"]], axis=0)

    # Combine wavelet data if available
    X_wav_all = None
    if data.get("X_2018_wav") is not None and data.get("X_2019_wav") is not None:
        X_wav_all = np.concatenate([data["X_2018_wav"], data["X_2019_wav"]], axis=0)
        logger.info(f"  Combined wavelets: {X_wav_all.shape}")
    elif data.get("X_2019_wav") is not None:
        # Only 2019 wavelets available — generate dummy for 2018
        logger.warning(
            "  Only X_2019_wavelet available. "
            "Run 'python scripts/generate_wavelets.py' to generate X_2018_wavelet."
        )
        X_wav_all = None

    logger.info(f"  Combined data: {X_all.shape}, labels: {y_all.shape}")
    logger.info(f"  Class distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")

    # Stratified split: train / temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all,
        test_size=(test_size + val_size),
        random_state=seed,
        stratify=y_all,
    )

    # Split temp → val / test
    relative_test = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test,
        random_state=seed,
        stratify=y_temp,
    )

    # Handle wavelet splits using indices
    X_wav_train = X_wav_val = X_wav_test = None
    if X_wav_all is not None:
        # We need to track indices through the splits
        indices = np.arange(len(X_all))
        idx_train, idx_temp = train_test_split(
            indices, test_size=(test_size + val_size),
            random_state=seed, stratify=y_all,
        )
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=relative_test,
            random_state=seed, stratify=y_all[idx_temp],
        )
        X_wav_train = X_wav_all[idx_train]
        X_wav_val = X_wav_all[idx_val]
        X_wav_test = X_wav_all[idx_test]

    logger.info(
        f"  Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    # Augmentation
    raman_aug = RamanAugmentation() if use_augmentation else None

    # Create datasets
    train_ds = PINNACLEDataset(
        X_train, y_train, X_wav_train,
        transform_raman=raman_aug, split="train",
    )
    val_ds = PINNACLEDataset(
        X_val, y_val, X_wav_val, split="val",
    )
    test_ds = PINNACLEDataset(
        X_test, y_test, X_wav_test, split="test",
    )

    # Create DataLoaders
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

    return train_loader, val_loader, test_loader, y_train
