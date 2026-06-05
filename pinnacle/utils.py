"""
PINNACLE — Utility functions
Seed management, logging, device detection for macOS / CPU / MPS.
"""

import os
import random
import logging
import torch
import numpy as np


def get_logger(name: str = "PINNACLE", level: int = logging.INFO) -> logging.Logger:
    """Create a formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger()


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS (Apple Silicon) doesn't support manual_seed_all but manual_seed covers it
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")


def get_device(requested: str = "auto") -> torch.device:
    """
    Auto-detect best available device.
    Priority: cuda > mps > cpu
    """
    if requested != "auto":
        device = torch.device(requested)
        logger.info(f"Using requested device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
