"""
PINNACLE — Visualization and figure generation.
Generates publication-quality figures matching draft.tex Figs 1–10.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for macOS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, Optional, List

from pinnacle.utils import logger


# Publication style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

SPECIES_NAMES = [
    "E. coli",
    "S. aureus",
    "P. aeruginosa",
    "K. pneumoniae",
    "E. faecalis",
]

SPECIES_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_single_spectrum(
    spectrum: np.ndarray,
    wavenumbers: Optional[np.ndarray] = None,
    output_path: str = "outputs/figures/fig1_single_spectrum.png",
):
    """Fig 1: Representative Raman spectrum with band annotations."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = wavenumbers if wavenumbers is not None else np.arange(len(spectrum))
    ax.plot(x, spectrum, color="#2c3e50", linewidth=1.0)

    # Annotate key bands (if wavenumber axis is available)
    if wavenumbers is not None:
        bands = [
            (800, 900, "Nucleic\nacids", "#e74c3c"),
            (1000, 1200, "Lipids\nC-C/C-H", "#27ae60"),
            (1200, 1350, "Amide III", "#2980b9"),
            (1350, 1400, "Pyocyanin\n(P. aeruginosa)", "#8e44ad"),
            (1600, 1700, "Amide I", "#e67e22"),
        ]
        for lo, hi, label, color in bands:
            ax.axvspan(lo, hi, alpha=0.15, color=color, label=label)

        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.legend(loc="upper right", fontsize=9)
    else:
        ax.set_xlabel("Spectral point index")

    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Representative Preprocessed Raman Spectrum")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.replace(".png", ".pdf"))
    plt.close(fig)
    logger.info(f"  📊 Fig 1 saved: {output_path}")


def plot_class_spectra(
    X: np.ndarray,
    y: np.ndarray,
    wavenumbers: Optional[np.ndarray] = None,
    output_path: str = "outputs/figures/fig2_class_spectra.png",
):
    """Fig 2: Mean spectra per class with ±1 SD bands."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = wavenumbers if wavenumbers is not None else np.arange(X.shape[1])
    classes = sorted(np.unique(y))

    for cls_idx, cls in enumerate(classes):
        mask = y == cls
        mean = X[mask].mean(axis=0)
        std = X[mask].std(axis=0)
        name = SPECIES_NAMES[cls_idx] if cls_idx < len(SPECIES_NAMES) else f"Class {cls}"
        color = SPECIES_COLORS[cls_idx % len(SPECIES_COLORS)]

        ax.plot(x, mean, color=color, linewidth=1.2, label=name)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)" if wavenumbers is not None else "Spectral index")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("Mean Raman Spectra by Bacterial Species (±1 SD)")
    ax.legend(loc="upper right")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.replace(".png", ".pdf"))
    plt.close(fig)
    logger.info(f"  📊 Fig 2 saved: {output_path}")


def plot_training_curves(
    history: Dict,
    output_path: str = "outputs/figures/fig4_training_curves.png",
):
    """Fig 4: Training and validation accuracy/loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_acc"]) + 1)

    # Accuracy
    ax1.plot(epochs, history["train_acc"], "b-", linewidth=1.5, label="Train")
    ax1.plot(epochs, history["val_acc"], "r-", linewidth=1.5, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Training & Validation Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, history["train_loss"], "b-", linewidth=1.5, label="Train")
    ax2.plot(epochs, history["val_loss"], "r-", linewidth=1.5, label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training & Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.replace(".png", ".pdf"))
    plt.close(fig)
    logger.info(f"  📊 Fig 4 saved: {output_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str = "outputs/figures/fig5_confusion_matrix.png",
):
    """Fig 5: Confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))

    n_classes = cm.shape[0]
    labels = SPECIES_NAMES[:n_classes]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — PINNACLE")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.replace(".png", ".pdf"))
    plt.close(fig)
    logger.info(f"  📊 Fig 5 saved: {output_path}")


def plot_perclass_metrics(
    per_class: Dict,
    output_path: str = "outputs/figures/fig6_perclass_metrics.png",
):
    """Fig 6: Per-class precision, recall, F1 bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(per_class.keys())
    n = len(names)
    x = np.arange(n)
    width = 0.25

    prec = [per_class[n]["precision"] for n in names]
    rec = [per_class[n]["recall"] for n in names]
    f1 = [per_class[n]["f1"] for n in names]

    ax.bar(x - width, prec, width, label="Precision", color="#3498db", alpha=0.85)
    ax.bar(x, rec, width, label="Recall", color="#e74c3c", alpha=0.85)
    ax.bar(x + width, f1, width, label="F1-Score", color="#2ecc71", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Per-Class Performance Metrics")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.replace(".png", ".pdf"))
    plt.close(fig)
    logger.info(f"  📊 Fig 6 saved: {output_path}")


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str = "outputs/figures/fig7_tsne_features.png",
):
    """Fig 7: t-SNE of learned feature embeddings."""
    logger.info("  Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))

    classes = sorted(np.unique(labels))
    for cls_idx, cls in enumerate(classes):
        mask = labels == cls
        name = SPECIES_NAMES[cls_idx] if cls_idx < len(SPECIES_NAMES) else f"Class {cls}"
        color = SPECIES_COLORS[cls_idx % len(SPECIES_COLORS)]
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=color, label=name, alpha=0.6, s=15, edgecolors="none",
        )

    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title("t-SNE of Fused Feature Space (Test Set)")
    ax.legend(markerscale=3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.replace(".png", ".pdf"))
    plt.close(fig)
    logger.info(f"  📊 Fig 7 saved: {output_path}")
