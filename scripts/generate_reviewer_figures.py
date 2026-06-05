#!/usr/bin/env python3
"""
PINNACLE — Reviewer-requested figure improvements.

Generates:
  Fig 5  : Confusion matrix (% annotations, fixed labels, bold off-diag)
  Fig 6  : Radar / spider chart for per-class P/R/F1
  Fig 7  : Side-by-side t-SNE: Raman-only (128-D) vs PINNACLE fused (256-D)
  Fig 8  : Attribution map — Integrated Gradients overlaid on Raman spectrum

All outputs saved to outputs/figures/ as .png and .pdf at 300 DPI.

Usage:
    python scripts/generate_reviewer_figures.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinnacle.utils import set_seed, get_device, logger
from pinnacle.model import PINNACLE
from pinnacle.dataset import load_data, create_dataloaders

# Constants
SPECIES = ["E. coli", "S. aureus", "P. aeruginosa", "K. pneumoniae", "E. faecalis"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
OUT_DIR = "outputs/figures"
CKPT = "outputs/checkpoints/best_model.pth"
PREDS = "outputs/predictions_current.npz"
DATA_DIR = "data"
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# Publication style tuned for single-column readability (larger final render)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "lines.linewidth": 2.2,
})


def _save(fig, stem):
    p = os.path.join(OUT_DIR, stem)
    fig.savefig(p + ".png", bbox_inches="tight")
    fig.savefig(p + ".pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {p}.png / .pdf")


def _refresh_predictions(model, test_loader, device):
    """Always refresh predictions from the loaded checkpoint for data traceability."""
    logger.info("Generating predictions from current checkpoint...")
    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for raman, scalogram, labels in test_loader:
            logits, _, _ = model(raman.to(device), scalogram.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(labels.numpy())
            all_probs.append(probs)

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    y_probs = np.concatenate(all_probs)
    acc = np.mean(y_pred == y_true) * 100
    logger.info(f"  Test accuracy: {acc:.2f}%")

    np.savez(PREDS, y_pred=y_pred, y_true=y_true, y_probs=y_probs)
    logger.info(f"  Predictions saved: {PREDS}")


def fig5_confusion_matrix():
    logger.info("Generating Fig 5: Confusion Matrix (improved)...")

    if not os.path.exists(PREDS):
        logger.warning(f"  Predictions file not found: {PREDS}")
        return

    data = np.load(PREDS)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    acc = np.mean(y_true == y_pred) * 100
    n_cls = cm.shape[0]
    labels = SPECIES[:n_cls]

    fig, ax = plt.subplots(figsize=(12.5, 11.0))

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (%)", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    thresh = 50.0
    for i in range(n_cls):
        for j in range(n_cls):
            val = cm_pct[i, j]
            count = cm[i, j]
            color = "white" if val > thresh else "black"
            is_diag = i == j
            fs = 15 if is_diag else 13
            fw = "bold" if (not is_diag and val > 1.0) else "normal"
            ax.text(
                j,
                i,
                f"{val:.1f}%\n({count})",
                ha="center",
                va="center",
                fontsize=fs,
                fontweight=fw,
                color=color,
            )

    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=16)
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel("Predicted Label", fontsize=20, labelpad=8)
    ax.set_ylabel("True Label", fontsize=20, labelpad=8)
    ax.set_title(
        f"PINNACLE Confusion Matrix — Test Accuracy: {acc:.2f}%\n"
        f"(1,250 test samples, seed 42)",
        fontsize=20,
        pad=10,
    )

    fig.tight_layout()
    _save(fig, "fig5_confusion_matrix")


def fig6_radar_perclass():
    logger.info("Generating Fig 6: Radar chart (per-class P/R/F1)...")

    if not os.path.exists(PREDS):
        logger.warning(f"  Predictions file not found: {PREDS}")
        return

    data = np.load(PREDS)
    y_true = data["y_true"]
    y_pred = data["y_pred"]

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    prec *= 100
    rec *= 100
    f1 *= 100

    n_cls = len(SPECIES)
    metrics = {"Precision": prec, "Recall": rec, "F1-Score": f1}
    metric_colors = {
        "Precision": "#3498db",
        "Recall": "#e74c3c",
        "F1-Score": "#2ecc71",
    }

    angles = np.linspace(0, 2 * np.pi, n_cls, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12.5, 12.0), subplot_kw=dict(polar=True))

    for metric_name, values in metrics.items():
        vals = list(values) + [values[0]]
        ax.plot(
            angles,
            vals,
            color=metric_colors[metric_name],
            linewidth=2.5,
            linestyle="-",
            label=metric_name,
        )
        ax.fill(angles, vals, color=metric_colors[metric_name], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(SPECIES[:n_cls], size=18)
    ax.set_rlabel_position(30)
    ax.set_ylim(80, 100)
    ax.set_yticks([82, 86, 90, 94, 98])
    ax.set_yticklabels(["82", "86", "90", "94", "98"], size=13, color="grey")

    ax.grid(color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.spines["polar"].set_visible(False)

    ax.set_title(
        "Per-Class Performance — PINNACLE\n(Precision / Recall / F1-Score, %)",
        size=22,
        pad=28,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.40, 1.17), fontsize=15)

    fig.tight_layout()
    _save(fig, "fig6_perclass_radar")


def _extract_embeddings(model, test_loader, device):
    """
    Returns:
        z_raman   : (N, 128)  — SpectralBranch pooled output
        z_fused   : (N, 256)  — SeparationCross fused output
        labels    : (N,)
    """
    model.eval()
    raman_embs, fused_embs, all_labels = [], [], []

    with torch.no_grad():
        for raman, scalogram, labels in test_loader:
            raman = raman.to(device)
            scalogram = scalogram.to(device)

            z_s = model.spectral_branch(raman)
            h_s = model.spectral_branch.forward_features(raman)
            h_w = model.scalogram_branch.forward_features(scalogram)
            z_fused, _, _ = model.fusion(h_s, h_w)

            raman_embs.append(z_s.cpu().numpy())
            fused_embs.append(z_fused.cpu().numpy())
            all_labels.append(labels.numpy())

    return (
        np.concatenate(raman_embs, axis=0),
        np.concatenate(fused_embs, axis=0),
        np.concatenate(all_labels, axis=0),
    )


def fig7_tsne_sidebyside(model, test_loader, device):
    logger.info("Generating Fig 7: Side-by-side t-SNE (Raman-only vs PINNACLE)...")

    z_raman, z_fused, labels = _extract_embeddings(model, test_loader, device)

    logger.info(f"  Running t-SNE on Raman-only ({z_raman.shape})...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=40, max_iter=1000)
    tsne_raman = tsne.fit_transform(z_raman)

    logger.info(f"  Running t-SNE on PINNACLE fused ({z_fused.shape})...")
    tsne2 = TSNE(n_components=2, random_state=SEED, perplexity=40, max_iter=1000)
    tsne_fused = tsne2.fit_transform(z_fused)

    fig, axes = plt.subplots(1, 2, figsize=(20.0, 9.5))
    panels = [
        (axes[0], tsne_raman, "1D Raman-Only Branch (128-D)"),
        (axes[1], tsne_fused, "PINNACLE Fused Representation (256-D)"),
    ]

    for ax, emb, title in panels:
        for cls_idx in range(len(SPECIES)):
            mask = labels == cls_idx
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                c=COLORS[cls_idx],
                label=SPECIES[cls_idx],
                alpha=0.65,
                s=32,
                edgecolors="none",
                linewidths=0,
            )
        ax.set_title(title, fontsize=18, pad=8)
        ax.set_xlabel("t-SNE Component 1", fontsize=16)
        ax.set_ylabel("t-SNE Component 2", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    patches = [mpatches.Patch(color=COLORS[i], label=SPECIES[i]) for i in range(len(SPECIES))]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=len(SPECIES),
        fontsize=15,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )
    fig.suptitle(
        "t-SNE Feature Visualisation: Raman-Only vs PINNACLE Fusion\n"
        "(Test Set, 1,250 samples, seed 42)",
        fontsize=19,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig7_tsne_sidebyside")


def _integrated_gradients(model, raman_input, scalogram_input, target_class, n_steps=50, device="cpu"):
    """
    Compute Integrated Gradients w.r.t. the Raman spectrum input.

    baseline = zero spectrum.
    Returns: attributions (seq_len,) numpy array.
    """
    baseline = torch.zeros_like(raman_input)
    raman_input = raman_input.to(device)
    scalogram_input = scalogram_input.to(device)
    baseline = baseline.to(device)

    alphas = torch.linspace(0, 1, n_steps).to(device)

    grads = []
    for alpha in alphas:
        interp = baseline + alpha * (raman_input - baseline)
        interp = interp.unsqueeze(0).requires_grad_(True)
        scal = scalogram_input.unsqueeze(0)

        logits, _, _ = model(interp, scal)
        score = logits[0, target_class]
        score.backward()
        grads.append(interp.grad.squeeze(0).cpu().detach().numpy())

    grads = np.stack(grads, axis=0)
    avg_g = grads.mean(axis=0)
    delta = (raman_input - baseline).cpu().numpy().squeeze()
    ig = avg_g * delta
    return ig


def fig8_attribution_maps(model, test_loader, device, wavenumbers=None):
    logger.info("Generating Fig 8: Attribution maps (Integrated Gradients)...")

    model.eval()

    samples = {}
    for raman, scalogram, labels in test_loader:
        for b in range(len(labels)):
            cls = int(labels[b].item())
            if cls not in samples and len(samples) < len(SPECIES):
                with torch.no_grad():
                    logits, _, _ = model(
                        raman[b:b+1].to(device),
                        scalogram[b:b+1].to(device),
                    )
                pred = logits.argmax(dim=1).item()
                if pred == cls:
                    samples[cls] = (raman[b].clone(), scalogram[b].clone())
        if len(samples) == len(SPECIES):
            break

    if not samples:
        logger.warning("  No correctly classified samples found.")
        return

    x_axis = wavenumbers if wavenumbers is not None else np.arange(1000)
    xlabel = "Wavenumber (cm$^{-1}$)" if wavenumbers is not None else "Spectral index"

    bands = [
        (785, 875, "Nucleic acids", "#e74c3c"),
        (1000, 1100, "Lipids C-C", "#27ae60"),
        (1200, 1350, "Amide III", "#2980b9"),
        (1350, 1420, "Pyocyanin", "#8e44ad"),
        (1600, 1700, "Amide I", "#e67e22"),
    ]

    n_cls = len(samples)
    fig, axes = plt.subplots(n_cls, 1, figsize=(18.0, 4.5 * n_cls), sharex=True)
    if n_cls == 1:
        axes = [axes]

    for cls_idx in sorted(samples.keys()):
        ax = axes[cls_idx]
        raman_t, scal_t = samples[cls_idx]

        ig = _integrated_gradients(model, raman_t, scal_t, cls_idx, n_steps=50, device=device)
        spectrum = raman_t.numpy()

        ig_norm = ig / (np.abs(ig).max() + 1e-9)

        ax.plot(x_axis, spectrum, color="#2c3e50", linewidth=1.4, zorder=3, label="Raman spectrum")

        pos_ig = np.where(ig_norm > 0, ig_norm, 0)
        neg_ig = np.where(ig_norm < 0, ig_norm, 0)
        ax.fill_between(
            x_axis,
            0,
            pos_ig * spectrum.max() * 0.5,
            color="#e74c3c",
            alpha=0.45,
            label="Positive attribution",
            zorder=2,
        )
        ax.fill_between(
            x_axis,
            0,
            neg_ig * spectrum.max() * 0.5,
            color="#3498db",
            alpha=0.35,
            label="Negative attribution",
            zorder=2,
        )

        if wavenumbers is not None:
            wn_min, wn_max = x_axis.min(), x_axis.max()
            for lo, hi, bname, bcolor in bands:
                if lo >= wn_min and hi <= wn_max:
                    ax.axvspan(lo, hi, alpha=0.08, color=bcolor, zorder=1)
                    mid = (lo + hi) / 2
                    ax.text(
                        mid,
                        spectrum.max() * 1.02,
                        bname,
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color=bcolor,
                        rotation=90,
                        clip_on=True,
                    )

        name = SPECIES[cls_idx] if cls_idx < len(SPECIES) else f"Class {cls_idx}"
        ax.set_ylabel("Intensity (a.u.)", fontsize=13)
        ax.set_title(f"{name}", fontsize=14, color=COLORS[cls_idx], fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(x_axis[0], x_axis[-1])

        if cls_idx == 0:
            ax.legend(loc="upper right", fontsize=11, framealpha=0.7)

    axes[-1].set_xlabel(xlabel, fontsize=14)
    fig.suptitle(
        "Integrated Gradients Attribution Maps — PINNACLE Spectral Branch\n"
        "(Shading: positive=red, negative=blue; band annotations shown for reference)",
        fontsize=16,
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig8_attribution_maps")


def main():
    set_seed(SEED)
    device = get_device("auto")

    logger.info("=" * 70)
    logger.info("PINNACLE — Reviewer Figure Generation")
    logger.info("=" * 70)

    if not os.path.exists(CKPT):
        logger.error(f"Checkpoint not found: {CKPT}")
        sys.exit(1)

    data = load_data(DATA_DIR, remap=True)
    _, _, test_loader, _ = create_dataloaders(
        data,
        batch_size=32,
        seed=SEED,
        num_workers=0,
        use_augmentation=False,
    )
    wavenumbers = data.get("wavenumbers", None)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = PINNACLE(num_classes=5, embed_dim=128, dropout=0.3, mode="fusion")
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    logger.info(
        f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
        f"best val acc = {ckpt.get('best_val_acc', '?'):.2f}%"
    )

    _refresh_predictions(model, test_loader, device)

    fig5_confusion_matrix()
    fig6_radar_perclass()
    fig7_tsne_sidebyside(model, test_loader, device)
    fig8_attribution_maps(model, test_loader, device, wavenumbers=wavenumbers)

    logger.info("=" * 70)
    logger.info("All reviewer figures generated.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
