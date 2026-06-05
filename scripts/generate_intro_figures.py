#!/usr/bin/env python3
"""
PINNACLE — Introduction figures (Fig 1, 2, 3), journal-quality.

Regenerates from REAL Bacteria-ID data:
  Fig 1 : Representative Raman spectrum (E. coli) with annotated
          biochemical bands (matches caption "with biochemical bands").
  Fig 2 : Mean Raman spectra for the five species (real names) with
          +/-1 SD shading and the pyocyanin band highlighted.
  Fig 3 : 1D spectrum -> 2D CWT scalogram pair (real precomputed CWT).

Outputs -> figures/  (and .pdf), 300 DPI.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

SPECIES = ["E. coli", "S. aureus", "P. aeruginosa", "K. pneumoniae", "E. faecalis"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Biochemical bands consistent with draft.tex Section 1 (cm^-1).
BANDS = [
    (800, 900, "Nucleic acids", "#e74c3c"),
    (1000, 1200, "Lipids (C--C/C--H)", "#27ae60"),
    (1200, 1350, "Amide III", "#2980b9"),
    (1350, 1400, "Pyocyanin (P. aeruginosa)", "#8e44ad"),
    (1600, 1700, "Amide I", "#e67e22"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 18,
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
})


def _save(fig, stem):
    fig.savefig(os.path.join(OUT, stem + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{stem}.png / .pdf")


def load_data():
    wn = np.load("data/wavenumbers.npy")
    X = np.load("data/X_2018_proc.npy")
    y = np.load("data/y_2018clinical.npy").astype(int)
    # Ensure strictly increasing wavenumber axis for plotting.
    if wn[0] > wn[-1]:
        wn = wn[::-1].copy()
        X = X[:, ::-1].copy()
    return wn, X, y


def fig1_single(wn, X, y):
    # Representative E. coli (class 0) spectrum.
    idx = np.where(y == 0)[0][5]
    spec = X[idx]
    fig, ax = plt.subplots(figsize=(15.0, 9.0))
    ax.plot(wn, spec, color="#2c3e50", linewidth=2.0, zorder=3)
    ymax = spec.max()
    # Stagger label heights so adjacent bands (Amide III / Pyocyanin) don't collide.
    label_heights = [0.99, 0.99, 0.99, 0.88, 0.99]
    for (lo, hi, label, color), h in zip(BANDS, label_heights):
        ax.axvspan(lo, hi, alpha=0.16, color=color, zorder=1)
        xc = 0.5 * (lo + hi)
        ax.annotate(label, xy=(xc, ymax * h), ha="center", va="top",
                fontsize=14.0, color=color, fontweight="bold", rotation=0)
    ax.set_xlim(wn.min(), wn.max())
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Normalized intensity (a.u.)")
    ax.set_title("Representative Raman Spectrum (\\textit{E. coli})".replace("\\textit{", "").replace("}", ""))
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig1_single_spectrum")


def fig2_class(wn, X, y):
    fig, ax = plt.subplots(figsize=(16.0, 9.0))
    for c in range(5):
        m = y == c
        mean = X[m].mean(axis=0)
        std = X[m].std(axis=0)
        ax.plot(wn, mean, color=COLORS[c], linewidth=2.0, label=SPECIES[c], alpha=0.95, zorder=3)
        ax.fill_between(wn, mean - 0.5 * std, mean + 0.5 * std, color=COLORS[c], alpha=0.12, zorder=1)
    # Highlight the pyocyanin band (P. aeruginosa virulence marker).
    ax.axvspan(1350, 1400, color="#8e44ad", alpha=0.12, zorder=0)
    ax.annotate("Pyocyanin\n(\\textit{P. aeruginosa})".replace("\\textit{", "").replace("}", ""),
                xy=(1375, ax.get_ylim()[1] * 0.92), ha="center", va="top",
                fontsize=14.0, color="#8e44ad", fontweight="bold")
    ax.set_xlim(wn.min(), wn.max())
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Mean normalized intensity (a.u.)")
    ax.set_title("Mean Raman Spectra by Bacterial Species ($\\pm$0.5 SD)")
    ax.legend(loc="upper right", framealpha=0.95, ncol=1)
    ax.grid(True, alpha=0.25, linestyle="--")
    _save(fig, "fig2_class_spectra")


def fig3_wavelet(wn, X, y):
    # Real precomputed CWT scalogram (2019 set, channel 0 = magnitude).
    Xw = np.load("data/X_2019_wavelet.npy", mmap_mode="r")
    yw = np.load("data/y_2019clinical.npy").astype(int)
    idx = int(np.where(yw == 0)[0][5])
    scal = np.array(Xw[idx, 0])
    # Matching 1D spectrum from the 2019 processed set.
    X19 = np.load("data/X_2019_proc.npy")
    wn3 = np.load("data/wavenumbers.npy")
    spec = X19[idx]
    if wn3[0] > wn3[-1]:
        wn3 = wn3[::-1].copy(); spec = spec[::-1].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 8.5))
    ax1.plot(wn3, spec, color="#1f77b4", linewidth=2.0)
    ax1.set_xlim(wn3.min(), wn3.max())
    ax1.set_title("(a) 1D Raman Spectrum")
    ax1.set_xlabel("Raman shift (cm$^{-1}$)")
    ax1.set_ylabel("Normalized intensity (a.u.)")
    ax1.grid(True, alpha=0.25, linestyle="--")

    im = ax2.imshow(scal, aspect="auto", origin="lower", cmap="magma",
                    extent=[wn3.min(), wn3.max(), 1, scal.shape[0]])
    ax2.set_title("(b) 2D CWT Scalogram (Morlet)")
    ax2.set_xlabel("Raman shift (cm$^{-1}$)")
    ax2.set_ylabel("Wavelet scale")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Coefficient magnitude")
    fig.tight_layout()
    _save(fig, "fig3_wavelet_example")


def fig4_training():
    import torch
    ck = torch.load("outputs/checkpoints/best_model.pth", map_location="cpu", weights_only=False)
    h = ck["history"]
    ep = np.arange(1, len(h["train_acc"]) + 1)
    best_ep = int(np.argmax(h["val_acc"])) + 1
    best_val = max(h["val_acc"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 8.5))
    ax1.plot(ep, h["train_acc"], color="#1f77b4", linewidth=2.2, label="Train")
    ax1.plot(ep, h["val_acc"], color="#d62728", linewidth=2.2, label="Validation")
    ax1.axvline(best_ep, color="grey", linestyle=":", linewidth=1.2)
    ax1.scatter([best_ep], [best_val], color="#d62728", zorder=5, s=40)
    ax1.annotate("best %.2f%%\n(epoch %d)" % (best_val, best_ep),
                 xy=(best_ep, best_val), xytext=(-10, -28),
                 textcoords="offset points", ha="right", fontsize=16, color="#d62728")
    ax1.set_title("(a) Training & Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(loc="lower right", framealpha=0.95)
    ax1.grid(True, alpha=0.25, linestyle="--")

    ax2.plot(ep, h["train_loss"], color="#1f77b4", linewidth=2.2, label="Train")
    ax2.plot(ep, h["val_loss"], color="#d62728", linewidth=2.2, label="Validation")
    ax2.set_title("(b) Training & Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-entropy loss")
    ax2.legend(loc="upper right", framealpha=0.95)
    ax2.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save(fig, "fig4_training_curves")


if __name__ == "__main__":
    wn, X, y = load_data()
    print("Generating Fig 1 (representative spectrum + bands)...")
    fig1_single(wn, X, y)
    print("Generating Fig 2 (per-species mean spectra)...")
    fig2_class(wn, X, y)
    print("Generating Fig 3 (1D -> 2D CWT)...")
    fig3_wavelet(wn, X, y)
    print("Generating Fig 4 (training curves)...")
    fig4_training()
    print("Done. All intro figures regenerated from real data.")
