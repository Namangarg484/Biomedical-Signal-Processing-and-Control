#!/usr/bin/env python3
"""
Generate a consolidated PINNACLE architecture diagram (fig_arch_combined.png/pdf).

Produces a single figure that covers both:
  - fig:arch_overview  (high-level dual-stream overview)
  - fig:branch_details (detailed branch + fusion internals)

These two were previously referenced as image files that did not exist
on disk. This script generates them programmatically and saves a single
combined figure to figures/fig_arch_combined.png/pdf.

Usage:
    cd /path/to/Raman
    python scripts/generate_arch_figure.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

OUT_DIR = "figures"
OUT_PNG = os.path.join(OUT_DIR, "fig_arch_combined.png")
OUT_PDF = os.path.join(OUT_DIR, "fig_arch_combined.pdf")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette (pastel, print-safe)
# ---------------------------------------------------------------------------
C = {
    "input":   "#DDEEFF",
    "spectral":"#C8E6C9",   # green
    "scalo":   "#FFF9C4",   # yellow
    "fusion":  "#F8BBD0",   # pink
    "class":   "#E1BEE7",   # purple
    "arrow":   "#444444",
    "edge":    "#555555",
    "label":   "#111111",
    "sub":     "#555555",
}

FONT = dict(fontsize=8, fontfamily="sans-serif", color=C["label"])
SMALL = dict(fontsize=6.5, fontfamily="sans-serif", color=C["sub"])


# ---------------------------------------------------------------------------
# Helper: draw a labelled rounded box
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, sub="", color="#FFFFFF", fontsize=8, subsize=6.5):
    fancy = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=0.8,
        edgecolor=C["edge"],
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(fancy)
    yoff = 0.025 if sub else 0.0
    ax.text(x, y + yoff, text,
            ha="center", va="center",
            fontsize=fontsize, fontfamily="sans-serif",
            color=C["label"], zorder=4, fontweight="bold")
    if sub:
        ax.text(x, y - 0.055, sub,
                ha="center", va="center",
                fontsize=subsize, fontfamily="sans-serif",
                color=C["sub"], zorder=4)


# ---------------------------------------------------------------------------
# Helper: annotated arrow
# ---------------------------------------------------------------------------
def arrow(ax, x0, y0, x1, y1, label="", lw=1.2):
    ax.annotate("",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw,
                                color=C["arrow"], mutation_scale=8),
                zorder=2)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.01, my, label,
                ha="left", va="center", fontsize=5.5,
                fontfamily="sans-serif", color=C["sub"], zorder=5)


# ===========================================================================
# Panel A — High-level dual-stream overview
# ===========================================================================
def draw_overview(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(a) PINNACLE: Dual-Stream Architecture",
                 fontsize=9, fontweight="bold", pad=4)

    # --- column positions ---
    xin   = 0.10   # input
    xsp   = 0.32   # spectral branch
    xscl  = 0.32   # scalogram branch
    xfus  = 0.62   # separation cross
    xcls  = 0.86   # classifier

    # --- row positions ---
    y_top = 0.72   # spectral row
    y_bot = 0.30   # scalogram row
    y_mid = 0.51   # fusion

    # Inputs
    box(ax, xin, y_top, 0.14, 0.10,
        "Raman\nSpectrum", "(1 × 1000)", color=C["input"], fontsize=7)
    box(ax, xin, y_bot, 0.14, 0.10,
        "CWT\nScalogram", "(3 × 224 × 224)", color=C["input"], fontsize=7)

    # Spectral branch
    box(ax, xsp, y_top, 0.22, 0.12,
        "Spectral Branch", "1D ResNet (4-layer)", color=C["spectral"], fontsize=7.5)

    # Scalogram branch
    box(ax, xscl, y_bot, 0.22, 0.12,
        "Scalogram Branch", "2D CNN (3-layer)", color=C["scalo"], fontsize=7.5)

    # SeparationCross
    box(ax, xfus, y_mid, 0.22, 0.26,
        "Separation\nCross", "α·h_s ×β·h_w\n→ Cross-Attn + γ", color=C["fusion"], fontsize=7.5)

    # Classifier
    box(ax, xcls, y_mid, 0.16, 0.14,
        "Classifier", "256→128→K", color=C["class"], fontsize=7.5)

    # Final output
    box(ax, 0.975, y_mid, 0.08, 0.08,
        "Class\nPred.", "", color="#F0F0F0", fontsize=6.5)

    # Arrows
    arrow(ax, xin + 0.07, y_top, xsp - 0.11, y_top)
    arrow(ax, xin + 0.07, y_bot, xscl - 0.11, y_bot)
    arrow(ax, xsp  + 0.11, y_top, xfus - 0.11, y_mid + 0.05, label="z_s (128)")
    arrow(ax, xscl + 0.11, y_bot, xfus - 0.11, y_mid - 0.05, label="z_w (128)")
    arrow(ax, xfus + 0.11, y_mid, xcls - 0.08, y_mid, label="z_f (256)")
    arrow(ax, xcls + 0.08, y_mid, 0.935, y_mid)

    # Parameter labels
    ax.text(xsp, y_top - 0.10, "140,608 params",
            ha="center", va="center", fontsize=5.5, color=C["sub"])
    ax.text(xscl, y_bot + 0.10, "93,472 params",
            ha="center", va="center", fontsize=5.5, color=C["sub"])
    ax.text(xfus, y_mid + 0.17, "82,561 params",
            ha="center", va="center", fontsize=5.5, color=C["sub"])
    ax.text(xcls, y_mid + 0.10, "33,541 params",
            ha="center", va="center", fontsize=5.5, color=C["sub"])

    # Total
    ax.text(0.5, 0.03,
            "Total: 350,182 trainable parameters  |  95.84% test accuracy  |  2.3 ms inference",
            ha="center", va="center", fontsize=7, color=C["sub"],
            fontstyle="italic")


# ===========================================================================
# Panel B — SpectralBranch detail
# ===========================================================================
def draw_spectral_branch(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) SpectralBranch (1D ResNet)", fontsize=9, fontweight="bold", pad=4)

    layers = [
        ("Input",        "(B,1,1000)",    C["input"]),
        ("Conv1d 1→64\nk=7, s=2", "(B,64,500)",  C["spectral"]),
        ("BN + ReLU",    "",               "#F5F5F5"),
        ("Conv1d 64→128\nk=5",  "(B,128,500)", C["spectral"]),
        ("BN  [residual skip]", "",          "#F5F5F5"),
        ("Conv1d 128→128\nk=3", "(B,128,500)", C["spectral"]),
        ("BN + ReLU\n+ Residual Add", "",   "#E8F5E9"),
        ("Conv1d 128→128\nk=3", "(B,128,500)", C["spectral"]),
        ("BN + ReLU",    "",               "#F5F5F5"),
        ("AdaptiveAvg\nPool1d(1)", "z_s (B,128)", C["fusion"]),
    ]

    n = len(layers)
    xs = np.linspace(0.05, 0.95, n)
    y = 0.50
    bw, bh = 0.085, 0.28

    for i, (txt, sub, col) in enumerate(layers):
        box(ax, xs[i], y, bw, bh, txt, sub, color=col, fontsize=6.5, subsize=5.5)
        if i < n - 1:
            arrow(ax, xs[i] + bw / 2, y, xs[i + 1] - bw / 2, y)

    # Residual arrow (skip from layer 3 to layer 6)
    ri, rj = 3, 6   # 0-indexed
    ax.annotate("",
                xy=(xs[rj], y + bh / 2 + 0.06),
                xytext=(xs[ri], y + bh / 2 + 0.06),
                arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                color="#E53935", mutation_scale=7,
                                connectionstyle="arc3,rad=-0.35"),
                zorder=2)
    ax.text((xs[ri] + xs[rj]) / 2, y + bh / 2 + 0.14,
            "residual skip", ha="center", va="center",
            fontsize=5.5, color="#E53935")


# ===========================================================================
# Panel C — ScalogramBranch detail
# ===========================================================================
def draw_scalogram_branch(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(c) ScalogramBranch (2D CNN)", fontsize=9, fontweight="bold", pad=4)

    layers = [
        ("Input",          "(B,3,224,224)", C["input"]),
        ("Conv2d 3→32\nk=3", "(B,32,224,224)", C["scalo"]),
        ("BN+ReLU\nMaxPool2d(2)", "(B,32,112,112)", "#F5F5F5"),
        ("Conv2d 32→64\nk=3", "(B,64,112,112)", C["scalo"]),
        ("BN+ReLU\nMaxPool2d(2)", "(B,64,56,56)", "#F5F5F5"),
        ("Conv2d 64→128\nk=3", "(B,128,56,56)", C["scalo"]),
        ("BN+ReLU\nMaxPool2d(2)", "(B,128,28,28)", "#F5F5F5"),
        ("AdaptiveAvg\nPool2d(1)", "z_w (B,128)", C["fusion"]),
    ]

    n = len(layers)
    xs = np.linspace(0.05, 0.95, n)
    y = 0.50
    bw, bh = 0.10, 0.30

    for i, (txt, sub, col) in enumerate(layers):
        box(ax, xs[i], y, bw, bh, txt, sub, color=col, fontsize=6.5, subsize=5.5)
        if i < n - 1:
            arrow(ax, xs[i] + bw / 2, y, xs[i + 1] - bw / 2, y)


# ===========================================================================
# Panel D — SeparationCross detail
# ===========================================================================
def draw_separation_cross(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(d) SeparationCross Fusion Module", fontsize=9, fontweight="bold", pad=4)

    # Left: inputs
    box(ax, 0.08, 0.70, 0.11, 0.09, "h_s (B,128,L)", "",   color=C["spectral"], fontsize=6.5)
    box(ax, 0.08, 0.30, 0.11, 0.09, "h_w (B,128,H×W)", "", color=C["scalo"],    fontsize=6.5)

    # Gates
    box(ax, 0.28, 0.70, 0.14, 0.09, "Gate α = σ(W_α·z_s)", "", color=C["spectral"], fontsize=6.5)
    box(ax, 0.28, 0.30, 0.14, 0.09, "Gate β = σ(W_β·z_w)", "", color=C["scalo"],    fontsize=6.5)

    # Gated sequences
    box(ax, 0.50, 0.70, 0.12, 0.09, "α·h_s → Q", "(B,S,128)", color=C["spectral"], fontsize=6.5)
    box(ax, 0.50, 0.30, 0.12, 0.09, "β·h_w → K,V", "(B,P,128)", color=C["scalo"],  fontsize=6.5)

    # Cross-attention
    box(ax, 0.70, 0.50, 0.14, 0.12, "Cross-\nAttention", "softmax(QKᵀ/√d)·V", color=C["fusion"], fontsize=6.5)

    # Residual + concat
    box(ax, 0.88, 0.68, 0.14, 0.08, "z_w + γ·pool(Attn)", "", color="#FFECB3", fontsize=6)
    box(ax, 0.88, 0.50, 0.14, 0.08, "Concat [ · ; z_w ]", "→ z_f (256)", color=C["fusion"], fontsize=6)

    # Arrows
    arrow(ax, 0.135, 0.70, 0.21, 0.70)
    arrow(ax, 0.135, 0.30, 0.21, 0.30)
    arrow(ax, 0.35,  0.70, 0.44, 0.70)
    arrow(ax, 0.35,  0.30, 0.44, 0.30)
    arrow(ax, 0.56,  0.70, 0.63, 0.55)
    arrow(ax, 0.56,  0.30, 0.63, 0.45)
    arrow(ax, 0.77,  0.50, 0.81, 0.68)
    arrow(ax, 0.81,  0.64, 0.81, 0.54)
    arrow(ax, 0.95,  0.50, 0.99, 0.50)

    # γ label
    ax.text(0.84, 0.61, "γ (init=0)", ha="center", va="center",
            fontsize=5.5, color="#E53935", fontstyle="italic")


# ===========================================================================
# Compose the full figure
# ===========================================================================
def main():
    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor("white")

    gs = GridSpec(
        2, 2,
        figure=fig,
        hspace=0.35,
        wspace=0.18,
        left=0.03, right=0.97,
        top=0.94, bottom=0.05,
    )

    ax_ov  = fig.add_subplot(gs[0, :])      # Panel A — full width
    ax_sp  = fig.add_subplot(gs[1, 0])      # Panel B
    ax_sc  = fig.add_subplot(gs[1, 1])      # Panel C

    draw_overview(ax_ov)
    draw_spectral_branch(ax_sp)
    draw_scalogram_branch(ax_sc)

    # Panel D (SeparationCross) in an inset below panels B/C
    ax_sep = fig.add_axes([0.03, 0.02, 0.94, 0.22])
    draw_separation_cross(ax_sep)
    ax_sep.set_title(
        "(d) SeparationCross Fusion Module",
        fontsize=9, fontweight="bold", pad=4,
    )

    fig.suptitle(
        "PINNACLE Architecture: Dual-Stream Spectral–Scalogram Fusion",
        fontsize=11, fontweight="bold", y=0.98,
    )

    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PNG}")
    print(f"Saved → {OUT_PDF}")


if __name__ == "__main__":
    main()
