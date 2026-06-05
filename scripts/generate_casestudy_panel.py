#!/usr/bin/env python3
"""
Generate a multi-species case-study panel (fig_casestudy_panel.png/pdf).

Shows 6 test samples — 3 "rescued" (PINNACLE correct, Raman-only wrong)
and 3 "failed" (PINNACLE wrong but representative of hard cases) — spread
across the 5 bacterial classes.

Each panel column shows:
  Row 1: Noisy 1D Raman spectrum (with wavenumbers)
  Row 2: CWT scalogram image
  Title: True | PINNACLE | Raman-only prediction

Usage:
    cd /path/to/Raman
    python scripts/generate_casestudy_panel.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
DATA_DIR    = "data/"
OUT_PNG     = "figures/fig_casestudy_panel.png"
OUT_PDF     = "figures/fig_casestudy_panel.pdf"

PINNACLE_PRED = "outputs/predictions_current.npz"
RAMAN_PRED    = "outputs/ablation_5class/raman_only/predictions.npz"

SEED = 42

CLASS_NAMES = [
    "E. coli",
    "S. aureus",
    "P. aeruginosa",
    "K. pneumoniae",
    "E. faecalis",
]

os.makedirs("figures", exist_ok=True)

# -------------------------------------------------------------------------
# Load data + predictions
# -------------------------------------------------------------------------
def load_test_set():
    X_2018 = np.load(os.path.join(DATA_DIR, "X_2018_proc.npy")).astype(np.float32)
    X_2019 = np.load(os.path.join(DATA_DIR, "X_2019_proc.npy")).astype(np.float32)
    y_2018 = np.load(os.path.join(DATA_DIR, "y_2018clinical.npy")).astype(np.int64)
    y_2019 = np.load(os.path.join(DATA_DIR, "y_2019clinical.npy")).astype(np.int64)

    X_all = np.concatenate([X_2018, X_2019], axis=0)
    y_all = np.concatenate([y_2018, y_2019], axis=0)

    wav_2018 = os.path.join(DATA_DIR, "X_2018_wavelet.npy")
    wav_2019 = os.path.join(DATA_DIR, "X_2019_wavelet.npy")
    W_all = None
    if os.path.exists(wav_2018):
        W_all = np.concatenate([
            np.load(wav_2018).astype(np.float32),
            np.load(wav_2019).astype(np.float32)
        ], axis=0)

    idx = np.arange(len(X_all))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=y_all)
    idx_val, idx_te  = train_test_split(idx_tmp, test_size=0.50, random_state=SEED, stratify=y_all[idx_tmp])

    X_test = X_all[idx_te]
    y_test = y_all[idx_te]
    W_test = W_all[idx_te] if W_all is not None else None

    wav_axis = None
    wav_path = os.path.join(DATA_DIR, "wavenumbers.npy")
    if os.path.exists(wav_path):
        wav_axis = np.load(wav_path)

    print(f"Test set: {len(X_test)} samples")
    return X_test, y_test, W_test, wav_axis


def pick_cases(y_true, pin_pred, ram_pred):
    """
    Pick 3 rescued + 3 failed cases, spread across classes.
    Rescued: PINNACLE correct & Raman wrong
    Failed:  PINNACLE wrong (and ideally Raman correct for contrast)
    """
    rescued = np.where((pin_pred == y_true) & (ram_pred != y_true))[0]
    failed  = np.where((pin_pred != y_true))[0]

    # Try to pick one from each class (for rescued), fallback to random
    chosen_rescued, used_classes = [], set()
    for cls in range(len(CLASS_NAMES)):
        cls_rescued = rescued[y_true[rescued] == cls]
        if len(cls_rescued) > 0 and cls not in used_classes:
            np.random.seed(SEED + cls)
            chosen_rescued.append(int(np.random.choice(cls_rescued)))
            used_classes.add(cls)
        if len(chosen_rescued) == 3:
            break
    # Fill any remaining with random
    if len(chosen_rescued) < 3:
        rem = [i for i in rescued if i not in chosen_rescued]
        np.random.seed(SEED + 99)
        np.random.shuffle(rem)
        chosen_rescued.extend(rem[:3 - len(chosen_rescued)])

    chosen_failed, used_classes2 = [], set()
    for cls in range(len(CLASS_NAMES)):
        cls_failed = failed[y_true[failed] == cls]
        if len(cls_failed) > 0 and cls not in used_classes2:
            np.random.seed(SEED + cls + 10)
            chosen_failed.append(int(np.random.choice(cls_failed)))
            used_classes2.add(cls)
        if len(chosen_failed) == 3:
            break
    if len(chosen_failed) < 3:
        rem = [i for i in failed if i not in chosen_failed]
        np.random.seed(SEED + 100)
        np.random.shuffle(rem)
        chosen_failed.extend(rem[:3 - len(chosen_failed)])

    return chosen_rescued[:3], chosen_failed[:3]


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------
def plot_spectrum(ax, spectrum, wav_axis, color, title=""):
    x = wav_axis if wav_axis is not None else np.arange(len(spectrum))
    ax.plot(x, spectrum, lw=1.4, color=color, alpha=0.85)
    ax.set_yticks([])
    if wav_axis is not None:
        ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=11)
    else:
        ax.set_xlabel("Feature index", fontsize=11)
    ax.tick_params(axis='x', labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10.5, pad=3)


def plot_scalogram(ax, scalo):
    # scalo: (3, H, W) or (H, W)
    if scalo.ndim == 3:
        # Use channel mean for display
        img = scalo.mean(axis=0)
    else:
        img = scalo
    ax.imshow(img, aspect="auto", origin="lower",
              cmap="inferno", interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Scale", fontsize=11)


def make_panel(indices, category, y_true, pin_pred, ram_pred,
               X_test, W_test, wav_axis):
    """Returns (fig, axes_list) for 3 cases."""
    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 4.0))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        true_c = y_true[idx]
        pin_c  = pin_pred[idx]
        ram_c  = ram_pred[idx]

        color_spec = "#2196F3" if category == "rescued" else "#F44336"

        # Title with truth / PINNACLE / Raman-only
        pin_ok  = "✓" if pin_c  == true_c else "✗"
        ram_ok  = "✓" if ram_c  == true_c else "✗"
        col_title = (
            f"True: {CLASS_NAMES[true_c]}\n"
            f"PINNACLE {pin_ok}: {CLASS_NAMES[pin_c]}\n"
            f"Raman-only {ram_ok}: {CLASS_NAMES[ram_c]}"
        )

        # Spectrum row
        ax_sp = axes[0, col]
        plot_spectrum(ax_sp, X_test[idx], wav_axis, color=color_spec)
        ax_sp.set_title(col_title, fontsize=6, pad=3,
                        color="#1a6b1a" if category == "rescued" else "#8b0000")

        # Scalogram row
        ax_sc = axes[1, col]
        if W_test is not None:
            plot_scalogram(ax_sc, W_test[idx])
        else:
            ax_sc.text(0.5, 0.5, "Scalogram\nnot available",
                       ha="center", va="center", transform=ax_sc.transAxes, fontsize=7)
            ax_sc.set_xticks([]); ax_sc.set_yticks([])

    # Row labels
    axes[0, 0].set_ylabel("Raman\nSpectrum", fontsize=7, labelpad=4)
    axes[1, 0].set_ylabel("CWT\nScalogram", fontsize=7, labelpad=4)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def main():
    X_test, y_test, W_test, wav_axis = load_test_set()

    pin_data = np.load(PINNACLE_PRED)
    ram_data = np.load(RAMAN_PRED)

    pin_pred = pin_data["y_pred"]
    ram_pred = ram_data["y_pred"]
    y_true   = pin_data["y_true"]

    # Verify alignment
    assert np.array_equal(y_true, ram_data["y_true"]), "Label mismatch between prediction files!"
    assert len(y_true) == len(X_test), "Prediction count doesn't match test set size!"

    rescued_idx, failed_idx = pick_cases(y_true, pin_pred, ram_pred)
    print(f"Rescued cases: {rescued_idx}")
    print(f"Failed cases:  {failed_idx}")

    # ---- Combined 6-panel figure ----
    fig = plt.figure(figsize=(26.0, 11.0))
    fig.patch.set_facecolor("white")

    gs_top = fig.add_gridspec(
        2, 6, hspace=0.55, wspace=0.12,
        left=0.06, right=0.98, top=0.85, bottom=0.12
    )

    cases = rescued_idx + failed_idx
    categories = ["rescued"] * 3 + ["failed"] * 3

    for col, (idx, cat) in enumerate(zip(cases, categories)):
        true_c = y_true[idx]
        pin_c  = pin_pred[idx]
        ram_c  = ram_pred[idx]

        pin_ok = "✓" if pin_c == true_c else "✗"
        ram_ok = "✓" if ram_c == true_c else "✗"
        col_title = (
            f"True: {CLASS_NAMES[true_c]}\n"
            f"PINNACLE {pin_ok}: {CLASS_NAMES[pin_c]}\n"
            f"Raman-only {ram_ok}: {CLASS_NAMES[ram_c]}"
        )
        title_color = "#2e7d32" if cat == "rescued" else "#b71c1c"
        spec_color  = "#1565C0" if cat == "rescued" else "#C62828"

        ax_sp = fig.add_subplot(gs_top[0, col])
        ax_sc = fig.add_subplot(gs_top[1, col])

        plot_spectrum(ax_sp, X_test[idx], wav_axis, color=spec_color)
        ax_sp.set_title(col_title, fontsize=10.5, pad=3, color=title_color)
        if col == 0:
            ax_sp.set_ylabel("Intensity\n(a.u.)", fontsize=12)
        if col == 0:
            ax_sc.set_ylabel("CWT Scale", fontsize=12)

        if W_test is not None:
            plot_scalogram(ax_sc, W_test[idx])
        else:
            ax_sc.text(0.5, 0.5, "N/A",
                       ha="center", va="center",
                       transform=ax_sc.transAxes, fontsize=7)
            ax_sc.set_xticks([]); ax_sc.set_yticks([])

    # Dividing line between rescued and failed groups
    fig.add_artist(matplotlib.lines.Line2D(
        [0.505, 0.505], [0.08, 0.90],
        transform=fig.transFigure,
        color="#AAAAAA", lw=1.0, linestyle="--",
    ))

    # Group labels
    fig.text(0.26, 0.93, "Rescued by Scalogram (PINNACLE correct, Raman-only wrong)",
             ha="center", va="bottom", fontsize=14, color="#2e7d32", fontweight="bold")
    fig.text(0.76, 0.93, "Hard Cases (PINNACLE incorrect — residual errors)",
             ha="center", va="bottom", fontsize=14, color="#b71c1c", fontweight="bold")

    fig.suptitle(
        "Multi-Species Case Study: SeparationCross Fusion Rescues Ambiguous Raman Spectra",
        fontsize=17, fontweight="bold", y=0.99,
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_PNG}")
    print(f"Saved → {OUT_PDF}")


import matplotlib.lines  # needed for Line2D

if __name__ == "__main__":
    main()
