#!/usr/bin/env python3
"""
PINNACLE — 5-Class Data-Scarcity Experiment
============================================
Trains three models at four training-data fractions (20/40/60/80%) and
evaluates each on the FIXED full test set (identical 80/10/10 stratified
split, seed=42). Produces the real, data-backed numbers for tab:scarcity.

Models
------
  raman_only — SpectralBranch + FC head        (RamanOnlyModel)
  no_fusion  — both branches, naive concat      (ConcatFusion)
  pinnacle   — full SeparationCross fusion       (PINNACLE)

Training fractions are stratified subsamples of the fixed train split,
drawn deterministically (seed=42) and NESTED (20% ⊂ 40% ⊂ 60% ⊂ 80%)
so smaller fractions are subsets of larger ones.

All hyper-parameters mirror scripts/ablation_5class.py exactly.

Usage
-----
  caffeinate -i python scripts/scarcity_5class.py
  caffeinate -i python scripts/scarcity_5class.py --no-cache
  python scripts/scarcity_5class.py --epochs 5   # quick smoke test

Output
------
  outputs/scarcity_5class/<model>_<frac>/best.pt
  outputs/scarcity_5class/<model>_<frac>/predictions.npz
  outputs/scarcity_5class/results.json
  LaTeX-ready table rows printed to stdout
"""
import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from pinnacle.utils import set_seed, get_device, logger
from pinnacle.model import PINNACLE
from pinnacle.dataset import PINNACLEDataset, RamanAugmentation, load_data
from scripts.ablation_5class import (
    RamanOnlyModel, ConcatFusion, train_model, wilson_ci, CFG,
)

SPECIES = ["E. coli", "S. aureus", "P. aeruginosa", "K. pneumoniae", "E. faecalis"]
FRACTIONS = [0.20, 0.40, 0.60, 0.80]
SEED = 42

CFG["out_dir"] = "outputs/scarcity_5class"

MODEL_FACTORIES = {
    "raman_only": lambda: RamanOnlyModel(num_classes=5),
    "no_fusion":  lambda: ConcatFusion(num_classes=5),
    "pinnacle":   lambda: PINNACLE(num_classes=5),
}


def build_full_splits(data):
    """Reproduce the exact 80/10/10 stratified split used everywhere else."""
    X_all = np.concatenate([data["X_2018"], data["X_2019"]], axis=0)
    y_all = np.concatenate([data["y_2018"], data["y_2019"]], axis=0)
    X_wav_all = np.concatenate([data["X_2018_wav"], data["X_2019_wav"]], axis=0)

    indices = np.arange(len(X_all))
    idx_train, idx_temp = train_test_split(
        indices, test_size=0.2, random_state=SEED, stratify=y_all)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=SEED, stratify=y_all[idx_temp])

    return {
        "X_train": X_all[idx_train], "y_train": y_all[idx_train],
        "Xw_train": X_wav_all[idx_train],
        "X_val": X_all[idx_val], "y_val": y_all[idx_val],
        "Xw_val": X_wav_all[idx_val],
        "X_test": X_all[idx_test], "y_test": y_all[idx_test],
        "Xw_test": X_wav_all[idx_test],
    }


def nested_fraction_indices(y_train, fractions, seed=SEED):
    """
    Return a dict frac -> indices (into the train arrays), stratified and
    NESTED: the indices for a smaller fraction are a subset of the larger.
    Built by sorting a single deterministic random key per class.
    """
    rng = np.random.RandomState(seed)
    # Per-sample priority key; lower key = selected earlier.
    key = rng.rand(len(y_train))
    classes = np.unique(y_train)
    out = {}
    for frac in fractions:
        sel = []
        for c in classes:
            c_idx = np.where(y_train == c)[0]
            order = c_idx[np.argsort(key[c_idx])]
            n_keep = max(1, int(round(frac * len(c_idx))))
            sel.append(order[:n_keep])
        out[frac] = np.sort(np.concatenate(sel))
    return out


def make_loaders(splits, train_idx, batch_size):
    raman_aug = RamanAugmentation()
    train_ds = PINNACLEDataset(
        splits["X_train"][train_idx], splits["y_train"][train_idx],
        splits["Xw_train"][train_idx], transform_raman=raman_aug, split="train")
    val_ds = PINNACLEDataset(
        splits["X_val"], splits["y_val"], splits["Xw_val"], split="val")
    test_ds = PINNACLEDataset(
        splits["X_test"], splits["y_test"], splits["Xw_test"], split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="PINNACLE 5-class data-scarcity experiment")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.epochs is not None:
        CFG["epochs"] = args.epochs

    set_seed(SEED)
    device = torch.device("cpu") if args.cpu else get_device()
    logger.info(f"Device: {device}")

    os.makedirs(CFG["out_dir"], exist_ok=True)

    logger.info("=" * 66)
    logger.info("Loading 5-class data and reproducing fixed split...")
    data = load_data(CFG["data_dir"])
    splits = build_full_splits(data)
    n_train_full = len(splits["y_train"])
    n_test = len(splits["y_test"])
    logger.info(f"Full train={n_train_full}  test={n_test}")

    frac_idx = nested_fraction_indices(splits["y_train"], FRACTIONS, seed=SEED)
    for f in FRACTIONS:
        logger.info(f"  frac {int(f*100)}%: {len(frac_idx[f])} train samples")
    logger.info("=" * 66)

    results = {}  # model -> {frac_pct: {acc, ci_lo, ci_hi, n_train}}
    for model_name, factory in MODEL_FACTORIES.items():
        results[model_name] = {}
        for frac in FRACTIONS:
            pct = int(round(frac * 100))
            run_name = f"{model_name}_{pct}"
            train_idx = frac_idx[frac]
            train_loader, val_loader, test_loader = make_loaders(
                splits, train_idx, CFG["batch_size"])

            logger.info("")
            logger.info("#" * 66)
            logger.info(f"#  {model_name}  @  {pct}% train ({len(train_idx)} samples)")
            logger.info("#" * 66)

            model = factory()
            preds, labels, test_acc = train_model(
                run_name, model, train_loader, val_loader, test_loader,
                device, CFG["out_dir"], no_cache=args.no_cache)

            ci_lo, ci_hi = wilson_ci(test_acc / 100.0, len(labels))
            results[model_name][pct] = {
                "acc": round(float(test_acc), 2),
                "ci_lo": round(float(ci_lo), 2),
                "ci_hi": round(float(ci_hi), 2),
                "n_train": int(len(train_idx)),
            }
            logger.info(f"  >>> {run_name}: {test_acc:.2f}% "
                        f"[{ci_lo:.2f}, {ci_hi:.2f}]")

    out_json = os.path.join(CFG["out_dir"], "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results -> {out_json}")

    # ---- LaTeX-ready rows ----
    pretty = {"raman_only": "Raman-only", "no_fusion": "No-fusion",
              "pinnacle": "PINNACLE"}
    print("\n" + "=" * 66)
    print("tab:scarcity rows (test acc %)")
    print("=" * 66)
    for m in ["raman_only", "no_fusion", "pinnacle"]:
        cells = " & ".join(f"{results[m][p]['acc']:.1f}"
                           for p in [20, 40, 60, 80])
        print(f"  {pretty[m]:16s} & {cells} \\\\")

    p20 = results["pinnacle"][20]["acc"]
    r20 = results["raman_only"][20]["acc"]
    n20 = results["no_fusion"][20]["acc"]
    print(f"\n  @20%: PINNACLE - Raman-only = {p20 - r20:+.1f} pp")
    print(f"  @20%: PINNACLE - No-fusion  = {p20 - n20:+.1f} pp")


if __name__ == "__main__":
    main()
