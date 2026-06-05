#!/usr/bin/env python3
"""
Compute the gate-discordance stratification claimed in draft.tex:
  "samples with |‖alpha‖_1 - ‖beta‖_1| < 0.2 achieve 96.3%, only 78.4%
   for high-discordance samples."

Loads the trained PINNACLE checkpoint, runs the fixed test set, collects
per-sample gates (alpha, beta), computes the L1-norm discordance, and
reports accuracy in the low- vs high-discordance strata.

No training. Read-only over outputs/checkpoints/best_model.pth.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from pinnacle.utils import set_seed, get_device, logger
from pinnacle.model import PINNACLE
from pinnacle.dataset import load_data, create_dataloaders

CKPT = "outputs/checkpoints/best_model.pth"
DATA_DIR = "data"
SEED = 42

set_seed(SEED)
device = get_device()

data = load_data(DATA_DIR, remap=True)
_, _, test_loader, _ = create_dataloaders(
    data, batch_size=64, seed=SEED, num_workers=0, use_augmentation=False)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = PINNACLE(num_classes=5, embed_dim=128, dropout=0.3, mode="fusion")
model.load_state_dict(ckpt["model_state"])
model = model.to(device).eval()

disc, correct = [], []
with torch.no_grad():
    for raman, scalo, labels in test_loader:
        logits, alpha, beta = model(raman.to(device), scalo.to(device))
        preds = logits.argmax(1).cpu()
        # alpha, beta: (B, D) gate activations
        a1 = alpha.abs().sum(dim=1).cpu().numpy()
        b1 = beta.abs().sum(dim=1).cpu().numpy()
        disc.append(np.abs(a1 - b1))
        correct.append((preds == labels).numpy())

disc = np.concatenate(disc)
correct = np.concatenate(correct).astype(bool)
n = len(disc)

print("=" * 60)
print(f"Test samples: {n}   overall acc: {100*correct.mean():.2f}%")
print(f"Discordance |‖a‖1-‖b‖1|: min={disc.min():.3f} med={np.median(disc):.3f} "
      f"max={disc.max():.3f}")
print("=" * 60)

# Report several thresholds so the draft can quote a real, honest split.
for thr in [0.2, np.median(disc)]:
    lo = disc < thr
    hi = ~lo
    lo_acc = 100 * correct[lo].mean() if lo.sum() else float("nan")
    hi_acc = 100 * correct[hi].mean() if hi.sum() else float("nan")
    print(f"thr={thr:.3f}: low-disc n={lo.sum():4d} acc={lo_acc:.2f}%  | "
          f"high-disc n={hi.sum():4d} acc={hi_acc:.2f}%")

# Also report by quartile of discordance.
q = np.quantile(disc, [0.25, 0.5, 0.75])
print("\nAccuracy by discordance quartile:")
edges = [-np.inf, q[0], q[1], q[2], np.inf]
for i in range(4):
    m = (disc >= edges[i]) & (disc < edges[i + 1])
    print(f"  Q{i+1}: n={m.sum():4d}  acc={100*correct[m].mean():.2f}%")
