#!/usr/bin/env python3
"""
Verify every quantitative claim in draft.tex against saved predictions/data.
Recomputes (NO retraining):
  - Accuracy + Wilson 95% CI for every saved model
  - McNemar p-value vs PINNACLE for every model
  - Per-class precision/recall/F1 for PINNACLE
  - Confusion-pair error counts
  - Inter-class mean-spectrum Pearson correlations (for the r=0.94 / r=0.89 claims)
All printed in a form directly comparable to the LaTeX tables.
"""
import os, sys, math, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from scipy.stats import chi2
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

SPECIES = ["E. coli", "S. aureus", "P. aeruginosa", "K. pneumoniae", "E. faecalis"]


def wilson_ci(acc_frac, n, z=1.96):
    p = acc_frac
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def mcnemar_p(preds_a, preds_b, labels):
    ca = (preds_a == labels)
    cb = (preds_b == labels)
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    if b + c == 0:
        return 1.0, b, c
    if b + c < 25:
        from scipy.stats import binomtest
        return float(binomtest(b, b + c, 0.5).pvalue), b, c
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return float(1.0 - chi2.cdf(stat, df=1)), b, c


def load_npz(path):
    d = np.load(path)
    keys = set(d.keys())
    pk = "y_pred" if "y_pred" in keys else ("preds" if "preds" in keys else "predictions")
    lk = "y_true" if "y_true" in keys else "labels"
    return d[pk].astype(int), d[lk].astype(int)


print("=" * 70)
print("1.  ACCURACY + WILSON CI + McNEMAR p vs PINNACLE")
print("=" * 70)

pinn_yp, pinn_yt = load_npz("outputs/predictions_current.npz")
n = len(pinn_yt)
pinn_acc = 100 * np.mean(pinn_yp == pinn_yt)
lo, hi = wilson_ci(pinn_acc / 100, n)
print(f"PINNACLE              acc={pinn_acc:.2f}  CI=[{lo:.2f},{hi:.2f}]  n={n}")

models = {
    "raman_only":      "outputs/ablation_5class/raman_only/predictions.npz",
    "scalogram_only":  "outputs/ablation_5class/scalogram_only/predictions.npz",
    "concat":          "outputs/ablation_5class/concat/predictions.npz",
    "gating_only":     "outputs/ablation_5class/gating_only/predictions.npz",
    "crossattn_only":  "outputs/ablation_5class/crossattn_only/predictions.npz",
    "ResNet-18-1D":    "outputs/sota_baselines/ResNet-18-1D (Ho et al.)/predictions.npz",
    "KirchhoffNet":    "outputs/sota_baselines/KirchhoffNet (CNN+Attn)/predictions.npz",
    "PureCrossAttn":   "outputs/sota_baselines/PureCrossAttn (no gating)/predictions.npz",
    "Raman+HeavyAug":  "outputs/robustness_baselines/aug_raman_only/predictions.npz",
    "GMF":             "outputs/robustness_baselines/gmf_shared_gating/predictions.npz",
}

for name, path in models.items():
    if not os.path.exists(path):
        print(f"{name:20s}  MISSING: {path}")
        continue
    yp, yt = load_npz(path)
    acc = 100 * np.mean(yp == yt)
    lo, hi = wilson_ci(acc / 100, len(yt))
    # align with PINNACLE labels if same test set
    if len(yt) == len(pinn_yt) and np.array_equal(yt, pinn_yt):
        p, b, c = mcnemar_p(yp, pinn_yp, pinn_yt)
        ptxt = f"p={p:.3f} (b={b},c={c})"
    else:
        ptxt = "diff test set / labels"
    print(f"{name:20s}  acc={acc:.2f}  CI=[{lo:.2f},{hi:.2f}]  n={len(yt)}  {ptxt}")

print()
print("=" * 70)
print("2.  PINNACLE PER-CLASS PRECISION / RECALL / F1")
print("=" * 70)
p, r, f, sup = precision_recall_fscore_support(pinn_yt, pinn_yp)
for i, s in enumerate(SPECIES):
    print(f"  {s:16s} P={p[i]*100:5.1f}  R={r[i]*100:5.1f}  F1={f[i]*100:5.1f}  n={sup[i]}")

print()
print("=" * 70)
print("3.  CONFUSION-PAIR ERROR COUNTS (symmetric)")
print("=" * 70)
cm = confusion_matrix(pinn_yt, pinn_yp)
pairs = []
for i in range(len(SPECIES)):
    for j in range(i + 1, len(SPECIES)):
        pairs.append((cm[i, j] + cm[j, i], SPECIES[i], SPECIES[j]))
for cnt, a, b in sorted(pairs, reverse=True)[:6]:
    print(f"  {a:16s} <-> {b:16s}  {cnt} errors")

print()
print("=" * 70)
print("4.  INTER-CLASS MEAN-SPECTRUM PEARSON CORRELATIONS")
print("    (checks draft claims r=0.94 Ec/Kp, r=0.89 Pa/Sa)")
print("=" * 70)
from pinnacle.dataset import load_data
data = load_data("data", remap=True)
X = np.concatenate([data["X_2018"], data["X_2019"]], axis=0)
y = np.concatenate([data["y_2018"], data["y_2019"]], axis=0)
means = {i: X[y == i].mean(axis=0) for i in range(len(SPECIES))}
def corr(i, j):
    return np.corrcoef(means[i], means[j])[0, 1]
named = {0: "E.coli", 1: "S.aureus", 2: "P.aer", 3: "K.pneu", 4: "E.faec"}
for i in range(len(SPECIES)):
    for j in range(i + 1, len(SPECIES)):
        print(f"  r({named[i]:8s},{named[j]:8s}) = {corr(i,j):.3f}")
print()
print(f"  >>> E.coli/K.pneumoniae r = {corr(0,3):.3f}  (draft says 0.94)")
print(f"  >>> P.aeruginosa/S.aureus r = {corr(2,1):.3f}  (draft says 0.89)")
print("=" * 70)
