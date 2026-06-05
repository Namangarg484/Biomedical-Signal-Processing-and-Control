"""
PINNACLE — Evaluation and metrics.
Confusion matrix, per-class metrics, statistical significance tests.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from typing import Dict, Optional, Tuple

from pinnacle.utils import logger


SPECIES_NAMES = [
    "E. coli",
    "S. aureus",
    "P. aeruginosa",
    "K. pneumoniae",
    "E. faecalis",
]


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    species_names: list = SPECIES_NAMES,
) -> Dict:
    """
    Comprehensive model evaluation.

    Returns dict with:
        accuracy, per_class_metrics, confusion_matrix,
        predictions, labels, classification_report
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for raman, scalogram, labels in test_loader:
            raman = raman.to(device)
            scalogram = scalogram.to(device)

            logits, _, _ = model(raman, scalogram)
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None
    )

    per_class = {}
    for i, name in enumerate(species_names[:len(np.unique(all_labels))]):
        per_class[name] = {
            "precision": float(precision[i] * 100),
            "recall": float(recall[i] * 100),
            "f1": float(f1[i] * 100),
            "support": int(support[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=species_names[:len(np.unique(all_labels))],
        digits=4,
    )

    # 95% Wilson confidence interval
    n = len(all_labels)
    p_hat = accuracy / 100
    z = 1.96
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    ci_lower = (center - margin) * 100
    ci_upper = (center + margin) * 100

    results = {
        "accuracy": accuracy,
        "ci_95": (ci_lower, ci_upper),
        "per_class": per_class,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
        "classification_report": report,
    }

    # Log summary
    logger.info("=" * 60)
    logger.info(f"🎯 Test Accuracy: {accuracy:.2f}%  [95% CI: {ci_lower:.1f}–{ci_upper:.1f}%]")
    logger.info("=" * 60)
    for name, m in per_class.items():
        logger.info(
            f"  {name:20s}  P={m['precision']:.1f}%  R={m['recall']:.1f}%  "
            f"F1={m['f1']:.1f}%  n={m['support']}"
        )
    logger.info("=" * 60)

    return results


def mcnemar_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """
    McNemar's test for comparing two classifiers.

    Returns:
        chi2 statistic, p-value
    """
    from scipy.stats import chi2 as chi2_dist

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # Contingency: b01 = A wrong, B right; b10 = A right, B wrong
    b01 = np.sum(~correct_a & correct_b)
    b10 = np.sum(correct_a & ~correct_b)

    if b01 + b10 == 0:
        return 0.0, 1.0

    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return float(chi2), float(p_value)
