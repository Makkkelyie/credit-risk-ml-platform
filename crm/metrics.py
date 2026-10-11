"""Evaluation metrics for a probability-of-default model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true, y_score) -> float:
    """Max separation between the score CDFs of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Weighted mean |observed default rate - mean predicted PD| over equal-width bins."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        mask = bins == b
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def evaluate(y_true, y_prob) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }


def bootstrap_auc_diff(y_true, p_new, p_old, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Paired bootstrap of AUC(new) - AUC(old) on the same rows: (mean, 2.5%, 97.5%)."""
    y_true, p_new, p_old = map(np.asarray, (y_true, p_new, p_old))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(roc_auc_score(y_true[idx], p_new[idx]) - roc_auc_score(y_true[idx], p_old[idx]))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
