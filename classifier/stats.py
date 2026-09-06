"""
classifier.stats
================
Statistical add-ons on top of point-estimate metrics.

    bootstrap_f1_ci(y_true, y_pred, ...)   -> {mean, lo, hi}
    paired_bootstrap(y_true, y_pred_a, y_pred_b, ...) -> {delta, ci, p_value}
    mcnemar(y_true, y_pred_a, y_pred_b) -> chi2, p_value  (binary tasks)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score


def bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    average: str = "macro",
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        f1s[i] = f1_score(y_true[idx], y_pred[idx],
                          average=average, zero_division=0)
    alpha = (1 - ci) / 2
    return {
        "mean":     float(f1s.mean()),
        "lo":       float(np.quantile(f1s, alpha)),
        "hi":       float(np.quantile(f1s, 1 - alpha)),
        "n":        int(n_resamples),
        "average":  average,
        "ci_level": ci,
    }


def paired_bootstrap(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_resamples: int = 1000,
    average: str = "macro",
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap: same resampled indices for both predictors, report
    the F1 difference distribution and a two-sided p-value for delta == 0.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        fa = f1_score(y_true[idx], y_pred_a[idx], average=average, zero_division=0)
        fb = f1_score(y_true[idx], y_pred_b[idx], average=average, zero_division=0)
        deltas[i] = fa - fb
    alpha = (1 - ci) / 2
    p_value = float(2 * min(
        (deltas <= 0).mean(),
        (deltas >= 0).mean(),
    ))
    return {
        "delta_mean": float(deltas.mean()),
        "delta_lo":   float(np.quantile(deltas, alpha)),
        "delta_hi":   float(np.quantile(deltas, 1 - alpha)),
        "p_value":    p_value,
        "n":          int(n_resamples),
    }


def mcnemar(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, float]:
    """
    Exact McNemar test (binary comparison of two classifiers on the same
    frames). Uses the continuity-corrected chi-square approximation.
    """
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)

    n01 = int(((~a_correct) & b_correct).sum())
    n10 = int((a_correct & (~b_correct)).sum())

    if (n01 + n10) == 0:
        return {"n01": 0, "n10": 0, "chi2": 0.0, "p_value": 1.0}

    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

    # Two-sided p-value from chi-square(1) survival function.
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, df=1))
    except ImportError:
        p = float("nan")

    return {"n01": n01, "n10": n10, "chi2": float(chi2), "p_value": p}
