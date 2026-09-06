"""
classifier.imbalance
====================
Class-imbalance strategies applied on the TRAINING partition only.

    - smote(X, y)            SMOTE oversampling (RF, DNN)
    - class_weights(y)       inverse-frequency weights (BiLSTM loss)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def smote(X: np.ndarray, y: np.ndarray,
          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE oversampling. Falls back to random oversampling if a class has
    fewer samples than the default k_neighbors of imblearn's SMOTE.

    Parameters
    ----------
    X : (N, D) float32
    y : (N,)   int64

    Returns
    -------
    X_balanced, y_balanced with equal counts per class.
    """
    from collections import Counter

    counts = Counter(y.tolist())
    if len(counts) < 2:
        return X, y

    min_count = min(counts.values())

    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. "
            "pip install imbalanced-learn"
        ) from exc

    # imblearn's SMOTE needs k_neighbors < min_count. Fall back safely.
    k = max(1, min(5, min_count - 1))
    if min_count <= 1:
        sampler = RandomOverSampler(random_state=random_state)
    else:
        sampler = SMOTE(random_state=random_state, k_neighbors=k)

    X_bal, y_bal = sampler.fit_resample(X, y)
    return X_bal.astype(np.float32), y_bal.astype(np.int64)


def undersample_majority(
    X: np.ndarray,
    y: np.ndarray,
    ratio: float = 2.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random undersampling of the majority class(es) so no class has more
    than `ratio * min_count` rows. Minority classes are kept in full.

    This is the standard imbalance strategy for large frame-level
    tabular data where SMOTE'ing the minority up to majority count
    produces tens of millions of synthetic rows and blows training time
    without helping generalisation.

    Parameters
    ----------
    X : (N, D) float32
    y : (N,)   int64
    ratio : float
        Max multiplier of the smallest class's count that any other
        class may exceed. ratio = 2.0 keeps majorities at 2x minority.
    random_state : int
        RNG seed for reproducible subsampling.

    Returns
    -------
    X_bal, y_bal
    """
    from collections import Counter

    counts = Counter(y.tolist())
    if len(counts) < 2:
        return X, y

    min_count = min(counts.values())
    cap = max(1, int(round(ratio * min_count)))

    rng = np.random.default_rng(random_state)
    keep_indices = []
    for cls, n in counts.items():
        idx = np.where(y == cls)[0]
        if n <= cap:
            keep_indices.append(idx)
        else:
            keep_indices.append(rng.choice(idx, size=cap, replace=False))

    keep = np.concatenate(keep_indices)
    rng.shuffle(keep)
    return X[keep].astype(np.float32), y[keep].astype(np.int64)


def class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Inverse-frequency weights, normalised so the smallest weight is 1.

    Suitable for torch loss `weight=` on a class-ordered tensor.
    """
    classes, counts = np.unique(y, return_counts=True)
    inv = counts.max() / counts
    return {int(c): float(w) for c, w in zip(classes, inv)}


def class_weight_tensor(y: np.ndarray, n_classes: int):
    """Return a torch tensor of length n_classes with per-class weights."""
    import torch

    weights = np.ones(n_classes, dtype=np.float32)
    for c, w in class_weights(y).items():
        if 0 <= c < n_classes:
            weights[c] = w
    return torch.from_numpy(weights)
