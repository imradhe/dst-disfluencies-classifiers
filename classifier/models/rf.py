"""
classifier.models.rf
====================
Random-Forest trainer + predictor for frame-level classification.

Follows the RF config from Garg 2021:
    - 100 trees, max_depth = 20
    - class_weight = "balanced"

SMOTE balancing is applied on the training set before fitting (spec
section 8, "SMOTE on train set only").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from classifier.config import RANDOM_SEED
from classifier.imbalance import smote


@dataclass
class RFConfig:
    n_estimators: int = 100
    max_depth: int = 20
    class_weight: str = "balanced"
    n_jobs: int = -1
    random_state: int = RANDOM_SEED


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: RFConfig = RFConfig(),
    apply_smote: bool = True,
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train a Random Forest on (SMOTE-balanced by default) frame features.

    Returns (fitted_model, info_dict).
    """
    info = {"n_train_raw": int(len(y_train))}

    if apply_smote:
        X_train, y_train = smote(X_train, y_train, random_state=cfg.random_state)
        info["n_train_balanced"] = int(len(y_train))
    else:
        info["n_train_balanced"] = int(len(y_train))

    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        class_weight=cfg.class_weight,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_train)
    return model, info


def predict_rf(
    model: RandomForestClassifier,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (y_pred, y_score).

    y_score is:
        - (N,) probability of the positive class for binary tasks
        - (N, C) full probability matrix for multi-class tasks
    """
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)     # (N, C)
    if proba.shape[1] == 2:
        y_score = proba[:, 1]                # positive-class prob
    else:
        y_score = proba
    return y_pred, y_score


def save_rf(model: RandomForestClassifier, path: Path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_rf(path: Path) -> RandomForestClassifier:
    import joblib
    return joblib.load(path)
