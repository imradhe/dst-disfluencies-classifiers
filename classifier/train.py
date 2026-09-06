"""
classifier.train
================
Per-run orchestration: build (X, y) from cache, fit a classifier, score
on test, persist model + predictions + metrics.

Currently wires the Random Forest path end-to-end. DNN and BiLSTM will
plug in through the same `Run` container.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from classifier.cache import stack_dataset
from classifier.config import (
    ALL_CLASSES,
    DISFLUENCY_CLASSES,
    FLUENT,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RESULTS_DIR,
)
from classifier.data import TASK_FVD, TASK_MULTI, TASK_FVC
from classifier.metrics import frame_metrics, summarise_metrics


# ---------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------

def task_info(task_id: str):
    """Return (task_type, class_names) for a task id."""
    if task_id == TASK_FVD:
        return "binary", ["Fluent", "Disfluent"]
    if task_id == TASK_MULTI:
        return "multiclass", ALL_CLASSES
    if task_id.startswith("fluent_vs_"):
        cls = task_id[len("fluent_vs_"):]
        return "binary", ["not_" + cls, cls]
    raise ValueError(f"Unknown task_id: {task_id}")


# ---------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------

def _fit_standardiser(X: np.ndarray) -> Dict:
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean, "std": std}


def _apply_standardiser(X: np.ndarray, stats: Dict) -> np.ndarray:
    return ((X - stats["mean"]) / stats["std"]).astype(np.float32)


# ---------------------------------------------------------------------
# Run container
# ---------------------------------------------------------------------

@dataclass
class Run:
    feature: str
    classifier: str
    task: str
    train_stems: List[str]
    val_stems:   List[str] = field(default_factory=list)
    test_stems:  List[str] = field(default_factory=list)

    @property
    def tag(self) -> str:
        return f"{self.feature}__{self.classifier}__{self.task}"


# ---------------------------------------------------------------------
# RF path
# ---------------------------------------------------------------------

def run_rf(r: Run,
           apply_smote: bool = True,
           standardise: bool = True) -> Dict:
    """
    Full Random-Forest training + evaluation for one Run.

    Writes:
        models/{tag}.joblib
        predictions/{tag}.npz    (y_true, y_pred, y_score, stem_index)
        results/{tag}.json       (metric bundle + config)
    """
    from classifier.models.rf import RFConfig, predict_rf, save_rf, train_rf

    task_type, class_names = task_info(r.task)
    t0 = time.time()

    # ---- Load & stack cached (features, labels) ----
    X_tr, y_tr, _s_tr = stack_dataset(r.train_stems, r.feature, r.task)
    X_te, y_te, s_te  = stack_dataset(r.test_stems,  r.feature, r.task)

    if X_tr.size == 0 or X_te.size == 0:
        raise RuntimeError(
            f"{r.tag}: empty split (train={len(y_tr)}, test={len(y_te)})"
        )

    # ---- Global standardisation fit on train, applied to test ----
    if standardise:
        stats = _fit_standardiser(X_tr)
        X_tr = _apply_standardiser(X_tr, stats)
        X_te = _apply_standardiser(X_te, stats)
    else:
        stats = None

    # ---- Fit ----
    model, info = train_rf(X_tr, y_tr,
                           cfg=RFConfig(), apply_smote=apply_smote)

    # ---- Predict + score ----
    y_pred, y_score = predict_rf(model, X_te)
    metrics = frame_metrics(
        y_true=y_te,
        y_pred=y_pred,
        task_type=task_type,
        class_names=class_names,
        y_score=y_score,
    )

    # ---- Persist artefacts ----
    save_rf(model, MODELS_DIR / f"{r.tag}.joblib")

    preds_path = PREDICTIONS_DIR / f"{r.tag}.npz"
    np.savez_compressed(
        preds_path,
        y_true=y_te.astype(np.int64),
        y_pred=y_pred.astype(np.int64),
        y_score=np.asarray(y_score, dtype=np.float32),
        stem_index=s_te.astype(np.int32),
    )

    elapsed = time.time() - t0
    result = {
        "tag":          r.tag,
        "feature":      r.feature,
        "classifier":   r.classifier,
        "task":         r.task,
        "task_type":    task_type,
        "class_names":  class_names,
        "n_train_raw":  info["n_train_raw"],
        "n_train_bal":  info["n_train_balanced"],
        "n_test":       int(len(y_te)),
        "apply_smote":  apply_smote,
        "standardise":  standardise,
        "elapsed_sec":  elapsed,
        "metrics":      metrics,
    }
    with open(RESULTS_DIR / f"{r.tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[run] {r.tag}  ({elapsed:.1f}s)  {summarise_metrics(metrics)}")
    return result
