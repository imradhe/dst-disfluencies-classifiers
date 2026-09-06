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
from classifier.imbalance import class_weight_tensor
from classifier.metrics import extended_metrics, frame_metrics, summarise_metrics


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
    metrics = extended_metrics(
        y_true=y_te, y_pred=y_pred, stem_index=s_te,
        task_type=task_type, class_names=class_names,
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


# ---------------------------------------------------------------------
# DNN path
# ---------------------------------------------------------------------

def run_dnn(r: Run,
            apply_smote: bool = True,
            standardise: bool = True,
            cfg=None) -> Dict:
    """Full DNN training + evaluation for one Run."""
    from classifier.imbalance import smote
    from classifier.models.dnn import DNNConfig, predict_dnn, save_dnn, train_dnn

    if cfg is None:
        cfg = DNNConfig()

    task_type, class_names = task_info(r.task)
    n_classes = len(class_names)
    t0 = time.time()

    X_tr, y_tr, _ = stack_dataset(r.train_stems, r.feature, r.task)
    X_te, y_te, s_te = stack_dataset(r.test_stems, r.feature, r.task)
    if r.val_stems:
        X_va, y_va, _ = stack_dataset(r.val_stems, r.feature, r.task)
    else:
        X_va = y_va = None

    if X_tr.size == 0 or X_te.size == 0:
        raise RuntimeError(
            f"{r.tag}: empty split (train={len(y_tr)}, test={len(y_te)})"
        )

    if standardise:
        stats = _fit_standardiser(X_tr)
        X_tr = _apply_standardiser(X_tr, stats)
        X_te = _apply_standardiser(X_te, stats)
        if X_va is not None:
            X_va = _apply_standardiser(X_va, stats)

    n_train_raw = int(len(y_tr))
    if apply_smote:
        X_tr, y_tr = smote(X_tr, y_tr, random_state=cfg.random_state)
    n_train_bal = int(len(y_tr))

    model, info = train_dnn(
        X_tr, y_tr, n_classes=n_classes, cfg=cfg,
        X_val=X_va, y_val=y_va,
    )
    y_pred, y_score = predict_dnn(model, X_te)

    metrics = extended_metrics(
        y_true=y_te, y_pred=y_pred, stem_index=s_te,
        task_type=task_type, class_names=class_names,
        y_score=y_score if task_type == "binary" else None,
    )

    save_dnn(model, MODELS_DIR / f"{r.tag}.pt")
    np.savez_compressed(
        PREDICTIONS_DIR / f"{r.tag}.npz",
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
        "n_train_raw":  n_train_raw,
        "n_train_bal":  n_train_bal,
        "n_test":       int(len(y_te)),
        "apply_smote":  apply_smote,
        "standardise":  standardise,
        "epochs_run":   info["epochs_run"],
        "elapsed_sec":  elapsed,
        "metrics":      metrics,
    }
    with open(RESULTS_DIR / f"{r.tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[run] {r.tag}  ({elapsed:.1f}s, {info['epochs_run']} ep)  "
          f"{summarise_metrics(metrics)}")
    return result


# ---------------------------------------------------------------------
# BiLSTM path
# ---------------------------------------------------------------------

def run_bilstm(r: Run,
               standardise: bool = True,
               cfg=None) -> Dict:
    """Full BiLSTM training + evaluation on per-file sequences."""
    from classifier.models.bilstm import (
        BiLSTMConfig, build_sequences, predict_bilstm, save_bilstm, train_bilstm,
    )

    if cfg is None:
        cfg = BiLSTMConfig()

    task_type, class_names = task_info(r.task)
    n_classes = len(class_names)
    t0 = time.time()

    train_seqs = build_sequences(r.train_stems, r.feature, r.task)
    val_seqs   = build_sequences(r.val_stems,   r.feature, r.task) if r.val_stems else []
    test_seqs  = build_sequences(r.test_stems,  r.feature, r.task)

    if not train_seqs or not test_seqs:
        raise RuntimeError(
            f"{r.tag}: empty split (train_seqs={len(train_seqs)}, "
            f"test_seqs={len(test_seqs)})"
        )

    # Standardise per-feature-dim using train stats.
    if standardise:
        X_stack = np.concatenate([s.X for s in train_seqs], axis=0)
        stats = _fit_standardiser(X_stack)
        for s in train_seqs + val_seqs + test_seqs:
            s.X = _apply_standardiser(s.X, stats)

    # Class weights (BiLSTM path uses weighted loss, not SMOTE).
    y_flat = np.concatenate([s.y for s in train_seqs])
    n_train_raw = int(len(y_flat))
    cw = class_weight_tensor(y_flat, n_classes=n_classes)

    model, info = train_bilstm(
        train_seqs, n_classes=n_classes, cfg=cfg,
        val_seqs=val_seqs, class_weight_tensor=cw,
    )

    y_pred, y_score = predict_bilstm(model, test_seqs)
    y_true = np.concatenate([s.y for s in test_seqs])
    s_idx  = np.concatenate([
        np.full(len(s.y), i, dtype=np.int32) for i, s in enumerate(test_seqs)
    ])

    metrics = extended_metrics(
        y_true=y_true, y_pred=y_pred, stem_index=s_idx,
        task_type=task_type, class_names=class_names,
        y_score=y_score if task_type == "binary" else None,
    )

    save_bilstm(model, MODELS_DIR / f"{r.tag}.pt")
    np.savez_compressed(
        PREDICTIONS_DIR / f"{r.tag}.npz",
        y_true=y_true.astype(np.int64),
        y_pred=y_pred.astype(np.int64),
        y_score=np.asarray(y_score, dtype=np.float32),
        stem_index=s_idx.astype(np.int32),
    )

    elapsed = time.time() - t0
    result = {
        "tag":          r.tag,
        "feature":      r.feature,
        "classifier":   r.classifier,
        "task":         r.task,
        "task_type":    task_type,
        "class_names":  class_names,
        "n_train_raw":  n_train_raw,
        "n_train_bal":  n_train_raw,       # no SMOTE for sequences
        "n_test":       int(len(y_true)),
        "apply_smote":  False,
        "standardise":  standardise,
        "epochs_run":   info["epochs_run"],
        "elapsed_sec":  elapsed,
        "metrics":      metrics,
    }
    with open(RESULTS_DIR / f"{r.tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"[run] {r.tag}  ({elapsed:.1f}s, {info['epochs_run']} ep)  "
          f"{summarise_metrics(metrics)}")
    return result


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def run_one(r: Run) -> Dict:
    """Route to the correct trainer based on r.classifier."""
    if r.classifier == "rf":
        return run_rf(r)
    if r.classifier == "dnn":
        return run_dnn(r)
    if r.classifier == "bilstm":
        return run_bilstm(r)
    raise ValueError(f"Unknown classifier {r.classifier!r}")
