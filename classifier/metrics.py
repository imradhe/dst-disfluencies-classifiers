"""
classifier.metrics
==================
Frame-level metrics for binary and multi-class tasks.

Every metric routine returns plain Python types so the results can be
JSON-serialised into the experiment log.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _majority_class_accuracy(y_true: np.ndarray) -> float:
    classes, counts = np.unique(y_true, return_counts=True)
    if len(counts) == 0:
        return 0.0
    return float(counts.max() / counts.sum())


def frame_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,                 # "binary" | "multiclass"
    class_names: List[str],
    y_score: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute the standard frame-level metric bundle.

    y_score is required for ROC/PR-AUC.
        - binary: probability of the positive class, shape (N,)
        - multiclass: soft probabilities, shape (N, C) [optional]
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    out: Dict = {
        "task_type": task_type,
        "n_samples": int(len(y_true)),
        "class_names": class_names,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "majority_baseline_accuracy": _majority_class_accuracy(y_true),
    }

    labels = list(range(len(class_names)))

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out["per_class"] = {
        class_names[i]: {
            "precision": float(p[i]),
            "recall":    float(r[i]),
            "f1":        float(f[i]),
            "support":   int(s[i]),
        }
        for i in range(len(class_names))
    }

    for avg in ("macro", "micro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0,
            labels=labels,
        )
        out[f"precision_{avg}"] = float(pa)
        out[f"recall_{avg}"]    = float(ra)
        out[f"f1_{avg}"]        = float(fa)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out["confusion_matrix"] = cm.tolist()

    if task_type == "binary":
        if y_score is not None and len(np.unique(y_true)) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            out["pr_auc"]  = float(average_precision_score(y_true, y_score))
        else:
            out["roc_auc"] = None
            out["pr_auc"]  = None

    return out


def extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stem_index: np.ndarray,
    task_type: str,
    class_names: List[str],
    y_score: Optional[np.ndarray] = None,
    bootstrap_n: int = 1000,
    median_window: int = 5,
    min_event_ms: float = 30.0,
) -> Dict:
    """
    Frame-level metrics + smoothed-frame metrics + event-level metrics +
    bootstrap 95% CI on macro F1.
    """
    from classifier.postproc import event_metrics, smooth_predictions
    from classifier.stats import bootstrap_f1_ci

    frame = frame_metrics(
        y_true=y_true, y_pred=y_pred, task_type=task_type,
        class_names=class_names, y_score=y_score,
    )

    # For binary tasks fluent_id = 0 (class_names[0] == 'Fluent'/'not_<cls>').
    fluent_id = 0

    y_pred_smooth = smooth_predictions(
        y_pred, stem_index=stem_index,
        median_window=median_window, min_event_ms=min_event_ms,
        fluent_id=fluent_id,
    )

    frame_smoothed = frame_metrics(
        y_true=y_true, y_pred=y_pred_smooth, task_type=task_type,
        class_names=class_names, y_score=y_score,
    )

    events = event_metrics(
        y_true=y_true, y_pred=y_pred_smooth,
        stem_index=stem_index, class_names=class_names, fluent_id=fluent_id,
    )

    ci = bootstrap_f1_ci(y_true, y_pred, n_resamples=bootstrap_n)

    return {
        **frame,
        "smoothed":     frame_smoothed,
        "events":       events,
        "f1_macro_ci":  ci,
        "postproc": {
            "median_window":   median_window,
            "min_event_ms":    min_event_ms,
        },
    }


def summarise_metrics(m: Dict) -> str:
    """Compact one-liner for logs."""
    if m["task_type"] == "binary":
        auc = "" if m.get("roc_auc") is None else f"  AUC={m['roc_auc']:.3f}"
        return (f"acc={m['accuracy']:.3f}  "
                f"F1={m['f1_macro']:.3f} (macro){auc}  "
                f"baseline={m['majority_baseline_accuracy']:.3f}")
    return (f"acc={m['accuracy']:.3f}  "
            f"F1_macro={m['f1_macro']:.3f}  F1_micro={m['f1_micro']:.3f}  "
            f"baseline={m['majority_baseline_accuracy']:.3f}")
