"""
classifier.postproc
===================
Post-process per-frame predictions before deriving event-level metrics.

Spec section 10c:
    - Median filter over N frames (default 5 = 50 ms).
    - Drop events shorter than 30 ms.

Both operations run *within a file* — never across file boundaries.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from classifier.config import HOP_LENGTH, SAMPLE_RATE


def median_filter_1d(y: np.ndarray, window: int) -> np.ndarray:
    """
    Odd-length majority-vote (median-mode) filter over an integer label array.
    """
    if window <= 1 or len(y) <= 1:
        return y.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2

    padded = np.concatenate([np.full(half, y[0]),
                             y,
                             np.full(half, y[-1])]).astype(np.int64)
    out = np.empty_like(y)
    for i in range(len(y)):
        window_view = padded[i:i + window]
        vals, counts = np.unique(window_view, return_counts=True)
        out[i] = int(vals[int(np.argmax(counts))])
    return out


def _runs(y: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Contiguous-run summary: list of (label, start_frame, end_frame_exclusive).
    """
    if len(y) == 0:
        return []
    edges = np.diff(y.astype(np.int64)) != 0
    starts = np.concatenate([[0], np.where(edges)[0] + 1])
    ends   = np.concatenate([starts[1:], [len(y)]])
    return [(int(y[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def drop_short_events(y: np.ndarray,
                      min_frames: int,
                      fluent_id: int = 0) -> np.ndarray:
    """
    Any run shorter than `min_frames` is relabelled to `fluent_id`.
    Runs of Fluent itself are left alone.
    """
    if min_frames <= 1 or len(y) == 0:
        return y.copy()
    out = y.copy()
    for lbl, s, e in _runs(out):
        if lbl != fluent_id and (e - s) < min_frames:
            out[s:e] = fluent_id
    return out


def smooth_predictions(
    y: np.ndarray,
    stem_index: np.ndarray,
    median_window: int = 5,
    min_event_ms: float = 30.0,
    fluent_id: int = 0,
) -> np.ndarray:
    """
    Apply median filter + short-event drop within each file (stem group).

    y            : (N,)  per-frame label ids
    stem_index   : (N,)  which file each frame belongs to
    median_window: N frames for the majority filter
    min_event_ms : events shorter than this are demoted to fluent
    fluent_id    : the class id representing fluent (default 0)
    """
    out = y.copy()
    frame_ms = 1000.0 * HOP_LENGTH / SAMPLE_RATE
    min_frames = max(1, int(round(min_event_ms / frame_ms)))

    for stem_id in np.unique(stem_index):
        mask = stem_index == stem_id
        yi = out[mask]
        yi = median_filter_1d(yi, median_window)
        yi = drop_short_events(yi, min_frames=min_frames, fluent_id=fluent_id)
        out[mask] = yi

    return out


# ---------------------------------------------------------------------
# Event-level metrics
# ---------------------------------------------------------------------

def _events_per_file(y: np.ndarray, stem_index: np.ndarray,
                     fluent_id: int
                     ) -> List[Tuple[int, int, int, int]]:
    """
    Return [(stem_id, label, start_frame, end_frame_exclusive), ...]
    excluding fluent runs.
    """
    events = []
    for stem_id in np.unique(stem_index):
        mask = stem_index == stem_id
        yi = y[mask]
        for lbl, s, e in _runs(yi):
            if lbl == fluent_id:
                continue
            events.append((int(stem_id), lbl, s, e))
    return events


def event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stem_index: np.ndarray,
    class_names: List[str],
    fluent_id: int = 0,
) -> dict:
    """
    Per-class event-level Precision / Recall / F1 using any-overlap match
    within a file. y_pred is expected to be already smoothed / short-event
    dropped.
    """
    true_events = _events_per_file(y_true, stem_index, fluent_id)
    pred_events = _events_per_file(y_pred, stem_index, fluent_id)

    per_class = {}
    for cid, cname in enumerate(class_names):
        if cid == fluent_id:
            continue
        gts = [(f, s, e) for f, l, s, e in true_events if l == cid]
        prs = [(f, s, e) for f, l, s, e in pred_events if l == cid]

        matched_pred = set()
        matched_gt = 0
        for gi, (fg, sg, eg) in enumerate(gts):
            for pi, (fp, sp, ep) in enumerate(prs):
                if pi in matched_pred or fp != fg:
                    continue
                if sp < eg and sg < ep:  # any overlap
                    matched_pred.add(pi)
                    matched_gt += 1
                    break

        tp = matched_gt
        fp = len(prs) - len(matched_pred)
        fn = len(gts) - matched_gt
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[cname] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": float(p), "recall": float(r), "f1": float(f),
            "n_true_events": len(gts),
            "n_pred_events": len(prs),
        }

    # Macro F1 across non-fluent classes
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()])) \
        if per_class else 0.0

    return {
        "per_class": per_class,
        "f1_macro": macro_f1,
    }
