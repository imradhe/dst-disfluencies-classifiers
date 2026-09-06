"""
classifier.cache
================
On-disk feature + label cache.

For each (feature_tag, file_stem):
    cache/features/{tag}/{stem}.npz  <- (D, T) float32 features
For each file_stem:
    cache/labels/{stem}.npz          <- class_ids (T,) int64, frame_times

Features and labels use the same frame timeline (see classifier.data),
so `features.shape[1]` always equals `labels.class_ids.shape[0]`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from classifier.config import FEATURES_CACHE, LABELS_CACHE, SAMPLE_RATE
from classifier.data import FileLabels, FilePair, labels_for_file
from classifier.features import extract_file


# ---------------------------------------------------------------------
# Labels cache
# ---------------------------------------------------------------------

def _labels_path(stem: str) -> Path:
    return LABELS_CACHE / f"{stem}.npz"


def cache_labels(pair: FilePair, overwrite: bool = False) -> FileLabels:
    """Compute labels for one file (or load from disk if cached)."""
    p = _labels_path(pair.stem)
    if p.exists() and not overwrite:
        data = np.load(p, allow_pickle=True)
        return FileLabels(
            stem=pair.stem,
            frame_times=data["frame_times"].astype(np.float64),
            raw_labels=data["raw_labels"].astype(object),
            class_ids=data["class_ids"].astype(np.int64),
        )

    fl = labels_for_file(pair)
    np.savez_compressed(
        p,
        frame_times=fl.frame_times,
        raw_labels=fl.raw_labels.astype(str),
        class_ids=fl.class_ids,
    )
    return fl


def load_labels(stem: str) -> FileLabels:
    p = _labels_path(stem)
    if not p.exists():
        raise FileNotFoundError(f"No cached labels for {stem}")
    data = np.load(p, allow_pickle=True)
    return FileLabels(
        stem=stem,
        frame_times=data["frame_times"].astype(np.float64),
        raw_labels=data["raw_labels"].astype(object),
        class_ids=data["class_ids"].astype(np.int64),
    )


# ---------------------------------------------------------------------
# Features cache
# ---------------------------------------------------------------------

def _feature_path(tag: str, stem: str) -> Path:
    d = FEATURES_CACHE / tag
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}.npz"


def cache_features(tag: str, pair: FilePair,
                   n_target_frames: int,
                   overwrite: bool = False) -> np.ndarray:
    """
    Compute frame-level features for (tag, file) and cache to disk.
    Returns (D, T) float32.
    """
    p = _feature_path(tag, pair.stem)
    if p.exists() and not overwrite:
        with np.load(p) as data:
            return data["features"]

    audio, sr = sf.read(str(pair.wav_path), dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    if sr != SAMPLE_RATE:
        raise ValueError(f"{pair.stem}: sr={sr}, expected {SAMPLE_RATE}")

    features = extract_file(tag, audio, n_target_frames=n_target_frames)

    np.savez_compressed(p, features=features.astype(np.float32))
    return features.astype(np.float32)


def load_features(tag: str, stem: str) -> np.ndarray:
    p = _feature_path(tag, stem)
    if not p.exists():
        raise FileNotFoundError(f"No cached features for {tag}/{stem}")
    with np.load(p) as data:
        return data["features"]


# ---------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------

def build_cache(
    pairs: List[FilePair],
    feature_tags: List[str],
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    For every (feature_tag, file), compute and cache to disk. Labels
    are cached once per file (they don't depend on the feature).
    """
    for i, pair in enumerate(pairs, start=1):
        fl = cache_labels(pair, overwrite=overwrite)
        n_frames = fl.n_frames
        for tag in feature_tags:
            try:
                cache_features(tag, pair, n_target_frames=n_frames,
                               overwrite=overwrite)
                status = "ok"
            except Exception as exc:
                status = f"FAILED: {exc}"
            if verbose:
                print(f"[{i:>4}/{len(pairs)}] {pair.stem:<40} {tag:<20} {status}")


def stack_dataset(
    stems: List[str],
    tag: str,
    task_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cached features + labels for a set of file stems and stack them
    into flat arrays for a given task.

    Returns
    -------
    X : (N, D) float32
    y : (N,)   int64  (task-specific labels, DROP frames already filtered)
    stem_index : (N,) int32
        Which file (by index into `stems`) each frame came from -- useful
        for per-file error analysis and event derivation.
    """
    from classifier.data import DROP_ID, task_labels

    X_parts, y_parts, stem_parts = [], [], []

    for idx, stem in enumerate(stems):
        feats = load_features(tag, stem)         # (D, T)
        lbls  = load_labels(stem)                # FileLabels
        task_y = task_labels(lbls.class_ids, task_id)

        keep = task_y != DROP_ID
        if not keep.any():
            continue

        X_parts.append(feats[:, keep].T.astype(np.float32))    # (T_keep, D)
        y_parts.append(task_y[keep].astype(np.int64))
        stem_parts.append(np.full(int(keep.sum()), idx, dtype=np.int32))

    if not X_parts:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int32))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    s = np.concatenate(stem_parts, axis=0)
    return X, y, s
