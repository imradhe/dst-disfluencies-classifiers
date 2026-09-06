"""
classifier.data
===============
Dataset discovery + frame-level label assignment.

Pipeline
--------
1. Walk DATASET_DIR to find (WAV, TXT) pairs.
2. Parse each TXT into a list of AnnotationRow.
3. Build a per-file frame timeline (centre times at HOP_MS spacing).
4. Assign each frame a raw label using `centre-in-interval`:
       - the annotation whose [start, end) contains t_centre.
       - if two or more annotations contain t_centre, the frame is
         marked with sentinel label DROP (kept in arrays, filtered at
         training time).
       - otherwise, if no annotation contains t_centre, the frame is
         labelled Fluent.

Frame times use librosa's centered semantics (matching how MFCC / SFFCC
frames are computed):
        t_centre[k] = k * HOP_LENGTH / SAMPLE_RATE

This is the same timeline every extractor uses, so labels align to
features by index.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

# Reuse the annotation parser from the extractor base module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Audio-Feature-Extractors"))
from base_feature_extractor import AnnotationParser, AnnotationRow  # noqa: E402

from classifier.config import (
    ALL_CLASSES,
    CLASS_TO_ID,
    DATASET_DIR,
    DISFLUENT_SET,
    FLUENT,
    HOP_LENGTH,
    SAMPLE_RATE,
)


# ---------------------------------------------------------------------
# Sentinel for frames dropped due to overlapping annotations
# ---------------------------------------------------------------------

DROP_LABEL = "__DROP__"
DROP_ID    = -1


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass
class FilePair:
    """One (wav, txt) pair plus derived properties."""
    stem: str
    wav_path: Path
    txt_path: Path
    n_samples: int
    duration: float

    @property
    def n_frames(self) -> int:
        # librosa-centred framing count for the whole file.
        return int(np.ceil(self.n_samples / HOP_LENGTH))


@dataclass
class FileLabels:
    """Per-frame labels for one file."""
    stem: str
    frame_times: np.ndarray      # (T,), seconds
    raw_labels: np.ndarray       # (T,) string, one of ALL_CLASSES or DROP_LABEL
    class_ids: np.ndarray        # (T,) int, matches CLASS_TO_ID (DROP_ID for dropped)

    @property
    def n_frames(self) -> int:
        return len(self.frame_times)


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------

def discover_pairs(dataset_dir: Path = DATASET_DIR) -> List[FilePair]:
    """
    Find every WAV under dataset_dir that has a matching .txt annotation
    beside it (same stem in the same directory).

    Raises FileNotFoundError if the dataset directory itself doesn't exist,
    but returns an empty list (with a stderr note) if no pairs are found.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    pairs: List[FilePair] = []
    for wav_path in sorted(dataset_dir.rglob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        info = sf.info(str(wav_path))
        if info.samplerate != SAMPLE_RATE:
            # Skip files at wrong sample rate. Callers can pre-resample.
            print(
                f"[data] Skipping {wav_path.name}: sr={info.samplerate} "
                f"(expected {SAMPLE_RATE})"
            )
            continue

        pairs.append(
            FilePair(
                stem=wav_path.stem,
                wav_path=wav_path,
                txt_path=txt_path,
                n_samples=int(info.frames),
                duration=float(info.frames) / info.samplerate,
            )
        )

    return pairs


# ---------------------------------------------------------------------
# Frame timeline
# ---------------------------------------------------------------------

def frame_times_for(n_samples: int, hop_length: int = HOP_LENGTH,
                    sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    librosa's centred-frame time grid: t[k] = k * hop / sr, for
    k in [0, ceil(n_samples / hop)).

    Matches librosa.frames_to_time(np.arange(n_frames), sr, hop_length).
    """
    n_frames = int(np.ceil(n_samples / hop_length))
    return (np.arange(n_frames) * hop_length / sr).astype(np.float64)


# ---------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------

def _assign_frame_labels(
    frame_times: np.ndarray,
    annotations: List[AnnotationRow],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centre-in-interval labeling with drop-frame tie-break.

    Returns
    -------
    raw_labels : np.ndarray of shape (T,), dtype object
        One of ALL_CLASSES or DROP_LABEL.
    class_ids  : np.ndarray of shape (T,), dtype int64
        CLASS_TO_ID for concrete labels; DROP_ID for dropped frames.
    """
    T = len(frame_times)
    raw = np.full(T, FLUENT, dtype=object)
    ids = np.full(T, CLASS_TO_ID[FLUENT], dtype=np.int64)

    if not annotations:
        return raw, ids

    # For each annotation, mark frames whose centre is inside [start, end).
    # Track how many annotations claim each frame so we can drop overlaps.
    hit_count = np.zeros(T, dtype=np.int32)
    first_hit = np.full(T, -1, dtype=np.int64)   # index into annotations

    for a_idx, ann in enumerate(annotations):
        # Skip unknown labels; the annotation parser already normalises,
        # but be safe.
        if ann.label not in CLASS_TO_ID:
            continue

        lo = np.searchsorted(frame_times, ann.start, side="left")
        hi = np.searchsorted(frame_times, ann.end,   side="left")

        if lo >= hi:
            continue

        hit_count[lo:hi] += 1
        # Record the first annotation index only; overlaps get dropped below.
        mask_first = first_hit[lo:hi] == -1
        first_hit[lo:hi] = np.where(mask_first, a_idx, first_hit[lo:hi])

    # Frames with exactly one annotation take that annotation's label.
    single = hit_count == 1
    for idx in np.where(single)[0]:
        ann = annotations[first_hit[idx]]
        raw[idx] = ann.label
        ids[idx] = CLASS_TO_ID[ann.label]

    # Frames with multiple annotations are dropped.
    multi = hit_count > 1
    raw[multi] = DROP_LABEL
    ids[multi] = DROP_ID

    # Frames with zero annotations stay Fluent (their init value).
    return raw, ids


def labels_for_file(pair: FilePair) -> FileLabels:
    """
    Full frame-level label assignment for one file.
    """
    parser = AnnotationParser()
    annotations = parser.parse_file(pair.txt_path)

    frame_times = frame_times_for(pair.n_samples)
    raw, ids = _assign_frame_labels(frame_times, annotations)

    return FileLabels(
        stem=pair.stem,
        frame_times=frame_times,
        raw_labels=raw,
        class_ids=ids,
    )


# ---------------------------------------------------------------------
# Task-specific label mappers
# ---------------------------------------------------------------------

# Task ids ------------------------------------------------------------

TASK_FVD = "fluent_vs_disfluent"

def TASK_FVC(cls: str) -> str:  # noqa: N802 (task id builder, not a class)
    return f"fluent_vs_{cls}"

TASK_MULTI = "multiclass"


def _fluent_vs_disfluent_ids(class_ids: np.ndarray) -> np.ndarray:
    """0 = fluent, 1 = disfluent (any of DISFLUENT_SET), -1 = dropped."""
    disfluent_ids = {CLASS_TO_ID[c] for c in DISFLUENT_SET}
    out = np.zeros_like(class_ids, dtype=np.int64)
    out[np.isin(class_ids, list(disfluent_ids))] = 1
    out[class_ids == DROP_ID] = -1
    return out


def _fluent_vs_class_ids(class_ids: np.ndarray, target: str) -> np.ndarray:
    """
    One-vs-rest binary (spec section 4 option B):
        1 = target class
        0 = anything else (fluent OR any other disfluency)
       -1 = dropped
    """
    tgt = CLASS_TO_ID[target]
    out = np.zeros_like(class_ids, dtype=np.int64)
    out[class_ids == tgt] = 1
    out[class_ids == DROP_ID] = -1
    return out


def _multiclass_ids(class_ids: np.ndarray) -> np.ndarray:
    """Return CLASS_TO_ID as-is; DROP_ID stays -1."""
    return class_ids.astype(np.int64, copy=True)


def task_labels(class_ids: np.ndarray, task: str) -> np.ndarray:
    """
    Map per-frame class ids to task-specific label ids.

    task strings:
        "fluent_vs_disfluent"
        "fluent_vs_<CLS>"  where <CLS> in DISFLUENCY_CLASSES
        "multiclass"
    """
    if task == TASK_FVD:
        return _fluent_vs_disfluent_ids(class_ids)
    if task == TASK_MULTI:
        return _multiclass_ids(class_ids)
    if task.startswith("fluent_vs_"):
        cls = task[len("fluent_vs_"):]
        if cls not in CLASS_TO_ID:
            raise ValueError(f"Unknown class in task id: {task}")
        return _fluent_vs_class_ids(class_ids, cls)
    raise ValueError(f"Unknown task: {task}")


def all_task_ids() -> List[str]:
    """The full list of task strings for the experiment grid."""
    return [
        TASK_FVD,
        TASK_MULTI,
        *[TASK_FVC(c) for c in ["I", "PR", "PhR", "WR", "PWR", "P"]],
    ]


# ---------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------

def summarise_labels(labels: List[FileLabels]) -> Dict[str, int]:
    """Frame-count per class across the whole (given) dataset partition."""
    from collections import Counter
    counter: Counter = Counter()
    for fl in labels:
        for c_id, count in zip(*np.unique(fl.class_ids, return_counts=True)):
            key = ID_TO_CLASS[int(c_id)] if int(c_id) != DROP_ID else DROP_LABEL
            counter[key] += int(count)
    return dict(counter)


# ID_TO_CLASS is re-exported for convenience
from classifier.config import ID_TO_CLASS  # noqa: E402
