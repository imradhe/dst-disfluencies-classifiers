"""
classifier.splits
=================
File-level splits, val = 10% of train.

Schemes:
    random_90_10           random file-level split
    speaker_held_out       no speaker in both train and test
                           (speaker id inferred from filename)

Persisted to `splits/{scheme}.json`. Split unit is always `stem` --
frames of a file never straddle partitions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from classifier.config import RANDOM_SEED, SPLITS_DIR


SPEAKER_RE = re.compile(r"speaker[_\-]?(\d+)", re.IGNORECASE)


def speaker_id(stem: str) -> Optional[str]:
    """
    Extract the speaker id from a filename stem.

    IED_Extended stems look like `IED_INTERN_1__speaker_1`; the speaker
    number is what makes a speaker unique across the whole dataset.

    Returns None if no `speaker_N` token is present.
    """
    m = SPEAKER_RE.search(stem)
    return m.group(1) if m else None


def _split_path(scheme: str) -> Path:
    return SPLITS_DIR / f"{scheme}.json"


def make_random_split(
    stems: List[str],
    test_frac: float = 0.10,
    val_frac_of_train: float = 0.10,
    scheme: str = "random_90_10",
    seed: int = RANDOM_SEED,
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    """
    Create (or reload) a random file-level split.

    Returns a dict {"train": [...], "val": [...], "test": [...]}.
    """
    p = _split_path(scheme)
    if p.exists() and not overwrite:
        with open(p) as f:
            return json.load(f)

    rng = np.random.default_rng(seed)
    order = np.array(sorted(stems))
    rng.shuffle(order)

    n_test = int(round(len(order) * test_frac))
    test = order[:n_test].tolist()
    trainval = order[n_test:]

    n_val = int(round(len(trainval) * val_frac_of_train))
    val = trainval[:n_val].tolist()
    train = trainval[n_val:].tolist()

    split = {"train": train, "val": val, "test": test}

    with open(p, "w") as f:
        json.dump(split, f, indent=2)

    return split


def make_speaker_split(
    stems: List[str],
    test_frac: float = 0.10,
    val_frac_of_train: float = 0.10,
    scheme: str = "speaker_held_out",
    seed: int = RANDOM_SEED,
    overwrite: bool = False,
) -> Dict[str, List[str]]:
    """
    Speaker-disjoint file-level split.

    All files from a given speaker land in the SAME partition (train,
    val, or test); no speaker appears in more than one partition.
    Files without a detectable speaker id fall back to a random pool
    and are split like the rest.
    """
    p = _split_path(scheme)
    if p.exists() and not overwrite:
        with open(p) as f:
            return json.load(f)

    # Group stems by speaker
    from collections import defaultdict
    by_speaker: Dict[str, List[str]] = defaultdict(list)
    unassigned: List[str] = []
    for stem in stems:
        sid = speaker_id(stem)
        if sid is None:
            unassigned.append(stem)
        else:
            by_speaker[sid].append(stem)

    speakers = sorted(by_speaker.keys())
    rng = np.random.default_rng(seed)
    speaker_order = np.array(speakers)
    rng.shuffle(speaker_order)

    n_test_speakers = max(1, int(round(len(speaker_order) * test_frac)))
    test_speakers = set(speaker_order[:n_test_speakers].tolist())
    trainval_speakers = speaker_order[n_test_speakers:]
    n_val_speakers = max(0, int(round(len(trainval_speakers) * val_frac_of_train)))
    val_speakers = set(trainval_speakers[:n_val_speakers].tolist())
    train_speakers = set(trainval_speakers[n_val_speakers:].tolist())

    train = sorted(s for sp in train_speakers for s in by_speaker[sp])
    val   = sorted(s for sp in val_speakers   for s in by_speaker[sp])
    test  = sorted(s for sp in test_speakers  for s in by_speaker[sp])

    # Sprinkle unassigned files across partitions proportionally to
    # the current sizes.
    if unassigned:
        rng.shuffle(unassigned)
        n_test_extra = int(round(len(unassigned) * test_frac))
        n_val_extra  = int(round(len(unassigned) * (1 - test_frac) * val_frac_of_train))
        test  += unassigned[:n_test_extra]
        val   += unassigned[n_test_extra:n_test_extra + n_val_extra]
        train += unassigned[n_test_extra + n_val_extra:]

    split = {
        "train": sorted(train),
        "val":   sorted(val),
        "test":  sorted(test),
        "_speakers_in_test": sorted(test_speakers),
        "_speakers_in_val":  sorted(val_speakers),
        "_speakers_in_train": sorted(train_speakers),
    }

    with open(p, "w") as f:
        json.dump(split, f, indent=2)

    return split


def load_split(scheme: str = "random_90_10") -> Dict[str, List[str]]:
    p = _split_path(scheme)
    if not p.exists():
        raise FileNotFoundError(
            f"No split at {p}. Call make_random_split() first."
        )
    with open(p) as f:
        return json.load(f)
