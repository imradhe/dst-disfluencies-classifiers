"""
classifier.splits
=================
File-level random 90:10 splits, val = 10% of train.

Persisted to `splits/{scheme}.json` so re-runs use the same partition.

Split unit is `stem` (filename without extension). All frames of a file
land in the same partition -- never split at frame level.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from classifier.config import RANDOM_SEED, SPLITS_DIR


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


def load_split(scheme: str = "random_90_10") -> Dict[str, List[str]]:
    p = _split_path(scheme)
    if not p.exists():
        raise FileNotFoundError(
            f"No split at {p}. Call make_random_split() first."
        )
    with open(p) as f:
        return json.load(f)
