"""
Central config for the IED-Extended classifier pipeline.

Every path here can be overridden by environment variables so the same
code runs on the researcher's laptop and on a shared machine without
edits.

Environment variables (all optional):
    IED_DATASET_DIR       -- where the WAV + TXT files live
    IED_OUTPUT_ROOT       -- root for cache / results / models / logs
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = Path(os.environ.get("IED_DATASET_DIR", PROJECT_ROOT / "IED"))
OUTPUT_ROOT = Path(os.environ.get("IED_OUTPUT_ROOT", PROJECT_ROOT))

CACHE_DIR      = OUTPUT_ROOT / "cache"
FEATURES_CACHE = CACHE_DIR / "features"
LABELS_CACHE   = CACHE_DIR / "labels"

SPLITS_DIR     = OUTPUT_ROOT / "splits"
RESULTS_DIR    = OUTPUT_ROOT / "results"
CONFUSION_DIR  = OUTPUT_ROOT / "confusion"
ROC_DIR        = OUTPUT_ROOT / "roc"
LOGS_DIR       = OUTPUT_ROOT / "logs"
MODELS_DIR     = OUTPUT_ROOT / "models"
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"

for _p in [
    CACHE_DIR, FEATURES_CACHE, LABELS_CACHE,
    SPLITS_DIR, RESULTS_DIR, RESULTS_DIR / "per_file",
    CONFUSION_DIR, ROC_DIR, LOGS_DIR,
    MODELS_DIR, PREDICTIONS_DIR,
]:
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Audio / frame parameters (spec section 5)
# ---------------------------------------------------------------------

SAMPLE_RATE = 16_000
WIN_MS      = 25.0
HOP_MS      = 10.0
WIN_LENGTH  = int(SAMPLE_RATE * WIN_MS / 1000.0)   # 400 samples
HOP_LENGTH  = int(SAMPLE_RATE * HOP_MS / 1000.0)   # 160 samples
WINDOW_FN   = "hamming"

FRAME_LABEL_RULE = "centre-in-interval"
OVERLAP_TIE_BREAK = "drop-frame"


# ---------------------------------------------------------------------
# Class handling (spec section 3)
# ---------------------------------------------------------------------

FLUENT = "Fluent"
DISFLUENCY_CLASSES = ["I", "PR", "PhR", "WR", "PWR", "P"]
ALL_CLASSES        = [FLUENT] + DISFLUENCY_CLASSES     # length 7
CLASS_TO_ID        = {c: i for i, c in enumerate(ALL_CLASSES)}
ID_TO_CLASS        = {i: c for c, i in CLASS_TO_ID.items()}

# For Fluent-vs-Disfluent, P counts as disfluent
DISFLUENT_SET = set(DISFLUENCY_CLASSES)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

RANDOM_SEED = 42
