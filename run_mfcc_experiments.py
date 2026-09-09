#!/usr/bin/env python3
"""
run_mfcc_experiments.py
=======================
One-shot driver for the MFCC-family experiment grid on IED-Extended.

Runs the EXACT configuration of the committed macOS "stage1" sweep,
extended to all three classifiers:

    features    = [mfcc45, mfcc_sdc_mod_k7]
                  # the two MFCC-family configs already cached / used:
                  #   mfcc45           -> 45-dim MFCC (Garg 2021)
                  #   mfcc_sdc_mod_k7  -> 14-dim base + modified SDC K=7 = 224 (Mehrotra 2022)
    classifiers = [rf, dnn, bilstm]                       # all combinations
    tasks       = fluent_vs_disfluent, multiclass,
                  fluent_vs_{I, PR, PhR, WR, PWR, P}      # 8 tasks
    split       = speaker_held_out  (reuses splits/speaker_held_out.json
                  verbatim if present, so train/val/test = 117/11/4 files
                  exactly as in the committed results)
    seed        = 42  (RFConfig / DNNConfig / BiLSTMConfig all pin
                  random_state = 42; numpy + torch are seeded here too)

    => 2 x 3 x 8 = 48 runs.  The 32 rf/dnn runs already exist in results/
    from the macOS run; the 16 bilstm runs are new.  By default this
    script re-runs all 48 so the result set is internally consistent on
    one machine.  Pass --skip-existing to keep prior results/<tag>.json
    and only fill the gaps (useful to resume a killed job).

Model hyper-parameters are the dataclass defaults in
classifier/models/{rf,dnn,bilstm}.py and are echoed at startup.  Device
is auto-selected there (cuda -> mps -> cpu); on the GPU box DNN and
BiLSTM use CUDA with no change.  RF is scikit-learn (CPU, RAM-heavy).

--------------------------------------------------------------------------
Server quick start
--------------------------------------------------------------------------
    # dataset is NOT in git -- put it in place first:
    #   <repo>/IED/*.wav + matching *.txt      (16 kHz mono)
    #   or:  export IED_DATASET_DIR=/data/IED
    #   optional:  export IED_OUTPUT_ROOT=/scratch/disfluency-v2

    python3.11 -m venv .venv && . .venv/bin/activate
    pip install -U pip && pip install -r requirements.txt

    python run_mfcc_experiments.py                 # full 48-run sweep
    python run_mfcc_experiments.py --dry-run       # print the grid, do nothing
    python run_mfcc_experiments.py --skip-existing # resume: skip finished tags
    python run_mfcc_experiments.py --no-cache-build
    python run_mfcc_experiments.py --classifiers bilstm
    python run_mfcc_experiments.py --tasks multiclass fluent_vs_I

Outputs (under IED_OUTPUT_ROOT, default = repo root) -- identical layout
to `python -m classifier.runner`:
    cache/features/<tag>/<stem>.npz   cache/labels/<stem>.npz
    results/<tag>.json    models/<tag>.{joblib,pt}
    predictions/<tag>.npz
    results/summary.md    results/all_runs.csv
    confusion/*.png       roc/*.png
    logs/mfcc_experiments_<timestamp>.log
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
import time
import traceback
from itertools import product
from pathlib import Path

import numpy as np

# --- make the repo's `classifier` package importable regardless of CWD ---
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ======================================================================
# Fixed configuration -- the "exact same" grid
# ======================================================================

FEATURES = ["mfcc45", "mfcc_sdc_mod_k7"]
CLASSIFIERS = ["rf", "dnn", "bilstm"]
SPLIT_SCHEME = "speaker_held_out"
SEED = 42


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ======================================================================
# Helpers
# ======================================================================

def _echo_model_configs(emit) -> None:
    """Print the dataclass defaults actually used for each classifier."""
    from classifier.models.rf import RFConfig
    from classifier.models.dnn import DNNConfig
    from classifier.models.bilstm import BiLSTMConfig

    for name, cfg in [("RFConfig", RFConfig()),
                      ("DNNConfig", DNNConfig()),
                      ("BiLSTMConfig", BiLSTMConfig())]:
        emit(f"  {name}:")
        for k, v in dataclasses.asdict(cfg).items():
            emit(f"      {k:<22} = {v}")


def _summarise_result(res: dict) -> str:
    """Short one-liner from a run result dict (as written to results/<tag>.json)."""
    m = res.get("metrics", {})
    acc = m.get("accuracy")
    f1m = m.get("f1_macro", m.get("macro_f1"))
    auc = m.get("roc_auc")
    parts = []
    if acc is not None:
        parts.append(f"acc={acc:.3f}")
    if f1m is not None:
        parts.append(f"F1_macro={f1m:.3f}")
    if auc is not None:
        parts.append(f"AUC={auc:.3f}")
    ep = res.get("epochs_run")
    if ep is not None:
        parts.append(f"{ep}ep")
    return "  ".join(parts) if parts else "(no metrics)"


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the MFCC-family x {rf,dnn,bilstm} x 8-task grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--features", nargs="+", default=FEATURES,
                    help=f"default: {FEATURES}")
    ap.add_argument("--classifiers", nargs="+", default=CLASSIFIERS,
                    choices=["rf", "dnn", "bilstm"],
                    help=f"default: {CLASSIFIERS}")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="default: all 8 (fluent_vs_disfluent, multiclass, "
                         "fluent_vs_{I,PR,PhR,WR,PWR,P})")
    ap.add_argument("--split-scheme", default=SPLIT_SCHEME,
                    choices=["speaker_held_out", "random_90_10"],
                    help=f"default: {SPLIT_SCHEME} (reuses splits/<scheme>.json)")
    ap.add_argument("--no-cache-build", action="store_true",
                    help="skip the feature-caching pass (cache/ must already exist)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip any (feature,classifier,task) that already has "
                         "results/<tag>.json  -- use to resume a killed run")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved grid and exit")
    args = ap.parse_args()

    _seed_everything(SEED)

    # Imports deferred so --help works without the full dependency stack.
    from classifier.config import LOGS_DIR, DATASET_DIR, OUTPUT_ROOT, RESULTS_DIR
    from classifier.cache import build_cache
    from classifier.data import all_task_ids, discover_pairs
    from classifier.report import build_full_report
    from classifier.splits import make_random_split, make_speaker_split
    from classifier.train import Run, run_one

    tasks = args.tasks or all_task_ids()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"mfcc_experiments_{ts}.log"
    logf = open(log_path, "w")

    def emit(msg: str = "") -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    emit("=" * 78)
    emit("MFCC-family experiment grid")
    emit("=" * 78)
    emit(f"  repo root      : {REPO_ROOT}")
    emit(f"  dataset dir    : {DATASET_DIR}")
    emit(f"  output root    : {OUTPUT_ROOT}")
    emit(f"  log file       : {log_path}")
    emit(f"  features       : {args.features}")
    emit(f"  classifiers    : {args.classifiers}")
    emit(f"  tasks          : {tasks}")
    emit(f"  split scheme   : {args.split_scheme}")
    emit(f"  seed           : {SEED}")
    emit(f"  cache build    : {'skipped' if args.no_cache_build else 'yes'}")
    emit(f"  skip existing  : {args.skip_existing}")
    emit("")
    emit("  model configs (dataclass defaults):")
    _echo_model_configs(emit)
    emit("")

    # ---- discover dataset ------------------------------------------------
    try:
        pairs = discover_pairs()
    except FileNotFoundError as exc:
        emit(f"[FATAL] {exc}")
        emit("        Put the IED-Extended dataset at the path above, or set "
             "IED_DATASET_DIR, then re-run.")
        logf.close()
        return 2
    if not pairs:
        emit("[FATAL] No (wav, txt) pairs found under the dataset dir "
             "(need 16 kHz mono wavs with same-stem .txt).")
        logf.close()
        return 2
    emit(f"  discovered {len(pairs)} (wav, txt) pairs")
    stems = [p.stem for p in pairs]

    # ---- split (reuse committed splits/<scheme>.json if present) -------
    if args.split_scheme == "speaker_held_out":
        split = make_speaker_split(stems, scheme=args.split_scheme, seed=SEED)
    else:
        split = make_random_split(stems, scheme=args.split_scheme, seed=SEED)
    emit(f"  split: train={len(split['train'])}  val={len(split['val'])}  "
         f"test={len(split['test'])} files")
    emit(f"  test files: {sorted(split['test'])}")
    emit("")

    # ---- resolve run grid --------------------------------------------------
    runs = [
        Run(feature=f, classifier=c, task=t,
            train_stems=split["train"], val_stems=split["val"],
            test_stems=split["test"])
        for f, c, t in product(args.features, args.classifiers, tasks)
    ]

    emit(f"  {len(runs)} runs in the grid:")
    for r in runs:
        done = (RESULTS_DIR / f"{r.tag}.json").exists()
        mark = "skip (exists)" if (done and args.skip_existing) else \
               "rerun" if done else "new"
        emit(f"      {r.tag:<48} [{mark}]")
    emit("")

    if args.dry_run:
        emit("  --dry-run: nothing executed.")
        logf.close()
        return 0

    # ---- build feature cache for just these features ------------------
    if not args.no_cache_build:
        emit("Building feature cache (only missing (feature,file) pairs are "
             "recomputed) ...")
        build_cache(pairs, list(args.features), overwrite=False, verbose=True)
        emit("")

    # ---- run -----------------------------------------------------------
    t_start = time.time()
    n_ok = n_fail = n_skip = 0
    completed: list[tuple[str, str]] = []

    for i, r in enumerate(runs, start=1):
        if args.skip_existing and (RESULTS_DIR / f"{r.tag}.json").exists():
            emit(f"[{i:>2}/{len(runs)}] {r.tag}  -- skip (results exist)")
            n_skip += 1
            continue

        emit(f"[{i:>2}/{len(runs)}] {r.tag}  -- start "
             f"({time.strftime('%H:%M:%S')})")
        t0 = time.time()
        try:
            res = run_one(r)
            dt = time.time() - t0
            line = _summarise_result(res)
            emit(f"[{i:>2}/{len(runs)}] {r.tag}  -- OK  ({dt:.1f}s)  {line}")
            completed.append((r.tag, line))
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 -- one bad run must not stop the sweep
            dt = time.time() - t0
            n_fail += 1
            emit(f"[{i:>2}/{len(runs)}] {r.tag}  -- FAIL ({dt:.1f}s): {exc}")
            logf.write(traceback.format_exc() + "\n")
            logf.flush()

    total_dt = time.time() - t_start

    # ---- report ------------------------------------------------------------
    emit("")
    emit("Rebuilding aggregate report (results/summary.md, all_runs.csv, "
         "confusion/*, roc/*) ...")
    try:
        build_full_report()
    except Exception as exc:  # noqa: BLE001
        emit(f"[warn] report build failed: {exc}")

    # ---- summary ---------------------------------------------------------
    emit("")
    emit("=" * 78)
    emit(f"DONE  ok={n_ok}  fail={n_fail}  skipped={n_skip}  "
         f"wall={total_dt/60:.1f} min")
    emit("=" * 78)
    for tag, line in completed:
        emit(f"  {tag:<48} {line}")
    emit("")
    emit(f"  full log      : {log_path}")
    emit(f"  per-task table: {RESULTS_DIR / 'summary.md'}")
    emit(f"  flat CSV      : {RESULTS_DIR / 'all_runs.csv'}")

    logf.close()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
