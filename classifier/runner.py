"""
classifier.runner
=================
Orchestrate the full experiment grid defined by spec section 6:

    features    = 6 configs (baselines only per spec)
    classifiers = [rf, dnn, bilstm]
    tasks       = fluent_vs_disfluent, multiclass,
                  and fluent_vs_{I, PR, PhR, WR, PWR, P}

Total = 6 x 3 x 8 = 144 runs. Runs are streamed to disk under
results/<tag>.json; failures are logged but never stop the sweep.

Usage
-----
    python -m classifier.runner --classifier rf         # just RF, all features + tasks
    python -m classifier.runner --feature mfcc45        # all classifiers on one feature
    python -m classifier.runner                         # full grid
    python -m classifier.runner --dry-run               # print the grid, do nothing
    python -m classifier.runner --report-only           # skip training, just re-aggregate CSV / md
"""

from __future__ import annotations

import argparse
import time
import traceback
from itertools import product
from typing import List

from classifier.cache import build_cache
from classifier.config import LOGS_DIR
from classifier.data import all_task_ids, discover_pairs
from classifier.report import build_full_report
from classifier.splits import make_random_split, make_speaker_split
from classifier.train import Run, run_one


BASELINE_FEATURES = [
    "mfcc45",
    "mfcc_sdc_mod_k7",
    "mfcc_sdc_conv_k7",
    "sffcc_sdc_mod_k7",
    "prosody32",
    "prosody96",
]

CLASSIFIERS = ["rf", "dnn", "bilstm"]


def _build_runs(
    features: List[str],
    classifiers: List[str],
    tasks: List[str],
    split: dict,
) -> List[Run]:
    runs = []
    for f, c, t in product(features, classifiers, tasks):
        runs.append(Run(
            feature=f, classifier=c, task=t,
            train_stems=split["train"],
            val_stems=split["val"],
            test_stems=split["test"],
        ))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", nargs="+", default=None)
    ap.add_argument("--classifier", nargs="+", default=None)
    ap.add_argument("--task", nargs="+", default=None)
    ap.add_argument("--split-scheme", default="random_90_10")
    ap.add_argument("--no-cache-build", action="store_true",
                    help="Skip the feature-caching pass (use existing cache).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        build_full_report()
        print("Report written.")
        return

    features    = args.feature    or BASELINE_FEATURES
    classifiers = args.classifier or CLASSIFIERS
    tasks       = args.task       or all_task_ids()

    print(f"Features   : {features}")
    print(f"Classifiers: {classifiers}")
    print(f"Tasks      : {tasks}")

    pairs = discover_pairs()
    if not pairs:
        raise RuntimeError("No (wav, txt) pairs discovered. "
                           "Check IED_DATASET_DIR.")
    print(f"Discovered {len(pairs)} files")

    stems = [p.stem for p in pairs]
    if args.split_scheme == "speaker_held_out":
        split = make_speaker_split(stems, scheme=args.split_scheme)
    else:
        split = make_random_split(stems, scheme=args.split_scheme)
    print(f"Split scheme={args.split_scheme}  "
          f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

    if not args.no_cache_build:
        print("Building feature cache ...")
        build_cache(pairs, features)

    runs = _build_runs(features, classifiers, tasks, split)
    print(f"Total runs : {len(runs)}")

    if args.dry_run:
        for r in runs:
            print(f"  {r.tag}")
        return

    log_path = LOGS_DIR / f"runner_{int(time.time())}.log"
    n_ok = 0
    n_fail = 0
    with open(log_path, "w") as log:
        for i, r in enumerate(runs, start=1):
            print(f"\n=== {i}/{len(runs)}  {r.tag} ===")
            try:
                run_one(r)
                n_ok += 1
                log.write(f"OK  {r.tag}\n")
            except Exception as exc:
                n_fail += 1
                log.write(f"FAIL {r.tag}: {exc}\n")
                log.write(traceback.format_exc() + "\n")
                print(f"[!] {r.tag} FAILED: {exc}")

    print(f"\nDone. OK={n_ok}  FAIL={n_fail}  Log: {log_path}")

    print("Building report ...")
    build_full_report()
    print(f"Report ready: results/summary.md and results/all_runs.csv")


if __name__ == "__main__":
    main()
