"""
Paper-reproduction test:
    Paper 1 (Garg 2021, NCC) Table V:
        MFCC 45-dim + Delta + Delta-Delta (=45) with +/-3 frame stacking (=315)
        Random Forest (100 trees, max_depth 20)
        SMOTE (upsample minority to majority count)
        Random 90:10 train-test split (paper says 80:20; we do 90:10 here
        to match our earlier speaker-held-out run; only difference: 90 vs 80).
        Task: Fluent vs Filled-Pause (I)
    Paper's reported F1 for I: 0.937

We compare that config against the SAME features + speaker-held-out
split, so the delta isolates the speaker-leakage contribution.

For tractability on 16 GB RAM at 315-dim, majority is undersampled to
`MAJ_MAX` frames before SMOTE (pure SMOTE at paper scale would need
~15 GB in RAM). This is noted honestly in the result.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier.cache import stack_dataset
from classifier.data import discover_pairs
from classifier.imbalance import smote, undersample_majority
from classifier.metrics import frame_metrics
from classifier.splits import make_random_split, make_speaker_split

from sklearn.ensemble import RandomForestClassifier


TAG        = "mfcc45_ctx3"
TASK       = "fluent_vs_I"
CLASS_NAMES = ["Fluent", "I"]

# Undersample majority to at most this many frames before SMOTE, to
# keep peak RAM usable. Paper had ~5.9M majority frames; at 315-dim
# that alone is ~7.5 GB, plus the SMOTE'd minority doubles it.
MAJ_MAX = 500_000


def _fit_predict(X_tr, y_tr, X_te, y_te, class_weight):
    print(f"  training RF (class_weight={class_weight}) on {X_tr.shape}...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=42,
    )
    rf.fit(X_tr, y_tr)
    dt = time.time() - t0
    print(f"  fit in {dt:.1f}s")
    y_pred = rf.predict(X_te)
    try:
        y_score = rf.predict_proba(X_te)[:, 1]
    except Exception:
        y_score = None
    m = frame_metrics(
        y_true=y_te,
        y_pred=y_pred,
        y_score=y_score,
        class_names=CLASS_NAMES,
        task_type="binary",
    )
    return m, dt


def _do_config(name, split, apply_smote, class_weight):
    print(f"\n=== {name} ===")
    print(f"  split: train={len(split['train'])}f, test={len(split['test'])}f")

    X_tr, y_tr, _ = stack_dataset(split['train'], TAG, TASK)
    X_te, y_te, _ = stack_dataset(split['test'],  TAG, TASK)
    print(f"  raw train: {X_tr.shape},  test: {X_te.shape}")
    print(f"  raw train class counts: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
    print(f"  test class counts     : {dict(zip(*np.unique(y_te, return_counts=True)))}")

    # Bound majority for tractability, then SMOTE minority up to it.
    from collections import Counter
    counts = Counter(y_tr.tolist())
    if apply_smote:
        maj_class = max(counts, key=counts.get)
        maj_count = counts[maj_class]
        if maj_count > MAJ_MAX:
            print(f"  undersampling majority ({maj_class}={maj_count}) -> {MAJ_MAX} before SMOTE")
            # Undersample majority only, keep minority as-is.
            rng = np.random.default_rng(42)
            keep_maj = rng.choice(np.where(y_tr == maj_class)[0], size=MAJ_MAX, replace=False)
            keep_min = np.where(y_tr != maj_class)[0]
            idx = np.concatenate([keep_maj, keep_min])
            rng.shuffle(idx)
            X_tr = X_tr[idx]; y_tr = y_tr[idx]
        print(f"  SMOTE'ing minority up to majority count ...")
        X_tr, y_tr = smote(X_tr, y_tr, random_state=42)
    print(f"  balanced train: {X_tr.shape}, counts: {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    m, dt = _fit_predict(X_tr, y_tr, X_te, y_te, class_weight)
    pc = m["per_class"]["I"]
    print(f"  RESULT  acc={m['accuracy']:.3f}  AUC={m.get('roc_auc'):.3f}  "
          f"pos-I  P={pc['precision']:.3f}  R={pc['recall']:.3f}  F1={pc['f1']:.3f}  "
          f"support={pc['support']}  (paper: F1=0.937)")

    return {
        "name": name,
        "accuracy": m["accuracy"],
        "roc_auc": m.get("roc_auc"),
        "pos_precision": pc["precision"],
        "pos_recall":    pc["recall"],
        "pos_f1":        pc["f1"],
        "pos_support":   pc["support"],
        "train_shape":   list(X_tr.shape),
        "test_shape":    list(X_te.shape),
        "elapsed":       dt,
    }


def main():
    pairs = discover_pairs()
    stems = [p.stem for p in pairs]

    # Random 90:10 file split (paper convention -- same speakers can appear in both).
    split_random = make_random_split(stems, test_frac=0.10, val_frac_of_train=0.0,
                                     scheme="paper_repro_random", seed=42, overwrite=True)
    # Speaker-held-out split for the honest comparison.
    split_speaker = make_speaker_split(stems, test_frac=0.10, val_frac_of_train=0.0,
                                       scheme="paper_repro_speaker_held_out",
                                       seed=42, overwrite=True)

    results = []
    # A: paper config -- SMOTE + no class_weight + random file split
    results.append(_do_config("A. Paper config (random split + SMOTE)",
                              split_random, apply_smote=True, class_weight=None))
    # B: same features + SMOTE but speaker-held-out
    results.append(_do_config("B. Speaker-held-out + SMOTE",
                              split_speaker, apply_smote=True, class_weight=None))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Paper 1 Table V (Fluent-vs-I, MFCC + RF + SMOTE):  F1 = 0.937\n")
    for r in results:
        print(f"  {r['name']:52}  pos-I F1 = {r['pos_f1']:.3f}  (P={r['pos_precision']:.3f}, R={r['pos_recall']:.3f})")

    Path("results").mkdir(exist_ok=True)
    with open("results/paper_repro_fluent_vs_I.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> results/paper_repro_fluent_vs_I.json")


if __name__ == "__main__":
    main()
