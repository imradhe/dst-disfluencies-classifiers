"""
Strict paper-reproduction test.

Target: Paper 1 (Garg 2021 NCC), Table V, row "Filled Pause / RF (SMOTE)"
Paper reported: accuracy 93.84%, F1 0.937

Paper's stated setup (Section IV):
    - Features: 13 MFCC + C0 + energy + Delta + Delta-Delta = 45 dim
    - Context: stack frames from 3 before and 3 after ("+/-3")
      -> 45 * 7 = 315-dim per frame                    (our `mfcc45_ctx3`)
    - Window / hop: 25 ms / 10 ms (Hamming)
    - Mean-variance normalization per file
    - Classifier: Random Forest, n=100, max_depth=20
    - Imbalance: SMOTE (upsample minority to majority count)
    - Split: "A train-test split of 80:20 was used"
      -- unit unspecified; we test three interpretations below
    - Task: Fluent vs Filled-Pause (I) binary classification

Configurations compared:
    A. FILE-level 80:20, SMOTE train-only, no class_weight
       -- most conservative interpretation of the paper's setup.
    B. FRAME-level shuffled 80:20, SMOTE train-only, no class_weight
       -- maximum leakage: adjacent frames of the same file split
       across train and test. Tests whether the paper's number
       requires frame-level shuffling.
    C. SMOTE the WHOLE dataset first, then a stratified 80:20 split
       (test now contains synthetic minority samples generated from
       real neighbours in the training data). Tests whether the
       paper's number requires this classic imbalanced-learning
       methodology error.

For RAM tractability at 315-dim, majority is capped at MAJ_MAX
frames before SMOTE. Config C SMOTE'es first, then splits, so its
majority cap applies before the split.

Runs on the full IED-Extended corpus (132 files, 20.5 hours).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier.cache import stack_dataset, load_features, load_labels
from classifier.data import discover_pairs, task_labels, DROP_ID
from classifier.imbalance import smote
from classifier.metrics import frame_metrics
from classifier.splits import make_random_split

from sklearn.ensemble import RandomForestClassifier


TAG        = "mfcc45_ctx3"
TASK       = "fluent_vs_I"
CLASS_NAMES = ["Fluent", "I"]

# Cap majority before SMOTE so 315-dim data fits in ~16 GB RAM.
# Pure paper SMOTE at 15M majority x 315 dim would need ~40 GB.
MAJ_MAX = 500_000
SEED = 42


def _fit_predict(X_tr, y_tr, X_te, y_te, name):
    print(f"  [{name}] fit RF on {X_tr.shape} -> test {X_te.shape}")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        class_weight=None,     # paper's SMOTE stands alone
        random_state=SEED,
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    y_score = rf.predict_proba(X_te)[:, 1]
    dt = time.time() - t0
    m = frame_metrics(
        y_true=y_te, y_pred=y_pred, y_score=y_score,
        class_names=CLASS_NAMES, task_type="binary",
    )
    pc = m["per_class"]["I"]
    print(f"  [{name}] fit+eval in {dt:.1f}s")
    print(f"  [{name}] RESULT  acc={m['accuracy']:.4f}  AUC={m.get('roc_auc'):.4f}  "
          f"pos-I  P={pc['precision']:.4f}  R={pc['recall']:.4f}  F1={pc['f1']:.4f}  "
          f"support={pc['support']}   (paper: F1=0.937 acc=0.9384)")
    return m, dt


def _cap_majority_then_smote(X, y, name):
    from collections import Counter
    counts = Counter(y.tolist())
    if len(counts) < 2:
        return X, y
    maj_class = max(counts, key=counts.get)
    if counts[maj_class] > MAJ_MAX:
        print(f"  [{name}] cap majority {maj_class}={counts[maj_class]} -> {MAJ_MAX}")
        rng = np.random.default_rng(SEED)
        keep_maj = rng.choice(np.where(y == maj_class)[0], size=MAJ_MAX, replace=False)
        keep_min = np.where(y != maj_class)[0]
        idx = np.concatenate([keep_maj, keep_min])
        rng.shuffle(idx)
        X = X[idx]; y = y[idx]
    print(f"  [{name}] SMOTE minority -> majority count ({dict(zip(*np.unique(y, return_counts=True)))})")
    X, y = smote(X, y, random_state=SEED)
    print(f"  [{name}] after SMOTE: {X.shape}  counts={dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


# -----------------------------------------------------------------
# Config A -- FILE-level split, SMOTE train only  (paper-fidelity)
# -----------------------------------------------------------------

def run_A():
    print("\n" + "=" * 78)
    print("A. FILE-level 80:20 split, SMOTE train only")
    print("=" * 78)
    pairs = discover_pairs()
    stems = [p.stem for p in pairs]
    split = make_random_split(stems, test_frac=0.20, val_frac_of_train=0.0,
                              scheme="paper_strict_A_file80", seed=SEED, overwrite=True)
    print(f"  train files: {len(split['train'])}  test files: {len(split['test'])}")

    X_tr, y_tr, _ = stack_dataset(split['train'], TAG, TASK)
    X_te, y_te, _ = stack_dataset(split['test'],  TAG, TASK)
    print(f"  raw train: {X_tr.shape}  test: {X_te.shape}")
    X_tr, y_tr = _cap_majority_then_smote(X_tr, y_tr, "A")
    return _fit_predict(X_tr, y_tr, X_te, y_te, "A")


# -----------------------------------------------------------------
# Config B -- FRAME-level shuffled 80:20, SMOTE train only
# -----------------------------------------------------------------

def run_B():
    print("\n" + "=" * 78)
    print("B. FRAME-level shuffled 80:20 split, SMOTE train only")
    print("=" * 78)
    pairs = discover_pairs()
    stems = [p.stem for p in pairs]

    # Load everything, concatenate, shuffle at the FRAME level.
    X_parts, y_parts = [], []
    for stem in stems:
        feats = load_features(TAG, stem)
        lbls  = load_labels(stem)
        y = task_labels(lbls.class_ids, TASK)
        keep = y != DROP_ID
        X_parts.append(feats[:, keep].T.astype(np.float32))
        y_parts.append(y[keep].astype(np.int64))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    print(f"  full frame set: {X.shape}")

    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = int(0.8 * len(y))
    tr_idx = idx[:cut]; te_idx = idx[cut:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    print(f"  raw train: {X_tr.shape}  test: {X_te.shape}")

    X_tr, y_tr = _cap_majority_then_smote(X_tr, y_tr, "B")
    return _fit_predict(X_tr, y_tr, X_te, y_te, "B")


# -----------------------------------------------------------------
# Config C -- SMOTE the WHOLE dataset first, THEN 80:20 split
# (classic "SMOTE-before-split" methodological error)
# -----------------------------------------------------------------

def run_C():
    print("\n" + "=" * 78)
    print("C. SMOTE(whole dataset), then stratified 80:20 split")
    print("=" * 78)
    pairs = discover_pairs()
    stems = [p.stem for p in pairs]

    X_parts, y_parts = [], []
    for stem in stems:
        feats = load_features(TAG, stem)
        lbls  = load_labels(stem)
        y = task_labels(lbls.class_ids, TASK)
        keep = y != DROP_ID
        X_parts.append(feats[:, keep].T.astype(np.float32))
        y_parts.append(y[keep].astype(np.int64))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    print(f"  raw full set: {X.shape}  counts={dict(zip(*np.unique(y, return_counts=True)))}")

    X, y = _cap_majority_then_smote(X, y, "C")

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y,
    )
    print(f"  post-split train: {X_tr.shape}  test: {X_te.shape}")
    print(f"  test counts    : {dict(zip(*np.unique(y_te, return_counts=True)))}")
    return _fit_predict(X_tr, y_tr, X_te, y_te, "C")


def main():
    Path("results").mkdir(exist_ok=True)
    results = {}

    for name, fn in [("A_file80_smote_train", run_A),
                     ("B_frame80_smote_train", run_B),
                     ("C_smote_all_then_split", run_C)]:
        try:
            m, dt = fn()
            results[name] = {
                "accuracy":     m["accuracy"],
                "roc_auc":      m.get("roc_auc"),
                "pos_precision": m["per_class"]["I"]["precision"],
                "pos_recall":    m["per_class"]["I"]["recall"],
                "pos_f1":        m["per_class"]["I"]["f1"],
                "pos_support":   m["per_class"]["I"]["support"],
                "n_test":        m["n_samples"],
                "elapsed_s":     dt,
            }
        except Exception as exc:
            print(f"  ! config {name} failed: {exc}")
            results[name] = {"error": str(exc)}

    with open("results/paper_repro_strict.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY  (paper 1 Table V Filled Pause: F1 = 0.937, acc = 0.9384)")
    print("=" * 78)
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:32}  ERROR: {r['error']}")
        else:
            print(f"  {name:32}  acc={r['accuracy']:.4f}  AUC={r['roc_auc']:.4f}  "
                  f"pos-I F1={r['pos_f1']:.4f}  P={r['pos_precision']:.4f}  R={r['pos_recall']:.4f}  "
                  f"support={r['pos_support']}")
    print(f"\nSaved -> results/paper_repro_strict.json")


if __name__ == "__main__":
    main()
