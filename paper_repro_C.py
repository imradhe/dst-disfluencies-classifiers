"""
Config C only, memory-safe: SMOTE the whole dataset, then split.
Also reports Fluent-class F1 alongside pos-I F1 to check the
'paper-reported-majority-F1' hypothesis directly.
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier.cache import load_features, load_labels
from classifier.data import discover_pairs, task_labels, DROP_ID
from classifier.imbalance import smote
from classifier.metrics import frame_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

TAG = "mfcc45_ctx3"
TASK = "fluent_vs_I"
MAJ_MAX = 500_000
SEED = 42


def _load_capped():
    """Per-file loading, cap majority at MAJ_MAX total; keep all minority."""
    pairs = discover_pairs()
    stems = [p.stem for p in pairs]

    all_X_min, all_X_maj_parts, all_y_maj_parts = [], [], []
    n_maj_total = 0
    n_min_total = 0

    rng = np.random.default_rng(SEED)
    for stem in stems:
        feats = load_features(TAG, stem)         # (D, T)
        lbls  = load_labels(stem)
        y = task_labels(lbls.class_ids, TASK)
        keep = y != DROP_ID
        y = y[keep]
        X = feats[:, keep].T.astype(np.float32)  # (T_keep, D)

        min_mask = y == 1
        maj_mask = y == 0

        if min_mask.any():
            all_X_min.append(X[min_mask])
            n_min_total += int(min_mask.sum())

        if maj_mask.any():
            all_X_maj_parts.append(X[maj_mask])
            n_maj_total += int(maj_mask.sum())

    X_min = np.concatenate(all_X_min, axis=0) if all_X_min else np.zeros((0, feats.shape[0]), dtype=np.float32)
    print(f"  raw minority frames total: {n_min_total}")
    print(f"  raw majority frames total: {n_maj_total}")

    if n_maj_total > MAJ_MAX:
        print(f"  subsampling majority {n_maj_total} -> {MAJ_MAX}")
        # Reservoir-style: sample proportionally from each file's block.
        pieces = []
        remaining = MAJ_MAX
        for i, part in enumerate(all_X_maj_parts):
            share = int(round(len(part) * MAJ_MAX / n_maj_total))
            share = min(share, len(part), remaining)
            if share <= 0: continue
            idx = rng.choice(len(part), size=share, replace=False)
            pieces.append(part[idx])
            remaining -= share
        X_maj = np.concatenate(pieces, axis=0)
    else:
        X_maj = np.concatenate(all_X_maj_parts, axis=0)

    X = np.concatenate([X_maj, X_min], axis=0)
    y = np.concatenate([
        np.zeros(len(X_maj), dtype=np.int64),
        np.ones (len(X_min), dtype=np.int64),
    ])
    print(f"  capped X: {X.shape}  counts={dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def _eval(rf, X_te, y_te, name, class_names=("Fluent", "I")):
    y_pred = rf.predict(X_te)
    y_score = rf.predict_proba(X_te)[:, 1]
    m = frame_metrics(y_true=y_te, y_pred=y_pred, y_score=y_score,
                      class_names=list(class_names), task_type="binary")
    pc_i = m["per_class"]["I"]
    pc_f = m["per_class"]["Fluent"]
    print(f"  [{name}] acc={m['accuracy']:.4f}  AUC={m.get('roc_auc'):.4f}  "
          f"pos-I  P={pc_i['precision']:.4f}  R={pc_i['recall']:.4f}  F1={pc_i['f1']:.4f}  supp={pc_i['support']}  "
          f"|  Fluent F1={pc_f['f1']:.4f}  supp={pc_f['support']}  "
          f"|  macro-F1={m['f1_macro']:.4f}  weighted-F1={m['f1_weighted']:.4f}")
    return m


def main():
    print("=" * 78)
    print("Config C: SMOTE(whole dataset) -> stratified 80:20 split -> RF")
    print("=" * 78)
    X, y = _load_capped()

    print("  SMOTE minority to majority count ...")
    X_s, y_s = smote(X, y, random_state=SEED)
    print(f"  post-SMOTE: {X_s.shape}  counts={dict(zip(*np.unique(y_s, return_counts=True)))}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y_s, test_size=0.20, random_state=SEED, stratify=y_s,
    )
    print(f"  train: {X_tr.shape}  test: {X_te.shape}")
    print(f"  test counts: {dict(zip(*np.unique(y_te, return_counts=True)))}")

    print("  fitting RF...")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1,
                                class_weight=None, random_state=SEED)
    rf.fit(X_tr, y_tr)
    print(f"  fit in {time.time()-t0:.1f}s")

    print("\n--- Evaluation on the SMOTE'd test (balanced) ---")
    m_bal = _eval(rf, X_te, y_te, "C-bal")

    # Also evaluate on the REAL (non-SMOTE'd) test = original data minus the training portion.
    # Simplest: evaluate on a fresh natural-prevalence hold-out drawn from the
    # non-SMOTE'd pool. We can construct that from `X` (pre-SMOTE) by removing
    # samples used in training. But since SMOTE creates synthetic samples with
    # new coordinates, we cannot recover which real samples ended up in train.
    # Instead: hold out the last 20% of the natural-prevalence dataset before
    # SMOTE'ing, so we have a clean natural-prevalence test set.
    print("\n--- Redo with natural-prevalence test set ---")
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y,
    )
    print(f"  natural train pre-SMOTE: {X_tr2.shape}  test: {X_te2.shape}  "
          f"test counts={dict(zip(*np.unique(y_te2, return_counts=True)))}")
    X_tr2s, y_tr2s = smote(X_tr2, y_tr2, random_state=SEED)
    print(f"  train post-SMOTE: {X_tr2s.shape}")

    print("  fitting RF (natural-test setup)...")
    t0 = time.time()
    rf2 = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1,
                                 class_weight=None, random_state=SEED)
    rf2.fit(X_tr2s, y_tr2s)
    print(f"  fit in {time.time()-t0:.1f}s")

    print("\n--- Evaluation on natural-prevalence test ---")
    m_nat = _eval(rf2, X_te2, y_te2, "C-nat")

    print("\n" + "=" * 78)
    print(f"PAPER 1 TABLE V FILLED PAUSE:  acc = 0.9384   F1 = 0.937")
    print("=" * 78)

    Path("results").mkdir(exist_ok=True)
    with open("results/paper_repro_C.json", "w") as f:
        json.dump({
            "C_smote_before_split_balanced_test": {
                "acc": m_bal["accuracy"],
                "auc": m_bal.get("roc_auc"),
                "pos_I": m_bal["per_class"]["I"],
                "Fluent": m_bal["per_class"]["Fluent"],
                "macro_f1": m_bal["f1_macro"],
                "weighted_f1": m_bal["f1_weighted"],
                "n_test": m_bal["n_samples"],
            },
            "C_smote_train_only_natural_test": {
                "acc": m_nat["accuracy"],
                "auc": m_nat.get("roc_auc"),
                "pos_I": m_nat["per_class"]["I"],
                "Fluent": m_nat["per_class"]["Fluent"],
                "macro_f1": m_nat["f1_macro"],
                "weighted_f1": m_nat["f1_weighted"],
                "n_test": m_nat["n_samples"],
            },
        }, f, indent=2, default=str)
    print(f"Saved -> results/paper_repro_C.json")


if __name__ == "__main__":
    main()
