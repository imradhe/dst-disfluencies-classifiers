"""
Paper 1 Table V, all 5 disfluency classes, honest paper-fidelity config:
    - features: mfcc45_ctx3 (315-dim, +/-3 stacked)
    - classifier: RF (100 trees, max_depth 20)
    - imbalance: SMOTE train only, no class_weight
    - split: file-level random 80:20
    - test: natural prevalence
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier.cache import stack_dataset
from classifier.data import discover_pairs
from classifier.imbalance import smote
from classifier.metrics import frame_metrics
from classifier.splits import make_random_split
from sklearn.ensemble import RandomForestClassifier

TAG = "mfcc45_ctx3"
MAJ_MAX = 500_000
SEED = 42

PAPER = {
    "I":   (0.9384, 0.937),
    "PR":  (0.9821, 0.979),
    "PWR": (0.9295, 0.927),
    "WR":  (0.9332, 0.931),
    "PhR": (0.8102, 0.814),
}


def _cap_and_smote(X, y):
    from collections import Counter
    counts = Counter(y.tolist())
    if len(counts) < 2:
        return X, y
    maj = max(counts, key=counts.get)
    if counts[maj] > MAJ_MAX:
        rng = np.random.default_rng(SEED)
        keep_maj = rng.choice(np.where(y == maj)[0], size=MAJ_MAX, replace=False)
        keep_min = np.where(y != maj)[0]
        idx = np.concatenate([keep_maj, keep_min]); rng.shuffle(idx)
        X = X[idx]; y = y[idx]
    return smote(X, y, random_state=SEED)


def main():
    pairs = discover_pairs()
    stems = [p.stem for p in pairs]
    split = make_random_split(
        stems, test_frac=0.20, val_frac_of_train=0.0,
        scheme="paper_allclasses_file80", seed=SEED, overwrite=True,
    )
    print(f"train={len(split['train'])}f  test={len(split['test'])}f\n")

    import gc
    # Reload prior partial results so we can resume without rerunning I.
    prev_path = Path("results/paper_repro_all_classes.json")
    results = {}
    if prev_path.exists():
        try:
            with open(prev_path) as f:
                results = json.load(f)
            print(f"Resuming: {list(results.keys())} already have results")
        except Exception:
            results = {}

    for cls in ["I", "PR", "PhR", "WR", "PWR"]:
        if cls in results and "error" not in results[cls]:
            print(f"=== fluent_vs_{cls}  (already done, skipping) ===\n")
            continue
        task = f"fluent_vs_{cls}"
        print(f"=== {task} ===")
        X_tr, y_tr, _ = stack_dataset(split['train'], TAG, task)
        X_te, y_te, _ = stack_dataset(split['test'],  TAG, task)
        print(f"  raw train: {X_tr.shape}  test: {X_te.shape}")
        print(f"  train counts: {dict(zip(*np.unique(y_tr, return_counts=True)))}")
        print(f"  test  counts: {dict(zip(*np.unique(y_te, return_counts=True)))}")

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            print(f"  ! skipping {cls}: one class has 0 samples in train or test")
            results[cls] = {"error": "one class absent"}
            continue

        X_bal, y_bal = _cap_and_smote(X_tr, y_tr)
        print(f"  post-SMOTE: {X_bal.shape}")

        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=20, n_jobs=-1,
            class_weight=None, random_state=SEED,
        )
        rf.fit(X_bal, y_bal)
        dt = time.time() - t0
        y_pred = rf.predict(X_te)
        y_score = rf.predict_proba(X_te)[:, 1]
        m = frame_metrics(y_true=y_te, y_pred=y_pred, y_score=y_score,
                          class_names=["Fluent", cls], task_type="binary")
        pc = m["per_class"][cls]
        fc = m["per_class"]["Fluent"]

        paper_acc, paper_f1 = PAPER.get(cls, (None, None))
        print(f"  RESULT  acc={m['accuracy']:.4f}  AUC={m.get('roc_auc'):.4f}  "
              f"pos-{cls} F1={pc['f1']:.4f} (P={pc['precision']:.4f}, R={pc['recall']:.4f})  "
              f"Fluent F1={fc['f1']:.4f}  (paper: acc={paper_acc}, F1={paper_f1})  [{dt:.0f}s]\n")

        results[cls] = {
            "paper_acc": paper_acc, "paper_f1_reported": paper_f1,
            "ours_acc": m["accuracy"], "ours_auc": m.get("roc_auc"),
            "ours_pos_f1": pc["f1"], "ours_pos_P": pc["precision"], "ours_pos_R": pc["recall"],
            "ours_pos_support": pc["support"],
            "ours_fluent_f1": fc["f1"], "ours_fluent_support": fc["support"],
            "ours_macro_f1": m["f1_macro"], "ours_weighted_f1": m["f1_weighted"],
            "fit_seconds": dt,
        }

        # Persist after each class so a crash doesn't wipe finished work.
        with open("results/paper_repro_all_classes.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Free large arrays before the next class -- macOS OOM-kills at ~14 GB.
        del X_tr, y_tr, X_te, y_te, X_bal, y_bal, rf, y_pred, y_score, m
        gc.collect()

    print("\n" + "=" * 90)
    print(f"{'class':<6} {'paper acc':>10} {'paper F1':>10}    {'ours acc':>9} {'ours AUC':>9} {'ours pos-F1':>12} {'ours Fluent-F1':>15}")
    print("=" * 90)
    for cls in ["I", "PR", "PhR", "WR", "PWR"]:
        r = results.get(cls, {})
        if "error" in r:
            print(f"  {cls:<6} skipped ({r['error']})")
            continue
        print(f"  {cls:<4} {r['paper_acc']*100:8.2f}% {r['paper_f1_reported']:10.3f}    "
              f"{r['ours_acc']*100:7.2f}% {r['ours_auc']:8.3f}   {r['ours_pos_f1']:9.3f}      {r['ours_fluent_f1']:9.3f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/paper_repro_all_classes.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> results/paper_repro_all_classes.json")


if __name__ == "__main__":
    main()
