"""
classifier.report
=================
Turn per-run JSON results into aggregated CSV + markdown + plots.

Outputs (under OUTPUT_ROOT):
    results/all_runs.csv          -- one row per (feature, classifier, task, class)
    results/summary.md            -- per-task markdown tables
    confusion/{tag}.png           -- multi-class confusion matrix
    roc/{tag}.png                 -- binary ROC + PR curves
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from classifier.config import (
    CONFUSION_DIR,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    ROC_DIR,
)


# ---------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------

def _iter_result_files() -> Iterable[Path]:
    for p in sorted(RESULTS_DIR.glob("*.json")):
        yield p


def load_all_results() -> List[Dict]:
    """
    Every grid-run result JSON in RESULTS_DIR.

    Files that are not grid runs (e.g. the ad-hoc `paper_repro_*.json`
    probes, which use a different schema and may even be a top-level
    list) are skipped rather than crashing the aggregator.
    """
    out = []
    for p in _iter_result_files():
        with open(p) as f:
            try:
                obj = json.load(f)
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict) and "metrics" in obj and "tag" in obj:
            out.append(obj)
    return out


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------

def _run_to_rows(r: Dict) -> List[Dict]:
    """
    Explode one run's metric bundle into rows keyed by class.
    Adds one summary row with class = '__all__'.
    """
    rows = []
    m = r["metrics"]

    summary = {
        "feature":     r["feature"],
        "classifier":  r["classifier"],
        "task":        r["task"],
        "task_type":   r["task_type"],
        "class":       "__all__",
        "accuracy":    m["accuracy"],
        "precision":   m.get("precision_macro"),
        "recall":      m.get("recall_macro"),
        "f1":          m.get("f1_macro"),
        "precision_micro":    m.get("precision_micro"),
        "recall_micro":       m.get("recall_micro"),
        "f1_micro":           m.get("f1_micro"),
        "precision_weighted": m.get("precision_weighted"),
        "recall_weighted":    m.get("recall_weighted"),
        "f1_weighted":        m.get("f1_weighted"),
        "roc_auc":     m.get("roc_auc"),
        "pr_auc":      m.get("pr_auc"),
        "baseline":    m.get("majority_baseline_accuracy"),
        "n_test":      r.get("n_test"),
        "n_train_raw": r.get("n_train_raw"),
        "n_train_bal": r.get("n_train_bal"),
        "elapsed_sec": r.get("elapsed_sec"),
    }
    rows.append(summary)

    for cname, cm in m.get("per_class", {}).items():
        rows.append({
            "feature":    r["feature"],
            "classifier": r["classifier"],
            "task":       r["task"],
            "task_type":  r["task_type"],
            "class":      cname,
            "precision":  cm["precision"],
            "recall":     cm["recall"],
            "f1":         cm["f1"],
            "support":    cm["support"],
        })
    return rows


def write_csv(path: Path = None) -> Path:
    if path is None:
        path = RESULTS_DIR / "all_runs.csv"
    rows = []
    for r in load_all_results():
        rows.extend(_run_to_rows(r))

    if not rows:
        path.write_text("")
        return path

    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ---------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------

def write_markdown(path: Path = None) -> Path:
    if path is None:
        path = RESULTS_DIR / "summary.md"

    results = load_all_results()
    if not results:
        path.write_text("# Results\n\nNo runs yet.\n")
        return path

    # Group by task -> (feature, classifier) -> summary metrics
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in results:
        by_task[r["task"]].append(r)

    lines = ["# Results", ""]
    for task, runs in sorted(by_task.items()):
        lines.append(f"## Task: `{task}`")
        lines.append("")
        lines.append("| Feature | Classifier | Acc | F1 macro | F1 micro | F1 weighted | AUC | Baseline |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        runs.sort(key=lambda r: (-r["metrics"]["f1_macro"], r["feature"]))
        for r in runs:
            m = r["metrics"]
            auc = m.get("roc_auc")
            lines.append(
                f"| `{r['feature']}` | `{r['classifier']}` | "
                f"{m['accuracy']:.3f} | {m['f1_macro']:.3f} | "
                f"{m.get('f1_micro', 0):.3f} | {m.get('f1_weighted', 0):.3f} | "
                f"{'' if auc is None else f'{auc:.3f}'} | "
                f"{m['majority_baseline_accuracy']:.3f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_confusion(result: Dict, out_dir: Path = None) -> Path:
    if out_dir is None:
        out_dir = CONFUSION_DIR
    if result["task_type"] != "multiclass":
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(result["metrics"]["confusion_matrix"], dtype=np.int64)
    classes = result["class_names"]

    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(classes),
                                    0.8 + 0.6 * len(classes)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                    fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(result["tag"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = out_dir / f"{result['tag']}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_roc_pr(result: Dict, out_dir: Path = None) -> Path:
    if out_dir is None:
        out_dir = ROC_DIR
    if result["task_type"] != "binary":
        return None

    preds_path = PREDICTIONS_DIR / f"{result['tag']}.npz"
    if not preds_path.exists():
        return None
    d = np.load(preds_path)
    y_true = d["y_true"]
    y_score = d["y_score"]
    if y_score.ndim != 1:
        return None

    from sklearn.metrics import (
        precision_recall_curve,
        roc_curve,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    axs[0].plot(fpr, tpr, label=f"AUC={result['metrics'].get('roc_auc', 0):.3f}")
    axs[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR")
    axs[0].set_title("ROC"); axs[0].legend()

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    axs[1].plot(rec, prec, label=f"AP={result['metrics'].get('pr_auc', 0):.3f}")
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
    axs[1].set_title("Precision-Recall"); axs[1].legend()

    fig.suptitle(result["tag"])
    fig.tight_layout()
    out = out_dir / f"{result['tag']}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def build_full_report() -> None:
    """Convenience: CSV + Markdown + confusion + ROC/PR for every run."""
    write_csv()
    write_markdown()
    for r in load_all_results():
        try:
            plot_confusion(r)
        except Exception as exc:
            print(f"[report] confusion failed for {r['tag']}: {exc}")
        try:
            plot_roc_pr(r)
        except Exception as exc:
            print(f"[report] roc/pr failed for {r['tag']}: {exc}")
