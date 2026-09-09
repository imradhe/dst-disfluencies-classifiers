# Disfluency Classifier v2 — IED (Extended)

Frame-level speech-disfluency classification on the IED-Extended corpus,
with paper-faithful feature extractors used as baselines.

- **Spec** (source of truth for every design choice): [`CLASSIFIER_SPEC.md`](CLASSIFIER_SPEC.md)
- **Current results snapshot**: [`results/summary.md`](results/summary.md), [`results/all_runs.csv`](results/all_runs.csv)
- **Papers** reproduced as feature baselines: [`Papers/`](Papers)
  - Garg et al. 2021 (NCC) — MFCC 45-dim + RF
  - Mehrotra et al. 2022 (IC3) — MFCC+SDC, SFFCC+SDC
  - Mehrotra et al. 2021 (APSIPA) — syllable prosody (32-dim)

## Status (as of this snapshot)

Pipeline runs end-to-end. One sweep is complete:
**`{mfcc45, mfcc_sdc_mod_k7} × {rf, dnn} × 8 tasks = 32 runs`**, speaker-held-out
split (train 117 / val 11 / test 4 files).

- Not yet run: **BiLSTM** (code present, never executed), and features
  `mfcc_sdc_conv_k7`, `sffcc_sdc_mod_k7`, `prosody32/96`, `mfcc_sdc_mod_k12`
  (only `prosody32` is partially cached: 6/132 files).
- Full spec grid is `6 features × 3 classifiers × 8 tasks = 144 runs` — 32 done.
- Every model so far scores **at or below the majority-class baseline**;
  minority-class recall on the speaker-disjoint test set is near zero.
- `paper_repro*.py` could **not** reproduce Garg 2021 Table V (F1 0.937 for
  Filled-Pause) under an honest frame-level protocol; the paper number is
  only approached (~0.88) with SMOTE-before-split leakage. See those
  scripts' docstrings and `results/paper_repro_*.json`.

Fuller TODO list at the bottom.

## What is / isn't in this repo

**Included** — all source, the spec, the papers, and the current-state
outputs for comparison: `splits/`, `results/`, `predictions/`,
`confusion/`, `roc/`, `logs/`.

**Not included** (regenerated on the GPU box):

| Path | Size | How to get it |
|---|---|---|
| `IED/` | ~3.3 GB | Proprietary dataset — copy it in separately (see below) |
| `cache/` | ~15 GB | Rebuilt from `IED/` by the runner (`build_cache`) |
| `models/` | ~4.7 GB | Retrained by the runner; `rsync` from the Mac if you want to skip RF retraining |

### Dataset placement

The runner discovers `(*.wav, *.txt)` pairs under `IED/` at the repo root.
Layout: one WAV + one same-stem TXT per recording; TXT is
`start<TAB>end<TAB>CLASS` per line (classes `I PR PhR WR PWR P`; everything
else is `Fluent`). 16 kHz mono; files at other rates are skipped.

Override the location without moving files:

```bash
export IED_DATASET_DIR=/data/IED
export IED_OUTPUT_ROOT=/scratch/disfluency-v2   # cache/results/models/... go here
```

## Setup

One command — creates `./.venv` (Python 3.11), installs everything,
verifies imports, writes `requirements-lock.txt`:

```bash
./setup.sh                 # CPU / Apple-MPS torch
./setup.sh --cuda cu124    # Linux GPU box: CUDA 12.4 torch wheels (also cu121, cu128, …)
./setup.sh --recreate      # wipe .venv and rebuild
./setup.sh --no-optional   # skip gammatone + imbalanced-learn
```

Manual equivalent:

```bash
python3.11 -m venv .venv && . .venv/bin/activate && pip install -U pip
# GPU box: CUDA torch first so pip keeps that build
pip install torch==2.13.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/cu124
pip install -r requirements.txt          # torch lines + -r core + -r optional
```

Requirements files:

| File | Contents |
|---|---|
| `requirements.txt` | one-shot: torch pins + `-r` the two below |
| `requirements-core.txt` | mandatory stack (numpy, scipy, sklearn, pandas, matplotlib, librosa, soundfile, parselmouth, …), exact pins |
| `requirements-optional.txt` | `gammatone` (prosody feats, git-only) + `imbalanced-learn` (`paper_repro` SMOTE) — **not** needed for the MFCC grid |
| `requirements-lock.txt` | full `pip freeze` of a verified `.venv`, (re)written by `setup.sh` |

Python 3.11 is the tested interpreter (`.python-version`). Verified clean
install on macOS/arm64: `numpy 2.4.6`, `librosa 0.11.0` (`numba 0.67`,
`llvmlite 0.49`), `scikit-learn 1.9.0`, `torch 2.13.0` — no resolver
conflicts.

Device is auto-selected (`cuda → mps → cpu`) in
[`classifier/models/dnn.py`](classifier/models/dnn.py) and
[`classifier/models/bilstm.py`](classifier/models/bilstm.py). RF is
scikit-learn (CPU, RAM-heavy — see note in `classifier/imbalance.py`).

> **BiLSTM on Apple MPS is effectively unusable** — the packed-sequence
> LSTM path stalls (an MPS/PyTorch limitation, not a bug here). Run the
> BiLSTM combos on the CUDA box; RF and DNN are fine on MPS/CPU.

## Running

### `run_mfcc_experiments.py` — the MFCC sweep for the GPU box

Self-contained driver: **`mfcc45` + `mfcc_sdc_mod_k7`  ×  `rf` + `dnn` + `bilstm`
×  8 tasks = 48 runs**, on the committed `speaker_held_out` split, seed 42,
with the dataclass-default model configs (echoed at startup). This is the
existing 32-run `rf`/`dnn` sweep plus the 16 never-run `bilstm` combos.

```bash
python run_mfcc_experiments.py                 # full 48-run sweep (builds cache first)
python run_mfcc_experiments.py --dry-run       # print the resolved grid, exit
python run_mfcc_experiments.py --skip-existing # resume: skip tags with a results JSON
python run_mfcc_experiments.py --no-cache-build
python run_mfcc_experiments.py --classifiers bilstm
```

It reuses `splits/speaker_held_out.json` verbatim, streams `results/<tag>.json`
+ models + predictions, rebuilds `results/summary.md` / `all_runs.csv` /
plots, and logs to `logs/mfcc_experiments_<ts>.log`. One failed run never
stops the sweep.

### Full grid via the package runner

```bash
# Build feature cache for all baseline features, then train.
python -m classifier.runner

# Subsets (each flag takes a space-separated list):
python -m classifier.runner --classifier bilstm
python -m classifier.runner --feature sffcc_sdc_mod_k7 prosody32
python -m classifier.runner --task multiclass fluent_vs_I
python -m classifier.runner --split-scheme speaker_held_out   # default: random_90_10
python -m classifier.runner --no-cache-build                  # reuse existing cache/
python -m classifier.runner --dry-run                         # print the grid only
python -m classifier.runner --report-only                     # re-aggregate CSV/MD/plots

# Paper-reproduction probes (Garg 2021 Table V), run individually:
python paper_repro.py
python paper_repro_strict.py
python paper_repro_all_classes.py
python paper_repro_C.py
```

Feature tags: `mfcc45`, `mfcc45_ctx3`, `mfcc_sdc_mod_k7`, `mfcc_sdc_mod_k12`,
`mfcc_sdc_conv_k7`, `sffcc_sdc_mod_k7`, `prosody32`, `prosody96`
(registry in [`classifier/features.py`](classifier/features.py)).
Tasks: `fluent_vs_disfluent`, `multiclass`, `fluent_vs_{I,PR,PhR,WR,PWR,P}`.

The runner streams per-run JSON to `results/<tag>.json`, saves models and
per-frame predictions, and rebuilds `results/summary.md` +
`results/all_runs.csv` + confusion/ROC plots at the end. Failures are
logged (`logs/runner_*.log`) and never stop the sweep.

To reproduce this snapshot's split exactly, keep `splits/` in place and
run with `--split-scheme speaker_held_out --no-cache-build` after the
cache is built.

## Outputs

```
cache/features/<tag>/<stem>.npz   frame-level (D, T) features
cache/labels/<stem>.npz           frame-level class ids + times
splits/<scheme>.json              persisted train/val/test file lists
results/<tag>.json                per-run metric bundle
results/all_runs.csv              one row per (feature, classifier, task, class)
results/summary.md                per-task tables
models/<tag>.{joblib,pt}          trained model
predictions/<tag>.npz             y_true / y_pred / y_score / stem index
confusion/<tag>.png               multi-class confusion matrix
roc/<tag>.png                     binary ROC + PR curves
logs/                             run logs
```

## TODO

1. **Finish the grid**: cache `prosody32` (126 files left), `prosody96`,
   `mfcc_sdc_conv_k7`, `sffcc_sdc_mod_k7`, `mfcc_sdc_mod_k12`; run BiLSTM;
   extend 32 → 144 runs.
2. **Decide on encoders**: Wav2Vec2 / Whisper / Conformer extractors exist
   under `Audio-Feature-Extractors/` but are `SKIP` in the spec and not in
   the feature registry.
3. **Modelling problem**: minority-class recall ≈ 0 on speaker-held-out —
   revisit features / sampling / thresholds / class weighting / sequence
   model before results are meaningful.
4. **Spec eval pieces not yet wired**: event-level P/R/F1, frame
   post-processing (median filter, min-duration), bootstrap 95% CIs,
   paired significance tests, PR curves, `results/per_file/` dumps.
5. **`P` class** is nearly absent from the test split (~105 frames) —
   check labelling and split stratification; `fluent_vs_P` AUC is noise.
6. **paper_repro**: `paper_repro_strict.py` configs B and C never finished
   (log truncated, no `paper_repro_strict.json`); rerun to completion.
7. **Spec `[FILL:]` slots** still open: DNN class weighting, BiLSTM
   sequence handling, P in the binary task, overlap priority order.
