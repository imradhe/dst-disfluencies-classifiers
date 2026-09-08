# Disfluency Classifier — Specification

Fill in every `[FILL: ...]` slot. `[DEFAULT: ...]` slots hold the value
from the source papers or a sensible convention; edit only if you want
to override.

Once complete, I will use this file verbatim to build the training,
evaluation, and reporting code.

---

## 1. Dataset

- Name: `IED (Extended)`
- Location on disk: `IED`
- Layout: `wav file with corresponding txt file with same name`
- Sample rate: `[DEFAULT: 16000]`
- Channels: `mono`
- Total hours / speakers / files (approx): `60`
- Speaker identity available? (needed for speaker-held-out splits)
  - `no`

---

## 2. Annotations (SRT)

- Format: 
Start End Class
```
19.176185	19.565329	I
19.676127	20.246332	WR
29.713400	30.007675	I
```
- Are compound labels possible in a single caption (e.g. `PR,I`)?
 no
- Are there explicit `Fluent` intervals in the SRT?
  - no
---

## 3. Class handling

Classes in play: `I, PR, PhR, WR, PWR, P` plus `Fluent`.

- **Where does Pause (P) go?**
  - In `Fluent-vs-Disfluent` binary: P is `[FILL: fluent / disfluent]`
  - In multi-class: P is `[FILL: its own class]`

- **Disfluent set** (for binary Fluent-vs-Disfluent):
  - `{I, PR, PhR, WR, PWR, P}`

- **Classes included in multi-class task** (final label list):
  - `[Fluent, I, PR, PhR, WR, PWR, P]`

---

## 4. Tasks

Confirm or edit:

1. **Binary — Fluent vs Disfluent**
   - Include this task: `yes`

2. **Binary — Fluent vs Class (one-vs-rest, per class)**
   - Include this task: `yes`
   - Negative-class definition: `B`
     - `A` — negative = fluent frames only (exclude all other disfluencies)
     - `B` — negative = fluent + all other disfluencies (**paper convention**)
   - Which classes get their own binary run: `[DEFAULT: all classes above]`

3. **Multi-class**
   - Include this task: `yes`
   - Label set: (see section 3)

---

## 5. Frame definition (all tasks are frame-level)

- Window / hop: `[25 ms window, 10 ms hop]`
- Window function: `[Hamming]`
- Sample rate for frame timing: `[16000]`
- **Frame-label rule** (given an annotation `[start, end]` and a frame at time `t`):
  - `[centre-in-interval]`
    - `centre-in-interval` — frame gets the label iff `t_centre ∈ [start, end)`
    - `majority-overlap` — frame gets label C iff ≥ 50% of the frame lies inside C
    - `any-overlap` — frame gets label C if the frame touches C at all
  - Suggested default: `centre-in-interval` (simplest and most common)
- **Overlapping-annotation tie-breaker** (frame overlaps two labels at once):
  - `[drop-frame]`
    - `priority-order` — first match in this order: `[FILL: e.g. I > PR > PWR > WR > PhR > P > Fluent]`
    - `drop-frame` — remove ambiguous frames from training and evaluation
    - `duplicate-frame` — one row per label (multi-label training)

---

## 6. Feature × task grid

Which extractors feed which classifiers. Mark `Y` / `N` / `SKIP`.

The three encoders (Whisper, Wav2Vec2, Conformer) are your **proposed**
features. The other four are **baselines**.

| Feature (D per frame) | RF | DNN | BiLSTM |
|---|---|---|---|
| #1 MFCC — 45-dim (or 315 with ±3 stacking) | `Y` | `Y` | `Y` |
| #2 MFCC + SDC — 224 (K=7) / 364 (K=12), modified | `Y` | `Y` | `Y` |
| #2 MFCC + SDC — conventional (baseline)      | `Y` | `Y` | `Y` |
| #3 SFFCC + SDC — 208 (K=7) modified          | `Y` | `Y` | `Y` |
| #4 Prosody — 32 per syllable                 | `Y` | `Y` | `Y` |
| #4 Prosody ±1 stacked — 96 per syllable      | `Y` | `Y` | `Y` |
| #5 Wav2Vec2 — 768                            | `SKIP` | `SKIP` | `SKIP` |
| #6 Whisper — 1024                            | `SKIP` | `SKIP` | `SKIP` |
| #7 Conformer — variable                      | `SKIP` | `SKIP` | `SKIP` |

- **Feature-level output required by frame-level tasks**: some extractors
  currently segment-pool. If you mark them `Y` above, they will be
  patched to emit `(D, T)` frame-level before training.
  - Patch #2 MFCC+SDC to frame-level output: `[yes]`
  - Patch #3 SFFCC+SDC to frame-level output: `[yes]`

- **Prosody (#4) at frame level**: syllable-level features need to be
  reconciled with frame-level classification. Choose one:
  - `B`
    - `A` — drop #4 from frame-level runs
    - `B` — upsample: each frame inside a syllable region gets that
      syllable's 32-dim (or 96-dim) vector

- **Feature preprocessing before classifier**:
  - Per-file MVN (mean/variance normalisation): `[DEFAULT: yes for cepstral; skip for SSL embeddings]`
  - Global standardisation (fit on train, apply to val/test): `yes`
  - PCA / whitening / dimensionality reduction: `[DEFAULT: no]`

---

## 7. Classifiers

### 7a. Random Forest

- Trees: `[DEFAULT: 100]`
- Max depth: `[DEFAULT: 20]`
- Class weight: `[DEFAULT: balanced]`
- n_jobs: `[DEFAULT: -1]`
- Random seed: (see section 12)

### 7b. DNN variants

Specify a grid. Cartesian product is trained per (feature, task).

- Hidden-layer configurations: `[DEFAULT: [(100, 50)]]`
  - Add more here if wanted, e.g. `(256, 128, 64)`
- Optimiser: `[DEFAULT: [Adam]]`  (options: Adam, RMSprop, SGD)
- Learning rate(s): `[DEFAULT: [1e-3]]`
- Batch size: `[DEFAULT: 32]`
- Max epochs: `[DEFAULT: 100]`
- Early stopping: `[DEFAULT: patience 10 on val loss]`
- Dropout: `[DEFAULT: 0.0]`  (papers used none)
- Activation: `[DEFAULT: ReLU]`
- Loss:
  - Binary tasks: `[DEFAULT: BCE with logits]`
  - Multi-class: `[DEFAULT: cross-entropy]`
- Class weighting in loss: `[FILL: none / inverse-frequency / focal]`

### 7c. BiLSTM

- Layers × hidden: `[DEFAULT: 2 × 90]` (paper) — or `[FILL: alternative, e.g. 2 × 256]`
- Dropout on layer 1: `[DEFAULT: 0.2]`
- Sequence handling:
  - `[FILL: exactly one]`
    - `full-file` — one sequence per audio file, dense per-frame outputs
    - `chunked` — fixed-length windows, e.g. `[FILL: N frames, stride S]`
- Head: `[DEFAULT: linear per-frame → sigmoid (binary) / softmax (multi)]`
- Optimiser / lr / batch / epochs / early stop: `[DEFAULT: same DNN block above]`

---

## 8. Class imbalance strategy

Pick one primary strategy per task; may differ per task.

- Binary Fluent-vs-Disfluent: `[DEFAULT: SMOTE on train set only]`
- Binary Fluent-vs-Class:     `[DEFAULT: SMOTE on train set only]`
- Multi-class:                 `[SMOTE]`

Additional options:
- Apply strategy on: `[DEFAULT: train set only, never on val/test]`
- BiLSTM-specific note: SMOTE is not straightforward for sequence
  models. If BiLSTM is used, fall back to `[DEFAULT: class-weighted loss]`.

---

## 9. Data splits

- Split scheme: `random-90-10`
  - `random-80-20` — random file-level split
  - `random-90-10`
  - `speaker-held-out` — no speaker appears in both train and test
  - `k-fold-speaker` — K speaker-disjoint folds, K = `[FILL]`
  - `k-fold-random`  — K random folds, K = `[FILL]`
- Validation set: `[DEFAULT: 10% of the train partition, stratified by class prevalence]`
- Persist split IDs to disk: `[DEFAULT: yes -> splits/{scheme}.json]`
- Split unit: `[DEFAULT: file]` (i.e. all frames of a file stay in the same partition — never split at frame level)

---

## 10. Evaluation metrics

### 10a. Per-run reported metrics

Frame-level (primary):
- Accuracy: `[DEFAULT: yes]`
- Precision / Recall / F1: `[DEFAULT: yes]`
- Averaging for multi-class: `[DEFAULT: report macro, micro, and weighted]`
- Per-class P/R/F1 breakdown: `[DEFAULT: yes]`
- Confusion matrix (multi-class): `[DEFAULT: yes]`
- ROC-AUC / PR-AUC (binary tasks): `[DEFAULT: yes]`
- Majority-class baseline accuracy: `[DEFAULT: report]`

### 10b. Event-level metrics (derived from frame predictions)

- Include event-level report: `yes`
- Event definition: contiguous run of frames predicted as class `C`.
- Match rule to ground-truth events: `[DEFAULT: any-overlap, per-class]`
- Metrics: event-level Precision / Recall / F1 per class.

### 10c. Frame-prediction post-processing (applied before event derivation)

- Median filter over N frames: `[DEFAULT: 5 frames = 50 ms]` — or `[FILL: N]`
- Minimum event duration: `[DEFAULT: drop events shorter than 30 ms]`

### 10d. Statistical reporting

- Bootstrap 95% CI on F1: `[DEFAULT: yes, 1000 resamples of the test set]`
- Paired significance test between classifiers on same test set: `yes`

---

## 11. Reproducibility

- Random seed: `[DEFAULT: 42]`
- Persist trained models: `[DEFAULT: yes -> models/{feature}_{classifier}_{task}.pkl or .pt]`
- Persist predictions per test frame: `[DEFAULT: yes -> predictions/…]`
- Environment: `[DEFAULT: use existing venv; freeze into requirements.txt after first run]`

---

## 12. Output artifacts

- `results/` — one CSV row per `(feature, classifier, task, class)` with all metrics.
- `results/summary.md` — human-readable table per task.
- `results/per_file/…` — per-file predictions for error analysis.
- `confusion/…` — confusion matrices (PNG + CSV) for multi-class runs.
- `roc/…` — ROC and PR curves (PNG) for binary runs.
- `logs/…` — training logs per model.

---

## 13. Anything else you want

- `[FILL: free-form notes, extra ablations, custom features to add later]`
