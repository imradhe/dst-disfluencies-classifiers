"""
classifier.models.bilstm
========================
PyTorch BiLSTM tagger operating on per-file sequences of frame features.

Matches Mehrotra 2021's BiLSTM shape:
    2 recurrent layers x 90 hidden units, dropout 0.2 on layer 1.
Head is a per-frame linear projection to `n_classes` logits.

Sequence handling: full-file (spec section 7c, IED clips are 8-12s).
Class imbalance: class-weighted loss (spec fallback for sequence models).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMTagger(nn.Module):
    def __init__(self, input_dim: int, hidden: int, n_layers: int,
                 n_classes: int, dropout_layer1: float = 0.2):
        super().__init__()
        # PyTorch's LSTM dropout applies between layers. For layers > 1
        # we route dropout at that shared knob; single-layer stacks
        # ignore it.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_layer1 if n_layers > 1 else 0.0,
        )
        out_dim = 1 if n_classes == 2 else n_classes
        self.head = nn.Linear(2 * hidden, out_dim)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, T_max, D)
        lengths : (B,)  actual lengths
        returns : (B, T_max, out_dim)
        """
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=x.size(1))
        return self.head(out)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class BiLSTMConfig:
    hidden: int = 90
    n_layers: int = 2
    dropout_layer1: float = 0.2
    lr: float = 1e-3
    optimiser: str = "adam"
    batch_size: int = 8               # sequences per batch
    max_epochs: int = 50
    early_stop_patience: int = 10
    val_frac_of_train: float = 0.10
    # Sequence handling (spec section 7c):
    #   chunk_frames = None -> full-file sequences (only viable when
    #                          files are short, e.g. 8-12s clips).
    #   chunk_frames = N    -> chunked mode with stride = chunk_frames
    #                          (non-overlapping), N frames per window.
    # Default 3000 frames = 30 s at 10 ms hop.
    chunk_frames: Optional[int] = 3000
    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    random_state: int = 42


# ---------------------------------------------------------------------
# Per-file sequence assembly
# ---------------------------------------------------------------------

@dataclass
class Sequence:
    """One file's frame sequence + task-label vector."""
    stem: str
    X: np.ndarray             # (T, D)
    y: np.ndarray             # (T,)  task label ids, DROP already removed
    keep_mask: np.ndarray     # (T,)  bool -- which original frames survived

    def __len__(self) -> int:
        return len(self.y)


def build_sequences(
    stems: List[str],
    feature_tag: str,
    task_id: str,
) -> List[Sequence]:
    """
    Assemble one Sequence per file. DROP frames are removed (kept in
    keep_mask so predictions can be aligned back to the file's timeline).
    """
    from classifier.cache import load_features, load_labels
    from classifier.data import DROP_ID, task_labels

    seqs: List[Sequence] = []
    for stem in stems:
        feats = load_features(feature_tag, stem)   # (D, T)
        lbls  = load_labels(stem)
        task_y = task_labels(lbls.class_ids, task_id)
        keep = task_y != DROP_ID
        if not keep.any():
            continue
        seqs.append(Sequence(
            stem=stem,
            X=feats[:, keep].T.astype(np.float32),
            y=task_y[keep].astype(np.int64),
            keep_mask=keep,
        ))
    return seqs


def chunk_sequences(seqs: List[Sequence],
                    chunk_frames: int) -> List[Sequence]:
    """
    Split each per-file Sequence into non-overlapping windows of at most
    `chunk_frames` frames. The parent stem is preserved so predictions
    can be concatenated back per file in order.
    """
    if chunk_frames is None or chunk_frames <= 0:
        return seqs
    out: List[Sequence] = []
    for s in seqs:
        T = len(s)
        if T <= chunk_frames:
            out.append(s)
            continue
        for start in range(0, T, chunk_frames):
            end = min(start + chunk_frames, T)
            out.append(Sequence(
                stem=s.stem,
                X=s.X[start:end],
                y=s.y[start:end],
                keep_mask=s.keep_mask.copy(),  # not sliced -- unused post-chunking
            ))
    return out


def _pad_batch(seqs: List[Sequence]
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded (X, y, lengths) tensors for a mini-batch."""
    D = seqs[0].X.shape[1]
    T_max = max(len(s) for s in seqs)
    B = len(seqs)

    X = torch.zeros(B, T_max, D, dtype=torch.float32)
    y = torch.full((B, T_max), -100, dtype=torch.long)   # -100 = ignore
    lengths = torch.zeros(B, dtype=torch.long)

    for i, s in enumerate(seqs):
        T = len(s)
        X[i, :T] = torch.from_numpy(s.X)
        y[i, :T] = torch.from_numpy(s.y)
        lengths[i] = T
    return X, y, lengths


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def _iter_batches(seqs: List[Sequence], batch_size: int, shuffle: bool,
                  rng: np.random.Generator):
    order = np.arange(len(seqs))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        yield [seqs[j] for j in order[i:i + batch_size]]


def train_bilstm(
    train_seqs: List[Sequence],
    n_classes: int,
    cfg: BiLSTMConfig = BiLSTMConfig(),
    val_seqs: Optional[List[Sequence]] = None,
    class_weight_tensor: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> Tuple[BiLSTMTagger, dict]:
    torch.manual_seed(cfg.random_state)
    rng = np.random.default_rng(cfg.random_state)
    device = torch.device(cfg.device)

    if not val_seqs:
        # Carve val out of train by file (never split within a file).
        rng2 = np.random.default_rng(cfg.random_state)
        idx = np.arange(len(train_seqs))
        rng2.shuffle(idx)
        n_val = max(1, int(round(cfg.val_frac_of_train * len(idx))))
        val_ids = set(idx[:n_val].tolist())
        val_seqs   = [s for i, s in enumerate(train_seqs) if i in val_ids]
        train_seqs = [s for i, s in enumerate(train_seqs) if i not in val_ids]

    # Chunk sequences AFTER the val split so val chunks stay from
    # file-disjoint speakers.
    train_seqs = chunk_sequences(train_seqs, cfg.chunk_frames)
    val_seqs   = chunk_sequences(val_seqs,   cfg.chunk_frames)

    D = train_seqs[0].X.shape[1]
    model = BiLSTMTagger(
        input_dim=D, hidden=cfg.hidden, n_layers=cfg.n_layers,
        n_classes=n_classes, dropout_layer1=cfg.dropout_layer1,
    ).to(device)

    from classifier.models.dnn import _make_optimiser
    opt = _make_optimiser(cfg.optimiser, model.parameters(), cfg.lr)

    binary = (n_classes == 2)
    if binary:
        pos_weight = None
        if class_weight_tensor is not None and len(class_weight_tensor) == 2:
            pos_weight = torch.tensor(
                [float(class_weight_tensor[1] / max(1e-9, class_weight_tensor[0]))]
            ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    else:
        w = class_weight_tensor.to(device) if class_weight_tensor is not None else None
        criterion = nn.CrossEntropyLoss(weight=w, ignore_index=-100, reduction="none")

    def _forward_loss(seqs: List[Sequence]) -> torch.Tensor:
        X, y, lengths = _pad_batch(seqs)
        X, y, lengths = X.to(device), y.to(device), lengths.to(device)
        logits = model(X, lengths)
        if binary:
            valid = (y != -100).float()
            yf = y.clamp_min(0).float()  # replace -100 with 0 safely
            loss = criterion(logits.squeeze(-1), yf) * valid
            return loss.sum() / valid.sum().clamp_min(1.0)
        else:
            B, T, C = logits.shape
            loss = criterion(logits.reshape(B * T, C), y.reshape(B * T))
            valid = (y.reshape(-1) != -100).float()
            return loss.sum() / valid.sum().clamp_min(1.0)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience = 0
    history = []

    for epoch in range(cfg.max_epochs):
        model.train()
        tr_losses = []
        for batch in _iter_batches(train_seqs, cfg.batch_size, True, rng):
            opt.zero_grad()
            loss = _forward_loss(batch)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")

        # Val loss (per-batch mean)
        model.eval()
        va_losses = []
        with torch.no_grad():
            for batch in _iter_batches(val_seqs, cfg.batch_size, False, rng):
                va_losses.append(_forward_loss(batch).item())
        val_loss = float(np.mean(va_losses)) if va_losses else tr_loss

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})
        if verbose:
            print(f"  epoch {epoch:>3}  train={tr_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    model.load_state_dict(best_state)
    return model, {
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "history": history,
        "device": str(device),
    }


def predict_bilstm(
    model: BiLSTMTagger,
    seqs: List[Sequence],
    batch_size: int = 8,
    chunk_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate per-frame predictions across all sequences in the same
    order the caller passed them (grouped by stem).

    If `chunk_frames` is given, each Sequence is chunked before
    inference and predictions are concatenated back per stem.
    """
    device = next(model.parameters()).device
    model.eval()

    if chunk_frames:
        chunked = chunk_sequences(seqs, chunk_frames)
    else:
        chunked = seqs

    preds_all, scores_all = [], []
    with torch.no_grad():
        for i in range(0, len(chunked), batch_size):
            batch = chunked[i:i + batch_size]
            X, _y, lengths = _pad_batch(batch)
            X, lengths = X.to(device), lengths.to(device)
            logits = model(X, lengths).cpu().numpy()
            for b, s in enumerate(batch):
                T = len(s)
                lg = logits[b, :T]
                if model.n_classes == 2:
                    prob = 1.0 / (1.0 + np.exp(-lg.squeeze(-1)))
                    preds_all.append((prob >= 0.5).astype(np.int64))
                    scores_all.append(prob.astype(np.float32))
                else:
                    ex = np.exp(lg - lg.max(axis=1, keepdims=True))
                    proba = ex / ex.sum(axis=1, keepdims=True)
                    preds_all.append(proba.argmax(axis=1).astype(np.int64))
                    scores_all.append(proba.astype(np.float32))

    y_pred = np.concatenate(preds_all)
    if model.n_classes == 2:
        y_score = np.concatenate(scores_all)
    else:
        y_score = np.concatenate(scores_all, axis=0)
    return y_pred, y_score


def save_bilstm(model: BiLSTMTagger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "n_classes":  model.n_classes,
    }, path)
