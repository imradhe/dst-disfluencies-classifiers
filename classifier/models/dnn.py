"""
classifier.models.dnn
=====================
PyTorch 2-hidden-layer MLP for frame-level classification.

Matches Garg 2021 / Mehrotra 2021 / 2022 DNN configs:
    hidden = (100, 50), ReLU, dropout 0.0
    optimiser = Adam(lr = 1e-3)
    batch = 32
    loss = BCEWithLogits (binary) / CrossEntropy (multi)
    early stopping patience = 10 epochs on val loss
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], n_classes: int,
                 dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        # For binary tasks we output a single logit (BCEWithLogits);
        # for multi-class we output n_classes logits (CrossEntropy).
        out_dim = 1 if n_classes == 2 else n_classes
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class DNNConfig:
    hidden: List[int] = field(default_factory=lambda: [100, 50])
    dropout: float = 0.0
    lr: float = 1e-3
    optimiser: str = "adam"       # "adam" | "rmsprop" | "sgd"
    batch_size: int = 32
    max_epochs: int = 100
    early_stop_patience: int = 10
    val_frac_of_train: float = 0.10   # carved from train if no val provided
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 42


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def _make_optimiser(name: str, params, lr: float):
    n = name.lower()
    if n == "adam":
        return torch.optim.Adam(params, lr=lr)
    if n == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimiser {name!r}")


def _split_val(X: np.ndarray, y: np.ndarray, val_frac: float,
               seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified carve-out of a val partition from (X, y)."""
    from sklearn.model_selection import train_test_split
    if val_frac <= 0:
        return X, y, np.empty((0, X.shape[1]), dtype=X.dtype), np.empty(0, dtype=y.dtype)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=val_frac, random_state=seed, stratify=y,
    )
    return X_tr, y_tr, X_va, y_va


def train_dnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    cfg: DNNConfig = DNNConfig(),
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[MLP, dict]:
    torch.manual_seed(cfg.random_state)
    device = torch.device(cfg.device)

    if X_val is None or len(X_val) == 0:
        X_train, y_train, X_val, y_val = _split_val(
            X_train, y_train, cfg.val_frac_of_train, cfg.random_state
        )

    input_dim = X_train.shape[1]
    model = MLP(input_dim, cfg.hidden, n_classes, dropout=cfg.dropout).to(device)
    opt = _make_optimiser(cfg.optimiser, model.parameters(), cfg.lr)

    binary = (n_classes == 2)
    if binary:
        criterion = nn.BCEWithLogitsLoss()
        y_train_t = torch.from_numpy(y_train.astype(np.float32))
        y_val_t = torch.from_numpy(y_val.astype(np.float32)) if len(y_val) else None
    else:
        criterion = nn.CrossEntropyLoss()
        y_train_t = torch.from_numpy(y_train.astype(np.int64))
        y_val_t = torch.from_numpy(y_val.astype(np.int64)) if len(y_val) else None

    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    X_val_t   = torch.from_numpy(X_val.astype(np.float32)) if len(X_val) else None

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=cfg.batch_size, shuffle=True,
    )

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    patience = 0
    history = []

    for epoch in range(cfg.max_epochs):
        model.train()
        tr_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            if binary:
                loss = criterion(out.squeeze(-1), yb)
            else:
                loss = criterion(out, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
            n_seen  += xb.size(0)
        tr_loss /= max(1, n_seen)

        if X_val_t is not None and len(X_val_t) > 0:
            model.eval()
            with torch.no_grad():
                out = model(X_val_t.to(device))
                if binary:
                    val_loss = criterion(out.squeeze(-1), y_val_t.to(device)).item()
                else:
                    val_loss = criterion(out, y_val_t.to(device)).item()
        else:
            val_loss = tr_loss   # no val -> use train loss as stopping proxy

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
    info = {
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "history": history,
        "device": str(device),
    }
    return model, info


def predict_dnn(
    model: MLP,
    X: np.ndarray,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_pred, y_score) matching the RF trainer's contract."""
    device = next(model.parameters()).device
    model.eval()

    X_t = torch.from_numpy(X.astype(np.float32))
    logits_parts = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            out = model(X_t[i:i + batch_size].to(device))
            logits_parts.append(out.cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0)

    if model.n_classes == 2:
        prob = 1.0 / (1.0 + np.exp(-logits.squeeze(-1)))
        pred = (prob >= 0.5).astype(np.int64)
        return pred, prob.astype(np.float32)

    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = ex / ex.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1).astype(np.int64)
    return pred, proba.astype(np.float32)


def save_dnn(model: MLP, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "n_classes":  model.n_classes,
        "arch":       [m.out_features if isinstance(m, nn.Linear) else None
                       for m in model.net],
    }, path)
