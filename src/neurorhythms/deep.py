"""Small 1-D convolutional network trained on the raw (z-scored) windows.

Architecture (about 50 k parameters):

    [Conv1d(k=7) -> BatchNorm -> ReLU -> MaxPool(2)] x 3   (16, 32, 64 channels)
    Conv1d(64, 64, k=7) -> BatchNorm -> ReLU -> global average pooling
    Dropout(0.3) -> Linear(64, 3)

Multi-channel windows (n, n_channels, n_samples) are handled by giving the first
convolution ``n_channels`` input channels (it then learns spatial filters as
well); single-channel windows (n, n_samples) give exactly the original model.

Training: AdamW, cross-entropy, a *fixed* number of epochs (no early stopping
on the test fold, so nothing is selected on held-out data). It runs on the
same grouped folds as the classical models.
"""
from __future__ import annotations

import time

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("The deep model requires PyTorch (pip install torch)") from exc
    return torch


def build_cnn(n_classes: int = 3, width: int = 16, kernel: int = 7, dropout: float = 0.3,
              in_channels: int = 1):
    torch = _torch()
    nn = torch.nn
    c1, c2, c3 = width, 2 * width, 4 * width
    pad = kernel // 2

    def block(cin, cout, pool=True):
        layers = [nn.Conv1d(cin, cout, kernel, padding=pad), nn.BatchNorm1d(cout), nn.ReLU()]
        if pool:
            layers.append(nn.MaxPool1d(2))
        return layers

    return nn.Sequential(*block(in_channels, c1), *block(c1, c2), *block(c2, c3),
                         *block(c3, c3, pool=False),
                         nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
                         nn.Linear(c3, n_classes))


def _as_channels(X) -> np.ndarray:
    """(n, n_samples) -> (n, 1, n_samples); (n, c, n_samples) unchanged."""
    X = np.asarray(X, dtype=np.float32)
    return X[:, None, :] if X.ndim == 2 else X


def train_predict(Xtr, ytr, Xte, epochs: int = 30, batch_size: int = 64, lr: float = 1e-3,
                  weight_decay: float = 1e-4, width: int = 16, seed: int = 0,
                  n_classes: int = 3, threads: int | None = None) -> np.ndarray:
    """Train a fresh CNN on (Xtr, ytr) and return class probabilities for Xte.

    X arrays are (n, n_samples) or (n, n_channels, n_samples)."""
    torch = _torch()
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xt = torch.as_tensor(_as_channels(Xtr))
    model = build_cnn(n_classes, width, in_channels=int(xt.shape[1]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = torch.nn.CrossEntropyLoss()
    yt = torch.as_tensor(np.asarray(ytr, dtype=np.int64))
    n = len(yt)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(n)
        for s in range(0, n, batch_size):
            idx = torch.as_tensor(order[s:s + batch_size])
            if len(idx) < 2:      # BatchNorm needs > 1 sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    xe = torch.as_tensor(_as_channels(Xte))
    out = []
    with torch.no_grad():
        for s in range(0, len(xe), 512):
            out.append(torch.softmax(model(xe[s:s + 512]), dim=1).numpy())
    return np.concatenate(out)


def cross_validate_cnn(X, y, folds, seed: int = 0, n_classes: int = 3, **train_kw) -> dict:
    """Out-of-fold CNN predictions on the given (grouped) folds."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    R, N = len(folds), len(y)
    pred = np.full((R, N), -1, dtype=int)
    proba = np.zeros((R, N, n_classes))
    fold_acc = []
    t0 = time.time()
    for r, rep in enumerate(folds):
        for k, (tr, te) in enumerate(rep):
            pr = train_predict(X[tr], y[tr], X[te], seed=seed + 1000 * r + k,
                               n_classes=n_classes, **train_kw)
            proba[r, te] = pr
            pred[r, te] = np.argmax(pr, axis=1)
            fold_acc.append(float(np.mean(pred[r, te] == y[te])))
    return {"pred": pred, "proba": proba, "fold_acc": fold_acc, "seconds": time.time() - t0}


def n_parameters(width: int = 16, in_channels: int = 1) -> int:
    return int(sum(p.numel() for p in build_cnn(width=width, in_channels=in_channels).parameters()))
