"""Metrics, group-bootstrap confidence intervals and the Go/No-Go rule.

The statistics mirror the GEM benchmark (Dcrr1717/gem-geophysical-descriptor,
``src/eval/hard_benchmark.py``):

* accuracy is computed on out-of-fold predictions, one value per CV repeat,
  and averaged over repeats;
* 95 % CIs resample *simulation runs* (the CV groups) with replacement
  (multinomial weights), so correlated windows of one run move together;
* comparisons between two feature sets are *paired*: both use the same folds
  and the same bootstrap resamples;
* GO iff margin = acc(candidate) - acc(reference) >= ``go_margin`` (+0.05).
  We additionally report whether the 95 % CI of the margin excludes 0.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def _group_index(groups):
    ug, ginv = np.unique(np.asarray(groups), return_inverse=True)
    return len(ug), ginv


def accuracy_per_repeat(pred, y) -> np.ndarray:
    pred = np.atleast_2d(pred)
    return (pred == np.asarray(y)[None, :]).mean(axis=1)


def macro_f1_per_repeat(pred, y) -> np.ndarray:
    pred = np.atleast_2d(pred)
    return np.array([f1_score(y, p, average="macro") for p in pred])


def run_level_accuracy(proba, y, groups) -> np.ndarray:
    """Per-repeat accuracy of run-level decisions (mean window probability per run)."""
    proba = np.asarray(proba)
    if proba.ndim == 2:
        proba = proba[None]
    G, ginv = _group_index(groups)
    y = np.asarray(y)
    run_y = np.zeros(G, dtype=int)
    run_y[ginv] = y
    accs = []
    for P in proba:
        S = np.zeros((G, P.shape[1]))
        np.add.at(S, ginv, P)
        accs.append(float(np.mean(S.argmax(axis=1) == run_y)))
    return np.array(accs)


def _correct_counts(pred, y, ginv, G):
    pred = np.atleast_2d(pred)
    return np.stack([np.bincount(ginv, weights=(p == y).astype(float), minlength=G)
                     for p in pred])                                   # (repeats, G)


def _weights(G, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)  # (B, G)


def bootstrap_accuracy(pred, y, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean accuracy over repeats and its 95 % group-bootstrap CI."""
    y = np.asarray(y)
    G, ginv = _group_index(groups)
    n_g = np.bincount(ginv, minlength=G).astype(float)
    c = _correct_counts(pred, y, ginv, G)
    W = _weights(G, n_boot, seed)
    acc = ((W @ c.T) / (W @ n_g)[:, None]).mean(axis=1)
    lo, hi = np.percentile(acc, [2.5, 97.5])
    point = float((c.sum(axis=1) / n_g.sum()).mean())
    return {"accuracy": point, "ci95": [float(lo), float(hi)]}


def paired_bootstrap_margin(pred_a, pred_b, y, groups, n_boot: int = 2000,
                            seed: int = 0) -> dict:
    """acc(A) - acc(B) with a paired 95 % group-bootstrap CI (as in GEM)."""
    y = np.asarray(y)
    G, ginv = _group_index(groups)
    n_g = np.bincount(ginv, minlength=G).astype(float)
    ca = _correct_counts(pred_a, y, ginv, G)
    cb = _correct_counts(pred_b, y, ginv, G)
    W = _weights(G, n_boot, seed)
    tot = W @ n_g
    diff = ((W @ ca.T) / tot[:, None]).mean(axis=1) - ((W @ cb.T) / tot[:, None]).mean(axis=1)
    point = float((ca.sum(1) / n_g.sum()).mean() - (cb.sum(1) / n_g.sum()).mean())
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"margin": point, "ci95": [float(lo), float(hi)],
            "p_le_0": float(np.mean(diff <= 0))}


def verdict(margin: float, ci95, go_margin: float) -> dict:
    return {"verdict": "GO" if margin >= go_margin else "NO-GO",
            "ci_excludes_0": bool(ci95[0] > 0 or ci95[1] < 0),
            "ci_above_0": bool(ci95[0] > 0)}


def confusion(pred, y, n_classes: int = 3) -> np.ndarray:
    """Row-normalised confusion matrix pooled over repeats."""
    pred = np.atleast_2d(pred)
    M = np.zeros((n_classes, n_classes))
    for p in pred:
        np.add.at(M, (np.asarray(y), p), 1)
    return M / M.sum(axis=1, keepdims=True)
