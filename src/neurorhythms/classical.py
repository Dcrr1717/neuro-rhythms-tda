"""Grouped cross-validation of the classical classifiers.

Folds come from ``StratifiedGroupKFold`` with the simulation run as group, so
all windows of one run are always on the same side of the split. The folds
depend only on (y, groups, seed), hence every feature set and the CNN see
*identical* splits, which makes paired comparisons valid.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_folds(y, groups, n_splits: int, n_repeats: int, seed: int):
    """List (per repeat) of lists of (train_idx, test_idx)."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    folds = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        folds.append([(tr, te) for tr, te in cv.split(np.zeros(len(y)), y, groups)])
    return folds


def check_folds(folds, groups) -> None:
    """Raise if any group appears in both the train and the test part of a fold."""
    groups = np.asarray(groups)
    for rep in folds:
        seen = np.zeros(len(groups), dtype=int)
        for tr, te in rep:
            if np.intersect1d(groups[tr], groups[te]).size:
                raise AssertionError("group leakage between train and test")
            seen[te] += 1
        if not np.all(seen == 1):
            raise AssertionError("each sample must be tested exactly once per repeat")


def build_classifier(name: str, seed: int, rf_trees: int = 300, n_jobs: int = 1):
    if name == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=5000))
    if name == "rf":
        return RandomForestClassifier(n_estimators=rf_trees, random_state=seed,
                                      n_jobs=n_jobs)
    raise ValueError(name)


def cross_validate(F, y, folds, model: str, seed: int = 0, rf_trees: int = 300,
                   n_jobs: int = 1, n_classes: int = 3) -> dict:
    """Out-of-fold predictions of one classifier for every repeat."""
    F = np.asarray(F, dtype=float)
    y = np.asarray(y)
    R, N = len(folds), len(y)
    pred = np.full((R, N), -1, dtype=int)
    proba = np.zeros((R, N, n_classes))
    fold_acc = []
    t0 = time.time()
    jobs = [(r, k, tr, te) for r, rep in enumerate(folds) for k, (tr, te) in enumerate(rep)]

    def fit_one(r, k, tr, te, inner_jobs):
        clf = build_classifier(model, seed + 1000 * r + k, rf_trees, inner_jobs)
        clf.fit(F[tr], y[tr])
        return clf.predict_proba(F[te])

    if model == "logreg" and n_jobs > 1 and len(jobs) > 1:
        # LogReg is single-threaded: parallelise over folds instead
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs)(delayed(fit_one)(r, k, tr, te, 1) for r, k, tr, te in jobs)
    else:
        out = [fit_one(r, k, tr, te, n_jobs) for r, k, tr, te in jobs]
    for (r, k, tr, te), pr in zip(jobs, out):
        proba[r, te] = pr
        pred[r, te] = np.argmax(pr, axis=1)
        fold_acc.append(float(np.mean(pred[r, te] == y[te])))
    return {"pred": pred, "proba": proba, "fold_acc": fold_acc,
            "seconds": time.time() - t0}
