"""Grouped (feature-family) permutation importance on held-out folds.

For the ``union`` feature set, each fitted fold model is scored on its test
fold before and after jointly permuting all columns of one feature family
(the same row permutation for every column of the family, so within-family
correlations are preserved). Importance = mean drop in accuracy. Jointly
permuting a family avoids the classic underestimation of importance among
correlated features (e.g. psd bins vs band powers).
"""
from __future__ import annotations

import numpy as np

from .classical import build_classifier


def family_permutation_importance(F, y, families, folds_one_repeat, model: str,
                                  n_permutations: int = 5, seed: int = 0,
                                  rf_trees: int = 300, n_jobs: int = 1) -> dict:
    F = np.asarray(F, dtype=float)
    y = np.asarray(y)
    families = np.asarray(families)
    fams = list(dict.fromkeys(families.tolist()))
    rng = np.random.default_rng(seed)
    drops = {f: [] for f in fams}
    base_accs = []
    for k, (tr, te) in enumerate(folds_one_repeat):
        clf = build_classifier(model, seed + k, rf_trees, n_jobs)
        clf.fit(F[tr], y[tr])
        Xte = F[te]
        base = float(np.mean(clf.predict(Xte) == y[te]))
        base_accs.append(base)
        for fam in fams:
            cols = np.nonzero(families == fam)[0]
            for _ in range(n_permutations):
                Xp = Xte.copy()
                perm = rng.permutation(len(te))
                Xp[:, cols] = Xte[perm][:, cols]
                drops[fam].append(base - float(np.mean(clf.predict(Xp) == y[te])))
    return {"base_accuracy": float(np.mean(base_accs)),
            "families": fams,
            "mean_drop": {f: float(np.mean(v)) for f, v in drops.items()},
            "std_drop": {f: float(np.std(v)) for f, v in drops.items()},
            "n_features": {f: int(np.sum(families == f)) for f in fams}}
