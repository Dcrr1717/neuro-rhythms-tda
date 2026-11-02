"""Descriptive check that each class shows its intended signature.

Computed on the *noise-free* model output of every run (before measurement
noise and windowing), so it validates the parameter regimes themselves and
involves no classifier.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.stats import skew

BANDS = {"slow (1-7.5 Hz)": (1.0, 7.5), "alpha (7.5-13 Hz)": (7.5, 13.0),
         "beta (13-30 Hz)": (13.0, 30.0)}

# Criteria (fixed, descriptive): which signature each class is meant to show.
CRITERIA = {
    "healthy": "alpha is the dominant band (largest relative power of slow/alpha/beta)",
    "epileptic": "slow band dominant AND skewness > 0.5 (tall positive spikes)",
    "parkinsonian": "beta relative power >= 0.10 (healthy median is ~0.03)",
}


def run_signatures(eeg: np.ndarray, fs: float) -> dict:
    x = eeg - eeg.mean(axis=1, keepdims=True)
    x /= x.std(axis=1, keepdims=True) + 1e-12
    f, p = sps.welch(x, fs=fs, nperseg=int(2 * fs), axis=-1)
    m = (f >= 1.0) & (f <= 45.0)
    tot = p[:, m].sum(axis=1)
    out = {k: p[:, (f >= lo) & (f < hi)].sum(axis=1) / tot for k, (lo, hi) in BANDS.items()}
    out["peak frequency (Hz)"] = f[m][np.argmax(p[:, m], axis=1)]
    out["skewness"] = skew(x, axis=1)
    return out


def signature_table(eeg: np.ndarray, label: np.ndarray, fs: float, classes) -> dict:
    sig = run_signatures(eeg, fs)
    bands = list(BANDS)
    rel = np.column_stack([sig[b] for b in bands])
    dom = np.argmax(rel, axis=1)
    holds = {
        "healthy": dom == 1,
        "epileptic": (dom == 0) & (sig["skewness"] > 0.5),
        "parkinsonian": sig["beta (13-30 Hz)"] >= 0.10,
    }
    table = {}
    for ci, cls in enumerate(classes):
        sel = label == ci
        row = {k: [float(np.median(v[sel])), float(np.quantile(v[sel], 0.1)),
                   float(np.quantile(v[sel], 0.9))] for k, v in sig.items()}
        row["fraction_signature_holds"] = float(np.mean(holds[cls][sel]))
        row["criterion"] = CRITERIA[cls]
        row["n_runs"] = int(sel.sum())
        table[cls] = row
    return table
