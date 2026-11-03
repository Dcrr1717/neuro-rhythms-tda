"""Non-topological features for the IAAFT surrogate benchmark.

``onepoint``  quantiles, skewness and kurtosis of the window's values: a
              function of the amplitude histogram only, which IAAFT surrogates
              preserve. Blind by construction; a validity check.
``nonlinear`` the classical, non-topological statistics used to detect
              nonlinear temporal structure in EEG and in surrogate-data tests:

  * normalised time-reversal asymmetry at lag tau,
    ``E[(x_{t+tau} - x_t)^3] / E[(x_{t+tau} - x_t)^2]^{3/2}``
    (Theiler et al. 1992; Diks et al. 1995; Schreiber & Schmitz 2000);
  * increment flatness (kurtosis of increments) at lag tau, sensitive to
    intermittent sharp transients;
  * third-order autocorrelation ``E[x_t x_{t+tau} x_{t+2 tau}]`` (Theiler et
    al. 1992), zero for any time-reversible Gaussian process;
  * rise fraction ``P(x_{t+1} > x_t)``, 0.5 for a time-reversible signal;
  * peak/trough sharpness asymmetry and rise/decay steepness asymmetry, the
    cycle-by-cycle waveform-shape measures of Cole et al. (2017) for
    non-sinusoidal beta in Parkinson's disease;
  * normalised line length (Esteller et al. 2001).

Lags (samples at 250 Hz): 1, 2, 3, 5, 8, 13, 21 (4-84 ms), fixed a priori.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

LAGS = (1, 2, 3, 5, 8, 13, 21)
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def onepoint_features(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    q = np.quantile(X, QUANTILES, axis=1).T
    F = np.column_stack([q, stats.skew(X, axis=1), stats.kurtosis(X, axis=1)])
    names = [f"q{int(round(100 * p)):02d}" for p in QUANTILES] + ["skewness", "kurtosis"]
    return F, names


def _sharpness(x: np.ndarray, k: int = 2):
    """Mean sharpness of local maxima and minima: x[i] - mean(x[i-k], x[i+k])."""
    i = np.arange(k, len(x) - k)
    peaks = i[(x[i] > x[i - 1]) & (x[i] >= x[i + 1])]
    troughs = i[(x[i] < x[i - 1]) & (x[i] <= x[i + 1])]
    sp = (x[peaks] - 0.5 * (x[peaks - k] + x[peaks + k])) if len(peaks) else np.array([0.0])
    st = (0.5 * (x[troughs - k] + x[troughs + k]) - x[troughs]) if len(troughs) else np.array([0.0])
    return float(np.mean(sp)), float(np.mean(st))


def nonlinear_features(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
    cols, names = [], []
    for tau in LAGS:
        d = X[:, tau:] - X[:, :-tau]
        m2 = (d ** 2).mean(axis=1) + 1e-12
        cols.append((d ** 3).mean(axis=1) / m2 ** 1.5)
        names.append(f"trev_{tau}")
        cols.append((d ** 4).mean(axis=1) / m2 ** 2)
        names.append(f"incr_flat_{tau}")
        cols.append((X[:, : -2 * tau] * X[:, tau:-tau] * X[:, 2 * tau:]).mean(axis=1))
        names.append(f"c3_{tau}")
    d1 = np.diff(X, axis=1)
    cols.append((d1 > 0).mean(axis=1))
    names.append("rise_fraction")
    # rise/decay steepness asymmetry: mean positive slope vs mean negative slope
    pos = np.where(d1 > 0, d1, np.nan)
    neg = np.where(d1 < 0, -d1, np.nan)
    cols.append(np.log((np.nanmean(pos, axis=1) + 1e-12) / (np.nanmean(neg, axis=1) + 1e-12)))
    names.append("log_rise_decay_steepness")
    sharp = np.array([_sharpness(x) for x in X])
    cols.append(np.log((sharp[:, 0] + 1e-6) / (sharp[:, 1] + 1e-6)))
    names.append("log_peak_trough_sharpness")
    cols.append(np.abs(d1).mean(axis=1))
    names.append("line_length")
    F = np.column_stack(cols)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0), names
