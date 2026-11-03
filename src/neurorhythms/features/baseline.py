"""Non-topological features: the strong baselines and the PSD part of the descriptor.

Views produced here (all computed on z-scored windows, so absolute amplitude is
never available):

``bandpower``  relative band powers (delta, theta, alpha, beta, gamma), log
               band ratios, spectral edge frequencies (SEF50/90/95), peak
               frequency and normalised spectral entropy.
``hjorth``     Hjorth mobility and complexity (Hjorth 1970). Activity is the
               variance, identically 1 after z-scoring, so it is omitted.
``waveform``   skewness, excess kurtosis and normalised line length: classic
               non-topological spike detectors (Esteller et al. 2001). Not in
               the brief's list, added to make the baseline *strict*.
``psd``        log10 relative Welch PSD in 1-Hz bins, 1-45 Hz: the spectral
               half of the topological-spectral descriptor.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy import stats

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
         "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
F_LO, F_HI = 1.0, 45.0


def welch_psd(X: np.ndarray, fs: float, seg_sec: float = 1.0, method: str = "welch"):
    """PSD of every row of X.

    ``method="welch"`` (default, used by the main benchmark): Hann segments of
    ``seg_sec`` s, 50 % overlap. ``method="periodogram"``: the raw periodogram of
    the whole window (rectangular taper) summed into 1-Hz bins centred on integer
    frequencies; this is exactly the quantity an IAAFT surrogate preserves.
    """
    if method == "welch":
        nper = int(round(seg_sec * fs))
        return sps.welch(X, fs=fs, nperseg=nper, noverlap=nper // 2, axis=-1)
    if method != "periodogram":
        raise ValueError(method)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    f = np.fft.rfftfreq(X.shape[-1], 1.0 / fs)
    p = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    edges = np.arange(0.5, fs / 2, 1.0)
    idx = np.digitize(f, edges)                     # bin k covers [k - 0.5, k + 0.5)
    k = len(edges) + 1
    pb = np.stack([np.bincount(idx, weights=row, minlength=k) for row in p])
    return np.arange(k, dtype=float), pb


def _band(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return p[:, m].sum(axis=1)


def bandpower_features(X: np.ndarray, fs: float, method: str = "welch"):
    f, p = welch_psd(X, fs, method=method)
    m = (f >= F_LO) & (f <= F_HI)
    fm, pm = f[m], p[:, m]
    total = pm.sum(axis=1) + 1e-20
    rel = {k: _band(f, p, lo, hi) / total for k, (lo, hi) in BANDS.items()}
    eps = 1e-12
    ratios = {
        "log_theta_alpha": np.log((rel["theta"] + eps) / (rel["alpha"] + eps)),
        "log_beta_alpha": np.log((rel["beta"] + eps) / (rel["alpha"] + eps)),
        "log_slow_fast": np.log((rel["delta"] + rel["theta"] + eps)
                                / (rel["alpha"] + rel["beta"] + eps)),
        "log_delta_alpha": np.log((rel["delta"] + eps) / (rel["alpha"] + eps)),
        "log_gamma_beta": np.log((rel["gamma"] + eps) / (rel["beta"] + eps)),
        "log_beta_theta": np.log((rel["beta"] + eps) / (rel["theta"] + eps)),
    }
    cum = np.cumsum(pm, axis=1) / total[:, None]
    sef = {f"sef{int(q * 100)}": fm[np.argmax(cum >= q, axis=1)] for q in (0.5, 0.9, 0.95)}
    pn = pm / total[:, None]
    sent = -(pn * np.log(pn + 1e-20)).sum(axis=1) / np.log(pn.shape[1])
    feats = {f"rel_{k}": v for k, v in rel.items()}
    feats.update(ratios)
    feats.update(sef)
    feats["peak_freq"] = fm[np.argmax(pm, axis=1)]
    feats["spectral_entropy"] = sent
    names = list(feats)
    return np.column_stack([feats[k] for k in names]), names


def hjorth_features(X: np.ndarray):
    d1 = np.diff(X, axis=1)
    d2 = np.diff(d1, axis=1)
    v0 = X.var(axis=1) + 1e-20
    v1 = d1.var(axis=1) + 1e-20
    v2 = d2.var(axis=1)
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / mobility
    return np.column_stack([mobility, complexity]), ["hjorth_mobility", "hjorth_complexity"]


def waveform_features(X: np.ndarray):
    sk = stats.skew(X, axis=1)
    ku = stats.kurtosis(X, axis=1)
    ll = np.abs(np.diff(X, axis=1)).mean(axis=1) / (X.std(axis=1) + 1e-12)
    return np.column_stack([sk, ku, ll]), ["skewness", "kurtosis", "line_length"]


def psd_features(X: np.ndarray, fs: float, method: str = "welch"):
    f, p = welch_psd(X, fs, method=method)
    m = (f >= F_LO) & (f <= F_HI)
    pm = p[:, m]
    rel = pm / (pm.sum(axis=1, keepdims=True) + 1e-20)
    return np.log10(rel + 1e-12), [f"psd_{fr:g}Hz" for fr in f[m]]
