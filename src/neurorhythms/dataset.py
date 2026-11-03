"""Synthetic EEG-like dataset: simulate runs, add measurement noise, cut windows.

One *run* is one independent simulation with its own parameter draw (a
virtual subject / recording). Each run is cut into non-overlapping windows;
all windows of a run share the run id, which is the grouping variable of the
cross-validation (windows of one run never appear in both train and test).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neural_mass import RunParams, highpass, pink_noise, simulate
from .regimes import CLASSES, DEFAULT_REGIMES, concat_params, sample_class_params


@dataclass
class RawRuns:
    eeg: np.ndarray          # (n_runs, n_samples) noiseless model output (mV)
    label: np.ndarray        # (n_runs,) class index
    fs: float
    meta: dict               # per-run parameters


@dataclass
class Dataset:
    X: np.ndarray            # (n_windows, n_samples) z-scored windows
    y: np.ndarray            # (n_windows,) class index
    groups: np.ndarray       # (n_windows,) run id
    fs: float
    classes: tuple
    run_meta: dict           # per-run arrays (class, parameters, noise levels)

    @property
    def n_runs(self) -> int:
        return int(len(np.unique(self.groups)))


def zscore_rows(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=-1, keepdims=True)
    return x / (x.std(axis=-1, keepdims=True) + 1e-12)


def simulate_runs(runs_per_class: int, duration: float, seed: int, fs: float = 250.0,
                  dt: float = 5e-4, transient: float = 2.0, regimes: dict | None = None,
                  batch_runs: int = 64) -> RawRuns:
    """Draw parameters for ``runs_per_class`` runs per class and integrate them."""
    rng = np.random.default_rng(seed)
    regimes = regimes or DEFAULT_REGIMES
    parts, meta_parts, labels = [], [], []
    for ci, cls in enumerate(CLASSES):
        p, m = sample_class_params(cls, runs_per_class, rng, regimes)
        parts.append(p)
        meta_parts.append(m)
        labels.append(np.full(runs_per_class, ci))
    params = concat_params(parts)
    label = np.concatenate(labels)
    n_runs = params.n_runs

    sigs = []
    for s in range(0, n_runs, batch_runs):
        sl = slice(s, min(n_runs, s + batch_runs))
        sub = RunParams(**{k: np.asarray(v)[sl] for k, v in params.__dict__.items()})
        sigs.append(simulate(sub, duration, rng, dt=dt, fs_out=fs, transient=transient))
    meta = params.as_dict()
    for key in meta_parts[0]:
        meta[key] = np.concatenate([m[key] for m in meta_parts])
    return RawRuns(eeg=np.concatenate(sigs, axis=0), label=label, fs=fs, meta=meta)


def measure_continuous(raw: RawRuns, seed: int, pink_level=(0.2, 0.8),
                       white_level=(0.05, 0.2), noise_scale: float = 1.0):
    """Measurement model on the whole run (before windowing).

    x = zscore(EEG) + c * (s_pink * pink + s_white * white), with s_pink, s_white
    drawn per run from ``pink_level`` / ``white_level`` (background 1/f activity
    and sensor noise) and c = ``noise_scale``, followed by a 0.5 Hz high-pass.

    Returns (x, s_pink, s_white) with x of shape (n_runs, n_samples).
    """
    rng = np.random.default_rng(seed)
    n_runs, n_samp = raw.eeg.shape
    s_pink = rng.uniform(*pink_level, size=n_runs)
    s_white = rng.uniform(*white_level, size=n_runs)
    noise = (s_pink[:, None] * pink_noise(n_runs, n_samp, rng)
             + s_white[:, None] * rng.standard_normal((n_runs, n_samp)))
    x = highpass(zscore_rows(raw.eeg) + noise_scale * noise, raw.fs, 0.5)
    return x, s_pink, s_white


def window_runs(x: np.ndarray, fs: float, window: float):
    """Cut every run into non-overlapping windows and z-score each window.

    Returns (X, run_index) with X of shape (n_runs * n_win, n_window_samples).
    """
    n_runs, n_samp = x.shape
    w = int(round(window * fs))
    n_win = n_samp // w
    if n_win < 1:
        raise ValueError("duration shorter than one window")
    X = zscore_rows(x[:, : n_win * w].reshape(n_runs * n_win, w))
    return X.astype(np.float32), np.repeat(np.arange(n_runs), n_win)


def measure(raw: RawRuns, window: float, seed: int, pink_level=(0.2, 0.8),
            white_level=(0.05, 0.2), noise_scale: float = 1.0) -> Dataset:
    """Measurement model (:func:`measure_continuous`) + windowing.

    Each window is z-scored, so absolute amplitude is *not* available to any
    classifier. The exploratory noise sweep raises ``noise_scale`` above 1.
    """
    x, s_pink, s_white = measure_continuous(raw, seed, pink_level, white_level, noise_scale)
    X, groups = window_runs(x, raw.fs, window)
    y = raw.label[groups]
    run_meta = dict(raw.meta)
    run_meta.update({"label": raw.label, "pink_level": s_pink, "white_level": s_white,
                     "noise_scale": np.full(len(raw.label), noise_scale)})
    return Dataset(X=X, y=y, groups=groups, fs=raw.fs, classes=CLASSES, run_meta=run_meta)


def make_dataset(runs_per_class: int, duration: float, window: float, seed: int,
                 fs: float = 250.0, dt: float = 5e-4, transient: float = 2.0,
                 pink_level=(0.2, 0.8), white_level=(0.05, 0.2),
                 regimes: dict | None = None) -> Dataset:
    """Convenience wrapper: :func:`simulate_runs` followed by :func:`measure`."""
    raw = simulate_runs(runs_per_class, duration, seed, fs=fs, dt=dt,
                        transient=transient, regimes=regimes)
    return measure(raw, window, seed + 1, pink_level, white_level)
