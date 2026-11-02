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


def measure(raw: RawRuns, window: float, seed: int, pink_level=(0.2, 0.8),
            white_level=(0.05, 0.2), noise_scale: float = 1.0) -> Dataset:
    """Measurement model + windowing.

    x = zscore(EEG) + c * (s_pink * pink + s_white * white), with s_pink, s_white
    drawn per run from ``pink_level`` / ``white_level`` (background 1/f activity
    and sensor noise) and c = ``noise_scale`` (1 for the main benchmark; the
    exploratory noise sweep raises it). Then a 0.5 Hz high-pass, non-overlapping
    windows of ``window`` seconds, and per-window z-scoring (absolute amplitude
    is therefore *not* available to any classifier).
    """
    rng = np.random.default_rng(seed)
    n_runs, n_samp = raw.eeg.shape
    fs = raw.fs
    s_pink = rng.uniform(*pink_level, size=n_runs)
    s_white = rng.uniform(*white_level, size=n_runs)
    noise = (s_pink[:, None] * pink_noise(n_runs, n_samp, rng)
             + s_white[:, None] * rng.standard_normal((n_runs, n_samp)))
    x = highpass(zscore_rows(raw.eeg) + noise_scale * noise, fs, 0.5)

    w = int(round(window * fs))
    n_win = n_samp // w
    if n_win < 1:
        raise ValueError("duration shorter than one window")
    X = zscore_rows(x[:, : n_win * w].reshape(n_runs * n_win, w))
    y = np.repeat(raw.label, n_win)
    groups = np.repeat(np.arange(n_runs), n_win)
    run_meta = dict(raw.meta)
    run_meta.update({"label": raw.label, "pink_level": s_pink, "white_level": s_white,
                     "noise_scale": np.full(n_runs, noise_scale)})
    return Dataset(X=X.astype(np.float32), y=y, groups=groups, fs=fs,
                   classes=CLASSES, run_meta=run_meta)


def make_dataset(runs_per_class: int, duration: float, window: float, seed: int,
                 fs: float = 250.0, dt: float = 5e-4, transient: float = 2.0,
                 pink_level=(0.2, 0.8), white_level=(0.05, 0.2),
                 regimes: dict | None = None) -> Dataset:
    """Convenience wrapper: :func:`simulate_runs` followed by :func:`measure`."""
    raw = simulate_runs(runs_per_class, duration, seed, fs=fs, dt=dt,
                        transient=transient, regimes=regimes)
    return measure(raw, window, seed + 1, pink_level, white_level)


# --------------------------------------------------------------------------
# Multi-channel montage (network of coupled columns, see network.py)
# --------------------------------------------------------------------------
@dataclass
class RawNetworkRuns:
    src: np.ndarray          # (n_runs, n_sources, n_samples) column outputs y1 - y2 (mV)
    label: np.ndarray        # (n_runs,)
    fs: float
    meta: dict               # per-run parameters (column parameters have shape (n_runs, n_sources))


@dataclass
class MultiChannelDataset:
    X: np.ndarray            # (n_windows, n_channels, n_samples), every channel z-scored per window
    y: np.ndarray
    groups: np.ndarray
    fs: float
    classes: tuple
    run_meta: dict

    @property
    def n_runs(self) -> int:
        return int(len(np.unique(self.groups)))

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[1])


def simulate_network_runs(runs_per_class: int, duration: float, seed: int, fs: float = 250.0,
                          dt: float = 5e-4, transient: float = 2.0, regimes: dict | None = None,
                          network: dict | None = None, batch_runs: int = 64) -> RawNetworkRuns:
    """Draw network parameters for ``runs_per_class`` runs per class and integrate them."""
    from .network import concat_network_params, simulate_network
    from .regimes import NETWORK_REGIMES, sample_network_params

    rng = np.random.default_rng(seed)
    regimes = regimes or DEFAULT_REGIMES
    network = network or NETWORK_REGIMES
    parts, meta_parts, labels = [], [], []
    for ci, cls in enumerate(CLASSES):
        p, m = sample_network_params(cls, runs_per_class, rng, regimes, network)
        parts.append(p)
        meta_parts.append(m)
        labels.append(np.full(runs_per_class, ci))
    params = concat_network_params(parts)
    label = np.concatenate(labels)
    sigs = []
    for s in range(0, params.n_runs, batch_runs):
        sub = params.subset(slice(s, min(params.n_runs, s + batch_runs)))
        sigs.append(simulate_network(sub, duration, rng, dt=dt, fs_out=fs, transient=transient))
    meta = {k: np.asarray(v) for k, v in params.__dict__.items()}
    for key in meta_parts[0]:
        meta[key] = np.concatenate([m[key] for m in meta_parts])
    return RawNetworkRuns(src=np.concatenate(sigs, axis=0), label=label, fs=fs, meta=meta)


def measure_multichannel(raw: RawNetworkRuns, window: float, seed: int, pink_level=(0.2, 0.8),
                         white_level=(0.05, 0.2), common_fraction=(0.2, 0.5),
                         network: dict | None = None,
                         noise_scale: float = 1.0) -> MultiChannelDataset:
    """Lead field + measurement noise + windowing for the multi-channel montage.

    sensors = L_run @ sources, with L_run the fixed Gaussian lead field times a
    per-run jitter; the montage is divided by its overall standard deviation
    (one scalar per run, so relative channel amplitudes survive), then

        x_c = sensors_c + s_pink (sqrt(1 - rho) pink_c + sqrt(rho) pink_common) + s_white white_c

    with s_pink, s_white drawn per run from the single-channel ranges and the
    spatially shared fraction rho of the 1/f background from ``common_fraction``.
    0.5 Hz high-pass, non-overlapping windows, and every channel of every
    window z-scored (exactly the single-channel convention: no classifier sees
    absolute or relative channel amplitude).
    """
    from .network import lead_field
    from .regimes import NETWORK_REGIMES

    net = network or NETWORK_REGIMES
    rng = np.random.default_rng(seed)
    n_runs, n_src, n_samp = raw.src.shape
    n_ch = int(net["n_channels"])
    fs = raw.fs
    L0 = lead_field(n_ch, n_src, float(net["lead_width"]))
    jitter = rng.uniform(*net["lead_jitter"], size=(n_runs, n_ch, n_src))
    L = L0[None] * jitter
    src = raw.src - raw.src.mean(axis=2, keepdims=True)
    sens = np.einsum("rcs,rst->rct", L, src)
    sens /= sens.std(axis=(1, 2), keepdims=True) + 1e-12
    s_pink = rng.uniform(*pink_level, size=n_runs)
    s_white = rng.uniform(*white_level, size=n_runs)
    rho = rng.uniform(*common_fraction, size=n_runs)
    pink_c = pink_noise(n_runs * n_ch, n_samp, rng).reshape(n_runs, n_ch, n_samp)
    pink_0 = pink_noise(n_runs, n_samp, rng)[:, None, :]
    white = rng.standard_normal((n_runs, n_ch, n_samp))
    noise = (s_pink[:, None, None] * (np.sqrt(1 - rho)[:, None, None] * pink_c
                                      + np.sqrt(rho)[:, None, None] * pink_0)
             + s_white[:, None, None] * white)
    x = highpass(sens + noise_scale * noise, fs, 0.5)

    w = int(round(window * fs))
    n_win = n_samp // w
    if n_win < 1:
        raise ValueError("duration shorter than one window")
    X = x[:, :, : n_win * w].reshape(n_runs, n_ch, n_win, w).transpose(0, 2, 1, 3)
    X = zscore_rows(X.reshape(n_runs * n_win, n_ch, w))
    y = np.repeat(raw.label, n_win)
    groups = np.repeat(np.arange(n_runs), n_win)
    run_meta = dict(raw.meta)
    run_meta.update({"label": raw.label, "pink_level": s_pink, "white_level": s_white,
                     "common_fraction": rho, "noise_scale": np.full(n_runs, noise_scale)})
    return MultiChannelDataset(X=X.astype(np.float32), y=y, groups=groups, fs=fs,
                               classes=CLASSES, run_meta=run_meta)
