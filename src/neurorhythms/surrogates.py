"""Iterative amplitude-adjusted Fourier-transform (IAAFT) surrogates.

An IAAFT surrogate (Schreiber & Schmitz 1996) of a signal keeps its power
spectrum (hence its autocorrelation) *and* its amplitude distribution, but
randomises the Fourier phases. Everything a linear Gaussian process observed
through a static nonlinearity can express is preserved; temporal structure
beyond that is destroyed: time-irreversibility (sharp rise / slow decay),
phase coupling between harmonics, waveform asymmetry, spike morphology. This
is the standard null model of the surrogate-data test for nonlinearity
(Theiler et al. 1992; Schreiber & Schmitz 2000).

The algorithm alternates two projections, starting from a random shuffle:
(1) impose the original Fourier amplitudes, keeping the current phases;
(2) impose the original value distribution by rank ordering. It cannot match
both exactly. ``final="spectrum"`` ends on step (1), so the spectrum is exact
and the histogram approximate; ``final="amplitude"`` ends on step (2), so the
histogram is exact and the spectrum approximate.

Vectorised over rows: ``x`` has shape (n_signals, n_samples).
"""
from __future__ import annotations

import numpy as np


def iaaft(x: np.ndarray, rng: np.random.Generator, n_iter: int = 100,
          final: str = "spectrum") -> np.ndarray:
    """IAAFT surrogates of every row of ``x``.

    Parameters
    ----------
    x : (n, T) array
    rng : numpy Generator
    n_iter : int
        Maximum number of iterations (stops early once no row changes rank order).
    final : {"spectrum", "amplitude"}
        Which constraint is imposed exactly by the last step.
    """
    if final not in ("spectrum", "amplitude"):
        raise ValueError("final must be 'spectrum' or 'amplitude'")
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n, T = x.shape
    amp = np.abs(np.fft.rfft(x, axis=1))
    sorted_x = np.sort(x, axis=1)
    rows = np.arange(n)[:, None]
    # start from a random permutation of each row
    s = x[rows, np.argsort(rng.random((n, T)), axis=1)]
    prev_rank = None
    for _ in range(n_iter):
        S = np.fft.rfft(s, axis=1)
        s = np.fft.irfft(amp * np.exp(1j * np.angle(S)), n=T, axis=1)
        rank = np.argsort(np.argsort(s, axis=1), axis=1)
        s = sorted_x[rows, rank]
        if prev_rank is not None and np.array_equal(rank, prev_rank):
            break
        prev_rank = rank
    if final == "spectrum":
        S = np.fft.rfft(s, axis=1)
        s = np.fft.irfft(amp * np.exp(1j * np.angle(S)), n=T, axis=1)
    return s


def surrogate_quality(x: np.ndarray, s: np.ndarray) -> dict:
    """How well a surrogate matches its original, row by row.

    spectrum_rel_err : relative L2 error of the Fourier amplitudes (excluding DC)
    histogram_err    : mean |sorted(x) - sorted(s)| in units of std(x)
    corr             : Pearson correlation between x and s (near 0: phases randomised)
    """
    x = np.atleast_2d(x)
    s = np.atleast_2d(s)
    ax = np.abs(np.fft.rfft(x, axis=1))[:, 1:]
    as_ = np.abs(np.fft.rfft(s, axis=1))[:, 1:]
    spec = np.linalg.norm(ax - as_, axis=1) / np.linalg.norm(ax, axis=1)
    hist = np.mean(np.abs(np.sort(x, axis=1) - np.sort(s, axis=1)), axis=1) / x.std(axis=1)
    xc = x - x.mean(axis=1, keepdims=True)
    sc = s - s.mean(axis=1, keepdims=True)
    corr = (xc * sc).sum(1) / np.sqrt((xc ** 2).sum(1) * (sc ** 2).sum(1))
    return {"spectrum_rel_err": spec, "histogram_err": hist, "corr": corr}
