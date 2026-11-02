"""Riemannian-geometry baseline: tangent-space features of SPD channel covariances.

The state-of-the-art "honest" baseline for multi-channel EEG classification
(Barachant et al. 2012, 2013; reviewed by Congedo, Barachant & Bhatia 2017 and
Lotte et al. 2018, who list Riemannian classifiers among the most accurate and
robust EEG classifiers):

1. **Filter bank.** Each multi-channel window is band-pass filtered in the
   same five bands as the band-power baseline (delta 1-4, theta 4-8, alpha
   8-13, beta 13-30, gamma 30-45 Hz; zero-phase 4th-order Butterworth).
2. **SPD covariance.** One ``n_channels x n_channels`` covariance per band,
   with Oracle-Approximating-Shrinkage regularisation (``pyriemann``
   ``Covariances(estimator="oas")``) so every matrix is well conditioned.
   Because every channel of a window is z-scored, the *diagonal* of a band's
   covariance is that channel's band power as a fraction of its total
   variance, i.e. exactly the relative band powers of the band-power
   baseline. The *off-diagonal* terms add band-limited cross-channel coupling.
   This baseline therefore subsumes band power and adds coupling.
3. **Tangent space.** The matrices of each band are mapped to the tangent
   space at their Riemannian (affine-invariant) mean (``TangentSpace(metric=
   "riemann")``), which turns the curved SPD manifold into a Euclidean vector
   space where a linear classifier is appropriate. The reference point is a
   statistic of the training set, so it is fitted *inside every CV fold*
   (:class:`TangentSpaceBlock` is the first step of the fold's pipeline);
   only the per-window covariances, which use no labels and no other window,
   are precomputed.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
except ImportError as exc:  # pragma: no cover
    raise ImportError("The Riemannian view requires pyriemann: pip install pyriemann") from exc

from .baseline import BANDS


def filterbank_covariances(X: np.ndarray, fs: float, bands: dict | None = None,
                           estimator: str = "oas", order: int = 4) -> tuple[np.ndarray, list]:
    """SPD covariances of band-passed multi-channel windows.

    X : (n_windows, n_channels, n_samples). ``bands=None`` uses the five
    band-power bands; ``bands={}`` gives a single broadband covariance.
    Returns (n_windows, n_bands, n_channels, n_channels) and the band names.
    """
    X = np.asarray(X, dtype=float)
    bands = BANDS if bands is None else bands
    covs, names = [], []
    if not bands:
        covs.append(Covariances(estimator=estimator).transform(X))
        names.append("broadband")
    for name, (lo, hi) in bands.items():
        sos = sps.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
        Xf = sps.sosfiltfilt(sos, X, axis=-1)
        covs.append(Covariances(estimator=estimator).transform(Xf))
        names.append(name)
    return np.stack(covs, axis=1), names


def covariance_block(X: np.ndarray, fs: float, bands: dict | None = None, prefix: str = "cov"):
    """Flattened covariances as a feature block -> (F, names, families)."""
    C, bnames = filterbank_covariances(X, fs, bands)
    n, b, c, _ = C.shape
    names = [f"{prefix}_{bn}_{i}{j}" for bn in bnames for i in range(c) for j in range(c)]
    return C.reshape(n, b * c * c), names, ["riemann"] * len(names)


class TangentSpaceBlock(BaseEstimator, TransformerMixin):
    """Fold-local tangent-space projection of the covariance columns of ``X``.

    The first ``n_bands * n_channels**2`` columns of ``X`` are flattened SPD
    matrices (one per band, as produced by :func:`covariance_block`); they are
    replaced by their tangent vectors at the Riemannian mean of the *training*
    matrices of each band. Any remaining columns pass through unchanged, so
    covariance and ordinary features can be combined in one pipeline.
    """

    def __init__(self, n_bands: int = 5, n_channels: int = 6, metric: str = "riemann"):
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.metric = metric

    @property
    def n_cov(self) -> int:
        return self.n_bands * self.n_channels ** 2

    def _mats(self, X):
        X = np.asarray(X, dtype=float)
        c = self.n_channels
        M = X[:, : self.n_cov].reshape(len(X), self.n_bands, c, c)
        return M, X[:, self.n_cov:]

    def fit(self, X, y=None):
        M, _ = self._mats(X)
        self.ts_ = [TangentSpace(metric=self.metric).fit(M[:, b]) for b in range(self.n_bands)]
        return self

    def transform(self, X):
        M, rest = self._mats(X)
        parts = [ts.transform(M[:, b]) for b, ts in enumerate(self.ts_)]
        return np.concatenate(parts + [rest], axis=1)
