"""Riemannian view: SPD covariances, fold-local tangent space, coupling information."""
from functools import partial

import numpy as np

from neurorhythms.classical import cross_validate, make_folds
from neurorhythms.features.baseline import bandpower_features
from neurorhythms.features.riemann import (TangentSpaceBlock, covariance_block,
                                           filterbank_covariances)

FS = 250.0


def _z(X):
    X = X - X.mean(axis=-1, keepdims=True)
    return X / X.std(axis=-1, keepdims=True)


def test_filterbank_covariances_are_spd():
    X = _z(np.random.default_rng(0).standard_normal((4, 5, 1000)))
    C, names = filterbank_covariances(X, FS)
    assert C.shape == (4, 5, 5, 5) and names == ["delta", "theta", "alpha", "beta", "gamma"]
    assert np.allclose(C, np.swapaxes(C, -1, -2))
    assert np.all(np.linalg.eigvalsh(C) > 0)
    Cb, nb = filterbank_covariances(X, FS, {})
    assert Cb.shape == (4, 1, 5, 5) and nb == ["broadband"]


def test_diagonal_is_relative_band_power():
    """With z-scored channels, diag of a band's covariance = fraction of variance in the band."""
    t = np.arange(1000) / FS
    rng = np.random.default_rng(1)
    X = np.stack([np.sin(2 * np.pi * 10 * t) + 0.6 * np.sin(2 * np.pi * 20 * t + 1)
                  + 0.2 * rng.standard_normal(1000) for _ in range(2)])[None]
    X = _z(X)
    C, names = filterbank_covariances(X, FS, estimator="scm")
    F, fn = bandpower_features(X[0], FS)
    rel = dict(zip(fn, F[0]))
    for b in ("alpha", "beta"):
        diag = C[0, names.index(b)].diagonal()
        assert np.allclose(diag, rel[f"rel_{b}"], atol=0.08)


def test_tangent_space_block_shapes_and_passthrough():
    X = _z(np.random.default_rng(2).standard_normal((12, 4, 500)))
    F, names, fams = covariance_block(X, FS)
    assert F.shape == (12, 5 * 16) and len(names) == len(fams) == F.shape[1]
    extra = np.arange(24.0).reshape(12, 2)
    ts = TangentSpaceBlock(n_bands=5, n_channels=4).fit(np.hstack([F, extra]))
    out = ts.transform(np.hstack([F, extra]))
    assert out.shape == (12, 5 * 10 + 2)
    assert np.array_equal(out[:, -2:], extra)


def test_riemann_sees_coupling_that_band_power_cannot():
    """Two classes with identical per-channel spectra, differing only in cross-channel
    coupling: band power is at chance, the Riemannian tangent space separates them."""
    rng = np.random.default_rng(3)
    n, c, w = 80, 4, 500
    t = np.arange(w) / FS
    X, y = [], []
    for i in range(n):
        cls = i % 2
        common = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.3))
        chans = []
        for _ in range(c):
            own = np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.3))
            chans.append((common if cls else own) + rng.standard_normal(w))
        X.append(chans)
        y.append(cls)
    X, y = _z(np.array(X)), np.array(y)
    groups = np.arange(n)
    folds = make_folds(y, groups, 4, 1, seed=0)
    Fb, _ = bandpower_features(X.reshape(n * c, w), FS)
    acc_bp = np.mean(cross_validate(Fb.reshape(n, -1), y, folds, "logreg",
                                    n_classes=2)["pred"][0] == y)
    Fc, _, _ = covariance_block(X, FS)
    pre = partial(TangentSpaceBlock, n_bands=5, n_channels=c)
    acc_rm = np.mean(cross_validate(Fc, y, folds, "logreg", n_classes=2, pre=pre)["pred"][0] == y)
    assert acc_rm > 0.9
    assert acc_bp < 0.75
