import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neurorhythms.classical import make_folds  # noqa: E402
from neurorhythms.deep import build_cnn, cross_validate_cnn, n_parameters  # noqa: E402


def test_cnn_shapes():
    out = build_cnn()(torch.zeros(4, 1, 1000))
    assert tuple(out.shape) == (4, 3)
    assert 20_000 < n_parameters() < 100_000


def test_cnn_learns_frequency_classes():
    rng = np.random.default_rng(0)
    t = np.arange(500) / 250.0
    X, y, g = [], [], []
    for run in range(24):
        cls = run % 3
        f = (4.0, 10.0, 22.0)[cls]
        for _ in range(4):
            x = np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3)) + 0.5 * rng.standard_normal(500)
            X.append((x - x.mean()) / x.std())
            y.append(cls)
            g.append(run)
    X, y, g = np.array(X, np.float32), np.array(y), np.array(g)
    folds = make_folds(y, g, 3, 1, seed=0)
    r = cross_validate_cnn(X, y, folds, epochs=12, batch_size=16, lr=3e-3, threads=1)
    assert np.mean(r["pred"][0] == y) > 0.8


def test_cnn_multichannel_input():
    out = build_cnn(in_channels=6)(torch.zeros(4, 6, 1000))
    assert tuple(out.shape) == (4, 3)
    assert n_parameters(in_channels=6) == n_parameters() + 5 * 16 * 7
