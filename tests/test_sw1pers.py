"""SW1PerS: periodicity scores behave as the theory says (Perea & Harer 2015)."""
import numpy as np

from neurorhythms.features.sw1pers import (SCALE_STATS, SQRT3, SW1PersConfig, betti1_curve,
                                           dominant_period, h1_diagram, scale_stats,
                                           sliding_window_cloud, sw1pers_feature_names,
                                           sw1pers_features, sw1pers_window)

FS = 250.0
T = np.arange(1000) / FS
CFG = SW1PersConfig(n_points=100, crocker_points=50)


def _z(x):
    return (x - x.mean()) / x.std()


def _mp(x, period, n_points=120):
    d = h1_diagram(sliding_window_cloud(_z(x), FS, period, n_points=n_points))
    p = np.sort(d[:, 1] - d[:, 0])[::-1]
    return np.concatenate([p, [0.0, 0.0]])[:2]


def test_cloud_on_unit_sphere():
    P = sliding_window_cloud(np.sin(2 * np.pi * 10 * T), FS, 0.1, n_harmonics=6, n_points=50)
    assert P.shape == (50, 13)
    assert np.allclose(np.linalg.norm(P, axis=1), 1.0)
    assert np.allclose(P.mean(axis=1), 0.0, atol=0.05)   # centred (up to mean-shift)


def test_sine_is_periodic_noise_is_not():
    mp_sine = _mp(np.sin(2 * np.pi * 10 * T), 0.1)
    mp_noise = _mp(np.random.default_rng(0).standard_normal(1000), 0.1)
    assert mp_sine[0] > 0.6 and mp_sine[1] < 0.05      # one dominant loop
    assert mp_sine[0] > 2 * mp_noise[0]
    assert mp_noise[0] < 0.4
    assert mp_sine[0] <= 1.0 + 1e-9                    # normalised by sqrt(3)


def test_two_incommensurate_rhythms_give_two_loops():
    x = np.sin(2 * np.pi * 10 * T) + 0.7 * np.sin(2 * np.pi * 21.3 * T)
    mp = _mp(x, 0.1)
    assert mp[1] > 0.2                                 # torus: second persistent H1 class


def test_dominant_period():
    for f in (4.0, 7.0, 12.0, 22.0):
        # integer-lag resolution: within one sample of the true period
        assert abs(dominant_period(np.sin(2 * np.pi * f * T), FS) - 1.0 / f) <= 1.0 / FS
    # monotone autocorrelation (no peak in range) falls back to 100 ms
    assert dominant_period(np.cumsum(np.ones(1000)), FS) == 0.1


def test_invariance_to_amplitude_and_offset():
    x = _z(np.sin(2 * np.pi * 9 * T) + 0.3 * np.random.default_rng(1).standard_normal(1000))
    assert np.allclose(sw1pers_window(x, FS, CFG), sw1pers_window(3.0 * x + 2.0, FS, CFG),
                       atol=1e-6)


def test_vectorisations_by_hand():
    d = np.array([[0.1, 0.9], [0.2, 0.4]])
    s = scale_stats(d)
    assert len(s) == len(SCALE_STATS)
    assert np.isclose(s[0], 0.8) and np.isclose(s[1], 0.2) and np.isclose(s[2], 0.1)
    assert np.isclose(s[3], 1.0)
    assert np.all(scale_stats(np.zeros((0, 2))) == 0)
    grid = SQRT3 * np.array([0.05, 0.3, 0.5, 0.95])
    assert np.array_equal(betti1_curve(d, grid), [0, 2, 1, 0])


def test_feature_shapes_single_and_multi():
    names, fams = sw1pers_feature_names(CFG)
    assert len(names) == len(set(names)) == len(fams)
    assert set(fams) == {"sw1pers", "crocker"}
    assert sum(f == "sw1pers" for f in fams) == len(CFG.scale_names) * len(SCALE_STATS)
    rng = np.random.default_rng(2)
    X = rng.standard_normal((2, 3, 1000))
    F, n, f = sw1pers_features(X, FS, CFG)
    assert F.shape == (2, 3 * len(names)) and len(n) == len(f) == F.shape[1]
    assert n[0].startswith("ch0_") and n[-1].startswith("ch2_")
    assert np.all(np.isfinite(F))
    F1, _, _ = sw1pers_features(X[:, 1], FS, CFG)
    assert np.allclose(F1, F[:, len(names):2 * len(names)])
