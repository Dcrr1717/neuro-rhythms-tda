"""IAAFT surrogates, the nonlinear statistics and the surrogate benchmark."""
import numpy as np
import pytest

from neurorhythms.features.baseline import bandpower_features, psd_features
from neurorhythms.features.nonlinear import LAGS, nonlinear_features, onepoint_features
from neurorhythms.surrogate_benchmark import end_matched_windows, load_config, run
from neurorhythms.surrogates import iaaft, surrogate_quality


def _sawtooth_like(n=4000, period=50, rng=None):
    """Sharp rise, slow decay (time-irreversible). With ``rng``, cycle lengths are
    jittered (aperiodic, broadband like a real rhythm) and a little noise is added."""
    if rng is None:
        t = np.arange(n) % period
        return np.where(t < 5, t / 5.0, 1.0 - (t - 5) / (period - 5.0))
    cycles = []
    while sum(len(c) for c in cycles) < n:
        p = int(rng.integers(35, 66))
        t = np.arange(p)
        cycles.append(np.where(t < 5, t / 5.0, 1.0 - (t - 5) / (p - 5.0)))
    return np.concatenate(cycles)[:n] + 0.05 * rng.standard_normal(n)


def _skewed_noise(rng, n=4000):
    """Correlated noise through a static exponential nonlinearity (skewed, linear dynamics)."""
    from scipy.signal import lfilter
    return np.exp(0.6 * lfilter([1.0], [1.0, -0.9], rng.standard_normal(n)) / 2.3)


def test_iaaft_spectrum_variant_preserves_spectrum_exactly(rng):
    x = np.vstack([_sawtooth_like(rng=rng), _skewed_noise(rng)])
    s = iaaft(x, rng, 100, "spectrum")
    q = surrogate_quality(x, s)
    assert np.all(q["spectrum_rel_err"] < 1e-10)
    assert np.all(q["histogram_err"] < 0.05)
    assert abs(q["corr"][1]) < 0.3          # phases randomised (aperiodic signal)


def test_iaaft_amplitude_variant_preserves_histogram_exactly(rng):
    x = np.vstack([_sawtooth_like(rng=rng), _skewed_noise(rng)])
    s = iaaft(x, rng, 100, "amplitude")
    assert np.allclose(np.sort(s, axis=1), np.sort(x, axis=1))
    assert np.all(surrogate_quality(x, s)["spectrum_rel_err"] < 0.05)


def test_iaaft_rejects_unknown_variant(rng):
    with pytest.raises(ValueError):
        iaaft(np.zeros((1, 16)), rng, final="phase")


def test_time_reversal_asymmetry_destroyed_by_surrogate(rng):
    x = _sawtooth_like(rng=rng)[None]
    s = iaaft(x, rng, 100, "spectrum")
    F, names = nonlinear_features(x)
    Fs, _ = nonlinear_features(s)
    j = names.index("trev_3")
    assert abs(F[0, j]) > 1.0             # sharp rise vs slow decay
    assert abs(Fs[0, j]) < 0.4            # surrogate is (nearly) time-reversible
    r = names.index("rise_fraction")
    assert abs(F[0, r] - 0.5) > abs(Fs[0, r] - 0.5) + 0.03


def test_nonlinear_features_of_reversible_signals(rng):
    t = np.arange(2000) / 250.0
    X = np.vstack([np.sin(2 * np.pi * 10 * t), rng.standard_normal(2000)])
    F, names = nonlinear_features(X)
    assert F.shape == (2, 3 * len(LAGS) + 4)
    for k in (f"trev_{tau}" for tau in LAGS):
        assert np.all(np.abs(F[:, names.index(k)]) < 0.15)
    assert np.all(np.abs(F[:, names.index("rise_fraction")] - 0.5) < 0.05)


def test_onepoint_features_are_histogram_functions(rng):
    x = rng.standard_normal((2, 1000))
    F, names = onepoint_features(x)
    Fp, _ = onepoint_features(x[:, rng.permutation(1000)])
    assert np.allclose(F, Fp) and len(names) == F.shape[1] == 11


def test_periodogram_features_blind_to_spectrum_surrogate(rng):
    x = np.vstack([_sawtooth_like(1000, rng=rng), rng.standard_normal(1000)])
    s = iaaft(x, rng, 100, "spectrum")
    for fn in (bandpower_features, psd_features):
        assert np.allclose(fn(x, 250.0, method="periodogram")[0],
                           fn(s, 250.0, method="periodogram")[0])


def test_end_matched_windows_are_smooth_and_disjoint(rng):
    t = np.arange(6000) / 250.0
    x = np.sin(2 * np.pi * 7.3 * t)[None] + 0.01 * rng.standard_normal((1, 6000))
    W, run, starts = end_matched_windows(x, 250.0, 4.0, 0.4)
    assert W.shape[1] == 1000 and np.all(run == 0)
    assert np.all(np.diff(starts) >= 1000)                         # non-overlapping
    assert np.median(np.abs(W[:, 0] - W[:, -1])) < 0.2              # ends join
    # without matching, an arbitrary 4 s cut of a 7.3 Hz sine leaves a jump
    assert abs(np.sin(2 * np.pi * 7.3 * 4.0) - 0.0) > 0.5


def test_surrogate_benchmark_end_to_end(tmp_path):
    cfg = load_config(None)
    cfg.update({"name": "tiny", "seed": 5, "n_jobs": 1, "cache_dir": None})
    cfg["simulation"].update({"runs_per_class": 4, "duration": 10.0})
    cfg["evaluation"].update({"n_splits": 2, "bootstrap": 50, "rf_trees": 10})
    cfg["importance"].update({"n_permutations": 1})
    cfg["cnn"].update({"epochs": 1})
    cfg["tasks"] = [
        {"name": "binary_pooled", "target": "binary", "classes": "all", "surrogate": "spectrum",
         "n_repeats": 1, "cnn": False, "sets": "all", "primary": True},
        {"name": "six_class", "target": "six", "classes": "all", "surrogate": "amplitude",
         "n_repeats": 1, "cnn": False, "sets": ["periodogram_psd", "nonlinear",
                                                "baselines_union", "topo", "union"]},
    ]
    cfg["outputs"] = {"out_dir": str(tmp_path), "figures_dir": str(tmp_path / "fig"),
                      "results_md": str(tmp_path / "RESULTS_SURROGATE.md")}
    R = run(cfg)
    P = R["tasks"]["binary_pooled"]
    assert R["final_verdict"] in ("GO", "NO-GO")
    assert P["n_windows"] == 2 * 12 * 2                     # 2 windows per run, x2 for surrogates
    assert P["summary"]["periodogram_psd"]["accuracy"] == pytest.approx(0.5, abs=0.15)
    assert R["data"]["quality"]["spectrum"]["spectrum_rel_err"]["max"] < 1e-10
    assert R["tasks"]["six_class"]["chance"] == pytest.approx(1 / 6)
    for f in ("surrogate_examples", "surrogate_accuracy_bars", "surrogate_margins",
              "surrogate_tasks", "surrogate_confusion_six_class", "surrogate_importance"):
        assert (tmp_path / "fig" / f"{f}.png").exists()
    assert "## Verdict" in (tmp_path / "RESULTS_SURROGATE.md").read_text()
