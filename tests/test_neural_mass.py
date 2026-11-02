"""Dynamics of the neural mass models reproduce the textbook signatures."""
import numpy as np
from scipy.signal import welch
from scipy.stats import skew

from neurorhythms.neural_mass import (GPE_B, GPE_M, JR_E0, JR_V0, STN_B, STN_M, RunParams,
                                      holgado_sigmoid, jr_sigmoid, pink_noise, simulate)
from neurorhythms.regimes import stn_gpe_hopf


def _params(n, A=3.25, p=220.0, rel=0.6, w_gg=2.0, d=6e-3, kappa=0.0, ctx_std=1.0):
    c = lambda v: np.full(n, float(v))  # noqa: E731
    g_crit, _ = stn_gpe_hopf(w_gg, d, d, 4e-3)
    w = np.sqrt(np.atleast_1d(rel) * g_crit) * np.ones(n)
    return RunParams(A=c(A) if np.isscalar(A) else np.asarray(A, float), B=c(22), a=c(100),
                     b=c(50), C=c(135), p_mean=c(p), p_std=c(57.7), w_sg=w, w_gs=w,
                     w_gg=c(w_gg), w_cs=c(2), ctx_drive=w * GPE_B / 2, ctx_std=c(ctx_std),
                     str_inh=w * STN_B - w_gg * GPE_B, d_sg=c(d), d_gs=c(d), d_gg=c(4e-3),
                     kappa=c(kappa), d_bc=c(1e-2))


def _peak(x, fs=250.0):
    f, p = welch(x - x.mean(), fs=fs, nperseg=500)
    m = f >= 1
    return f[m][np.argmax(p[m])]


def test_sigmoids():
    assert np.isclose(jr_sigmoid(np.array(JR_V0)), JR_E0)
    assert np.isclose(holgado_sigmoid(np.array(0.0), STN_M, STN_B), STN_B)
    assert np.isclose(holgado_sigmoid(np.array(0.0), GPE_M, GPE_B), GPE_B)
    assert holgado_sigmoid(np.array(1e4), STN_M, STN_B) > 0.99 * STN_M


def test_jansen_rit_standard_parameters_give_alpha(rng):
    eeg = simulate(_params(2), 8.0, rng)
    for x in eeg:
        assert 8.0 <= _peak(x) <= 12.5


def test_high_excitatory_gain_gives_slow_spikes(rng):
    eeg = simulate(_params(2, A=3.9, p=120.0), 8.0, rng)
    for x in eeg:
        assert _peak(x) < 7.0
        assert skew(x) > 0.5          # tall positive spikes, shallow troughs


def test_stn_gpe_hopf_threshold_matches_simulation(rng):
    rel = np.array([0.7, 1.3])
    P = _params(2, rel=rel, w_gg=3.0, d=5e-3)
    _, stn = simulate(P, 6.0, rng, transient=3.0, return_stn=True)
    assert stn[0].std() < 1.0         # below threshold: damped
    assert stn[1].std() > 3.0         # above threshold: sustained oscillation
    _, f_pred = stn_gpe_hopf(3.0, 5e-3, 5e-3, 4e-3)
    assert 13.0 <= _peak(stn[1]) <= 30.0
    assert abs(_peak(stn[1]) - f_pred) < 3.0


def test_simulation_is_reproducible():
    a = simulate(_params(2), 2.0, np.random.default_rng(0))
    b = simulate(_params(2), 2.0, np.random.default_rng(0))
    assert np.array_equal(a, b)
    assert a.shape == (2, 500)


def test_pink_noise_unit_variance_and_slope(rng):
    x = pink_noise(4, 4096, rng)
    assert np.allclose(x.std(axis=1), 1.0)
    f, p = welch(x, nperseg=1024, axis=-1)
    m = (f > 0.01) & (f < 0.3)
    slope = np.polyfit(np.log(f[m]), np.log(p.mean(0)[m]), 1)[0]
    assert -1.4 < slope < -0.6
