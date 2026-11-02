"""Multi-channel network: geometry, coupling, focus, measurement model."""
import numpy as np
from scipy.signal import coherence, welch
from scipy.stats import skew

from neurorhythms.dataset import (RawNetworkRuns, measure_multichannel,
                                  simulate_network_runs)
from neurorhythms.network import lead_field, ring_weights, simulate_network
from neurorhythms.regimes import NETWORK_REGIMES, sample_network_params


def test_ring_weights_and_lead_field(rng):
    W = ring_weights(4, rng, 5)
    assert W.shape == (5, 4, 4)
    assert np.allclose(W.sum(axis=2), 1.0)
    assert np.all(W[:, np.arange(4), np.arange(4)] == 0)
    assert np.all(W[:, 0, 1] > W[:, 0, 2]) and np.all(W[:, 0, 3] > W[:, 0, 2])  # neighbours > opposite
    L = lead_field(6, 4, 0.6)
    assert L.shape == (6, 4) and np.all((L > 0) & (L <= 1))
    # sensors 1, 2, 4, 5 sit next to one source each; sensors 0 and 3 between two
    assert list(np.argmax(L, axis=1)[[1, 2, 4, 5]]) == [0, 1, 2, 3]
    assert np.isclose(L[0, 0], L[0, 3]) and np.isclose(L[3, 1], L[3, 2])


def test_sample_network_params(rng):
    for cls in ("healthy", "epileptic", "parkinsonian"):
        p, meta = sample_network_params(cls, 6, rng)
        assert p.A.shape == (6, NETWORK_REGIMES["n_sources"]) and p.K.shape == (6,)
        assert p.W.shape == p.D.shape == (6, 4, 4)
        n_focus = meta["focus"].sum(axis=1)
        if cls == "epileptic":
            assert np.all(n_focus == 1)
            assert np.all(p.A[meta["focus"]] >= 3.6) and np.all(p.A[~meta["focus"]] <= 3.4)
        else:
            assert np.all(n_focus == 0) and np.all(p.A <= 3.4)
        assert (meta["bg_gain_rel"] > 1).all() == (cls == "parkinsonian")


def test_simulate_network_shapes_and_reproducible():
    p, _ = sample_network_params("healthy", 2, np.random.default_rng(1))
    a = simulate_network(p, 2.0, np.random.default_rng(0))
    b = simulate_network(p, 2.0, np.random.default_rng(0))
    assert a.shape == (2, 4, 500)
    assert np.array_equal(a, b)


def test_coupling_raises_alpha_coherence():
    p, _ = sample_network_params("healthy", 2, np.random.default_rng(2))
    p.kappa[:] = 0.0                 # no common basal-ganglia input
    p.K[:] = np.array([0.0, 30.0])   # run 0 uncoupled, run 1 strongly coupled
    src = simulate_network(p, 8.0, np.random.default_rng(3))
    coh = []
    for r in range(2):
        f, c = coherence(src[r, 0], src[r, 1], fs=250.0, nperseg=250)
        coh.append(c[(f >= 8) & (f <= 13)].mean())
        fw, pw = welch(src[r] - src[r].mean(1, keepdims=True), fs=250.0, nperseg=500)
        m = fw >= 1
        assert np.all((fw[m][np.argmax(pw[:, m], axis=1)] >= 8) &
                      (fw[m][np.argmax(pw[:, m], axis=1)] <= 13))   # every column in alpha
    assert coh[1] > coh[0] + 0.2


def test_epileptic_focus_spikes(rng):
    p, meta = sample_network_params("epileptic", 2, rng)
    p.A[meta["focus"]] = 3.9         # well inside the spike regime
    p.p_mean[meta["focus"]] = 120.0
    src = simulate_network(p, 8.0, np.random.default_rng(4))
    for r in range(2):
        f = np.nonzero(meta["focus"][r])[0][0]
        others = [c for c in range(4) if c != f]
        sk = skew(src[r], axis=1)
        assert sk[f] > 0.5 and sk[f] > sk[others].max()
        assert src[r, f].std() > src[r, others].std(axis=1).max()


def test_measure_multichannel_shapes_and_zscore():
    raw = simulate_network_runs(1, 8.0, seed=5)
    assert raw.src.shape == (3, 4, 2000) and list(raw.label) == [0, 1, 2]
    ds = measure_multichannel(raw, 4.0, seed=6)
    assert ds.X.shape == (6, 6, 1000) and ds.n_channels == 6 and ds.n_runs == 3
    assert np.allclose(ds.X.mean(axis=2), 0, atol=1e-4)
    assert np.allclose(ds.X.std(axis=2), 1, atol=1e-3)
    assert list(ds.groups) == [0, 0, 1, 1, 2, 2] and list(ds.y) == [0, 0, 1, 1, 2, 2]
    ds2 = measure_multichannel(RawNetworkRuns(raw.src, raw.label, raw.fs, raw.meta), 4.0, seed=6)
    assert np.array_equal(ds.X, ds2.X)
    # volume conduction + shared background: neighbouring channels are correlated
    r = np.corrcoef(ds.X[0])
    assert np.mean(r[np.triu_indices(6, 1)]) > -0.2
