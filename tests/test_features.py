import numpy as np

from neurorhythms.features import FEATURE_SETS, assemble, compute_blocks
from neurorhythms.features.baseline import (bandpower_features, hjorth_features,
                                            psd_features, waveform_features)
from neurorhythms.features.topology import (TopoConfig, alpha_diagrams, betti_curve,
                                            delay_embedding, diagram_stats, landscapes,
                                            sublevel_diagram, topo_feature_families,
                                            topo_feature_names, topo_features)

FS = 250.0


def _sine(f, n=1000, phase=0.0):
    return np.sin(2 * np.pi * f * np.arange(n) / FS + phase)


def test_bandpower_on_alpha_sine():
    F, names = bandpower_features(_sine(10.0)[None], FS)
    d = dict(zip(names, F[0]))
    assert d["peak_freq"] == 10.0
    assert d["rel_alpha"] > 0.9
    assert 9.0 <= d["sef50"] <= 11.0
    assert d["log_beta_alpha"] < -3
    F, _ = bandpower_features(_sine(20.0)[None], FS)
    assert dict(zip(names, F[0]))["rel_beta"] > 0.9


def test_hjorth_mobility_of_sine():
    f = 12.0
    F, _ = hjorth_features(_sine(f, 5000)[None])
    assert np.isclose(F[0, 0], 2 * np.sin(np.pi * f / FS), rtol=1e-2)
    assert np.isclose(F[0, 1], 1.0, atol=0.02)       # complexity of a pure sine = 1


def test_waveform_detects_positive_spikes():
    x = np.zeros(1000)
    x[::83] = 8.0
    x = x + 0.1 * np.random.default_rng(0).standard_normal(1000)
    F, names = waveform_features(((x - x.mean()) / x.std())[None])
    d = dict(zip(names, F[0]))
    assert d["skewness"] > 3 and d["kurtosis"] > 10


def test_psd_view_shape():
    F, names = psd_features(np.vstack([_sine(5.0), _sine(25.0)]), FS)
    assert F.shape == (2, 45) and len(names) == 45
    assert names[np.argmax(F[0])] == "psd_5Hz" and names[np.argmax(F[1])] == "psd_25Hz"


def test_sublevel_persistence_of_sine_counts_cycles():
    x = _sine(5.0)                                  # 20 cycles in 4 s
    d = sublevel_diagram(x)
    p = d[:, 1] - d[:, 0]
    assert np.sum(p > 1.5) in (19, 20)              # one valley per cycle (global min infinite)
    assert np.all(np.isfinite(d))


def test_embedding_h1_of_sine_has_one_dominant_loop():
    x = _sine(10.0)
    _, h1 = alpha_diagrams(delay_embedding(x, 6, 3))
    p = np.sort(h1[:, 1] - h1[:, 0])[::-1]
    assert p[0] > 0.5 and p[0] > 5 * p[1]


def test_vectorisations_by_hand():
    d = np.array([[0.0, 2.0], [1.0, 2.0]])
    assert np.array_equal(betti_curve(d, 0.0, 2.0, 5), [1, 1, 2, 2, 0])
    L = landscapes(d, 0.0, 2.0, 5, 2).reshape(2, 5)
    assert np.allclose(L[0], [0, 0.5, 1.0, 0.5, 0])
    assert np.allclose(L[1], [0, 0, 0, 0.5, 0])
    s = diagram_stats(d, threshold=1.5)
    assert s[0] == 2 and s[1] == 1 and np.isclose(s[2], 3.0) and np.isclose(s[3], 2.0)
    assert np.all(diagram_stats(np.zeros((0, 2)), 0.1) == 0)


def test_topo_features_shape_and_families():
    cfg = TopoConfig()
    X = np.vstack([_sine(10.0), _sine(3.0) + 0.1 * np.random.default_rng(1).standard_normal(1000)])
    X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    F, names = topo_features(X, cfg)
    assert F.shape == (2, len(names)) == (2, len(topo_feature_names(cfg)))
    assert len(topo_feature_families(cfg)) == len(names)
    assert np.all(np.isfinite(F))
    assert len(set(names)) == len(names)


def test_compute_blocks_and_feature_sets():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((3, 1000))
    blocks = compute_blocks(X, FS)
    for name in FEATURE_SETS:
        F, names, fams = assemble(blocks, name)
        assert F.shape[0] == 3 and F.shape[1] == len(names) == len(fams)
        assert len(set(names)) == len(names)
    _, names, _ = assemble(blocks, "union_summary")
    assert all(not n.startswith(("sub_h0_betti", "emb2_h1_land")) for n in names)
