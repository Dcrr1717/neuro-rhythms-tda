import numpy as np
import pytest

from neurorhythms.regimes import CLASSES, DEFAULT_REGIMES, sample_class_params, stn_gpe_hopf


def test_hopf_closed_form_without_self_inhibition():
    # w_gg = 0: K = |1 + i w tS| |1 + i w tG| at the phase-crossing frequency
    g, f = stn_gpe_hopf(0.0, 6e-3, 6e-3, 4e-3)
    assert 25.0 < g < 28.0
    assert 14.5 < f < 16.5


def test_hopf_frequency_increases_with_gpe_self_inhibition():
    fs = [stn_gpe_hopf(w, 5e-3, 5e-3, 4e-3)[1] for w in (1.0, 3.0, 5.0)]
    assert fs[0] < fs[1] < fs[2]
    assert 13.0 < fs[0] and fs[2] < 35.0


@pytest.mark.parametrize("cls", CLASSES)
def test_sample_class_params_respects_ranges(cls, rng):
    p, meta = sample_class_params(cls, 20, rng)
    spec = {**DEFAULT_REGIMES["shared"], **DEFAULT_REGIMES[cls]}
    assert p.n_runs == 20
    for k in ("A", "p_mean", "b", "kappa"):
        lo, hi = spec[k]
        v = getattr(p, k)
        assert np.all((v >= lo) & (v <= hi))
    lo, hi = spec["bg_gain_rel"]
    assert np.all((meta["bg_gain_rel"] >= lo) & (meta["bg_gain_rel"] <= hi))
    assert np.allclose(p.w_sg * p.w_gs, meta["bg_gain_rel"] * meta["bg_g_crit"])


def test_parkinsonian_is_supercritical_and_others_subcritical(rng):
    for cls in CLASSES:
        _, meta = sample_class_params(cls, 30, rng)
        if cls == "parkinsonian":
            assert np.all(meta["bg_gain_rel"] > 1.0)
            assert np.all((meta["bg_f_hopf"] > 13.0) & (meta["bg_f_hopf"] < 35.0))
        else:
            assert np.all(meta["bg_gain_rel"] < 1.0)


def test_unknown_class_raises(rng):
    with pytest.raises(ValueError):
        sample_class_params("migraine", 2, rng)
