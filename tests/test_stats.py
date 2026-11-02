import numpy as np
import pytest

from neurorhythms.classical import check_folds, cross_validate, make_folds
from neurorhythms.stats import (bootstrap_accuracy, confusion, paired_bootstrap_margin,
                                run_level_accuracy, verdict)


@pytest.fixture
def toy():
    groups = np.repeat(np.arange(30), 5)
    y = np.repeat(np.arange(30) % 3, 5)
    return y, groups


def test_folds_are_grouped_and_reproducible(toy):
    y, g = toy
    f1 = make_folds(y, g, 5, 2, seed=0)
    f2 = make_folds(y, g, 5, 2, seed=0)
    check_folds(f1, g)
    for r1, r2 in zip(f1, f2):
        for (a, b), (c, d) in zip(r1, r2):
            assert np.array_equal(a, c) and np.array_equal(b, d)
    with pytest.raises(AssertionError):
        check_folds([[(np.arange(0, 100), np.arange(98, 150))]], g)


def test_identical_predictions_have_zero_margin(toy):
    y, g = toy
    pred = np.where(np.random.default_rng(0).random(len(y)) < 0.8, y, (y + 1) % 3)[None]
    m = paired_bootstrap_margin(pred, pred, y, g, n_boot=200)
    assert m["margin"] == 0.0 and m["ci95"] == [0.0, 0.0]


def test_perfect_vs_chance_margin_is_positive(toy):
    y, g = toy
    chance = np.random.default_rng(1).integers(0, 3, size=(2, len(y)))
    m = paired_bootstrap_margin(np.vstack([y, y]), chance, y, g, n_boot=500)
    assert m["margin"] > 0.4 and m["ci95"][0] > 0.3 and m["p_le_0"] == 0.0


def test_bootstrap_accuracy_point_estimate(toy):
    y, g = toy
    pred = y.copy()
    pred[:15] = (pred[:15] + 1) % 3
    b = bootstrap_accuracy(pred[None], y, g, n_boot=300)
    assert np.isclose(b["accuracy"], 1 - 15 / len(y))
    assert b["ci95"][0] <= b["accuracy"] <= b["ci95"][1]


def test_verdict_rule():
    assert verdict(0.06, [0.01, 0.1], 0.05)["verdict"] == "GO"
    v = verdict(0.02, [0.005, 0.04], 0.05)
    assert v["verdict"] == "NO-GO" and v["ci_above_0"]
    assert not verdict(0.0, [-0.01, 0.01], 0.05)["ci_excludes_0"]


def test_run_level_accuracy_and_confusion(toy):
    y, g = toy
    proba = np.eye(3)[y] * 0.6 + 0.4 / 3
    proba[0] = [0, 0, 1]                      # one wrong window does not flip its run
    assert run_level_accuracy(proba, y, g)[0] == 1.0
    M = confusion(np.argmax(proba, 1)[None], y)
    assert np.allclose(M.sum(1), 1.0) and M[0, 0] < 1.0


def test_cross_validate_separable_problem(toy):
    y, g = toy
    rng = np.random.default_rng(0)
    F = np.eye(3)[y] * 3 + rng.standard_normal((len(y), 3))
    folds = make_folds(y, g, 3, 1, seed=0)
    for model in ("logreg", "rf"):
        r = cross_validate(F, y, folds, model, rf_trees=30)
        assert np.mean(r["pred"][0] == y) > 0.9
        assert np.allclose(r["proba"][0].sum(1), 1.0)
