"""End-to-end smoke test of the SW1PerS / Riemannian benchmark on a tiny configuration."""
import importlib.util
import json
import os

from neurorhythms.sw1pers_bench import SETS, load_config, run


def test_sw1pers_benchmark_end_to_end(tmp_path):
    cfg = load_config(None)
    cfg.update({"name": "tiny", "seed": 3, "n_jobs": 2, "cache_dir": None})
    cfg["simulation"].update({"runs_per_class": 4, "duration": 8.0})
    cfg["topology"] = {"delays": [5]}
    cfg["sw1pers"] = {"n_points": 40, "crocker_points": 30}
    cfg["evaluation"].update({"n_splits": 2, "n_repeats": 1, "bootstrap": 50, "rf_trees": 10})
    cfg["cnn"].update({"enabled": importlib.util.find_spec("torch") is not None, "epochs": 1})
    cfg["importance"].update({"n_permutations": 1})
    cfg["outputs"] = {"out_dir": str(tmp_path), "figures_dir": str(tmp_path / "fig"),
                      "results_md": str(tmp_path / "SW1PERS_RESULTS.md")}
    res = run(cfg)
    assert set(res["answers"]) == {"Q1", "Q2", "Q3"}
    for t in ("single", "multi"):
        tr = res["tracks"][t]
        assert tr["dataset"]["n_windows"] == 24
        assert set(SETS[t]) <= set(tr["summary"])
        for s in tr["summary"].values():
            assert 0.0 <= s["accuracy"] <= 1.0 and s["ci95"][0] <= s["ci95"][1]
        qs = {c["question"] for c in tr["comparisons"] if c["primary"]}
        assert qs == ({"Q1", "Q2", "Q3"} if t == "multi" else {"Q1"})
        pairs = [(c["candidate"], c["reference"]) for c in tr["comparisons"]]
        assert len(pairs) == len(set(pairs))
    assert res["tracks"]["multi"]["dataset"]["n_channels"] == 6
    assert res["tracks"]["multi"]["summary"]["riemann"]["n_features"] == 5 * 21
    with open(tmp_path / "results.json") as fh:
        assert json.load(fh)["config"]["name"] == "tiny"
    for f in ("example_multichannel", "sw1pers_clouds", "accuracy_bars", "margins",
              "confusion_matrices_single", "confusion_matrices_multi", "feature_importance_multi"):
        assert os.path.exists(tmp_path / "fig" / f"{f}.png"), f
    text = (tmp_path / "SW1PERS_RESULTS.md").read_text()
    assert "## Answers" in text and "riemann" in text and "Q3" in text
