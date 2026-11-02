"""End-to-end smoke test of the benchmark on a tiny configuration."""
import importlib.util
import json
import os

from neurorhythms.pipeline import load_config, run


def test_pipeline_end_to_end(tmp_path):
    cfg = load_config(None)
    cfg.update({"name": "tiny", "seed": 3, "n_jobs": 1, "cache_dir": None})
    cfg["simulation"].update({"runs_per_class": 4, "duration": 8.0})
    cfg["evaluation"].update({"n_splits": 2, "n_repeats": 2, "bootstrap": 50, "rf_trees": 10})
    cfg["cnn"].update({"enabled": importlib.util.find_spec("torch") is not None, "epochs": 1})
    cfg["importance"].update({"n_permutations": 1})
    cfg["outputs"] = {"out_dir": str(tmp_path), "figures_dir": str(tmp_path / "fig"),
                      "results_md": str(tmp_path / "RESULTS.md")}
    res = run(cfg)
    assert res["dataset"]["n_windows"] == 12 * 2
    assert res["gonogo"]["final_verdict"] in ("GO", "NO-GO")
    assert res["gonogo"]["as_briefed"]["verdict"] in ("GO", "NO-GO")
    for k in ("baselines_union", "topo", "descriptor", "union"):
        s = res["summary"][k]
        assert 0.0 <= s["accuracy"] <= 1.0 and s["ci95"][0] <= s["ci95"][1]
    with open(tmp_path / "results.json") as fh:
        assert json.load(fh)["config"]["name"] == "tiny"
    for f in ("example_traces", "accuracy_bars", "margins", "confusion_matrix",
              "feature_importance"):
        assert os.path.exists(tmp_path / "fig" / f"{f}.png")
    text = (tmp_path / "RESULTS.md").read_text()
    assert "## Verdict" in text and "baselines_union" in text
