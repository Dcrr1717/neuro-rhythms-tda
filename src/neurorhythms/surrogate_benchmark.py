"""IAAFT surrogate benchmark: can topology see nonlinear temporal structure
that the spectrum and the amplitude distribution cannot?

Design (mirrors the GEM hard benchmark, `Dcrr1717/gem-geophysical-descriptor`):

* The same 240 simulation runs as the main benchmark (same seed and
  measurement model). Each run is cut into 4 s windows whose start is shifted
  by up to 0.4 s so that the window's ends join smoothly (value and slope);
  this end-matching avoids the end-point artefact of Fourier surrogates
  (Schreiber & Schmitz 2000).
* Every window gets its own IAAFT surrogate: same periodogram, same amplitude
  histogram, random phases. Primary variant ``final="spectrum"`` (periodogram
  exact, histogram to ~1e-3 std); robustness variant ``final="amplitude"``
  (histogram exact, periodogram to ~1e-3).
* Tasks: original vs surrogate pooled over the three rhythms (primary), the
  same per rhythm, a 6-class task (rhythm x original/surrogate) and the
  histogram-exact robustness variant.
* Grouped CV: a window, its surrogate and every other window of the same run
  share a group, so they are always on the same side of a split.
* Validity: spectral features computed from the full-window periodogram,
  one-point statistics and Hjorth parameters are blind by construction and
  must score chance.
* Verdict (GEM rule): GO iff a topology-containing feature set beats the union
  of all non-topological features (including the classical nonlinear
  statistics: time-reversal asymmetry, third-order autocorrelation, increment
  flatness, waveform sharpness) by >= +0.05 accuracy.

Usage::

    python -m neurorhythms.surrogate_benchmark                                # small config
    python -m neurorhythms.surrogate_benchmark --config configs/surrogate_full.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from types import SimpleNamespace

import numpy as np
import yaml

from . import plots
from .classical import check_folds, cross_validate, make_folds
from .dataset import measure_continuous, zscore_rows
from .features import summary_block
from .features.baseline import bandpower_features, psd_features
from .features.nonlinear import nonlinear_features, onepoint_features
from .importance import family_permutation_importance
from .pipeline import _json_default, _log, _merge, get_blocks, get_raw
from .regimes import CLASSES
from .stats import (accuracy_per_repeat, bootstrap_accuracy, confusion, macro_f1_per_repeat,
                    paired_bootstrap_margin, run_level_accuracy, verdict)
from .surrogates import iaaft, surrogate_quality

# --------------------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------------------
FEATURE_SETS = {
    # blind by construction (validity checks): functions of what IAAFT preserves
    "periodogram_bandpower": ("periodogram_bandpower",),
    "periodogram_psd": ("periodogram_psd",),
    "onepoint": ("onepoint",),
    "hjorth": ("hjorth",),
    # the brief-named EEG baselines, with the usual Welch estimator
    "bandpower": ("bandpower",),
    "bandpower+hjorth": ("bandpower", "hjorth"),
    "psd": ("psd",),
    # classical nonlinear time-series statistics (the strict competitor)
    "nonlinear": ("nonlinear",),
    "baselines_union": ("bandpower", "hjorth", "psd", "onepoint", "nonlinear"),
    # topology
    "topo": ("topo",),
    "topo_summary": ("topo_summary",),
    "descriptor": ("psd", "topo"),
    "union": ("bandpower", "hjorth", "psd", "onepoint", "nonlinear", "topo"),
    "union_summary": ("bandpower", "hjorth", "psd", "onepoint", "nonlinear", "topo_summary"),
}
VALIDITY_SETS = ("periodogram_bandpower", "periodogram_psd", "onepoint", "hjorth")
BRIEF_SETS = ("bandpower", "hjorth", "bandpower+hjorth")
STRICT_SETS = ("bandpower", "hjorth", "bandpower+hjorth", "psd", "onepoint", "nonlinear",
               "baselines_union")
TOPO_SETS = ("topo", "topo_summary", "descriptor", "union", "union_summary")
KEY_SETS = ("periodogram_psd", "onepoint", "bandpower+hjorth", "nonlinear", "baselines_union",
            "topo", "topo_summary", "union", "union_summary")

DEFAULT_CONFIG: dict = {
    "name": "surrogate_small",
    "seed": 0,
    "n_jobs": 4,
    "cache_dir": "data/cache",
    "simulation": {"runs_per_class": 12, "duration": 24.0, "fs": 250.0, "dt": 5e-4,
                   "transient": 2.0},
    "measurement": {"window": 4.0, "pink_level": [0.2, 0.8], "white_level": [0.05, 0.2]},
    "surrogate": {"n_iter": 100, "end_match_search": 0.4},
    "topology": {},
    "evaluation": {"n_splits": 4, "bootstrap": 500, "rf_trees": 200, "go_margin": 0.05},
    "cnn": {"epochs": 8, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4, "width": 16},
    "importance": {"enabled": True, "n_permutations": 2},
    "tasks": [
        {"name": "binary_pooled", "target": "binary", "classes": "all", "surrogate": "spectrum",
         "n_repeats": 1, "cnn": True, "sets": "all", "primary": True},
        {"name": "six_class", "target": "six", "classes": "all", "surrogate": "spectrum",
         "n_repeats": 1, "cnn": False, "sets": "key"},
    ],
    "outputs": {"out_dir": "results/surrogate_small",
                "figures_dir": "results/surrogate_small/figures", "results_md": None},
}


def load_config(path: str | None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path) as fh:
        cfg = _merge(DEFAULT_CONFIG, yaml.safe_load(fh) or {})
    return cfg


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def end_matched_windows(x: np.ndarray, fs: float, window: float, search: float):
    """Non-overlapping windows whose start is chosen within ``search`` seconds so
    that the periodic continuation of the window is smooth (Schreiber & Schmitz
    2000). Returns (windows, run index, start sample)."""
    n_runs, T = x.shape
    W = int(round(window * fs))
    S = max(1, int(round(search * fs)))
    step = W + S
    n_win = (T - S) // step
    if n_win < 1:
        raise ValueError("runs too short for one end-matched window")
    out, run, starts = [], [], []
    d = np.diff(x, axis=1)
    for r in range(n_runs):
        for k in range(n_win):
            s0 = np.arange(k * step, k * step + S)
            cost = (x[r, s0] - x[r, s0 + W]) ** 2 + (d[r, s0] - d[r, s0 + W]) ** 2
            s = int(s0[np.argmin(cost)])
            out.append(x[r, s:s + W])
            run.append(r)
            starts.append(s)
    return np.array(out), np.array(run), np.array(starts)


def build_surrogate_data(cfg: dict) -> dict:
    """Original windows and their IAAFT surrogates (both variants)."""
    raw = get_raw(cfg)
    meas = cfg["measurement"]
    x, _, _ = measure_continuous(raw, cfg["seed"] + 1, meas["pink_level"], meas["white_level"])
    Xw, run, starts = end_matched_windows(x, raw.fs, meas["window"],
                                          cfg["surrogate"]["end_match_search"])
    n_iter = cfg["surrogate"]["n_iter"]
    rng = np.random.default_rng(cfg["seed"] + 101)
    t0 = time.time()
    S_spec = iaaft(Xw, rng, n_iter, "spectrum")
    S_amp = iaaft(Xw, rng, n_iter, "amplitude")
    _log(f"IAAFT surrogates for {len(Xw)} windows x 2 variants in {time.time() - t0:.1f} s")
    q_spec, q_amp = surrogate_quality(Xw, S_spec), surrogate_quality(Xw, S_amp)
    mism = np.abs(Xw[:, 0] - Xw[:, -1]) / Xw.std(axis=1)
    quality = {
        "spectrum": {k: _q(v) for k, v in q_spec.items()},
        "amplitude": {k: _q(v) for k, v in q_amp.items()},
        "end_mismatch_sd": _q(mism),
    }
    return {"raw": raw, "fs": raw.fs, "run": run, "label": raw.label[run], "starts": starts,
            "X": {"original": zscore_rows(Xw).astype(np.float32),
                  "spectrum": zscore_rows(S_spec).astype(np.float32),
                  "amplitude": zscore_rows(S_amp).astype(np.float32)},
            "quality": quality}


def _q(v):
    v = np.asarray(v, dtype=float)
    return {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)),
            "max": float(np.max(np.abs(v)))}


def compute_all_blocks(data: dict, cfg: dict) -> dict:
    out = {}
    for variant, X in data["X"].items():
        b, _ = get_blocks(SimpleNamespace(X=X, fs=data["fs"]), cfg, f"surr_{variant}")
        b = dict(b)
        Xd = X.astype(float)
        F, n = bandpower_features(Xd, data["fs"], method="periodogram")
        b["periodogram_bandpower"] = (F, n, ["periodogram_bandpower"] * len(n))
        F, n = psd_features(Xd, data["fs"], method="periodogram")
        b["periodogram_psd"] = (F, n, ["periodogram_psd"] * len(n))
        F, n = onepoint_features(Xd)
        b["onepoint"] = (F, n, ["onepoint"] * len(n))
        F, n = nonlinear_features(Xd)
        b["nonlinear"] = (F, n, ["nonlinear"] * len(n))
        b["topo_summary"] = summary_block(b["topo"])
        out[variant] = b
    return out


def assemble(blocks: dict, name: str, rows):
    parts = FEATURE_SETS[name]
    F = np.concatenate([blocks[p][0][rows] for p in parts], axis=1)
    fams = sum((list(blocks[p][2]) for p in parts), [])
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0), fams


# --------------------------------------------------------------------------
# One task
# --------------------------------------------------------------------------
def task_arrays(task: dict, data: dict):
    cls = list(range(len(CLASSES))) if task["classes"] == "all" else [
        CLASSES.index(c) for c in task["classes"]]
    rows = np.nonzero(np.isin(data["label"], cls))[0]
    n = len(rows)
    variant = np.r_[np.zeros(n, int), np.ones(n, int)]
    lab = np.r_[data["label"][rows], data["label"][rows]]
    if task["target"] == "binary":
        y, names = variant, ["original", "surrogate"]
    elif task["target"] == "six":
        y = 2 * lab + variant
        names = [f"{c}/{v}" for c in CLASSES for v in ("orig", "surr")]
    else:
        raise ValueError(task["target"])
    groups = np.r_[data["run"][rows], data["run"][rows]]
    units = 2 * groups + variant          # run-level unit: (run, original/surrogate)
    X = np.vstack([data["X"]["original"][rows], data["X"][task["surrogate"]][rows]])
    return rows, y, names, groups, units, X


def run_task(task: dict, data: dict, blocks: dict, cfg: dict) -> dict:
    ev = cfg["evaluation"]
    seed, n_boot, go = cfg["seed"], ev["bootstrap"], ev["go_margin"]
    rows, y, names, groups, units, X = task_arrays(task, data)
    k = len(names)
    sets = (list(FEATURE_SETS) if task.get("sets", "key") == "all"
            else [s for s in KEY_SETS] if task.get("sets", "key") == "key" else list(task["sets"]))
    folds = make_folds(y, groups, ev["n_splits"], task["n_repeats"], seed)
    check_folds(folds, groups)
    _log(f"task {task['name']}: {len(y)} windows, {len(np.unique(groups))} runs, "
         f"{k} classes, {ev['n_splits']} folds x {task['n_repeats']} repeats")
    bo, bs = blocks["original"], blocks[task["surrogate"]]
    preds, probas, summary, fams_union = {}, {}, {}, None
    for s in sets:
        Fo, fams = assemble(bo, s, rows)
        Fs, _ = assemble(bs, s, rows)
        F = np.vstack([Fo, Fs])
        if s == "union":
            fams_union = fams
        res = {m: cross_validate(F, y, folds, m, seed=seed, rf_trees=ev["rf_trees"],
                                 n_jobs=cfg["n_jobs"], n_classes=k) for m in ("logreg", "rf")}
        acc = {m: float(accuracy_per_repeat(res[m]["pred"], y).mean()) for m in res}
        best = max(acc, key=acc.get)
        preds[s], probas[s] = res[best]["pred"], res[best]["proba"]
        summary[s] = _summ(preds[s], probas[s], y, groups, units, n_boot, seed,
                           {"best_classifier": best, "logreg_accuracy": acc["logreg"],
                            "rf_accuracy": acc["rf"], "n_features": int(F.shape[1])})
        _log(f"  {s:>22s} [{F.shape[1]:4d}]  LR {acc['logreg']:.3f}  RF {acc['rf']:.3f}")
    if task.get("cnn"):
        from .deep import cross_validate_cnn
        c = cfg["cnn"]
        r = cross_validate_cnn(X, y, folds, seed=seed, n_classes=k, epochs=c["epochs"],
                               batch_size=c["batch_size"], lr=c["lr"],
                               weight_decay=c["weight_decay"], width=c["width"],
                               threads=cfg["n_jobs"])
        preds["cnn"], probas["cnn"] = r["pred"], r["proba"]
        summary["cnn"] = _summ(r["pred"], r["proba"], y, groups, units, n_boot, seed,
                               {"seconds": r["seconds"]})
        _log(f"  {'cnn':>22s} [raw]   acc {summary['cnn']['accuracy']:.3f} ({r['seconds']:.0f} s)")

    def cmp(a, b):
        m = paired_bootstrap_margin(preds[a], preds[b], y, groups, n_boot, seed)
        return {"candidate": a, "reference": b, "candidate_accuracy": summary[a]["accuracy"],
                "reference_accuracy": summary[b]["accuracy"], **m,
                **verdict(m["margin"], m["ci95"], go)}

    def best(names_):
        av = [n for n in names_ if n in summary]
        return max(av, key=lambda n: summary[n]["accuracy"]) if av else None

    out = {"task": task, "n_windows": int(len(y)), "n_runs": int(len(np.unique(groups))),
           "labels": names, "chance": 1.0 / k, "summary": summary, "comparisons": {}}
    comps = out["comparisons"]
    vb = best(VALIDITY_SETS)
    if vb:
        out["validity"] = {"best_blind_set": vb, "accuracy": summary[vb]["accuracy"],
                           "ci95": summary[vb]["ci95"], "chance": 1.0 / k,
                           "all": {s: summary[s]["accuracy"] for s in VALIDITY_SETS if s in summary}}
    bb = best(BRIEF_SETS)
    if bb and "descriptor" in summary:
        comps["descriptor_vs_best_brief"] = cmp("descriptor", bb)
    if bb and "topo" in summary:
        comps["topo_vs_best_brief"] = cmp("topo", bb)
    for s in ("topo", "topo_summary"):
        if s in summary and "nonlinear" in summary:
            comps[f"{s}_vs_nonlinear"] = cmp(s, "nonlinear")
    strict = {s: cmp(s, "baselines_union") for s in TOPO_SETS if s in summary}
    for s, r in strict.items():
        comps[f"{s}_vs_baselines_union"] = r
    if "cnn" in summary:
        comps["cnn_vs_baselines_union"] = cmp("cnn", "baselines_union")
    u = summary["baselines_union"]["accuracy"]
    go_sets = [s for s, r in strict.items() if r["verdict"] == "GO"]
    best_strict = max(strict.values(), key=lambda r: r["margin"])
    out["gonogo"] = {
        "go_margin": go,
        "as_briefed": comps.get("descriptor_vs_best_brief"),
        "strict": {"baselines_union_accuracy": u, "headroom": 1.0 - u,
                   "go_arithmetically_possible": bool(1.0 - u >= go),
                   "best": best_strict, "sets_reaching_go_margin": go_sets,
                   "verdict": "GO" if go_sets else "NO-GO"},
        "incremental": strict.get("union"),
    }
    if "union" in preds:
        out["confusion"] = {"union": confusion(preds["union"], y, k).tolist()}
        if "cnn" in preds:
            out["confusion"]["cnn"] = confusion(preds["cnn"], y, k).tolist()
    if task.get("primary") and cfg["importance"]["enabled"] and fams_union is not None:
        Fo, _ = assemble(bo, "union", rows)
        Fs, _ = assemble(bs, "union", rows)
        F = np.vstack([Fo, Fs])
        out["importance"] = {m: family_permutation_importance(
            F, y, fams_union, folds[0], m, cfg["importance"]["n_permutations"], seed,
            ev["rf_trees"], cfg["n_jobs"]) for m in ("logreg", "rf")}
    _log(f"  verdict {task['name']}: strict {out['gonogo']['strict']['verdict']} "
         f"(best {best_strict['candidate']} {best_strict['margin']:+.3f}, "
         f"headroom {1 - u:.3f})")
    return out


def _summ(pred, proba, y, groups, units, n_boot, seed, extra):
    b = bootstrap_accuracy(pred, y, groups, n_boot, seed)
    out = {"accuracy": b["accuracy"], "ci95": b["ci95"],
           "macro_f1": float(macro_f1_per_repeat(pred, y).mean()),
           "unit_level_accuracy": float(run_level_accuracy(proba, y, units).mean())}
    out.update(extra)
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    t0 = time.time()
    out_dir = cfg["outputs"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    _log(f"surrogate benchmark '{cfg['name']}'")
    data = build_surrogate_data(cfg)
    _log(f"{len(data['run'])} end-matched windows from {len(data['raw'].label)} runs; "
         f"quality {json.dumps(data['quality'], default=float)[:300]}")
    blocks = compute_all_blocks(data, cfg)
    tasks = {}
    for task in cfg["tasks"]:
        tasks[task["name"]] = run_task(task, data, blocks, cfg)
    primary = next((t["name"] for t in cfg["tasks"] if t.get("primary")), cfg["tasks"][0]["name"])
    results = {
        "config": cfg,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "data": {"n_runs": int(len(data["raw"].label)), "n_windows": int(len(data["run"])),
                 "windows_per_run": int(len(data["run"]) // len(data["raw"].label)),
                 "quality": data["quality"]},
        "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()},
        "primary_task": primary,
        "tasks": tasks,
        "final_verdict": tasks[primary]["gonogo"]["strict"]["verdict"],
        "runtime_s": time.time() - t0,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=_json_default)
    render(results, data)
    _log(f"done in {time.time() - t0:.0f} s -> {out_dir}")
    return results


def render(results: dict, data: dict) -> None:
    from .surrogate_report import write_surrogate_report
    cfg = results["config"]
    fig = cfg["outputs"]["figures_dir"]
    go = cfg["evaluation"]["go_margin"]
    P = results["tasks"][results["primary_task"]]
    plots.surrogate_examples(data["X"]["original"], data["X"]["spectrum"], data["label"],
                             data["run"], data["fs"], CLASSES,
                             os.path.join(fig, "surrogate_examples.png"), seed=cfg["seed"])
    plots.accuracy_bars(P["summary"], "baselines_union", go,
                        os.path.join(fig, "surrogate_accuracy_bars.png"), chance=P["chance"])
    rows = [(f"{c['candidate']}  vs  {c['reference']}", c["margin"], c["ci95"])
            for c in P["comparisons"].values()]
    plots.margin_forest(rows, go, os.path.join(fig, "surrogate_margins.png"))
    plots.task_overview(results["tasks"], os.path.join(fig, "surrogate_tasks.png"))
    six = results["tasks"].get("six_class")
    if six and six.get("confusion"):
        mats = {f"union ({six['summary']['union']['best_classifier']})":
                np.array(six["confusion"]["union"])}
        if "cnn" in six["confusion"]:
            mats["1D-CNN"] = np.array(six["confusion"]["cnn"])
        plots.confusion_panels(mats, six["labels"],
                               os.path.join(fig, "surrogate_confusion_six_class.png"))
    if P.get("importance"):
        plots.importance_bars(P["importance"], os.path.join(fig, "surrogate_importance.png"))
    md = cfg["outputs"].get("results_md") or os.path.join(cfg["outputs"]["out_dir"], "RESULTS.md")
    write_surrogate_report(results, md, fig)
    _log(f"report -> {md}")


def replot(cfg: dict) -> None:
    with open(os.path.join(cfg["outputs"]["out_dir"], "results.json")) as fh:
        results = json.load(fh)
    results["config"]["outputs"] = cfg["outputs"]
    rc = dict(results["config"])
    rc["cache_dir"] = cfg.get("cache_dir")
    render(results, build_surrogate_data(rc))


def main(argv=None):
    ap = argparse.ArgumentParser(description="IAAFT surrogate benchmark")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--replot", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.out_dir:
        cfg["outputs"] = {"out_dir": a.out_dir, "figures_dir": os.path.join(a.out_dir, "figures"),
                          "results_md": None}
    if a.no_cache:
        cfg["cache_dir"] = None
    if a.no_cnn:
        for t in cfg["tasks"]:
            t["cnn"] = False
    if a.n_jobs:
        cfg["n_jobs"] = a.n_jobs
    replot(cfg) if a.replot else run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
