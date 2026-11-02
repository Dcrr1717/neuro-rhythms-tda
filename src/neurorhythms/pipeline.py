"""End-to-end benchmark: simulate -> features -> grouped CV -> statistics -> report.

Usage::

    python -m neurorhythms                         # small default config (CI)
    python -m neurorhythms --config configs/full.yaml

The Go/No-Go protocol follows the GEM benchmark: repeated StratifiedGroupKFold
(group = simulation run), LogReg and RandomForest per feature set with the
better of the two reported, paired group-bootstrap 95 % CIs and the rule
"GO iff margin >= go_margin (+0.05 accuracy)".
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import time

import numpy as np
import yaml

from . import plots
from .classical import check_folds, cross_validate, make_folds
from .dataset import RawRuns, measure, simulate_runs
from .features import (BRIEF_BASELINES, FEATURE_SETS, STRICT_BASELINES, TOPO_SETS,
                       assemble, compute_blocks)
from .features.topology import TopoConfig
from .importance import family_permutation_importance
from .regimes import CLASSES, DEFAULT_REGIMES
from .signatures import signature_table
from .stats import (accuracy_per_repeat, bootstrap_accuracy, confusion, macro_f1_per_repeat,
                    paired_bootstrap_margin, run_level_accuracy, verdict)

DEFAULT_CONFIG: dict = {
    "name": "small",
    "seed": 0,
    "n_jobs": 4,
    "cache_dir": "data/cache",
    "simulation": {"runs_per_class": 12, "duration": 24.0, "fs": 250.0, "dt": 5e-4,
                   "transient": 2.0},
    "measurement": {"window": 4.0, "pink_level": [0.2, 0.8], "white_level": [0.05, 0.2]},
    "topology": {},
    "evaluation": {"n_splits": 4, "n_repeats": 1, "bootstrap": 500, "rf_trees": 200,
                   "go_margin": 0.05},
    "cnn": {"enabled": True, "epochs": 10, "batch_size": 64, "lr": 1e-3,
            "weight_decay": 1e-4, "width": 16},
    "sanity": {"enabled": True},
    "importance": {"enabled": True, "n_permutations": 3},
    "noise_sweep": {"enabled": False, "scales": [2.0, 4.0], "n_repeats": 1,
                    "sets": ["bandpower+hjorth", "baselines_union", "topo", "descriptor", "union"],
                    "cnn": True},
    "outputs": {"out_dir": "results/small", "figures_dir": "results/small/figures",
                "results_md": None},
}


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path) as fh:
        return _merge(DEFAULT_CONFIG, yaml.safe_load(fh) or {})


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Data + features (cached)
# --------------------------------------------------------------------------
def get_raw(cfg) -> RawRuns:
    sim = cfg["simulation"]
    key = _hash({"sim": sim, "seed": cfg["seed"], "regimes": DEFAULT_REGIMES})
    path = os.path.join(cfg["cache_dir"], f"raw_{key}.npz") if cfg.get("cache_dir") else None
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        meta = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
        return RawRuns(eeg=z["eeg"], label=z["label"], fs=float(z["fs"]), meta=meta)
    t0 = time.time()
    raw = simulate_runs(sim["runs_per_class"], sim["duration"], cfg["seed"], fs=sim["fs"],
                        dt=sim["dt"], transient=sim["transient"])
    _log(f"simulated {len(raw.label)} runs x {sim['duration']} s in {time.time() - t0:.1f} s")
    if path:
        os.makedirs(cfg["cache_dir"], exist_ok=True)
        np.savez_compressed(path, eeg=raw.eeg, label=raw.label, fs=raw.fs,
                            **{f"meta_{k}": v for k, v in raw.meta.items()})
    return raw


def get_blocks(ds, cfg, tag: str):
    topo_cfg = TopoConfig(**{k: tuple(v) if isinstance(v, list) else v
                             for k, v in cfg["topology"].items()})
    key = _hash({"X": hashlib.sha1(ds.X.tobytes()).hexdigest(), "topo": topo_cfg.__dict__})
    path = (os.path.join(cfg["cache_dir"], f"features_{tag}_{key}.npz")
            if cfg.get("cache_dir") else None)
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        blocks = {}
        for b in ("bandpower", "hjorth", "waveform", "psd", "topo"):
            blocks[b] = (z[f"{b}_F"], list(z[f"{b}_names"]), list(z[f"{b}_fams"]))
        return blocks, topo_cfg
    t0 = time.time()
    blocks = compute_blocks(ds.X, ds.fs, topo_cfg, n_jobs=cfg["n_jobs"])
    _log(f"features for {len(ds.X)} windows in {time.time() - t0:.1f} s "
         f"({', '.join(f'{b}={v[0].shape[1]}' for b, v in blocks.items())})")
    if path:
        os.makedirs(cfg["cache_dir"], exist_ok=True)
        np.savez_compressed(path, **{f"{b}_{s}": np.asarray(v[i]) for b, v in blocks.items()
                                     for i, s in enumerate(("F", "names", "fams"))})
    return blocks, topo_cfg


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def evaluate_sets(blocks, ds, folds, cfg, sets) -> dict:
    ev = cfg["evaluation"]
    res = {}
    for s in sets:
        F, _, _ = assemble(blocks, s)
        res[s] = {}
        for model in ("logreg", "rf"):
            res[s][model] = cross_validate(F, ds.y, folds, model, seed=cfg["seed"],
                                           rf_trees=ev["rf_trees"], n_jobs=cfg["n_jobs"])
        a_lr = accuracy_per_repeat(res[s]["logreg"]["pred"], ds.y).mean()
        a_rf = accuracy_per_repeat(res[s]["rf"]["pred"], ds.y).mean()
        res[s]["best"] = "logreg" if a_lr >= a_rf else "rf"
        _log(f"  {s:>24s} [{F.shape[1]:4d} feats]  LR {a_lr:.3f}  RF {a_rf:.3f}")
    return res


def evaluate_cnn(ds, folds, cfg) -> dict:
    from .deep import cross_validate_cnn
    c = cfg["cnn"]
    out = cross_validate_cnn(ds.X, ds.y, folds, seed=cfg["seed"], epochs=c["epochs"],
                             batch_size=c["batch_size"], lr=c["lr"],
                             weight_decay=c["weight_decay"], width=c["width"],
                             threads=cfg["n_jobs"])
    _log(f"  {'cnn':>24s} [raw signal]  acc {accuracy_per_repeat(out['pred'], ds.y).mean():.3f}"
         f"  ({out['seconds']:.0f} s)")
    return out


def summarize(pred, proba, ds, n_boot, seed, extra=None) -> dict:
    b = bootstrap_accuracy(pred, ds.y, ds.groups, n_boot, seed)
    acc_r = accuracy_per_repeat(pred, ds.y)
    out = {"accuracy": b["accuracy"], "ci95": b["ci95"],
           "accuracy_repeat_std": float(acc_r.std()),
           "macro_f1": float(macro_f1_per_repeat(pred, ds.y).mean()),
           "run_level_accuracy": float(run_level_accuracy(proba, ds.y, ds.groups).mean())}
    out.update(extra or {})
    return out


def best_of(summary, names):
    n = max(names, key=lambda k: summary[k]["accuracy"])
    return n, summary[n]["accuracy"]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    t_start = time.time()
    ev = cfg["evaluation"]
    out_dir = cfg["outputs"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    seed, n_boot, go = cfg["seed"], ev["bootstrap"], ev["go_margin"]

    _log(f"config '{cfg['name']}': {cfg['simulation']['runs_per_class']} runs/class, "
         f"{ev['n_splits']} folds x {ev['n_repeats']} repeats")
    raw = get_raw(cfg)
    meas = cfg["measurement"]
    ds = measure(raw, meas["window"], seed + 1, meas["pink_level"], meas["white_level"])
    _log(f"dataset: {len(ds.y)} windows of {meas['window']} s from {ds.n_runs} runs")
    blocks, topo_cfg = get_blocks(ds, cfg, "main")

    folds = make_folds(ds.y, ds.groups, ev["n_splits"], ev["n_repeats"], seed)
    check_folds(folds, ds.groups)

    _log("grouped CV, classical models")
    res = evaluate_sets(blocks, ds, folds, cfg, list(FEATURE_SETS))
    preds = {s: res[s][res[s]["best"]]["pred"] for s in res}
    probas = {s: res[s][res[s]["best"]]["proba"] for s in res}

    summary = {}
    for s in res:
        lr_acc = float(accuracy_per_repeat(res[s]["logreg"]["pred"], ds.y).mean())
        rf_acc = float(accuracy_per_repeat(res[s]["rf"]["pred"], ds.y).mean())
        summary[s] = summarize(preds[s], probas[s], ds, n_boot, seed,
                               {"best_classifier": res[s]["best"], "logreg_accuracy": lr_acc,
                                "rf_accuracy": rf_acc,
                                "n_features": int(assemble(blocks, s)[0].shape[1])})

    cnn = None
    if cfg["cnn"]["enabled"]:
        _log("grouped CV, 1D-CNN")
        cnn = evaluate_cnn(ds, folds, cfg)
        preds["cnn"], probas["cnn"] = cnn["pred"], cnn["proba"]
        from .deep import n_parameters
        summary["cnn"] = summarize(cnn["pred"], cnn["proba"], ds, n_boot, seed,
                                   {"n_parameters": n_parameters(cfg["cnn"]["width"]),
                                    "seconds": cnn["seconds"]})

    # ---------------- Go / No-Go --------------------------------------------
    def cmp(a, b):
        m = paired_bootstrap_margin(preds[a], preds[b], ds.y, ds.groups, n_boot, seed)
        return {"candidate": a, "reference": b, "candidate_accuracy": summary[a]["accuracy"],
                "reference_accuracy": summary[b]["accuracy"], **m, **verdict(m["margin"], m["ci95"], go)}

    best_brief, _ = best_of(summary, BRIEF_BASELINES)
    best_strict, _ = best_of(summary, STRICT_BASELINES)
    union_acc = summary["baselines_union"]["accuracy"]
    comparisons = {
        "descriptor_vs_best_brief_baseline": cmp("descriptor", best_brief),
        "topo_vs_best_brief_baseline": cmp("topo", best_brief),
        "bandpower+hjorth+topo_vs_bandpower+hjorth": cmp("bandpower+hjorth+topo", "bandpower+hjorth"),
        "descriptor_vs_best_strict_baseline": cmp("descriptor", best_strict),
        "topo_vs_best_strict_baseline": cmp("topo", best_strict),
        "descriptor_vs_baselines_union": cmp("descriptor", "baselines_union"),
        "union_vs_baselines_union": cmp("union", "baselines_union"),
        "topo_vs_psd": cmp("topo", "psd"),
    }
    if cnn is not None:
        comparisons["cnn_vs_baselines_union"] = cmp("cnn", "baselines_union")
        best_classical, _ = best_of(summary, [s for s in summary if s != "cnn"])
        comparisons["cnn_vs_best_classical"] = cmp("cnn", best_classical)

    brief_rule = comparisons["descriptor_vs_best_brief_baseline"]
    strict_rows = {s: cmp(s, "baselines_union") for s in TOPO_SETS}
    strict_go = [s for s, r in strict_rows.items() if r["verdict"] == "GO"]
    gonogo = {
        "go_margin": go,
        "as_briefed": {"rule": "descriptor (psd+topo) vs best brief-named baseline "
                               "(bandpower / hjorth / bandpower+hjorth)",
                       "best_brief_baseline": best_brief, **brief_rule},
        "strict": {"rule": "any topology-containing set vs baselines_union "
                           "(all non-topological features)",
                   "baselines_union_accuracy": union_acc, "headroom": 1.0 - union_acc,
                   "go_arithmetically_possible": bool(1.0 - union_acc >= go),
                   "rows": strict_rows, "sets_reaching_go_margin": strict_go,
                   "verdict": "GO" if strict_go else "NO-GO"},
        "incremental_gain_over_baselines_union": comparisons["union_vs_baselines_union"],
    }
    gonogo["final_verdict"] = gonogo["strict"]["verdict"]
    _log(f"Go/No-Go: as briefed {brief_rule['verdict']} ({brief_rule['margin']:+.3f}), "
         f"strict {gonogo['strict']['verdict']} (headroom {1 - union_acc:.3f})")

    # ---------------- Sanity checks -----------------------------------------
    sanity = {}
    if cfg["sanity"]["enabled"]:
        _log("sanity checks")
        rng = np.random.default_rng(seed + 7)
        run_lab = np.zeros(ds.n_runs, dtype=int)
        run_lab[ds.groups] = ds.y
        y_perm = rng.permutation(run_lab)[ds.groups]
        Fu, _, _ = assemble(blocks, "union")
        fp = make_folds(y_perm, ds.groups, ev["n_splits"], 1, seed)
        r = cross_validate(Fu, y_perm, fp, "logreg", seed=seed)
        sanity["permuted_run_labels_union_logreg"] = float(np.mean(r["pred"][0] == y_perm))
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=ev["n_splits"], shuffle=True, random_state=seed)
        ung = [[(tr, te) for tr, te in skf.split(np.zeros(len(ds.y)), ds.y)]]
        for s in ("topo", "baselines_union"):
            F, _, _ = assemble(blocks, s)
            r = cross_validate(F, ds.y, ung, "rf", seed=seed, rf_trees=ev["rf_trees"],
                               n_jobs=cfg["n_jobs"])
            sanity[f"ungrouped_cv_rf_{s}"] = float(np.mean(r["pred"][0] == ds.y))
            sanity[f"grouped_cv_rf_{s}"] = float(
                accuracy_per_repeat(res[s]["rf"]["pred"][:1], ds.y)[0])
        _log(f"  {sanity}")

    # ---------------- Importance ---------------------------------------------
    importance = {}
    if cfg["importance"]["enabled"]:
        _log("feature-family permutation importance (union)")
        Fu, _, fams = assemble(blocks, "union")
        for m in ("logreg", "rf"):
            importance[m] = family_permutation_importance(
                Fu, ds.y, fams, folds[0], m, cfg["importance"]["n_permutations"], seed,
                ev["rf_trees"], cfg["n_jobs"])

    # ---------------- Exploratory noise sweep --------------------------------
    sweep = None
    ns = cfg["noise_sweep"]
    if ns["enabled"]:
        _log("exploratory noise sweep")
        sweep = {"scales": [1.0] + list(ns["scales"]), "n_repeats": ns["n_repeats"],
                 "sets": {}, "margins_vs_baselines_union": {}}
        sets = list(ns["sets"]) + (["cnn"] if ns["cnn"] and cfg["cnn"]["enabled"] else [])
        for s in sets:
            r = summary[s]
            sweep["sets"][s] = [(r["accuracy"], r["ci95"][0], r["ci95"][1])]
        for s in ns["sets"]:
            if s != "baselines_union":
                m = paired_bootstrap_margin(preds[s], preds["baselines_union"], ds.y, ds.groups,
                                            n_boot, seed)
                sweep["margins_vs_baselines_union"][s] = [(m["margin"], *m["ci95"])]
        for scale in ns["scales"]:
            _log(f" noise scale x{scale}")
            dsn = measure(raw, meas["window"], seed + 1, meas["pink_level"],
                          meas["white_level"], noise_scale=scale)
            bn, _ = get_blocks(dsn, cfg, f"noise{scale:g}")
            fn = make_folds(dsn.y, dsn.groups, ev["n_splits"], ns["n_repeats"], seed)
            rn = evaluate_sets(bn, dsn, fn, cfg, ns["sets"])
            pn = {s: rn[s][rn[s]["best"]]["pred"] for s in rn}
            if "cnn" in sets:
                pn["cnn"] = evaluate_cnn(dsn, fn, cfg)["pred"]
            for s in sets:
                b = bootstrap_accuracy(pn[s], dsn.y, dsn.groups, n_boot, seed)
                sweep["sets"][s].append((b["accuracy"], *b["ci95"]))
            for s in ns["sets"]:
                if s != "baselines_union":
                    m = paired_bootstrap_margin(pn[s], pn["baselines_union"], dsn.y,
                                                dsn.groups, n_boot, seed)
                    sweep["margins_vs_baselines_union"][s].append((m["margin"], *m["ci95"]))

    # ---------------- Persist -------------------------------------------------
    conf = {"best_brief_baseline": confusion(preds[best_brief], ds.y).tolist(),
            "union": confusion(preds["union"], ds.y).tolist()}
    if cnn is not None:
        conf["cnn"] = confusion(preds["cnn"], ds.y).tolist()
    results = {
        "config": cfg,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "dataset": {"n_windows": int(len(ds.y)), "n_runs": ds.n_runs,
                    "windows_per_run": int(len(ds.y) // ds.n_runs), "classes": list(CLASSES),
                    "fs": ds.fs, "window_s": meas["window"],
                    "class_counts": np.bincount(ds.y).tolist(),
                    "beta_onset_freq_hz_parkinsonian": {
                        "min": float(raw.meta["bg_f_hopf"][raw.label == 2].min()),
                        "max": float(raw.meta["bg_f_hopf"][raw.label == 2].max())}},
        "regimes": DEFAULT_REGIMES,
        "signatures": signature_table(raw.eeg, raw.label, raw.fs, CLASSES),
        "topology_config": {k: list(v) if isinstance(v, tuple) else v
                            for k, v in topo_cfg.__dict__.items()},
        "summary": summary,
        "comparisons": comparisons,
        "gonogo": gonogo,
        "sanity": sanity,
        "importance": importance,
        "confusion": conf,
        "noise_sweep": sweep,
        "runtime_s": time.time() - t_start,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=_json_default)
    np.savez_compressed(os.path.join(out_dir, "oof_predictions.npz"), y=ds.y, groups=ds.groups,
                        **{f"pred_{k}": v for k, v in preds.items()})

    # ---------------- Figures + report -----------------------------------------
    render(results, ds)
    _log(f"done in {time.time() - t_start:.0f} s -> {out_dir}")
    return results


def render(results: dict, ds) -> None:
    """Figures + Markdown report from a results dict (also used by ``--replot``)."""
    from .report import write_report
    cfg = results["config"]
    fig_dir = cfg["outputs"]["figures_dir"]
    go = cfg["evaluation"]["go_margin"]
    summary, comparisons, conf = results["summary"], results["comparisons"], results["confusion"]
    _log("figures")
    plots.example_traces(ds.X, ds.y, ds.groups, ds.fs, CLASSES,
                         os.path.join(fig_dir, "example_traces.png"), seed=cfg["seed"])
    plots.accuracy_bars(summary, "baselines_union", go, os.path.join(fig_dir, "accuracy_bars.png"))
    rows = [(f"{v['candidate']}  vs  {v['reference']}", v["margin"], v["ci95"])
            for v in comparisons.values()]
    plots.margin_forest(rows, go, os.path.join(fig_dir, "margins.png"))
    bb = results["gonogo"]["as_briefed"]["best_brief_baseline"]
    mats = {f"best brief baseline: {bb} ({summary[bb]['best_classifier']})":
            np.array(conf["best_brief_baseline"]),
            f"union ({summary['union']['best_classifier']})": np.array(conf["union"])}
    if "cnn" in conf:
        mats["1D-CNN (raw signal)"] = np.array(conf["cnn"])
    plots.confusion_panels(mats, CLASSES, os.path.join(fig_dir, "confusion_matrix.png"))
    if results.get("importance"):
        plots.importance_bars(results["importance"],
                              os.path.join(fig_dir, "feature_importance.png"))
    if results.get("noise_sweep"):
        plots.noise_sweep_plot(results["noise_sweep"], os.path.join(fig_dir, "noise_sweep.png"))
    md_path = cfg["outputs"].get("results_md") or os.path.join(cfg["outputs"]["out_dir"],
                                                                "RESULTS.md")
    write_report(results, md_path, fig_dir)
    _log(f"report -> {md_path}")


def replot(cfg: dict) -> None:
    """Re-render figures and report from an existing results.json (no re-training)."""
    with open(os.path.join(cfg["outputs"]["out_dir"], "results.json")) as fh:
        results = json.load(fh)
    results["config"]["outputs"] = cfg["outputs"]
    rc = results["config"]
    raw = get_raw({**rc, "cache_dir": cfg.get("cache_dir")})
    meas = rc["measurement"]
    ds = measure(raw, meas["window"], rc["seed"] + 1, meas["pink_level"], meas["white_level"])
    if "signatures" not in results:
        results["signatures"] = signature_table(raw.eeg, raw.label, raw.fs, CLASSES)
        with open(os.path.join(cfg["outputs"]["out_dir"], "results.json"), "w") as fh:
            json.dump(results, fh, indent=1, default=_json_default)
    render(results, ds)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None, help="YAML config (default: built-in small)")
    ap.add_argument("--out-dir", default=None, help="override outputs.out_dir")
    ap.add_argument("--figures-dir", default=None, help="override outputs.figures_dir")
    ap.add_argument("--results-md", default=None, help="override outputs.results_md")
    ap.add_argument("--no-cache", action="store_true", help="do not read/write the cache")
    ap.add_argument("--no-cnn", action="store_true", help="skip the deep model")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--replot", action="store_true",
                    help="only re-render figures + report from an existing results.json")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.out_dir:
        cfg["outputs"]["out_dir"] = a.out_dir
        if not a.figures_dir:
            cfg["outputs"]["figures_dir"] = os.path.join(a.out_dir, "figures")
        if not a.results_md:
            cfg["outputs"]["results_md"] = None
    if a.figures_dir:
        cfg["outputs"]["figures_dir"] = a.figures_dir
    if a.results_md:
        cfg["outputs"]["results_md"] = a.results_md
    if a.no_cache:
        cfg["cache_dir"] = None
    if a.no_cnn:
        cfg["cnn"]["enabled"] = False
    if a.n_jobs:
        cfg["n_jobs"] = a.n_jobs
    if a.replot:
        replot(cfg)
    else:
        run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
