"""Second benchmark: SW1PerS topology and a Riemannian SOTA baseline, single- and multi-channel.

The first benchmark (``pipeline.py``) found NO-GO: sublevel-set and Alpha-
complex persistence of the raw trace re-encode what band power already
measures. This benchmark asks the fair questions:

1. Does **SW1PerS** (sliding-window embedding + 1-persistence, the topological
   method designed for signals; Perea & Harer 2015) add value *over band power*?
2. Does the honest state-of-the-art baseline, **Riemannian tangent-space
   features of SPD channel covariances** (``pyriemann``), beat band power and
   the existing topological-spectral descriptor?
3. Does adding topology to the Riemannian features give any incremental value?

Two tracks share one protocol (repeated StratifiedGroupKFold, group =
simulation run; LogReg and RandomForest, the better of the two per feature
set; paired group-bootstrap CIs; GO iff margin >= +0.05 accuracy):

``single``  the original single-channel dataset (same simulator, same seed
            convention as ``pipeline.py``, so with ``configs/sw1pers_full.yaml``
            it is *the* dataset of ``docs/RESULTS.md``);
``multi``   a 6-channel montage of 4 coupled Jansen-Rit columns sharing one
            STN-GPe loop (``network.py``), so cross-channel structure exists.

Usage::

    python -m neurorhythms.sw1pers_bench                                   # small (CI)
    python -m neurorhythms.sw1pers_bench --config configs/sw1pers_full.yaml
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import time
from functools import partial

import numpy as np
import yaml

from .classical import check_folds, cross_validate, make_folds
from .dataset import RawNetworkRuns, measure, measure_multichannel, simulate_network_runs
from .features import compute_blocks
from .features.riemann import TangentSpaceBlock, covariance_block
from .features.sw1pers import SW1PersConfig, sw1pers_features
from .features.topology import TopoConfig
from .importance import family_permutation_importance
from .pipeline import _hash, _json_default, _log, best_of, get_raw, summarize
from .regimes import CLASSES, DEFAULT_REGIMES, NETWORK_REGIMES
from .stats import (accuracy_per_repeat, confusion,
                    paired_bootstrap_margin, verdict)

TRACKS = ("single", "multi")

DEFAULT_CONFIG: dict = {
    "name": "sw1pers_small",
    "seed": 0,
    "n_jobs": 4,
    "cache_dir": "data/cache",
    "tracks": ["single", "multi"],
    "simulation": {"runs_per_class": 12, "duration": 24.0, "fs": 250.0, "dt": 5e-4,
                   "transient": 2.0},
    "measurement": {"window": 4.0, "pink_level": [0.2, 0.8], "white_level": [0.05, 0.2],
                    "common_fraction": [0.2, 0.5]},
    "network": {},          # overrides of regimes.NETWORK_REGIMES
    "topology": {},         # overrides of TopoConfig (the existing descriptor)
    "sw1pers": {},          # overrides of SW1PersConfig
    "evaluation": {"n_splits": 4, "n_repeats": 1, "bootstrap": 500, "rf_trees": 200,
                   "go_margin": 0.05},
    "cnn": {"enabled": True, "epochs": 10, "batch_size": 64, "lr": 1e-3,
            "weight_decay": 1e-4, "width": 16},
    "sanity": {"enabled": True},
    "importance": {"enabled": True, "n_permutations": 3},
    "outputs": {"out_dir": "results/sw1pers_small", "figures_dir": "results/sw1pers_small/figures",
                "results_md": None},
}

# --------------------------------------------------------------------------
# Feature sets (blocks are concatenated in this order; a covariance block,
# if any, must come first because the fold-local tangent space reads it there)
# --------------------------------------------------------------------------
BASE = ("bandpower", "hjorth", "waveform", "psd")
SW = ("sw1pers", "crocker")
SETS_COMMON = {
    "bandpower": ("bandpower",),
    "baselines_union": BASE,
    "topo": ("topo",),
    "descriptor": ("psd", "topo"),
    "sw1pers_core": ("sw1pers",),
    "sw1pers": SW,
    "bandpower+sw1pers": ("bandpower",) + SW,
    "baselines_union+sw1pers": BASE + SW,
}
SETS = {
    "single": {**SETS_COMMON, "union": BASE + ("topo",) + SW},
    "multi": {**SETS_COMMON,
              "riemann": ("riemann",),
              "riemann_broadband": ("riemann_bb",),
              "riemann+sw1pers": ("riemann",) + SW,
              "riemann+topo": ("riemann", "topo"),
              "riemann+baselines_union": ("riemann",) + BASE,
              "union": ("riemann",) + BASE + ("topo",) + SW},
}
COV_BLOCKS = ("riemann", "riemann_bb")

# Pre-specified comparisons: (key, candidate, reference, question, primary?)
QUESTIONS = {
    "Q1": "Does SW1PerS add value over band power? (the fair topology test)",
    "Q2": "Does Riemannian covariance beat band power and the existing descriptor?",
    "Q3": "Does adding topology to the Riemannian features give any incremental value?",
    "CNN": "Where does the 1D-CNN on the raw signal stand?",
}


def comparison_plan(track: str, has_cnn: bool) -> list[tuple]:
    rows = [
        ("Q1", "bandpower+sw1pers", "bandpower", True),
        ("Q1", "sw1pers", "bandpower", False),
        ("Q1", "baselines_union+sw1pers", "baselines_union", False),
        ("Q1", "sw1pers", "topo", False),
        ("Q1", "sw1pers_core", "bandpower", False),
    ]
    if track == "multi":
        rows += [
            ("Q2", "riemann", "bandpower", True),
            ("Q2", "riemann", "descriptor", True),
            ("Q2", "riemann", "baselines_union", False),
            ("Q2", "riemann_broadband", "bandpower", False),
            ("Q3", "riemann+sw1pers", "riemann", True),
            ("Q3", "riemann+topo", "riemann", False),
            ("Q3", "union", "riemann+baselines_union", False),
        ]
    else:
        rows += [("Q1", "union", "baselines_union", False)]
    if has_cnn:
        rows += [("CNN", "cnn", "bandpower", False)]
        rows += [("CNN", "cnn", "riemann", False)] if track == "multi" else []
        rows += [("CNN", "cnn", "__best_classical__", False)]
    return rows


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


def _tupled(d: dict) -> dict:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def network_regimes(cfg) -> dict:
    return {**NETWORK_REGIMES, **cfg.get("network", {})}


# --------------------------------------------------------------------------
# Data (cached)
# --------------------------------------------------------------------------
def get_network_raw(cfg) -> RawNetworkRuns:
    sim, net = cfg["simulation"], network_regimes(cfg)
    seed = cfg["seed"] + 100
    key = _hash({"sim": sim, "seed": seed, "regimes": DEFAULT_REGIMES, "network": net})
    path = os.path.join(cfg["cache_dir"], f"rawnet_{key}.npz") if cfg.get("cache_dir") else None
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        meta = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
        return RawNetworkRuns(src=z["src"], label=z["label"], fs=float(z["fs"]), meta=meta)
    t0 = time.time()
    raw = simulate_network_runs(sim["runs_per_class"], sim["duration"], seed, fs=sim["fs"],
                                dt=sim["dt"], transient=sim["transient"], network=net)
    _log(f"simulated {len(raw.label)} network runs ({raw.src.shape[1]} columns) x "
         f"{sim['duration']} s in {time.time() - t0:.1f} s")
    if path:
        os.makedirs(cfg["cache_dir"], exist_ok=True)
        np.savez_compressed(path, src=raw.src, label=raw.label, fs=raw.fs,
                            **{f"meta_{k}": v for k, v in raw.meta.items()})
    return raw


def get_dataset(cfg, track: str):
    meas = cfg["measurement"]
    if track == "single":
        raw = get_raw(cfg)
        ds = measure(raw, meas["window"], cfg["seed"] + 1, meas["pink_level"],
                     meas["white_level"])
    else:
        raw = get_network_raw(cfg)
        ds = measure_multichannel(raw, meas["window"], cfg["seed"] + 101, meas["pink_level"],
                                  meas["white_level"], meas["common_fraction"],
                                  network_regimes(cfg))
    return raw, ds


# --------------------------------------------------------------------------
# Features (cached)
# --------------------------------------------------------------------------
def _per_channel(block, n, c):
    F, names, fams = block
    F = F.reshape(n, c * F.shape[1])
    return F, [f"ch{i}_{nm}" for i in range(c) for nm in names], list(fams) * c


def compute_track_blocks(ds, cfg, track: str) -> dict:
    topo_cfg = TopoConfig(**_tupled(cfg["topology"]))
    sw_cfg = SW1PersConfig(**_tupled(cfg["sw1pers"]))
    nj = cfg["n_jobs"]
    t0 = time.time()
    if track == "single":
        blocks = compute_blocks(ds.X, ds.fs, topo_cfg, n_jobs=nj)
    else:
        n, c, w = ds.X.shape
        flat = compute_blocks(ds.X.reshape(n * c, w), ds.fs, topo_cfg, n_jobs=nj)
        blocks = {b: _per_channel(v, n, c) for b, v in flat.items()}
    _log(f"  [{track}] spectral + sublevel/Alpha topology in {time.time() - t0:.0f} s")
    t0 = time.time()
    F, names, fams = sw1pers_features(ds.X, ds.fs, sw_cfg, n_jobs=nj)
    fams = np.asarray(fams)
    for fam in ("sw1pers", "crocker"):
        m = fams == fam
        blocks[fam] = (F[:, m], [nm for nm, k in zip(names, m) if k], [fam] * int(m.sum()))
    _log(f"  [{track}] SW1PerS for {np.prod(ds.X.shape[:-1])} channel-windows in "
         f"{time.time() - t0:.0f} s")
    if track == "multi":
        blocks["riemann"] = covariance_block(ds.X, ds.fs, None, "cov")
        blocks["riemann_bb"] = covariance_block(ds.X, ds.fs, {}, "covbb")
    return blocks


def get_track_blocks(ds, cfg, track: str) -> dict:
    key = _hash({"X": hashlib.sha1(ds.X.tobytes()).hexdigest(), "track": track,
                 "topo": cfg["topology"], "sw": cfg["sw1pers"], "v": 1})
    path = (os.path.join(cfg["cache_dir"], f"features_sw1pers_{track}_{key}.npz")
            if cfg.get("cache_dir") else None)
    if path and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        bnames = sorted({k.rsplit("_", 1)[0] for k in z.files})
        return {b: (z[f"{b}_F"], list(z[f"{b}_names"]), list(z[f"{b}_fams"])) for b in bnames}
    blocks = compute_track_blocks(ds, cfg, track)
    if path:
        os.makedirs(cfg["cache_dir"], exist_ok=True)
        np.savez_compressed(path, **{f"{b}_{s}": np.asarray(v[i]) for b, v in blocks.items()
                                     for i, s in enumerate(("F", "names", "fams"))})
    return blocks


def assemble(blocks: dict, parts: tuple):
    F = np.concatenate([blocks[b][0] for b in parts], axis=1)
    names = sum((list(blocks[b][1]) for b in parts), [])
    fams = sum((list(blocks[b][2]) for b in parts), [])
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0), names, fams


def preprocessor(blocks: dict, parts: tuple, n_channels: int):
    """Fold-local tangent-space factory if the set starts with a covariance block."""
    if parts[0] not in COV_BLOCKS:
        if any(p in COV_BLOCKS for p in parts):
            raise ValueError("a covariance block must come first in a feature set")
        return None
    n_bands = blocks[parts[0]][0].shape[1] // n_channels ** 2
    return partial(TangentSpaceBlock, n_bands=n_bands, n_channels=n_channels)


def n_model_features(blocks, parts, n_channels) -> int:
    """Number of features the classifier sees (tangent vectors, not raw matrices)."""
    n = 0
    for b in parts:
        k = blocks[b][0].shape[1]
        if b in COV_BLOCKS:
            nb = k // n_channels ** 2
            k = nb * n_channels * (n_channels + 1) // 2
        n += k
    return n


# --------------------------------------------------------------------------
# Evaluation of one track
# --------------------------------------------------------------------------
def evaluate_track(cfg, track: str) -> tuple[dict, object, object]:
    ev = cfg["evaluation"]
    seed, n_boot, go = cfg["seed"], ev["bootstrap"], ev["go_margin"]
    raw, ds = get_dataset(cfg, track)
    n_ch = 1 if ds.X.ndim == 2 else ds.X.shape[1]
    _log(f"[{track}] dataset: {len(ds.y)} windows x {n_ch} channel(s) from {ds.n_runs} runs")
    blocks = get_track_blocks(ds, cfg, track)
    folds = make_folds(ds.y, ds.groups, ev["n_splits"], ev["n_repeats"], seed)
    check_folds(folds, ds.groups)
    sets = SETS[track]

    _log(f"[{track}] grouped CV, classical models")
    preds, probas, summary = {}, {}, {}
    for s, parts in sets.items():
        F, _, _ = assemble(blocks, parts)
        pre = preprocessor(blocks, parts, n_ch)
        res = {m: cross_validate(F, ds.y, folds, m, seed=seed, rf_trees=ev["rf_trees"],
                                 n_jobs=cfg["n_jobs"], pre=pre) for m in ("logreg", "rf")}
        acc = {m: float(accuracy_per_repeat(res[m]["pred"], ds.y).mean()) for m in res}
        best = "logreg" if acc["logreg"] >= acc["rf"] else "rf"
        preds[s], probas[s] = res[best]["pred"], res[best]["proba"]
        summary[s] = summarize(preds[s], probas[s], ds, n_boot, seed,
                               {"best_classifier": best, "logreg_accuracy": acc["logreg"],
                                "rf_accuracy": acc["rf"], "blocks": list(parts),
                                "n_features": n_model_features(blocks, parts, n_ch)})
        _log(f"  {s:>26s} [{summary[s]['n_features']:5d} feats]  LR {acc['logreg']:.3f}  "
             f"RF {acc['rf']:.3f}")

    if cfg["cnn"]["enabled"]:
        from .deep import cross_validate_cnn, n_parameters
        c = cfg["cnn"]
        out = cross_validate_cnn(ds.X, ds.y, folds, seed=seed, epochs=c["epochs"],
                                 batch_size=c["batch_size"], lr=c["lr"],
                                 weight_decay=c["weight_decay"], width=c["width"],
                                 threads=cfg["n_jobs"])
        preds["cnn"], probas["cnn"] = out["pred"], out["proba"]
        summary["cnn"] = summarize(out["pred"], out["proba"], ds, n_boot, seed,
                                   {"n_parameters": n_parameters(c["width"], n_ch),
                                    "seconds": out["seconds"]})
        _log(f"  {'cnn':>26s} [raw signal]  acc {summary['cnn']['accuracy']:.3f} "
             f"({out['seconds']:.0f} s)")

    # ---------------- pre-specified comparisons ------------------------------
    best_classical, _ = best_of(summary, [s for s in summary if s != "cnn"])
    comparisons = []
    for q, cand, ref, primary in comparison_plan(track, "cnn" in summary):
        ref = best_classical if ref == "__best_classical__" else ref
        if cand == ref or any(c["candidate"] == cand and c["reference"] == ref
                              for c in comparisons):
            continue
        m = paired_bootstrap_margin(preds[cand], preds[ref], ds.y, ds.groups, n_boot, seed)
        comparisons.append({"question": q, "primary": primary, "candidate": cand,
                            "reference": ref, "candidate_accuracy": summary[cand]["accuracy"],
                            "reference_accuracy": summary[ref]["accuracy"],
                            "headroom": 1.0 - summary[ref]["accuracy"], **m,
                            **verdict(m["margin"], m["ci95"], go)})

    # ---------------- sanity checks (leak detectors) --------------------------
    sanity = {}
    if cfg["sanity"]["enabled"]:
        rng = np.random.default_rng(seed + 7)
        run_lab = np.zeros(ds.n_runs, dtype=int)
        run_lab[ds.groups] = ds.y
        y_perm = rng.permutation(run_lab)[ds.groups]
        fp = make_folds(y_perm, ds.groups, ev["n_splits"], 1, seed)
        for s in ("union",) + (("riemann",) if track == "multi" else ()):
            F, _, _ = assemble(blocks, sets[s])
            r = cross_validate(F, y_perm, fp, "logreg", seed=seed, n_jobs=cfg["n_jobs"],
                               pre=preprocessor(blocks, sets[s], n_ch))
            sanity[f"permuted_run_labels_{s}_logreg"] = float(np.mean(r["pred"][0] == y_perm))
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=ev["n_splits"], shuffle=True, random_state=seed)
        ung = [[(tr, te) for tr, te in skf.split(np.zeros(len(ds.y)), ds.y)]]
        for s in ("bandpower", "sw1pers") + (("riemann",) if track == "multi" else ()):
            F, _, _ = assemble(blocks, sets[s])
            pre = preprocessor(blocks, sets[s], n_ch)
            r_u = cross_validate(F, ds.y, ung, "rf", seed=seed, rf_trees=ev["rf_trees"],
                                 n_jobs=cfg["n_jobs"], pre=pre)
            r_g = cross_validate(F, ds.y, [folds[0]], "rf", seed=seed, rf_trees=ev["rf_trees"],
                                 n_jobs=cfg["n_jobs"], pre=pre)
            sanity[f"ungrouped_cv_rf_{s}"] = float(np.mean(r_u["pred"][0] == ds.y))
            sanity[f"grouped_cv_rf_{s}"] = float(np.mean(r_g["pred"][0] == ds.y))
        _log(f"  [{track}] sanity {sanity}")

    # ---------------- importance (union) ---------------------------------------
    importance = {}
    if cfg["importance"]["enabled"]:
        F, _, fams = assemble(blocks, sets["union"])
        pre = preprocessor(blocks, sets["union"], n_ch)
        for m in ("logreg", "rf"):
            importance[m] = family_permutation_importance(
                F, ds.y, fams, folds[0], m, cfg["importance"]["n_permutations"], seed,
                ev["rf_trees"], cfg["n_jobs"], pre)

    conf_sets = ["bandpower", "sw1pers"] + (["riemann"] if track == "multi" else []) + ["union"]
    conf_sets += ["cnn"] if "cnn" in preds else []
    dataset_info = {"n_windows": int(len(ds.y)), "n_runs": ds.n_runs, "n_channels": n_ch,
                    "windows_per_run": int(len(ds.y) // ds.n_runs), "fs": ds.fs,
                    "window_s": cfg["measurement"]["window"],
                    "class_counts": np.bincount(ds.y).tolist()}
    if track == "multi":
        dataset_info["n_sources"] = int(raw.src.shape[1])
        dataset_info["source_signatures"] = network_signatures(raw)
    result = {"dataset": dataset_info, "summary": summary, "comparisons": comparisons,
              "best_classical": best_classical, "sanity": sanity, "importance": importance,
              "confusion": {s: confusion(preds[s], ds.y).tolist() for s in conf_sets},
              "post_hoc_signature_breakdown": signature_breakdown(
                  signature_holds(raw, track), ds.y, ds.groups, preds)}
    return result, ds, preds


def _network_signature_parts(raw: RawNetworkRuns):
    from .signatures import run_signatures
    n, c, t = raw.src.shape
    sig = run_signatures(raw.src.reshape(n * c, t), raw.fs)
    slow = sig["slow (1-7.5 Hz)"].reshape(n, c)
    beta = sig["beta (13-30 Hz)"].reshape(n, c)
    sk = sig["skewness"].reshape(n, c)
    spiking = (slow > 0.5) & (sk > 0.5)                 # epileptic column signature
    return beta, spiking


def network_signatures(raw: RawNetworkRuns) -> dict:
    """Descriptive, classifier-free check of the noise-free column outputs."""
    beta, spiking = _network_signature_parts(raw)
    n, c = beta.shape
    focus = np.asarray(raw.meta.get("focus", np.zeros((n, c), bool)), bool)
    out = {}
    for ci, cls in enumerate(CLASSES):
        sel = raw.label == ci
        row = {"n_runs": int(sel.sum()),
               "median_column_beta_fraction": float(np.median(beta[sel])),
               "fraction_runs_any_column_beta_ge_0.10": float(np.mean((beta[sel] >= 0.10).any(1))),
               "fraction_runs_any_spiking_column": float(np.mean(spiking[sel].any(1)))}
        if cls == "epileptic":
            row["fraction_focus_columns_spiking"] = float(np.mean(spiking[sel][focus[sel]]))
        out[cls] = row
    return out


def signature_holds(raw, track: str) -> np.ndarray:
    """Per run: does the noise-free model output show its own class signature?

    single: the criteria of ``signatures.CRITERIA``; multi: epileptic = some
    column spikes, parkinsonian = some column has beta fraction >= 0.10,
    healthy = neither.
    """
    if track == "single":
        from .signatures import run_signature_holds
        return run_signature_holds(raw.eeg, raw.label, raw.fs, CLASSES)
    beta, spiking = _network_signature_parts(raw)
    sp, bt = spiking.any(1), (beta >= 0.10).any(1)
    lab = np.asarray(raw.label)
    return np.where(lab == 0, ~sp & ~bt, np.where(lab == 1, sp, bt))


BREAKDOWN_SETS = ("bandpower", "baselines_union", "descriptor", "sw1pers", "bandpower+sw1pers",
                  "riemann", "riemann+sw1pers", "union", "cnn")


def signature_breakdown(holds_run: np.ndarray, y, groups, preds: dict) -> dict:
    """Post hoc: accuracy on runs that show their class signature vs borderline runs."""
    y, groups = np.asarray(y), np.asarray(groups)
    hw = np.asarray(holds_run, bool)[groups]
    run_lab = np.zeros(len(holds_run), dtype=int)
    run_lab[groups] = y
    out = {"n_runs_signature_holds": [int(np.sum(holds_run & (run_lab == c))) for c in range(3)],
           "n_runs_borderline": [int(np.sum(~holds_run & (run_lab == c))) for c in range(3)],
           "sets": {}}
    for s in BREAKDOWN_SETS:
        if s not in preds:
            continue
        P = np.atleast_2d(preds[s])
        ok = P == y[None, :]
        row = {"acc_signature_runs": float(ok[:, hw].mean()),
               "acc_borderline_runs": float(ok[:, ~hw].mean()) if (~hw).any() else None,
               "borderline_recall": {}}
        for c, cls in enumerate(CLASSES):
            m = ~hw & (y == c)
            if m.any():
                row["borderline_recall"][cls] = float(ok[:, m].mean())
        out["sets"][s] = row
    return out


# --------------------------------------------------------------------------
# Verdicts for the three questions
# --------------------------------------------------------------------------
def _classify(row, go):
    lo, hi = row["ci95"]
    if row["margin"] >= go:
        return "GO (CI above 0)" if lo > 0 else "GO by the margin rule, but the CI includes 0"
    if lo > 0:
        return "detectable gain below the GO margin"
    if hi < 0:
        return "worse"
    return "no detectable difference"


def _detail(rows: dict) -> str:
    return "; ".join(f"{k}: {r['label']} ({r['margin']:+.3f} {_fmt_ci(r['ci95'])})"
                     for k, r in rows.items())


def _fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def answer_questions(tracks: dict, go: float) -> dict:
    """Machine-generated answers from the primary comparisons (no hand-typed numbers).

    A "YES" requires the GEM margin (>= ``go``) *and* a 95 % CI above 0.
    """
    def rows(track, q, primary_only=True):
        return [r for r in tracks.get(track, {}).get("comparisons", [])
                if r["question"] == q and (r["primary"] or not primary_only)]

    def labelled(rs):
        return {f"[{r['_track']}] {r['candidate']} vs {r['reference']}":
                {k: v for k, v in r.items() if k != "_track"} | {"label": _classify(r, go)}
                for r in rs}

    def tag(t, rs):
        return [{**r, "_track": t} for r in rs]

    def strong(r):
        return r["margin"] >= go and r["ci95"][0] > 0

    ans = {}
    q1 = sum((tag(t, rows(t, "Q1")[:1]) for t in tracks), [])
    if q1:
        if all(strong(r) for r in q1):
            word = "YES"
        elif any(strong(r) for r in q1):
            word = "PARTLY (GO in some tracks only)"
        else:
            word = "NO-GO"
        lab = labelled(q1)
        ans["Q1"] = {"question": QUESTIONS["Q1"], "answer": f"{word}. {_detail(lab)}",
                     "headline": word, "rows": lab}
    if "multi" in tracks:
        q2 = tag("multi", rows("multi", "Q2"))
        lab = labelled(q2)
        if all(r["ci95"][0] > 0 for r in q2):
            word = ("YES, by at least the GO margin against both" if all(strong(r) for r in q2)
                    else "YES (CI above 0 against both), but not by the GO margin against both")
        else:
            word = "NO (not detectably better than both references)"
        ans["Q2"] = {"question": QUESTIONS["Q2"], "answer": f"{word}. {_detail(lab)}",
                     "headline": word, "rows": lab}
        q3 = tag("multi", rows("multi", "Q3")[:1])
        q3_all = tag("multi", rows("multi", "Q3", primary_only=False))
        r = q3[0]
        word = ("YES" if strong(r) else
                "NO-GO (a statistically detectable gain, below the GO margin)"
                if r["ci95"][0] > 0 else "NO")
        lab = labelled(q3_all)
        ans["Q3"] = {"question": QUESTIONS["Q3"],
                     "answer": f"{word}. {_detail(labelled(q3))}", "headline": word, "rows": lab}
    return ans


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(cfg: dict) -> dict:
    t_start = time.time()
    ev = cfg["evaluation"]
    out_dir = cfg["outputs"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    _log(f"config '{cfg['name']}': {cfg['simulation']['runs_per_class']} runs/class, "
         f"{ev['n_splits']} folds x {ev['n_repeats']} repeats, tracks {cfg['tracks']}")
    tracks, datasets, all_preds = {}, {}, {}
    for track in cfg["tracks"]:
        if track not in TRACKS:
            raise ValueError(f"unknown track {track!r}")
        t0 = time.time()
        tracks[track], datasets[track], all_preds[track] = evaluate_track(cfg, track)
        tracks[track]["runtime_s"] = time.time() - t0
    answers = answer_questions(tracks, ev["go_margin"])
    for q, a in answers.items():
        _log(f"{q}: {a['answer']}")
    results = {
        "config": cfg,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "classes": list(CLASSES),
        "regimes": DEFAULT_REGIMES,
        "network_regimes": network_regimes(cfg),
        "topology_config": {k: list(v) if isinstance(v, tuple) else v
                            for k, v in TopoConfig(**_tupled(cfg["topology"])).__dict__.items()},
        "sw1pers_config": {k: list(v) if isinstance(v, tuple) else v
                           for k, v in SW1PersConfig(**_tupled(cfg["sw1pers"])).__dict__.items()},
        "questions": QUESTIONS,
        "answers": answers,
        "tracks": tracks,
        "runtime_s": time.time() - t_start,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=_json_default)
    np.savez_compressed(os.path.join(out_dir, "oof_predictions.npz"),
                        **{f"{t}_{k}": v for t, p in all_preds.items() for k, v in p.items()},
                        **{f"{t}_y": datasets[t].y for t in datasets},
                        **{f"{t}_groups": datasets[t].groups for t in datasets})
    render(results, datasets)
    _log(f"done in {time.time() - t_start:.0f} s -> {out_dir}")
    return results


def render(results: dict, datasets: dict) -> None:
    from . import plots_sw1pers as P
    from .report_sw1pers import write_report
    cfg = results["config"]
    fig_dir = cfg["outputs"]["figures_dir"]
    _log("figures")
    P.render_all(results, datasets, fig_dir, SW1PersConfig(**_tupled(cfg["sw1pers"])))
    md = cfg["outputs"].get("results_md") or os.path.join(cfg["outputs"]["out_dir"],
                                                           "SW1PERS_RESULTS.md")
    write_report(results, md, fig_dir)
    _log(f"report -> {md}")


def replot(cfg: dict) -> None:
    """Re-render figures + report from an existing results.json (no re-training)."""
    with open(os.path.join(cfg["outputs"]["out_dir"], "results.json")) as fh:
        results = json.load(fh)
    results["config"]["outputs"] = cfg["outputs"]
    rc = {**results["config"], "cache_dir": cfg.get("cache_dir")}
    datasets, changed = {}, False
    oof = os.path.join(cfg["outputs"]["out_dir"], "oof_predictions.npz")
    for t, tr in results["tracks"].items():
        raw, datasets[t] = get_dataset(rc, t)
        if "post_hoc_signature_breakdown" not in tr and os.path.exists(oof):
            z = np.load(oof, allow_pickle=False)
            pre = f"{t}_"
            preds = {k[len(pre):]: z[k] for k in z.files
                     if k.startswith(pre) and k not in (f"{t}_y", f"{t}_groups")}
            tr["post_hoc_signature_breakdown"] = signature_breakdown(
                signature_holds(raw, t), z[f"{t}_y"], z[f"{t}_groups"], preds)
            changed = True
    if changed:
        with open(os.path.join(cfg["outputs"]["out_dir"], "results.json"), "w") as fh:
            json.dump(results, fh, indent=1, default=_json_default)
    render(results, datasets)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=None, help="YAML config (default: built-in small)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--results-md", default=None)
    ap.add_argument("--tracks", default=None, help="comma-separated subset of single,multi")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--replot", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.out_dir:
        cfg["outputs"]["out_dir"] = a.out_dir
        cfg["outputs"]["figures_dir"] = a.figures_dir or os.path.join(a.out_dir, "figures")
        cfg["outputs"]["results_md"] = a.results_md
    if a.figures_dir:
        cfg["outputs"]["figures_dir"] = a.figures_dir
    if a.results_md:
        cfg["outputs"]["results_md"] = a.results_md
    if a.tracks:
        cfg["tracks"] = [t.strip() for t in a.tracks.split(",") if t.strip()]
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
