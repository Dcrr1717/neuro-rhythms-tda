"""Markdown report of the IAAFT surrogate benchmark (every number read from results.json)."""
from __future__ import annotations

import os
import time

from .report import _ci, _m, _mci, _verdict_sentence

SET_ORDER = ("periodogram_bandpower", "periodogram_psd", "onepoint", "hjorth", "bandpower",
             "bandpower+hjorth", "psd", "nonlinear", "baselines_union", "topo", "topo_summary",
             "descriptor", "union", "union_summary", "cnn")
GROUP = {"periodogram_bandpower": "blind by construction", "periodogram_psd": "blind by construction",
         "onepoint": "blind by construction", "hjorth": "blind by construction",
         "bandpower": "brief-named (Welch)", "bandpower+hjorth": "brief-named (Welch)",
         "psd": "Welch spectrum", "nonlinear": "classical nonlinear",
         "baselines_union": "all non-topological", "topo": "topological",
         "topo_summary": "topological", "descriptor": "topological-spectral",
         "union": "all + topology", "union_summary": "all + topology", "cnn": "deep"}
OVERVIEW = ("periodogram_psd", "onepoint", "bandpower+hjorth", "nonlinear", "baselines_union",
            "topo", "union", "cnn")


def _row(c, go):
    return (f"| {c['candidate']} | {c['reference']} | {c['candidate_accuracy']:.3f} | "
            f"{c['reference_accuracy']:.3f} | {_m(c['margin'])} {_mci(c['ci95'])} | {c['verdict']} |")


def write_surrogate_report(R: dict, path: str, fig_dir: str) -> None:
    cfg = R["config"]
    go = cfg["evaluation"]["go_margin"]
    rel = os.path.relpath(fig_dir, os.path.dirname(os.path.abspath(path)) or ".")
    P = R["tasks"][R["primary_task"]]
    G = P["gonogo"]
    L = []
    w = L.append
    w(f"# Results — IAAFT surrogate benchmark (config `{cfg['name']}`)\n")
    w(f"_Generated automatically by `python -m neurorhythms.surrogate_benchmark` on "
      f"{time.strftime('%Y-%m-%d')}; runtime {R['runtime_s'] / 60:.1f} min. Every number below "
      f"is read from `results.json`; do not edit by hand._\n")
    w("**Question.** Each 4 s window is paired with an IAAFT surrogate that has the same "
      "periodogram and the same amplitude histogram but random Fourier phases. Telling them "
      "apart requires *nonlinear temporal structure*: time-irreversibility, phase coupling, "
      "waveform shape. Does the topological view capture that structure better than the "
      "classical nonlinear time-series statistics?\n")

    # ---------------------------------------------------------------- verdict
    w("## Verdict\n")
    s = G["strict"]
    w(f"**Final (strict) verdict on the primary task (`{R['primary_task']}`): {R['final_verdict']}.** "
      f"Rule, as in the GEM benchmark: GO iff a topology-containing feature set beats the union of "
      f"all non-topological features by >= +{go:.2f} accuracy (paired group-bootstrap 95 % CI "
      f"alongside).\n")
    w(f"| question | candidate | reference | candidate acc | reference acc | margin [95% CI] | rule (>= +{go:.2f}) |")
    w("|---|---|---|---|---|---|---|")
    C = P["comparisons"]
    rows = [("As briefed: descriptor vs band-power / Hjorth (Welch)", G.get("as_briefed")),
            ("Strict: best topology set vs all non-topological", s["best"]),
            ("Incremental: all features + topology vs all non-topological", G.get("incremental")),
            ("Head-to-head: topology vs classical nonlinear statistics", C.get("topo_vs_nonlinear")),
            ("Deep: 1D-CNN vs all non-topological", C.get("cnn_vs_baselines_union"))]
    for q, c in rows:
        if c:
            w(f"| {q} | {c['candidate']} | {c['reference']} | {c['candidate_accuracy']:.3f} | "
              f"{c['reference_accuracy']:.3f} | {_m(c['margin'])} {_mci(c['ci95'])} | "
              f"**{c['verdict']}** |")
    w("")
    v = P.get("validity")
    if v:
        w(f"- **Validity.** The best feature set that is blind by construction (`{v['best_blind_set']}`) "
          f"scores {v['accuracy']:.3f} {_ci(v['ci95'])}; chance is {v['chance']:.3f}. The task "
          f"cannot be solved from what the surrogate preserves.")
    if G.get("as_briefed"):
        w(f"- As briefed, the descriptor {_verdict_sentence(G['as_briefed'], go)}.")
    w(f"- Strict: `baselines_union` scores {s['baselines_union_accuracy']:.3f}; headroom "
      f"{s['headroom']:.3f}" + ("" if s["go_arithmetically_possible"] else
                                f" < {go:.2f}, so a GO is arithmetically impossible") + ". "
      f"The best topology-containing set ({s['best']['candidate']}) "
      f"{_verdict_sentence(s['best'], go)}.")
    if C.get("topo_vs_nonlinear"):
        w(f"- Head-to-head, topology alone vs the classical nonlinear statistics alone: "
          f"{_m(C['topo_vs_nonlinear']['margin'])} {_mci(C['topo_vs_nonlinear']['ci95'])}.")
    w("")

    # ---------------------------------------------------------------- setup
    d, q = R["data"], R["data"]["quality"]
    ev = cfg["evaluation"]
    w("## What was run\n")
    w(f"- **Signals.** The same simulation runs and measurement model as the main benchmark "
      f"(config seed {cfg['seed']}, {d['n_runs']} runs). Each run is cut into "
      f"{d['windows_per_run']} non-overlapping {cfg['measurement']['window']:.0f} s windows whose "
      f"start is shifted by at most {cfg['surrogate']['end_match_search']:.1f} s so that the "
      f"window's ends join smoothly (median end mismatch {q['end_mismatch_sd']['median']:.2f} sd), "
      f"which avoids the end-point artefact of Fourier surrogates. {d['n_windows']} windows.")
    w(f"- **Surrogates.** One IAAFT surrogate per window ({cfg['surrogate']['n_iter']} iterations). "
      f"Primary variant (final step = spectrum): periodogram relative error "
      f"{q['spectrum']['spectrum_rel_err']['max']:.1e} (max), histogram error "
      f"{q['spectrum']['histogram_err']['median']:.1e} sd (median). Robustness variant (final step "
      f"= rank order): histogram exact, periodogram error "
      f"{q['amplitude']['spectrum_rel_err']['median']:.1e} (median). Correlation surrogate vs "
      f"original: median |r| {q['spectrum']['corr']['median']:.2f}.")
    w(f"- **Protocol.** StratifiedGroupKFold ({ev['n_splits']} folds; repeats per task below), "
      f"group = simulation run, so a window and its surrogate are always on the same side. "
      f"LogReg and RandomForest ({ev['rf_trees']} trees), the better of the two per feature set; "
      f"paired group-bootstrap CIs ({ev['bootstrap']} resamples of runs).")
    w("- **Feature sets.** *Blind by construction*: spectral features from the full-window "
      "periodogram, one-point statistics (quantiles, skewness, kurtosis), Hjorth parameters. "
      "*Brief-named*: band powers / ratios / SEF with the usual Welch estimator, Hjorth. *Classical "
      "nonlinear*: time-reversal asymmetry, increment flatness and third-order autocorrelation at "
      "7 lags, rise fraction, rise/decay and peak/trough sharpness asymmetry, line length. "
      "*Topology*: the main benchmark's GUDHI view (sublevel-set + delay-embedding persistence).")
    w("")

    # ---------------------------------------------------------------- primary table
    w(f"## Primary task (`{R['primary_task']}`): accuracy of every view\n")
    w(_task_table(P))
    w(f"![accuracy]({rel}/surrogate_accuracy_bars.png)\n")
    w("Paired comparisons on the primary task:\n")
    w("| candidate | reference | candidate acc | reference acc | margin [95% CI] | rule |")
    w("|---|---|---|---|---|---|")
    for c in C.values():
        w(_row(c, go))
    w("")
    w(f"![margins]({rel}/surrogate_margins.png)\n")

    # ---------------------------------------------------------------- all tasks
    w("## All tasks\n")
    keys = [k for k in OVERVIEW if any(k in T["summary"] for T in R["tasks"].values())]
    w("| task | windows | repeats | chance | " + " | ".join(keys)
      + " | best topology set vs all non-topological [95% CI] | verdict |")
    w("|---|---|---|---|" + "---|" * len(keys) + "---|---|")
    for tn, T in R["tasks"].items():
        b = T["gonogo"]["strict"]["best"]
        cells = [f"{T['summary'][k]['accuracy']:.3f}" if k in T["summary"] else "–" for k in keys]
        w(f"| {tn} | {T['n_windows']} | {T['task']['n_repeats']} | {T['chance']:.3f} | "
          + " | ".join(cells)
          + f" | {b['candidate']}: {_m(b['margin'])} {_mci(b['ci95'])} | "
          f"{T['gonogo']['strict']['verdict']} |")
    w("")
    w(f"![tasks]({rel}/surrogate_tasks.png)\n")

    six = R["tasks"].get("six_class")
    if six and six.get("confusion"):
        w("## Six-class task: confusion matrices\n")
        w(f"![confusion]({rel}/surrogate_confusion_six_class.png)\n")

    if P.get("importance"):
        imp = P["importance"]
        models = list(imp)
        w("## Which feature families carry the signal? (primary task, `union`)\n")
        w("| family | # features | " + " | ".join(f"{m} drop" for m in models) + " |")
        w("|---|---|" + "---|" * len(models))
        for f in imp[models[0]]["families"]:
            w(f"| {f} | {imp[models[0]]['n_features'][f]} | "
              + " | ".join(f"{imp[m]['mean_drop'][f]:+.3f}" for m in models) + " |")
        w("")
        w("Permutation importance measures how much the *fitted* model relies on a family, not "
          "its unique information (see `docs/RESULTS.md`).\n")
        w(f"![importance]({rel}/surrogate_importance.png)\n")

    w("## Example windows and surrogates\n")
    w(f"![examples]({rel}/surrogate_examples.png)\n")
    w("## Caveats\n")
    w("- The surrogate test asks for *any* departure from a linear Gaussian process seen through "
      "a static nonlinearity, within one window: nonlinear dynamics, but also within-window "
      "non-stationarity (e.g. one burst of spikes). It does not tell which of the two a feature "
      "picked up.")
    w("- Welch-based spectral features are *not* blind: averaging Hann-windowed 1-s segments "
      "makes them sensitive to the slow waxing and waning of the oscillation envelope, which "
      "phase randomisation changes. They are reported as the brief-named baselines, and the "
      "periodogram versions serve as the validity check.")
    w("- Choosing the better of LogReg / RF per feature set is mildly optimistic and applied to "
      "every set alike (GEM convention).")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def _task_table(T: dict) -> str:
    out = ["| group | feature set | # features | LogReg | RF | best [95% CI] | macro-F1 |",
           "|---|---|---|---|---|---|---|"]
    for n in SET_ORDER:
        if n not in T["summary"]:
            continue
        r = T["summary"][n]
        if n == "cnn":
            out.append(f"| {GROUP[n]} | **cnn** | raw signal | – | – | {r['accuracy']:.3f} "
                       f"{_ci(r['ci95'])} | {r['macro_f1']:.3f} |")
        else:
            out.append(f"| {GROUP[n]} | **{n}** | {r['n_features']} | {r['logreg_accuracy']:.3f} | "
                       f"{r['rf_accuracy']:.3f} | {r['accuracy']:.3f} {_ci(r['ci95'])} "
                       f"({r['best_classifier']}) | {r['macro_f1']:.3f} |")
    return "\n".join(out) + "\n"
