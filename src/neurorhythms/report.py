"""Generate the Markdown results report from ``results.json``.

Every number in the report is read from the results dictionary; nothing is
typed by hand, so the report cannot drift from the run that produced it.
"""
from __future__ import annotations

import os
import time

GROUPS = [
    ("Brief-named baselines", ("bandpower", "hjorth", "bandpower+hjorth")),
    ("Strict non-topological baselines", ("waveform", "psd", "baselines_union")),
    ("Topological / topological-spectral", ("topo", "descriptor", "topo_summary")),
    ("Incremental (topology added to baselines)", ("bandpower+hjorth+topo", "union",
                                                   "union_summary")),
    ("Deep model", ("cnn",)),
]


def _ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _m(x):
    return f"{x:+.3f}"


def _mci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _verdict_sentence(row, go):
    m, ci = row["margin"], row["ci95"]
    if row["verdict"] == "GO":
        s = f"reaches the +{go:.2f} GO margin"
    else:
        s = f"does not reach the +{go:.2f} GO margin"
    if ci[0] > 0:
        s += "; the 95 % CI lies above 0 (a statistically detectable but "
        s += "practically small gain)" if m < go else ")"
    elif ci[1] < 0:
        s += "; the 95 % CI lies below 0 (the candidate is worse)"
    else:
        s += "; the 95 % CI includes 0 (no detectable difference)"
    return s


def write_report(R: dict, path: str, fig_dir: str) -> None:
    cfg, S, C, G = R["config"], R["summary"], R["comparisons"], R["gonogo"]
    ev = cfg["evaluation"]
    go = G["go_margin"]
    rel = os.path.relpath(fig_dir, os.path.dirname(os.path.abspath(path)) or ".")
    d = R["dataset"]
    L = []
    w = L.append
    w(f"# Results — neural-mass rhythm benchmark (config `{cfg['name']}`)\n")
    w(f"_Generated automatically by `python -m neurorhythms --config ...` on "
      f"{time.strftime('%Y-%m-%d')}; runtime {R['runtime_s'] / 60:.1f} min. "
      f"Every number below is read from `results.json`; do not edit by hand._\n")

    # ------------------------------------------------------------------ verdict
    a, s = G["as_briefed"], G["strict"]
    inc = G["incremental_gain_over_baselines_union"]
    w("## Verdict\n")
    w(f"**Final (strict) verdict: {G['final_verdict']}.** Rule, as in the GEM benchmark: "
      f"GO iff a configuration that contains the topological view beats the reference by "
      f">= +{go:.2f} accuracy (paired group-bootstrap 95 % CI reported alongside).\n")
    w(f"| Question | candidate | reference | candidate acc | reference acc | margin [95% CI] | rule (>= +{go:.2f}) |")
    w("|---|---|---|---|---|---|---|")
    w(f"| As briefed: topological-spectral descriptor vs band-power / Hjorth baselines | "
      f"descriptor | {a['best_brief_baseline']} | {a['candidate_accuracy']:.3f} | "
      f"{a['reference_accuracy']:.3f} | {_m(a['margin'])} {_mci(a['ci95'])} | **{a['verdict']}** |")
    best_row = max(s["rows"].values(), key=lambda r: r["margin"])
    w(f"| Strict: best topology-containing set vs all non-topological baselines | "
      f"{best_row['candidate']} | baselines_union | {best_row['candidate_accuracy']:.3f} | "
      f"{best_row['reference_accuracy']:.3f} | {_m(best_row['margin'])} {_mci(best_row['ci95'])} | "
      f"**{s['verdict']}** |")
    w(f"| Incremental: baselines_union + topology vs baselines_union | union | baselines_union | "
      f"{inc['candidate_accuracy']:.3f} | {inc['reference_accuracy']:.3f} | {_m(inc['margin'])} "
      f"{_mci(inc['ci95'])} | {inc['verdict']} |")
    w("")
    w(f"- As briefed, the descriptor {_verdict_sentence(a, go)}.")
    w(f"- Strict: `baselines_union` scores {s['baselines_union_accuracy']:.3f}, so the headroom "
      f"(largest margin any feature set could reach) is {s['headroom']:.3f}"
      + ("" if s["go_arithmetically_possible"] else
         f" < {go:.2f}: a GO is arithmetically impossible on this task") + ".")
    w(f"- Adding topology to every baseline changes accuracy by {_m(inc['margin'])} "
      f"{_mci(inc['ci95'])}: it {_verdict_sentence(inc, go)}.")
    if "cnn_vs_baselines_union" in C:
        c = C["cnn_vs_baselines_union"]
        w(f"- The 1D-CNN on the raw signal reaches {c['candidate_accuracy']:.3f} "
          f"({_m(c['margin'])} {_mci(c['ci95'])} vs `baselines_union`).")
    w("")

    # ------------------------------------------------------------------ setup
    w("## What was run\n")
    sim, meas = cfg["simulation"], cfg["measurement"]
    w(f"- **Data.** {d['n_runs']} independent simulation runs ({sim['runs_per_class']} per class) "
      f"of the coupled Jansen-Rit cortical column + STN-GPe loop, {sim['duration']:.0f} s each "
      f"after a {sim['transient']:.0f} s transient, Euler-Maruyama at dt = {sim['dt'] * 1e3:.1f} ms, "
      f"sampled at {d['fs']:.0f} Hz. Measurement model: z-scored model output + 1/f noise "
      f"(level U{tuple(meas['pink_level'])}) + white noise (U{tuple(meas['white_level'])}), 0.5 Hz "
      f"high-pass, {d['window_s']:.0f} s non-overlapping windows, each z-scored. "
      f"{d['n_windows']} windows ({d['windows_per_run']} per run, classes {d['class_counts']}). "
      f"Predicted beta frequency at onset in the parkinsonian runs: "
      f"{d['beta_onset_freq_hz_parkinsonian']['min']:.1f}-"
      f"{d['beta_onset_freq_hz_parkinsonian']['max']:.1f} Hz.")
    w(f"- **Protocol.** Repeated StratifiedGroupKFold ({ev['n_splits']} folds x {ev['n_repeats']} "
      f"repeats), group = simulation run, identical folds for every feature set and the CNN. "
      f"LogReg (standardised, C = 1) and RandomForest ({ev['rf_trees']} trees); the better of "
      f"the two is reported per feature set (as in GEM). Paired group-bootstrap CIs "
      f"({ev['bootstrap']} resamples of runs). Primary metric: window-level accuracy "
      f"(classes are balanced, so it equals balanced accuracy).")
    if "cnn" in S:
        cc = cfg["cnn"]
        w(f"- **Deep model.** 1D-CNN, {S['cnn']['n_parameters']:,} parameters, "
          f"{cc['epochs']} epochs (fixed, no early stopping), AdamW lr {cc['lr']}, batch "
          f"{cc['batch_size']}, on the raw z-scored window ({int(d['fs'] * d['window_s'])} samples).")
    w("")

    if R.get("signatures"):
        sg = R["signatures"]
        w("## Do the classes show their intended signatures?\n")
        w("Computed on the noise-free model output of every run (no classifier involved): "
          "median [10th-90th percentile] over runs.\n")
        keys = [k for k in next(iter(sg.values())) if isinstance(next(iter(sg.values()))[k], list)]
        w("| class | " + " | ".join(keys) + " | signature holds | criterion |")
        w("|---|" + "---|" * (len(keys) + 2))
        for cls_, row in sg.items():
            cells = [f"{row[k][0]:.2f} [{row[k][1]:.2f}, {row[k][2]:.2f}]" for k in keys]
            w(f"| {cls_} | " + " | ".join(cells)
              + f" | {100 * row['fraction_signature_holds']:.0f} % of {row['n_runs']} runs | "
              f"{row['criterion']} |")
        w("")

    # ------------------------------------------------------------------ tables
    w("## Accuracy of every view\n")
    w("| group | feature set | # features | LogReg | RF | best [95% CI] | macro-F1 | run-level acc |")
    w("|---|---|---|---|---|---|---|---|")
    for gname, names in GROUPS:
        for n in names:
            if n not in S:
                continue
            r = S[n]
            if n == "cnn":
                w(f"| {gname} | **{n}** | raw signal | – | – | {r['accuracy']:.3f} {_ci(r['ci95'])} | "
                  f"{r['macro_f1']:.3f} | {r['run_level_accuracy']:.3f} |")
            else:
                w(f"| {gname} | **{n}** | {r['n_features']} | {r['logreg_accuracy']:.3f} | "
                  f"{r['rf_accuracy']:.3f} | {r['accuracy']:.3f} {_ci(r['ci95'])} "
                  f"({r['best_classifier']}) | {r['macro_f1']:.3f} | {r['run_level_accuracy']:.3f} |")
    w("")
    w(f"![accuracy]({rel}/accuracy_bars.png)\n")

    w("## Paired comparisons\n")
    w("| candidate | reference | candidate acc | reference acc | margin [95% CI] | P(margin <= 0) | rule |")
    w("|---|---|---|---|---|---|---|")
    for k, c in C.items():
        w(f"| {c['candidate']} | {c['reference']} | {c['candidate_accuracy']:.3f} | "
          f"{c['reference_accuracy']:.3f} | {_m(c['margin'])} {_mci(c['ci95'])} | "
          f"{c['p_le_0']:.3f} | {c['verdict']} |")
    w("")
    w("Every topology-containing set against `baselines_union` (the strict rule):\n")
    w("| set | accuracy | margin vs baselines_union [95% CI] | rule |")
    w("|---|---|---|---|")
    for n, r in s["rows"].items():
        w(f"| {n} | {r['candidate_accuracy']:.3f} | {_m(r['margin'])} {_mci(r['ci95'])} | {r['verdict']} |")
    w("")
    w(f"![margins]({rel}/margins.png)\n")

    w("## Confusion matrices (row-normalised, pooled over repeats)\n")
    w(f"![confusion]({rel}/confusion_matrix.png)\n")
    cls = d["classes"]
    for name, M in R["confusion"].items():
        w(f"<details><summary>{name}</summary>\n")
        w("| true \\ predicted | " + " | ".join(cls) + " |")
        w("|---|" + "---|" * len(cls))
        for i, row in enumerate(M):
            w(f"| {cls[i]} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
        w("\n</details>\n")

    if R.get("importance"):
        w("## Which feature families carry the signal?\n")
        w("Grouped permutation importance on the `union` set (all columns of a family "
          "permuted together on held-out folds of repeat 1; mean accuracy drop).\n")
        imp = R["importance"]
        models = list(imp)
        w("| family | # features | " + " | ".join(f"{m} drop" for m in models) + " |")
        w("|---|---|" + "---|" * len(models))
        for f in imp[models[0]]["families"]:
            w(f"| {f} | {imp[models[0]]['n_features'][f]} | "
              + " | ".join(f"{imp[m]['mean_drop'][f]:+.3f}" for m in models) + " |")
        w("")
        w(f"![importance]({rel}/feature_importance.png)\n")

    if R.get("sanity"):
        sa = R["sanity"]
        w("## Sanity checks (leak detectors)\n")
        w("| check | value | expectation |")
        w("|---|---|---|")
        if "permuted_run_labels_union_logreg" in sa:
            w(f"| union / LogReg with run labels permuted | "
              f"{sa['permuted_run_labels_union_logreg']:.3f} | chance (0.333) |")
        for s_ in ("topo", "baselines_union"):
            if f"ungrouped_cv_rf_{s_}" in sa:
                w(f"| {s_} / RF, grouped CV (repeat 1) vs ungrouped CV | "
                  f"{sa[f'grouped_cv_rf_{s_}']:.3f} vs {sa[f'ungrouped_cv_rf_{s_}']:.3f} | "
                  f"ungrouped >= grouped (window leakage inflates) |")
        w("")

    sw = R.get("noise_sweep")
    if sw:
        w("## Exploratory: robustness to measurement noise (post hoc, not part of the verdict)\n")
        w(f"The same simulated runs, with the pink + white measurement noise multiplied by "
          f"the scale factor ({sw['n_repeats']} CV repeat(s) for scales > 1).\n")
        names = list(sw["sets"])
        w("| noise scale | " + " | ".join(names) + " |")
        w("|---|" + "---|" * len(names))
        for i, sc in enumerate(sw["scales"]):
            w(f"| x{sc:g} | " + " | ".join(f"{sw['sets'][n][i][0]:.3f}" for n in names) + " |")
        w("")
        w("Margin vs `baselines_union` [95% CI]:\n")
        mn = list(sw["margins_vs_baselines_union"])
        w("| noise scale | " + " | ".join(mn) + " |")
        w("|---|" + "---|" * len(mn))
        for i, sc in enumerate(sw["scales"]):
            w(f"| x{sc:g} | " + " | ".join(
                f"{sw['margins_vs_baselines_union'][n][i][0]:+.3f} "
                f"[{sw['margins_vs_baselines_union'][n][i][1]:+.3f}, "
                f"{sw['margins_vs_baselines_union'][n][i][2]:+.3f}]" for n in mn) + " |")
        w("")
        w(f"![noise]({rel}/noise_sweep.png)\n")

    w("## Example signals\n")
    w(f"![traces]({rel}/example_traces.png)\n")
    w("## Caveats\n")
    w("- Synthetic data: the classes are defined by model parameter regimes, so this "
      "benchmark measures how well each representation recovers *dynamical regimes of "
      "neural-mass models*, not clinical diagnostic accuracy. See `docs/SCIENCE.md`.")
    w("- Choosing the better of LogReg / RF per feature set on the same folds is mildly "
      "optimistic; it is applied identically to every feature set (GEM convention).")
    w("- Windows are the classification unit; runs are the resampling unit of the CIs. "
      "Run-level accuracy (mean window probability per run) is reported as a secondary metric.")
    w("- The noise sweep (if present) is exploratory and was not part of the pre-specified rule.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
