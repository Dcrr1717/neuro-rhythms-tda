"""Generate ``SW1PERS_RESULTS.md`` from the results of ``sw1pers_bench``.

Every number is read from the results dictionary; the verdict sentences are
produced by fixed rules (``sw1pers_bench.answer_questions``), so the report
cannot drift from the run that produced it.
"""
from __future__ import annotations

import os
import time

SET_GROUPS = [
    ("baseline", ("bandpower", "baselines_union")),
    ("existing topology", ("topo", "descriptor")),
    ("SW1PerS", ("sw1pers_core", "sw1pers", "bandpower+sw1pers", "baselines_union+sw1pers")),
    ("Riemannian", ("riemann", "riemann_broadband", "riemann+baselines_union")),
    ("Riemannian + topology", ("riemann+sw1pers", "riemann+topo")),
    ("everything", ("union",)),
    ("deep model", ("cnn",)),
]


def _ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _m(x):
    return f"{x:+.3f}"


def _mci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _row(r):
    return (f"{r['candidate_accuracy']:.3f} | {r['reference_accuracy']:.3f} | "
            f"{_m(r['margin'])} {_mci(r['ci95'])} | {r['p_le_0']:.3f}")


def write_report(R: dict, path: str, fig_dir: str) -> None:
    cfg, T, A = R["config"], R["tracks"], R["answers"]
    ev = cfg["evaluation"]
    go = ev["go_margin"]
    rel = os.path.relpath(fig_dir, os.path.dirname(os.path.abspath(path)) or ".")
    sw = R["sw1pers_config"]
    L = []
    w = L.append
    w(f"# Results — SW1PerS topology vs a Riemannian baseline (config `{cfg['name']}`)\n")
    w(f"_Generated automatically by `python -m neurorhythms.sw1pers_bench --config ...` on "
      f"{time.strftime('%Y-%m-%d')}; runtime {R['runtime_s'] / 60:.1f} min. Every number below "
      f"is read from `results.json`, and the verdicts are produced by fixed rules; do not edit "
      f"by hand. Interpretation: [`SCIENCE.md`](SCIENCE.md) section 8._\n")

    # ------------------------------------------------------------- answers
    w("## Answers\n")
    w(f"Rule (GEM, as in the first benchmark): a comparison is **GO** iff the margin "
      f"(candidate − reference accuracy) is ≥ +{go:.2f}. Each margin has a paired 95 % "
      f"group-bootstrap CI (resampling simulation runs). Labels: *GO*; *detectable gain below "
      f"the GO margin* (CI above 0, margin < {go:.2f}); *no detectable difference* (CI "
      f"includes 0); *worse* (CI below 0). Primary comparisons were fixed before the full run.\n")
    for q, a in A.items():
        w(f"### {q}. {a['question']}\n")
        w(f"**Answer: {a['answer']}.**\n")
        w("| comparison | candidate acc | reference acc | margin [95% CI] | P(margin ≤ 0) | label |")
        w("|---|---|---|---|---|---|")
        for key, r in a["rows"].items():
            w(f"| {key} | {_row(r)} | {r['label']} |")
        for r in [r for r in a["rows"].values() if r["headroom"] < go]:
            w(f"\n_Headroom note: `{r['reference']}` scores {r['reference_accuracy']:.3f}, "
              f"so a +{go:.2f} margin over it is arithmetically impossible._")
        w("")
    if "CNN" in {c["question"] for t in T.values() for c in t["comparisons"]}:
        w("### Deep model\n")
        w("| track | comparison | candidate acc | reference acc | margin [95% CI] | P(margin ≤ 0) | rule |")
        w("|---|---|---|---|---|---|---|")
        for t, tr in T.items():
            for r in tr["comparisons"]:
                if r["question"] == "CNN":
                    w(f"| {t} | {r['candidate']} vs {r['reference']} | {_row(r)} | {r['verdict']} |")
        w("")

    # ------------------------------------------------------------- setup
    sim, meas, net = cfg["simulation"], cfg["measurement"], R["network_regimes"]
    w("## What was run\n")
    for t, tr in T.items():
        d = tr["dataset"]
        if t == "single":
            w(f"- **Single-channel track.** The dataset of the first benchmark: {d['n_runs']} "
              f"runs ({sim['runs_per_class']} per class) of one Jansen-Rit column driven by the "
              f"STN-GPe loop, {sim['duration']:.0f} s each, {d['fs']:.0f} Hz, 1/f + white "
              f"measurement noise, {d['window_s']:.0f} s windows z-scored: {d['n_windows']} "
              f"windows.")
        else:
            w(f"- **Multi-channel track.** {d['n_runs']} runs of a network of "
              f"{d['n_sources']} coupled Jansen-Rit columns (ring connectivity, inter-column "
              f"gain K ~ U{tuple(net['K'])}, delays U{tuple(net['delay'])} s) sharing one "
              f"STN-GPe loop (BG gain to each column x U{tuple(net['kappa_col'])}), seen by "
              f"{d['n_channels']} sensors through a Gaussian lead field (width "
              f"{net['lead_width']}, per-run jitter U{tuple(net['lead_jitter'])}). Measurement "
              f"noise as in the single track, with a spatially shared fraction "
              f"U{tuple(meas['common_fraction'])} of the 1/f background. Epileptic runs have "
              f"{net['epileptic_n_focus']} randomly placed focus column(s) in the epileptic "
              f"regime; parkinsonian runs have a supercritical loop. Every channel of every "
              f"{d['window_s']:.0f} s window is z-scored: {d['n_windows']} windows x "
              f"{d['n_channels']} channels.")
    w(f"- **Protocol.** Repeated StratifiedGroupKFold ({ev['n_splits']} folds x "
      f"{ev['n_repeats']} repeats), group = simulation run, identical folds for every view and "
      f"the CNN within a track. LogReg (standardised, C = 1) and RandomForest "
      f"({ev['rf_trees']} trees), the better of the two per set; {ev['bootstrap']} paired "
      f"group-bootstrap resamples.")
    scales = ", ".join(f"{f:g} Hz" for f in sw["target_freqs"]) + (
        " and the window's own dominant period" if sw["adaptive"] else "")
    w(f"- **SW1PerS** (Perea & Harer 2015). Per channel: cubic spline, sliding-window "
      f"embedding with M = 2N = {2 * sw['n_harmonics']} and window length P·M/(M+1) for "
      f"target periods P at {scales}; {sw['n_points']} points; pointwise centring and "
      f"normalisation; {sw['knn_denoise']}-NN mean-shift; Rips H1 over Z_{sw['coeff']} "
      f"(ripser). Features per scale: max persistence mp1 (= 1 − periodicity score), second "
      f"persistence mp2 (quasi-periodicity), birth of the top class, total persistence, "
      f"persistent entropy. **CROCKER**: {sw['crocker_segment']:g} s segments every "
      f"{sw['crocker_hop']:g} s, Betti-1 on {sw['crocker_resolution']} Rips scales in "
      f"{tuple(sw['crocker_grid'])}, mean and std over segments, plus mean/std/min of the "
      f"segment mp1. `sw1pers` = scales + CROCKER; `sw1pers_core` = scales only.")
    w("- **Riemannian** (`pyriemann`). Per window, OAS-shrunk covariance of the band-passed "
      "channels in the five band-power bands (`riemann`) or of the broadband signal "
      "(`riemann_broadband`); tangent space at the Riemannian mean of the training fold "
      "(fitted inside every fold). With z-scored channels the diagonal of each band's "
      "covariance is the channel's relative band power, so `riemann` contains the relative "
      "band powers (and their ratios) and adds cross-channel coupling; it does not contain "
      "the finer spectral shape of `bandpower` (peak frequency, spectral edge, entropy).")
    w("- **Existing views**: `bandpower` (relative band powers, ratios, SEF, peak frequency, "
      "spectral entropy), `baselines_union` (+ Hjorth, skewness/kurtosis/line length, "
      "log-PSD), `topo` (sublevel-set + Alpha-complex delay-embedding persistence) and "
      "`descriptor` (log-PSD + topo), all per channel and concatenated in the multi track.")
    if any("cnn" in tr["summary"] for tr in T.values()):
        cc = cfg["cnn"]
        w(f"- **1D-CNN** on the raw z-scored window (all channels as input channels in the "
          f"multi track), {cc['epochs']} epochs, AdamW lr {cc['lr']}, fixed schedule.")
    w("")

    if "multi" in T and T["multi"]["dataset"].get("source_signatures"):
        sg = T["multi"]["dataset"]["source_signatures"]
        w("### Do the network classes show their signatures? (noise-free columns, no classifier)\n")
        w("| class | runs | median column beta fraction | runs with a column beta ≥ 0.10 | "
          "runs with a spiking column | focus columns spiking |")
        w("|---|---|---|---|---|---|")
        for cls, r in sg.items():
            fs = r.get("fraction_focus_columns_spiking")
            w(f"| {cls} | {r['n_runs']} | {r['median_column_beta_fraction']:.3f} | "
              f"{100 * r['fraction_runs_any_column_beta_ge_0.10']:.0f} % | "
              f"{100 * r['fraction_runs_any_spiking_column']:.0f} % | "
              f"{'' if fs is None else f'{100 * fs:.0f} %'} |")
        w("\nA column is *spiking* if its slow (1-7.5 Hz) power fraction exceeds 0.5 and its "
          "skewness exceeds 0.5.\n")

    # ------------------------------------------------------------- tables
    w("## Accuracy of every view\n")
    for t, tr in T.items():
        S = tr["summary"]
        w(f"### {t}-channel track\n")
        w("| group | feature set | # features | LogReg | RF | best [95% CI] | macro-F1 | run-level acc |")
        w("|---|---|---|---|---|---|---|---|")
        for g, names in SET_GROUPS:
            for n in names:
                if n not in S:
                    continue
                r = S[n]
                if n == "cnn":
                    w(f"| {g} | **{n}** | raw signal ({r['n_parameters']:,} params) | – | – | "
                      f"{r['accuracy']:.3f} {_ci(r['ci95'])} | {r['macro_f1']:.3f} | "
                      f"{r['run_level_accuracy']:.3f} |")
                else:
                    w(f"| {g} | **{n}** | {r['n_features']} | {r['logreg_accuracy']:.3f} | "
                      f"{r['rf_accuracy']:.3f} | {r['accuracy']:.3f} {_ci(r['ci95'])} "
                      f"({r['best_classifier']}) | {r['macro_f1']:.3f} | "
                      f"{r['run_level_accuracy']:.3f} |")
        w("")
    w(f"![accuracy]({rel}/accuracy_bars.png)\n")

    w("## Every pre-specified comparison\n")
    w("`*` = primary comparison of its question.\n")
    w("| track | question | candidate | reference | candidate acc | reference acc | margin [95% CI] | P(margin ≤ 0) | rule |")
    w("|---|---|---|---|---|---|---|---|---|")
    for t, tr in T.items():
        for r in tr["comparisons"]:
            star = " *" if r["primary"] else ""
            w(f"| {t} | {r['question']}{star} | {r['candidate']} | {r['reference']} | "
              f"{_row(r)} | {r['verdict']} |")
    w("")
    w(f"![margins]({rel}/margins.png)\n")

    w("## Confusion matrices (row-normalised, pooled over repeats)\n")
    cls = R["classes"]
    for t, tr in T.items():
        w(f"![confusion {t}]({rel}/confusion_matrices_{t}.png)\n")
        for name, M in tr["confusion"].items():
            w(f"<details><summary>{t}: {name}</summary>\n")
            w("| true \\ predicted | " + " | ".join(cls) + " |")
            w("|---|" + "---|" * len(cls))
            for i, row in enumerate(M):
                w(f"| {cls[i]} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
            w("\n</details>\n")

    if any(tr.get("post_hoc_signature_breakdown") for tr in T.values()):
        w("## Post hoc: where are the errors? (exploratory, not part of the verdicts)\n")
        w("Each run is classified, from its *noise-free* model output and without any "
          "classifier, as showing its class signature or as *borderline*. Single track: the "
          "criteria of the first benchmark (alpha-dominant / slow + skewed / beta ≥ 0.10). "
          "Multi track: epileptic = some column spikes (slow fraction > 0.5 and skewness > "
          "0.5), parkinsonian = some column has beta fraction ≥ 0.10, healthy = neither. "
          "Window accuracy is then split by run type, using the out-of-fold predictions of "
          "the main run (no re-training).\n")
        for t, tr in T.items():
            b = tr.get("post_hoc_signature_breakdown")
            if not b:
                continue
            nb = b["n_runs_borderline"]
            w(f"**{t}-channel track**: borderline runs per class "
              + ", ".join(f"{c} {n} of {n + h}" for c, n, h in
                          zip(cls, nb, b["n_runs_signature_holds"])) + ".\n")
            border_cls = [c for c, n in zip(cls, nb) if n]
            w("| feature set | acc, runs with signature | acc, borderline runs | "
              + " | ".join(f"borderline {c} recall" for c in border_cls) + " |")
            w("|---|---|---|" + "---|" * len(border_cls))
            for s_, r in b["sets"].items():
                ab = "–" if r["acc_borderline_runs"] is None else f"{r['acc_borderline_runs']:.3f}"
                w(f"| {s_} | {r['acc_signature_runs']:.3f} | {ab} | "
                  + " | ".join(f"{r['borderline_recall'].get(c, float('nan')):.3f}"
                               for c in border_cls) + " |")
            w("")

    if any(tr.get("importance") for tr in T.values()):
        w("## Which feature families does the `union` model rely on?\n")
        w("Grouped permutation importance on held-out folds of repeat 1 (all columns of a "
          "family permuted together; mean accuracy drop). It measures reliance of a fitted "
          "model, not unique information: redundant families can each look unimportant to a "
          "random forest.\n")
        for t, tr in T.items():
            imp = tr.get("importance")
            if not imp:
                continue
            models = list(imp)
            w(f"**{t}-channel track**\n")
            w("| family | # columns | " + " | ".join(f"{m} drop" for m in models) + " |")
            w("|---|---|" + "---|" * len(models))
            for f in imp[models[0]]["families"]:
                w(f"| {f} | {imp[models[0]]['n_features'][f]} | "
                  + " | ".join(f"{imp[m]['mean_drop'][f]:+.3f}" for m in models) + " |")
            w(f"\n![importance {t}]({rel}/feature_importance_{t}.png)\n")
        w("(`riemann` columns are the raw covariance entries; the tangent-space projection is "
          "refitted inside the fold pipeline, so permuting them swaps whole matrices between "
          "windows.)\n")

    if any(tr.get("sanity") for tr in T.values()):
        w("## Sanity checks (leak detectors)\n")
        w("| track | check | value | expectation |")
        w("|---|---|---|---|")
        for t, tr in T.items():
            sa = tr.get("sanity", {})
            for k, v in sa.items():
                if k.startswith("permuted"):
                    s = k[len("permuted_run_labels_"):].rsplit("_", 1)[0]
                    w(f"| {t} | {s} / LogReg with run labels permuted | {v:.3f} | chance (0.333) |")
            for k in sa:
                if k.startswith("ungrouped"):
                    s = k[len("ungrouped_cv_rf_"):]
                    w(f"| {t} | {s} / RF, grouped (repeat 1) vs ungrouped CV | "
                      f"{sa[f'grouped_cv_rf_{s}']:.3f} vs {sa[k]:.3f} | ungrouped ≥ grouped |")
        w("")

    w("## Figures\n")
    if "multi" in T:
        w(f"![multi-channel traces]({rel}/example_multichannel.png)\n")
    w(f"![SW1PerS point clouds]({rel}/sw1pers_clouds.png)\n")
    w("## Caveats\n")
    w("- Synthetic data: classes are parameter regimes of neural-mass models; this measures "
      "which representation recovers the regime, not clinical diagnostic accuracy.")
    w("- Best-of-two classifiers per set is mildly optimistic, applied identically to every set.")
    w("- The multi-channel montage is a minimal network (4 columns, 6 sensors, Gaussian lead "
      "field), not a head model. Its network ranges were set from physiology and a pilot "
      "(seed 999) that ran only the band-power reference, to confirm there was headroom above "
      "it; no candidate view was run on the pilot to choose data or hyperparameters.")
    w("- Windows are the classification unit; runs are the resampling unit of every CI.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
