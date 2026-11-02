"""Figures (static PNG, matplotlib).

Colour use: one fixed categorical order (blue, orange, aqua) for the three
classes, and the same three slots for feature-set *kind* in the accuracy plots
(non-topological = blue, contains topology = orange, CNN = aqua). Text is
always neutral ink; grids are recessive.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .features.baseline import welch_psd  # noqa: E402
from .features.topology import delay_embedding, window_diagrams  # noqa: E402

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
CLASS_COLORS = (C1, C2, C3)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def example_traces(X, y, groups, fs, classes, path, seed: int = 0, tau: int = 5):
    """Per class: one window, run-mean PSD, delay embedding and two persistence diagrams."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(len(classes), 5, figsize=(17, 2.9 * len(classes)),
                           gridspec_kw={"width_ratios": [3.0, 1.25, 1.25, 1.25, 1.25]})
    t = np.arange(X.shape[1]) / fs
    last = len(classes) - 1
    for ci, cls in enumerate(classes):
        run = rng.choice(np.unique(groups[y == ci]))
        idx = np.nonzero(groups == run)[0]
        x = X[idx[0]].astype(float)
        col = CLASS_COLORS[ci]
        a = ax[ci, 0]
        a.plot(t, x, color=col, lw=0.8)
        a.set_ylabel(f"{cls}\n(z-score)")
        a.set_xlim(0, t[-1])
        if ci == 0:
            a.set_title("one 4-s window")
        if ci == last:
            a.set_xlabel("time (s)")
        # PSD
        a = ax[ci, 1]
        f, p = welch_psd(X[idx].astype(float), fs)
        m = (f >= 1) & (f <= 45)
        a.axvspan(8, 13, color="#e6e5e1", lw=0)
        a.axvspan(13, 30, color="#f3f2ef", lw=0)
        a.semilogy(f[m], p[:, m].mean(axis=0), color=col, lw=1.5)
        if ci == 0:
            a.set_title("PSD, run mean")
            a.text(10.5, 1.0, "alpha", transform=a.get_xaxis_transform(), ha="center",
                   va="top", fontsize=7, color=INK2)
            a.text(21.5, 1.0, "beta", transform=a.get_xaxis_transform(), ha="center",
                   va="top", fontsize=7, color=INK2)
        if ci == last:
            a.set_xlabel("frequency (Hz)")
        # embedding
        a = ax[ci, 2]
        E = delay_embedding(x, tau, 2)
        a.plot(E[:, 0], E[:, 1], color=col, lw=0.4, alpha=0.8)
        a.set_aspect("equal", adjustable="datalim")
        if ci == 0:
            a.set_title(f"delay embedding, tau={tau / fs * 1e3:.0f} ms")
        a.set_ylabel("x(t + tau)")
        if ci == last:
            a.set_xlabel("x(t)")
        dg = window_diagrams(x)
        # peaks: superlevel H0 = (peak height, prominence)
        a = ax[ci, 3]
        sup = dg["sup_h0"]
        a.scatter(-sup[:, 0], sup[:, 1] - sup[:, 0], s=9, color=col, alpha=0.7, lw=0)
        a.set_xlim(-2.5, 4.5)
        a.set_ylim(-0.1, 5.5)
        a.set_ylabel("prominence")
        if ci == 0:
            a.set_title("peaks: superlevel H0")
        if ci == last:
            a.set_xlabel("peak height (z)")
        # loops: embedding H1
        a = ax[ci, 4]
        h1 = dg[f"emb{tau}_h1"] if f"emb{tau}_h1" in dg else dg["emb5_h1"]
        a.plot([0, 0.9], [0, 0.9], color=INK2, lw=0.8)
        a.scatter(h1[:, 0], h1[:, 1], s=9, color=col, alpha=0.7, lw=0)
        a.set_xlim(0, 0.9)
        a.set_ylim(0, 0.9)
        a.set_aspect("equal")
        a.set_ylabel("death (radius)")
        if ci == 0:
            a.set_title("loops: H1 of embedding")
        if ci == last:
            a.set_xlabel("birth (radius)")
    fig.tight_layout()
    _save(fig, path)


def _kind(name):
    from .features import TOPO_SETS
    if name == "cnn":
        return "cnn"
    return "topo" if name in TOPO_SETS else "base"


def accuracy_bars(summary, ref_name, go_margin, path, chance=1 / 3):
    names = list(summary)
    acc = np.array([summary[n]["accuracy"] for n in names])
    lo = np.array([summary[n]["ci95"][0] for n in names])
    hi = np.array([summary[n]["ci95"][1] for n in names])
    col = {"base": C1, "topo": C2, "cnn": C3}
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(names) + 1.6))
    ypos = np.arange(len(names))[::-1]
    for yp, n, a_, l_, h_ in zip(ypos, names, acc, lo, hi):
        c = col[_kind(n)]
        ax.barh(yp, a_ - chance, left=chance, height=0.6, color=c, alpha=0.85)
        ax.plot([l_, h_], [yp, yp], color=INK, lw=1.2)
        lab = f"{a_:.3f} [{l_:.3f}, {h_:.3f}]"
        if "best_classifier" in summary[n]:
            lab += f"  {summary[n]['best_classifier']}"
        ax.text(1.02, yp, lab, va="center", fontsize=7.5, color=INK,
                transform=ax.get_yaxis_transform())
    ref = summary[ref_name]["accuracy"]
    ax.axvline(ref, color=INK2, lw=1, ls="--")
    ax.axvline(min(ref + go_margin, 1.0), color=C2, lw=1, ls=":")
    ax.text(ref, len(names) - 0.35, f"{ref_name} ", fontsize=7, color=INK2, ha="right")
    ax.text(min(ref + go_margin, 1.0), len(names) - 0.35, " +GO margin", fontsize=7, color=C2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlim(chance, 1.0)
    ax.set_xlabel("window-level accuracy (grouped CV, 95% group-bootstrap CI)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C1, label="non-topological"),
                       Patch(color=C2, label="contains topology"),
                       Patch(color=C3, label="1D-CNN (raw signal)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, path)


def margin_forest(rows, go_margin, path):
    """rows: list of (label, margin, [lo, hi])."""
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(rows) + 1.8))
    yp = np.arange(len(rows))[::-1]
    for y_, (lab, m, ci) in zip(yp, rows):
        c = C2 if ci[0] > 0 else (C1 if ci[1] < 0 else INK2)
        ax.plot(ci, [y_, y_], color=c, lw=2)
        ax.plot([m], [y_], "o", color=c, ms=7, mec="white", mew=1.5)
        ax.text(1.02, y_, f"{m:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]", va="center",
                fontsize=7.5, color=INK, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK2, lw=1)
    ax.axvline(go_margin, color=C2, lw=1, ls=":")
    ax.text(go_margin, len(rows) - 0.4, " GO margin", color=C2, fontsize=7.5)
    ax.set_yticks(yp)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    lo = min(-0.06, min(r[2][0] for r in rows) - 0.01)
    hi = max(go_margin + 0.05, max(r[2][1] for r in rows) + 0.06)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("accuracy margin, candidate - reference (paired group bootstrap, 95% CI)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=C2, lw=2, label="CI above 0"),
                       Line2D([], [], color=INK2, lw=2, label="CI includes 0"),
                       Line2D([], [], color=C1, lw=2, label="CI below 0")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, path)


def confusion_panels(mats: dict, classes, path):
    fig, ax = plt.subplots(1, len(mats), figsize=(3.6 * len(mats), 3.4))
    ax = np.atleast_1d(ax)
    for a, (title, M) in zip(ax, mats.items()):
        a.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                       color="white" if M[i, j] > 0.6 else INK)
        a.set_xticks(range(len(classes)))
        a.set_xticklabels([c[:5] for c in classes])
        a.set_yticks(range(len(classes)))
        a.set_yticklabels([c[:5] for c in classes])
        a.set_xlabel("predicted")
        a.set_ylabel("true")
        a.set_title(title, fontsize=9)
        a.grid(False)
    fig.tight_layout()
    _save(fig, path)


def importance_bars(imp: dict, path):
    """imp: {model: result of family_permutation_importance}."""
    models = list(imp)
    fams = imp[models[0]]["families"]
    fig, ax = plt.subplots(1, len(models), figsize=(5.2 * len(models), 0.4 * len(fams) + 1.4),
                           sharey=True)
    ax = np.atleast_1d(ax)
    yp = np.arange(len(fams))[::-1]
    for a, m in zip(ax, models):
        d = imp[m]["mean_drop"]
        s = imp[m]["std_drop"]
        vals = np.array([d[f] for f in fams])
        errs = np.array([s[f] for f in fams])
        cols = [C2 if f.startswith("topo") else C1 for f in fams]
        a.barh(yp, vals, xerr=errs, color=cols, height=0.6,
               error_kw={"ecolor": INK2, "lw": 0.8})
        a.axvline(0, color=INK2, lw=0.8)
        a.set_title(f"{m} on union (base acc {imp[m]['base_accuracy']:.3f})", fontsize=9)
        a.set_xlabel("accuracy drop when family is permuted")
        a.grid(axis="y", visible=False)
    ax[0].set_yticks(yp)
    ax[0].set_yticklabels([f"{f} ({imp[models[0]]['n_features'][f]})" for f in fams])
    fig.tight_layout()
    _save(fig, path)


def noise_sweep_plot(sweep: dict, path):
    """sweep: {'scales': [...], 'sets': {name: [(acc, lo, hi), ...]}}."""
    scales = sweep["scales"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    style = {"bandpower+hjorth": (C1, "--", "o"), "baselines_union": (C1, "-", "o"),
             "topo": (C2, "--", "s"), "descriptor": (C2, ":", "s"), "union": (C2, "-", "s"),
             "cnn": (C3, "-", "^")}
    for name, vals in sweep["sets"].items():
        c, ls, mk = style.get(name, (INK2, "-", "o"))
        a = np.array(vals)
        ax.plot(scales, a[:, 0], ls=ls, marker=mk, color=c, lw=2, ms=7, mec="white", label=name)
        ax.fill_between(scales, a[:, 1], a[:, 2], color=c, alpha=0.12, lw=0)
    ax.set_xlabel("measurement-noise scale (x baseline pink+white noise)")
    ax.set_ylabel("accuracy (grouped CV)")
    ax.set_xticks(scales)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
