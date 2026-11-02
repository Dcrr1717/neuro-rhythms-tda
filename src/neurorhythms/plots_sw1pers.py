"""Figures of the SW1PerS / Riemannian benchmark (static PNG, matplotlib).

Colour use follows ``plots.py``: the three validated categorical slots (blue,
orange, aqua) mark the three classes in signal figures and the feature-set
*kind* in accuracy plots (non-topological = blue, contains topology = orange,
CNN = aqua). A fourth hue would fail the palette's all-pairs separation
checks, so Riemannian sets are marked with a hatch texture instead (and every
bar is named on its axis, so identity is never colour alone). Correlations use
a diverging blue - neutral grey - orange map; counts use one sequential hue.
"""
from __future__ import annotations

import os

import numpy as np

from .plots import C1, C2, C3, CLASS_COLORS, INK, INK2, _save, confusion_panels, plt
from .plots import importance_bars, margin_forest

DIVERGING = None


def _diverging():
    global DIVERGING
    if DIVERGING is None:
        from matplotlib.colors import LinearSegmentedColormap
        DIVERGING = LinearSegmentedColormap.from_list("blue_grey_orange",
                                                      [C1, "#e6e5e1", C2])
    return DIVERGING


TOPO_BLOCKS = ("topo", "sw1pers", "crocker")


def _kind(summary_row, name):
    if name == "cnn":
        return "cnn"
    blocks = summary_row.get("blocks", [])
    return "topo" if any(b in TOPO_BLOCKS for b in blocks) else "base"


def _is_riemann(summary_row):
    return any(b.startswith("riemann") for b in summary_row.get("blocks", []))


# --------------------------------------------------------------------------
# Example multi-channel traces + band correlation matrices
# --------------------------------------------------------------------------
def example_multichannel(X, y, groups, fs, classes, path, seed: int = 0):
    from .features.riemann import filterbank_covariances
    rng = np.random.default_rng(seed)
    n_ch = X.shape[1]
    fig, ax = plt.subplots(len(classes), 3, figsize=(14, 3.3 * len(classes)),
                           gridspec_kw={"width_ratios": [3.4, 1.0, 1.0]})
    t = np.arange(X.shape[2]) / fs
    bands = {"alpha (8-13 Hz)": (8.0, 13.0), "beta (13-30 Hz)": (13.0, 30.0)}
    im = None
    for ci, cls in enumerate(classes):
        run = rng.choice(np.unique(groups[y == ci]))
        idx = np.nonzero(groups == run)[0]
        x = X[idx[0]].astype(float)
        a = ax[ci, 0]
        for c in range(n_ch):
            a.plot(t, x[c] - 4.0 * c, color=CLASS_COLORS[ci], lw=0.6)
        a.set_yticks(-4.0 * np.arange(n_ch))
        a.set_yticklabels([f"ch{c}" for c in range(n_ch)])
        a.set_xlim(0, t[-1])
        a.set_ylabel(cls, fontsize=10, fontweight="bold")
        a.grid(axis="y", visible=False)
        if ci == 0:
            a.set_title(f"one 4-s window, {n_ch} channels (each z-scored)")
        if ci == len(classes) - 1:
            a.set_xlabel("time (s)")
        # mean band correlation over the run's windows
        Cb, _ = filterbank_covariances(X[idx].astype(float), fs, bands)
        for k, bname in enumerate(bands):
            M = Cb[:, k].mean(axis=0)
            d = np.sqrt(np.diag(M))
            R = M / np.outer(d, d)
            a = ax[ci, 1 + k]
            im = a.imshow(R, cmap=_diverging(), vmin=-1, vmax=1)
            a.set_xticks(range(n_ch))
            a.set_yticks(range(n_ch))
            a.set_xticklabels(range(n_ch), fontsize=7)
            a.set_yticklabels(range(n_ch), fontsize=7)
            a.grid(False)
            if ci == 0:
                a.set_title(f"{bname}\nchannel correlation", fontsize=9)
            if ci == len(classes) - 1:
                a.set_xlabel("channel")
    fig.tight_layout()
    cb = fig.colorbar(im, ax=ax[:, 1:].ravel().tolist(), shrink=0.5, pad=0.02)
    cb.set_label("band-limited correlation (run mean)")
    _save(fig, path)


# --------------------------------------------------------------------------
# SW1PerS point clouds, diagrams and CROCKER matrices
# --------------------------------------------------------------------------
def _pca2(P):
    Q = P - P.mean(axis=0)
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    return Q @ vt[:2].T


def sw1pers_clouds(X, y, groups, fs, classes, path, sw_cfg, seed: int = 0, channel: int = 1):
    from .features.sw1pers import h1_diagram, sliding_window_cloud, sw1pers_window
    rng = np.random.default_rng(seed)
    t = np.arange(1000) / fs
    noise = np.random.default_rng(seed + 1).standard_normal(1000)
    refs = [("reference: 10 Hz sine", np.sin(2 * np.pi * 10 * t), INK2),
            ("reference: 10 + 21.3 Hz\n(quasi-periodic)",
             np.sin(2 * np.pi * 10 * t) + 0.7 * np.sin(2 * np.pi * 21.3 * t), INK2),
            ("reference: white noise", noise, INK2)]
    rows = []
    for name, x, col in refs:
        rows.append((name, (x - x.mean()) / x.std(), col))
    Xc = X[:, channel] if X.ndim == 3 else X
    for ci, cls in enumerate(classes):
        run = rng.choice(np.unique(groups[y == ci]))
        i = np.nonzero(groups == run)[0][0]
        rows.append((f"{cls}\n(ch{channel})" if X.ndim == 3 else cls,
                     Xc[i].astype(float), CLASS_COLORS[ci]))
    fig, ax = plt.subplots(len(rows), 4, figsize=(14, 2.55 * len(rows)),
                           gridspec_kw={"width_ratios": [2.2, 1.1, 1.1, 1.5]})
    for r, (name, x, col) in enumerate(rows):
        v, dg, p_dom = sw1pers_window(x, fs, sw_cfg, return_diagrams=True)
        a = ax[r, 0]
        n1 = int(fs)
        a.plot(t[:n1], x[:n1], color=col, lw=0.9)
        a.set_ylabel(name, fontsize=8.5)
        a.set_xlim(0, t[n1 - 1])
        if r == 0:
            a.set_title("first second of the window")
        if r == len(rows) - 1:
            a.set_xlabel("time (s)")
        # point cloud at the dominant period
        P = sliding_window_cloud(x, fs, p_dom, sw_cfg.n_harmonics, sw_cfg.n_points,
                                 sw_cfg.knn_denoise, sw_cfg.seed)
        Z = _pca2(P)
        a = ax[r, 1]
        a.scatter(Z[:, 0], Z[:, 1], s=8, color=col, alpha=0.75, lw=0)
        a.set_aspect("equal", adjustable="datalim")
        a.set_xticks([])
        a.set_yticks([])
        a.text(0.02, 0.98, f"period {p_dom * 1e3:.0f} ms", transform=a.transAxes, va="top",
               fontsize=7.5, color=INK2)
        if r == 0:
            a.set_title(f"SW cloud, PCA (M={2 * sw_cfg.n_harmonics})")
        # H1 diagram (sqrt(3) units)
        d = h1_diagram(P, sw_cfg.coeff)
        a = ax[r, 2]
        a.plot([0, 1], [0, 1], color=INK2, lw=0.8)
        a.scatter(d[:, 0], d[:, 1], s=12, color=col, alpha=0.8, lw=0)
        p = np.sort(d[:, 1] - d[:, 0])[::-1] if len(d) else np.zeros(1)
        mp2 = p[1] if len(p) > 1 else 0.0
        a.text(0.97, 0.03, f"mp1 {p[0]:.2f}\nmp2 {mp2:.2f}", transform=a.transAxes,
               ha="right", va="bottom", fontsize=8, color=INK)
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.set_aspect("equal")
        if r == 0:
            a.set_title("Rips H1 (units of sqrt 3)")
        if r == len(rows) - 1:
            a.set_xlabel("birth")
        a.set_ylabel("death", fontsize=8)
        # CROCKER (segment x scale)
        a = ax[r, 3]
        B = dg["crocker"]
        grid = np.linspace(*sw_cfg.crocker_grid, sw_cfg.crocker_resolution)
        segs = np.arange(B.shape[0]) * sw_cfg.crocker_hop
        a.imshow(B.T, origin="lower", aspect="auto", cmap="Blues", vmin=0,
                 vmax=max(3, B.max()),
                 extent=[segs[0], segs[-1] + sw_cfg.crocker_hop, grid[0], grid[-1]])
        a.grid(False)
        a.set_ylabel("Rips scale", fontsize=8)
        if r == 0:
            a.set_title("CROCKER: Betti-1 over time x scale")
        if r == len(rows) - 1:
            a.set_xlabel("segment start (s)")
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------
# Accuracy bars (one panel per track)
# --------------------------------------------------------------------------
def accuracy_bars(tracks: dict, go_margin: float, path, chance=1 / 3):
    names_per = {t: list(tr["summary"]) for t, tr in tracks.items()}
    n_max = max(len(v) for v in names_per.values())
    fig, axes = plt.subplots(1, len(tracks), figsize=(8.0 * len(tracks), 0.42 * n_max + 1.9),
                             squeeze=False)
    col = {"base": C1, "topo": C2, "cnn": C3}
    lo_x = min(min(r["ci95"][0] for r in tr["summary"].values()) for tr in tracks.values())
    lo_x = max(chance, np.floor((lo_x - 0.03) * 20) / 20)
    for ax, (t, tr) in zip(axes[0], tracks.items()):
        S = tr["summary"]
        names = names_per[t]
        ypos = np.arange(len(names))[::-1]
        for yp, n in zip(ypos, names):
            r = S[n]
            k = _kind(r, n)
            ax.barh(yp, r["accuracy"] - lo_x, left=lo_x, height=0.62, color=col[k], alpha=0.9,
                    hatch="////" if _is_riemann(r) else None, edgecolor="white", lw=0)
            ax.plot(r["ci95"], [yp, yp], color=INK, lw=1.2)
            ax.text(1.005, yp, f"{r['accuracy']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]",
                    va="center", fontsize=7.5, color=INK, transform=ax.get_yaxis_transform())
        ref = S["bandpower"]["accuracy"]
        ax.axvline(ref, color=INK2, lw=1, ls="--")
        ax.axvline(min(ref + go_margin, 1.0), color=C2, lw=1, ls=":")
        ax.text(ref, len(names) - 0.4, "bandpower ", fontsize=7, color=INK2, ha="right")
        ax.text(min(ref + go_margin, 1.0), len(names) - 0.4, " +GO margin", fontsize=7, color=C2)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(lo_x, 1.0)
        d = tr["dataset"]
        ax.set_title(f"{t}-channel track ({d['n_channels']} channel"
                     f"{'s' if d['n_channels'] > 1 else ''}, {d['n_runs']} runs)")
        ax.set_xlabel("window-level accuracy (grouped CV, 95% group-bootstrap CI)")
        ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    handles = [Patch(color=C1, label="non-topological"), Patch(color=C2, label="contains topology"),
               Patch(color=C3, label="1D-CNN (raw signal)"),
               Patch(facecolor="#b9b8b3", hatch="////", edgecolor="white",
                     label="Riemannian (hatched)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def comparison_rows(tracks: dict) -> list:
    rows = []
    for t, tr in tracks.items():
        for c in tr["comparisons"]:
            star = " *" if c["primary"] else ""
            rows.append((f"[{t}] {c['question']}: {c['candidate']} vs {c['reference']}{star}",
                         c["margin"], c["ci95"]))
    return rows


def confusion_grid(tracks: dict, classes, path):
    for t, tr in tracks.items():
        mats = {f"[{t}] {s}": np.array(M) for s, M in tr["confusion"].items()}
        base, ext = os.path.splitext(path)
        confusion_panels(mats, classes, f"{base}_{t}{ext}")


def render_all(results: dict, datasets: dict, fig_dir: str, sw_cfg) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    cfg = results["config"]
    tracks = results["tracks"]
    go = cfg["evaluation"]["go_margin"]
    classes = results["classes"]
    if "multi" in datasets:
        ds = datasets["multi"]
        example_multichannel(ds.X, ds.y, ds.groups, ds.fs, classes,
                             os.path.join(fig_dir, "example_multichannel.png"), seed=cfg["seed"])
    key = "multi" if "multi" in datasets else next(iter(datasets))
    ds = datasets[key]
    sw1pers_clouds(ds.X, ds.y, ds.groups, ds.fs, classes,
                   os.path.join(fig_dir, "sw1pers_clouds.png"), sw_cfg, seed=cfg["seed"])
    accuracy_bars(tracks, go, os.path.join(fig_dir, "accuracy_bars.png"))
    margin_forest(comparison_rows(tracks), go, os.path.join(fig_dir, "margins.png"))
    confusion_grid(tracks, classes, os.path.join(fig_dir, "confusion_matrices.png"))
    for t, tr in tracks.items():
        if tr.get("importance"):
            importance_bars(tr["importance"], os.path.join(fig_dir, f"feature_importance_{t}.png"))
