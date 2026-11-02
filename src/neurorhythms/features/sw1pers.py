"""SW1PerS: sliding windows and 1-persistence scoring of (quasi-)periodicity.

This is the topological method designed for *signals* (Perea & Harer 2015,
"Sliding windows and persistence: an application of topological methods to
signal analysis", Found Comput Math 15:799-838, arXiv:1307.6188; pipeline of
Perea, Deckard, Haase & Harer 2015, BMC Bioinformatics 16:257):

1. **Curve fitting.** Cubic-spline interpolation of the window, so the
   sliding window can use a non-integer delay ``tau``.
2. **Sliding-window (time-delay) embedding.**
   ``SW_{M,tau} f(t) = (f(t), f(t + tau), ..., f(t + M tau))`` in R^{M+1}, at
   ``T`` start times ``t`` spread over the window. ``M = 2N`` where ``N`` is
   the number of Fourier harmonics to resolve, and ``M tau = P M / (M + 1)``
   for a target period ``P`` (Perea & Harer 2015, the window length that
   maximises the persistence of a period-``P`` signal).
3. **Pointwise centring and normalisation.** Every SW vector is mean-centred
   and divided by its norm: the cloud lies on the unit sphere, so the score
   is invariant to amplitude, offset and slow amplitude modulation.
4. **Denoising.** One step of k-nearest-neighbour mean-shift (each point is
   replaced by the mean of its ``k`` nearest neighbours, then re-normalised).
5. **Persistent homology.** Vietoris-Rips H1 with coefficients in Z_11
   (``ripser``). A periodic signal traces a closed curve; its H1 class is born
   early and dies late. The maximum persistence ``mp1`` divided by sqrt(3)
   (the largest persistence a circle on the unit sphere can have under the
   Rips diameter convention) is in [0, 1]; the SW1PerS periodicity score is
   ``1 - mp1``. Two incommensurate rhythms embed on a torus, which shows as a
   *second* persistent H1 class (``mp2``): the quasi-periodicity signature
   (Perea & Harer 2015, section 5; Tralie & Perea 2018).

Scales. Rhythms of interest differ by class (2-6 Hz spike-and-wave, ~10 Hz
alpha, 13-30 Hz beta), so each window is scored at fixed target periods
(``target_freqs``) and at its own dominant period, estimated without labels
from the highest autocorrelation peak in ``adaptive_range``.

CROCKER-style temporal summary (Topaz, Ziegelmeier & Halverson 2015). The
window is cut into overlapping sub-segments; at the dominant period each
segment gets its own SW1PerS H1 diagram, and the Betti-1 number is read on a
fixed grid of Rips scales. The resulting (segment x scale) matrix is a CROCKER
plot of how the loop structure evolves over the trace; we keep its mean and
standard deviation over segments, plus the mean / std / min of the
per-segment ``mp1`` (stability of the periodicity over time).

All parameters are fixed from the literature defaults above, never fitted to
labelled data; grids are fixed in normalised units, so nothing leaks across
CV folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

try:
    from ripser import ripser
except ImportError as exc:  # pragma: no cover
    raise ImportError("The SW1PerS view requires ripser: pip install ripser") from exc

SQRT3 = float(np.sqrt(3.0))
SCALE_STATS = ("mp1", "mp2", "birth1", "total_pers", "entropy")


@dataclass
class SW1PersConfig:
    target_freqs: tuple = (4.0, 10.0, 20.0)   # fixed scales (Hz): spike-wave, alpha, beta
    adaptive: bool = True                       # + the window's own dominant period
    adaptive_range: tuple = (2.0, 35.0)         # Hz, search range of the dominant period
    n_harmonics: int = 6                        # N; embedding dimension M = 2N
    n_points: int = 160                         # T, sliding-window points per cloud
    knn_denoise: int = 8                        # mean-shift neighbours (0 = off)
    coeff: int = 11                             # Z_p coefficients, as in SW1PerS
    crocker: bool = True
    crocker_segment: float = 1.0                # s
    crocker_hop: float = 0.5                    # s
    crocker_points: int = 80
    crocker_grid: tuple = (0.1, 1.7)            # Rips scale range (unit-sphere chord units)
    crocker_resolution: int = 10
    seed: int = 0                               # start-time sampling (label independent)

    @property
    def scale_names(self) -> list[str]:
        names = [f"f{f:g}Hz" for f in self.target_freqs]
        return names + (["dominant"] if self.adaptive else [])


# --------------------------------------------------------------------------
# Core: one point cloud, one diagram
# --------------------------------------------------------------------------
def sliding_window_cloud(x: np.ndarray, fs: float, period: float, n_harmonics: int = 6,
                         n_points: int = 160, knn: int = 8, seed: int = 0,
                         spline: CubicSpline | None = None, t0: float = 0.0,
                         t1: float | None = None) -> np.ndarray:
    """SW1PerS point cloud of ``x`` for a target ``period`` (s), on the unit sphere.

    Start times are drawn uniformly (fixed seed) from ``[t0, t1 - window]``, so
    the sampling cannot lock onto the period (no aliasing).
    """
    x = np.asarray(x, dtype=float)
    M = 2 * int(n_harmonics)
    win = period * M / (M + 1.0)
    tau = win / M
    t = np.arange(len(x)) / fs
    cs = spline if spline is not None else CubicSpline(t, x)
    t1 = t[-1] if t1 is None else t1
    if t1 - t0 <= win:
        raise ValueError("segment shorter than one sliding window")
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(t0, t1 - win, size=int(n_points)))
    pts = cs(starts[:, None] + tau * np.arange(M + 1)[None, :])
    pts -= pts.mean(axis=1, keepdims=True)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    if knn and knn > 1:
        sim = pts @ pts.T
        nn = np.argpartition(-sim, kth=knn - 1, axis=1)[:, :knn]
        pts = pts[nn].mean(axis=1)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12
    return pts


def h1_diagram(points: np.ndarray, coeff: int = 11) -> np.ndarray:
    """Rips H1 diagram (finite bars), in units of sqrt(3) (so persistence <= 1)."""
    d = ripser(points, maxdim=1, coeff=coeff)["dgms"][1]
    d = np.asarray(d, dtype=float).reshape(-1, 2)
    d = d[np.isfinite(d).all(axis=1)]
    return d / SQRT3


def scale_stats(d: np.ndarray) -> np.ndarray:
    """(mp1, mp2, birth of the most persistent class, total persistence, entropy)."""
    if len(d) == 0:
        return np.zeros(len(SCALE_STATS))
    p = d[:, 1] - d[:, 0]
    order = np.argsort(p)[::-1]
    ps = p[order]
    tot = p.sum()
    q = p / tot if tot > 0 else np.full_like(p, 1.0 / len(p))
    ent = float(-(q * np.log(q + 1e-20)).sum())
    return np.array([ps[0], ps[1] if len(ps) > 1 else 0.0, d[order[0], 0], tot, ent])


def dominant_period(x: np.ndarray, fs: float, f_range=(2.0, 35.0)) -> float:
    """Lag of the highest autocorrelation peak with frequency in ``f_range`` (s).

    Falls back to the 10 Hz period when the autocorrelation has no local
    maximum in range (e.g. white noise with a monotone decay).
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(np.abs(f) ** 2)[:n]
    ac /= ac[0] + 1e-20
    lo = max(1, int(np.floor(fs / f_range[1])))
    hi = min(n - 2, int(np.ceil(fs / f_range[0])))
    seg = ac[lo - 1: hi + 2]
    peaks = np.nonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + lo
    if len(peaks) == 0:
        return 0.1
    return float(peaks[np.argmax(ac[peaks])] / fs)


def betti1_curve(d: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Betti-1 at each Rips scale of ``grid`` (diagram in sqrt(3) units, grid in chord units)."""
    if len(d) == 0:
        return np.zeros(len(grid))
    g = np.asarray(grid) / SQRT3
    return ((d[:, :1] <= g[None, :]) & (d[:, 1:2] > g[None, :])).sum(axis=0).astype(float)


# --------------------------------------------------------------------------
# Feature vector of one single-channel window
# --------------------------------------------------------------------------
CROCKER_STATS = ("seg_mp1_mean", "seg_mp1_std", "seg_mp1_min")


def sw1pers_feature_names(cfg: SW1PersConfig | None = None) -> tuple[list[str], list[str]]:
    """(names, families) of the per-channel feature vector."""
    cfg = cfg or SW1PersConfig()
    names, fams = [], []
    for s in cfg.scale_names:
        names += [f"sw_{s}_{k}" for k in SCALE_STATS]
        fams += ["sw1pers"] * len(SCALE_STATS)
    if cfg.crocker:
        r = cfg.crocker_resolution
        names += [f"crocker_mean_b1_{i}" for i in range(r)]
        names += [f"crocker_std_b1_{i}" for i in range(r)]
        names += [f"crocker_{k}" for k in CROCKER_STATS]
        fams += ["crocker"] * (2 * r + len(CROCKER_STATS))
    return names, fams


def sw1pers_window(x: np.ndarray, fs: float, cfg: SW1PersConfig | None = None,
                   return_diagrams: bool = False):
    """SW1PerS feature vector of one z-scored single-channel window."""
    cfg = cfg or SW1PersConfig()
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x)) / fs
    cs = CubicSpline(t, x)
    p_dom = dominant_period(x, fs, cfg.adaptive_range)
    periods = [1.0 / f for f in cfg.target_freqs] + ([p_dom] if cfg.adaptive else [])
    feats, diagrams = [], {}
    for name, P in zip(cfg.scale_names, periods):
        pts = sliding_window_cloud(x, fs, P, cfg.n_harmonics, cfg.n_points, cfg.knn_denoise,
                                   cfg.seed, spline=cs)
        d = h1_diagram(pts, cfg.coeff)
        diagrams[name] = d
        feats.append(scale_stats(d))
    if cfg.crocker:
        grid = np.linspace(*cfg.crocker_grid, cfg.crocker_resolution)
        starts = np.arange(0.0, t[-1] - cfg.crocker_segment + 1e-9, cfg.crocker_hop)
        rows, mps = [], []
        for s in starts:
            pts = sliding_window_cloud(x, fs, p_dom, cfg.n_harmonics, cfg.crocker_points,
                                       cfg.knn_denoise, cfg.seed, spline=cs, t0=s,
                                       t1=s + cfg.crocker_segment)
            d = h1_diagram(pts, cfg.coeff)
            rows.append(betti1_curve(d, grid))
            mps.append(scale_stats(d)[0])
        B = np.array(rows)
        mps = np.array(mps)
        diagrams["crocker"] = B
        feats.append(np.concatenate([B.mean(axis=0), B.std(axis=0),
                                     [mps.mean(), mps.std(), mps.min()]]))
    v = np.concatenate(feats)
    if return_diagrams:
        return v, diagrams, p_dom
    return v


def _chunk(args):
    X, fs, cfg = args
    return np.stack([sw1pers_window(x, fs, cfg) for x in X])


def sw1pers_features(X: np.ndarray, fs: float, cfg: SW1PersConfig | None = None,
                     n_jobs: int = 1):
    """SW1PerS features of single-channel windows (n, n_samples) or of every
    channel of multi-channel windows (n, n_channels, n_samples).

    Multi-channel input gives the per-channel vectors concatenated in channel
    order (names prefixed ``ch{c}_``). Returns (F, names, families).
    """
    cfg = cfg or SW1PersConfig()
    X = np.asarray(X, dtype=float)
    multi = X.ndim == 3
    flat = X.reshape(-1, X.shape[-1])
    if n_jobs == 1 or len(flat) < 8:
        F = _chunk((flat, fs, cfg))
    else:
        from concurrent.futures import ProcessPoolExecutor
        chunks = np.array_split(flat, max(1, 8 * n_jobs))
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            F = np.concatenate(list(ex.map(_chunk, [(c, fs, cfg) for c in chunks if len(c)])))
    names, fams = sw1pers_feature_names(cfg)
    if not multi:
        return F, names, fams
    n, c = X.shape[:2]
    F = F.reshape(n, c * F.shape[1])
    names = [f"ch{i}_{nm}" for i in range(c) for nm in names]
    return F, names, fams * c
