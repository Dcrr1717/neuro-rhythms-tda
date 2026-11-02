"""Topological view: persistent homology of each window, computed with GUDHI.

Two complementary filtrations are used:

1. **Sublevel-set filtration of the signal** (a 1-D cubical complex on the
   samples, ``gudhi.CubicalComplex(vertices=x)``). Its H0 diagram pairs every
   local minimum with a local maximum; the persistence of a pair is the height
   of the corresponding oscillation "valley". The superlevel filtration
   (sublevel of ``-x``) does the same for peaks. Sharp, tall spikes and
   asymmetric spike-and-wave complexes leave a very different trace from
   symmetric alpha / beta waves.
2. **Time-delay (Takens) embedding** ``(x_t, x_{t+tau}, x_{t+2 tau})`` at
   several fixed delays, filtered by the Alpha complex
   (``gudhi.AlphaComplex``). A rhythm traces a loop in the embedding; the H1
   diagram measures how large and how clean the loop is, H0 measures how the
   point cloud clusters (e.g. the long slow segments between spikes).

Each persistence diagram is vectorised into a fixed-length vector:
summary statistics, a Betti curve and the first persistence landscapes
(Bubenik 2015), all on *fixed* grids chosen before any classification (no
range is fitted on the data, so nothing leaks across folds).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # GUDHI is a hard requirement of the topological view
    import gudhi
except ImportError as exc:  # pragma: no cover - exercised only without gudhi
    raise ImportError("The topological view requires GUDHI: pip install gudhi") from exc


@dataclass
class TopoConfig:
    delays: tuple = (2, 5, 10)          # embedding delays in samples (8, 20, 40 ms at 250 Hz)
    embed_dim: int = 3
    # Fixed grids. Chosen once from the value ranges of an unlabeled pilot
    # simulation (seed 999, 12 runs; ~1st-99th percentiles), never refitted.
    sublevel_range: tuple = (-3.0, 3.0)  # z-units
    alpha_range_h0: tuple = (0.0, 0.35)  # radius units (z-scored signal)
    alpha_range_h1: tuple = (0.0, 0.8)
    resolution: int = 16                 # grid points of Betti curves / landscapes
    n_landscapes: int = 2
    # persistence floors for the "n_long" statistic (~ pilot median / q99)
    count_threshold_sublevel: float = 0.5
    count_threshold_alpha_h0: float = 0.15
    count_threshold_alpha_h1: float = 0.05


# --------------------------------------------------------------------------
# Persistence diagrams
# --------------------------------------------------------------------------
def _finite(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float).reshape(-1, 2)
    return d[np.isfinite(d).all(axis=1)]


def sublevel_diagram(x: np.ndarray) -> np.ndarray:
    """H0 diagram of the sublevel-set filtration of a 1-D signal (finite bars)."""
    cc = gudhi.CubicalComplex(vertices=np.asarray(x, dtype=float))
    cc.compute_persistence()
    return _finite(cc.persistence_intervals_in_dimension(0))


def delay_embedding(x: np.ndarray, tau: int, dim: int = 3) -> np.ndarray:
    n = len(x) - (dim - 1) * tau
    if n <= dim:
        raise ValueError("window too short for this embedding")
    return np.column_stack([x[i * tau: i * tau + n] for i in range(dim)])


def alpha_diagrams(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """H0 and H1 diagrams of the Alpha filtration, in radius (not squared) units."""
    st = gudhi.AlphaComplex(points=points).create_simplex_tree()
    st.compute_persistence()
    h0 = np.sqrt(np.maximum(_finite(st.persistence_intervals_in_dimension(0)), 0.0))
    h1 = np.sqrt(np.maximum(_finite(st.persistence_intervals_in_dimension(1)), 0.0))
    return h0, h1


# --------------------------------------------------------------------------
# Vectorisations
# --------------------------------------------------------------------------
STAT_NAMES = ("n", "n_long", "total_pers", "max_pers", "second_pers", "mean_pers",
              "std_pers", "pers_entropy", "mean_birth", "mean_death")


def diagram_stats(d: np.ndarray, threshold: float) -> np.ndarray:
    if len(d) == 0:
        return np.zeros(len(STAT_NAMES))
    p = d[:, 1] - d[:, 0]
    ps = np.sort(p)[::-1]
    tot = p.sum()
    q = p / tot if tot > 0 else np.full_like(p, 1.0 / len(p))
    ent = float(-(q * np.log(q + 1e-20)).sum())
    return np.array([len(p), (p > threshold).sum(), tot, ps[0],
                     ps[1] if len(ps) > 1 else 0.0, p.mean(), p.std(), ent,
                     d[:, 0].mean(), d[:, 1].mean()], dtype=float)


def betti_curve(d: np.ndarray, lo: float, hi: float, res: int) -> np.ndarray:
    t = np.linspace(lo, hi, res)
    if len(d) == 0:
        return np.zeros(res)
    return ((d[:, :1] <= t[None, :]) & (d[:, 1:2] > t[None, :])).sum(axis=0).astype(float)


def landscapes(d: np.ndarray, lo: float, hi: float, res: int, k: int) -> np.ndarray:
    """First ``k`` persistence landscapes sampled on ``res`` points (Bubenik 2015)."""
    t = np.linspace(lo, hi, res)
    if len(d) == 0:
        return np.zeros(k * res)
    tents = np.clip(np.minimum(t[None, :] - d[:, :1], d[:, 1:2] - t[None, :]), 0.0, None)
    tents = -np.sort(-tents, axis=0)
    out = np.zeros((k, res))
    m = min(k, tents.shape[0])
    out[:m] = tents[:m]
    return out.ravel()


def vectorize(d: np.ndarray, rng: tuple, cfg: TopoConfig, threshold: float) -> np.ndarray:
    lo, hi = rng
    return np.concatenate([diagram_stats(d, threshold),
                           betti_curve(d, lo, hi, cfg.resolution),
                           landscapes(d, lo, hi, cfg.resolution, cfg.n_landscapes)])


def _block_names(prefix: str, cfg: TopoConfig) -> list[str]:
    names = [f"{prefix}_{s}" for s in STAT_NAMES]
    names += [f"{prefix}_betti{i}" for i in range(cfg.resolution)]
    names += [f"{prefix}_land{k}_{i}" for k in range(cfg.n_landscapes) for i in range(cfg.resolution)]
    return names


def diagram_blocks(cfg: TopoConfig) -> list[tuple[str, str]]:
    """(block prefix, family) for every diagram, in feature order."""
    blocks = [("sub_h0", "topo_sublevel"), ("sup_h0", "topo_sublevel")]
    for tau in cfg.delays:
        blocks += [(f"emb{tau}_h0", "topo_embed_h0"), (f"emb{tau}_h1", "topo_embed_h1")]
    return blocks


def topo_feature_names(cfg: TopoConfig | None = None) -> list[str]:
    cfg = cfg or TopoConfig()
    out = []
    for prefix, _ in diagram_blocks(cfg):
        out += _block_names(prefix, cfg)
    return out


def topo_feature_families(cfg: TopoConfig | None = None) -> list[str]:
    cfg = cfg or TopoConfig()
    per = len(STAT_NAMES) + cfg.resolution * (1 + cfg.n_landscapes)
    return [fam for _, fam in diagram_blocks(cfg) for _ in range(per)]


def window_diagrams(x: np.ndarray, cfg: TopoConfig | None = None) -> dict:
    """All persistence diagrams of one (z-scored) window."""
    cfg = cfg or TopoConfig()
    out = {"sub_h0": sublevel_diagram(x), "sup_h0": sublevel_diagram(-x)}
    for tau in cfg.delays:
        h0, h1 = alpha_diagrams(delay_embedding(x, tau, cfg.embed_dim))
        out[f"emb{tau}_h0"] = h0
        out[f"emb{tau}_h1"] = h1
    return out


def topo_features_window(x: np.ndarray, cfg: TopoConfig | None = None) -> np.ndarray:
    cfg = cfg or TopoConfig()
    dg = window_diagrams(np.asarray(x, dtype=float), cfg)
    parts = []
    for prefix, _ in diagram_blocks(cfg):
        if prefix.startswith(("sub", "sup")):
            parts.append(vectorize(dg[prefix], cfg.sublevel_range, cfg, cfg.count_threshold_sublevel))
        elif prefix.endswith("h0"):
            parts.append(vectorize(dg[prefix], cfg.alpha_range_h0, cfg, cfg.count_threshold_alpha_h0))
        else:
            parts.append(vectorize(dg[prefix], cfg.alpha_range_h1, cfg, cfg.count_threshold_alpha_h1))
    return np.concatenate(parts)


def _chunk(args):
    X, cfg = args
    return np.stack([topo_features_window(x, cfg) for x in X])


def topo_features(X: np.ndarray, cfg: TopoConfig | None = None, n_jobs: int = 1):
    """Topological feature matrix (n_windows, n_features) and feature names."""
    cfg = cfg or TopoConfig()
    X = np.asarray(X, dtype=float)
    if n_jobs == 1 or len(X) < 8:
        F = _chunk((X, cfg))
    else:
        from concurrent.futures import ProcessPoolExecutor
        chunks = np.array_split(X, max(1, 4 * n_jobs))
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            F = np.concatenate(list(ex.map(_chunk, [(c, cfg) for c in chunks if len(c)])))
    return F, topo_feature_names(cfg)
