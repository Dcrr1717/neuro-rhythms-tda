"""Feature views and their combinations.

Base blocks
-----------
bandpower, hjorth, waveform, psd  (non-topological, ``baseline.py``)
topo                               (persistent homology, ``topology.py``)

Feature sets evaluated (concatenations of blocks)
-------------------------------------------------
Brief-named baselines (what the task statement lists):
    bandpower, hjorth, bandpower+hjorth
Strict non-topological baselines:
    waveform, psd, baselines_union = bandpower+hjorth+waveform+psd
Topological-spectral descriptor and its parts:
    topo, descriptor = psd+topo
Incremental tests:
    bandpower+hjorth+topo  (topology added to the brief-named baselines)
    union = baselines_union+topo  (topology added to every baseline)
Sensitivity (compact topological vector, summary statistics only):
    topo_summary, union_summary = baselines_union+topo_summary
"""
from __future__ import annotations

import numpy as np

from .baseline import bandpower_features, hjorth_features, psd_features, waveform_features
from .topology import STAT_NAMES, TopoConfig, topo_feature_families, topo_features

BLOCKS = ("bandpower", "hjorth", "waveform", "psd", "topo", "topo_summary")

FEATURE_SETS = {
    "bandpower": ("bandpower",),
    "hjorth": ("hjorth",),
    "bandpower+hjorth": ("bandpower", "hjorth"),
    "waveform": ("waveform",),
    "psd": ("psd",),
    "baselines_union": ("bandpower", "hjorth", "waveform", "psd"),
    "topo": ("topo",),
    "descriptor": ("psd", "topo"),
    "bandpower+hjorth+topo": ("bandpower", "hjorth", "topo"),
    "union": ("bandpower", "hjorth", "waveform", "psd", "topo"),
    # sensitivity variant (declared before the full run): persistence summary
    # statistics only (10 per diagram), no Betti curves / landscapes
    "topo_summary": ("topo_summary",),
    "union_summary": ("bandpower", "hjorth", "waveform", "psd", "topo_summary"),
}
BRIEF_BASELINES = ("bandpower", "hjorth", "bandpower+hjorth")
STRICT_BASELINES = BRIEF_BASELINES + ("waveform", "psd", "baselines_union")
TOPO_SETS = ("topo", "descriptor", "bandpower+hjorth+topo", "union", "topo_summary",
             "union_summary")


def compute_blocks(X: np.ndarray, fs: float, topo_cfg: TopoConfig | None = None,
                   n_jobs: int = 1) -> dict:
    """Compute every base block. Returns {block: (F, names, families)}."""
    X = np.asarray(X, dtype=float)
    out = {}
    F, n = bandpower_features(X, fs)
    out["bandpower"] = (F, n, ["bandpower"] * len(n))
    F, n = hjorth_features(X)
    out["hjorth"] = (F, n, ["hjorth"] * len(n))
    F, n = waveform_features(X)
    out["waveform"] = (F, n, ["waveform"] * len(n))
    F, n = psd_features(X, fs)
    out["psd"] = (F, n, ["psd"] * len(n))
    topo_cfg = topo_cfg or TopoConfig()
    F, n = topo_features(X, topo_cfg, n_jobs=n_jobs)
    out["topo"] = (F, n, topo_feature_families(topo_cfg))
    return out


def summary_block(topo_block):
    """Columns of the topological block that are diagram summary statistics."""
    F, names, fams = topo_block
    keep = [i for i, n in enumerate(names) if n.split("_", 2)[-1] in STAT_NAMES]
    return F[:, keep], [names[i] for i in keep], [fams[i] for i in keep]


def assemble(blocks: dict, feature_set: str):
    """Concatenate the blocks of one feature set -> (F, names, families)."""
    parts = FEATURE_SETS[feature_set]
    if "topo_summary" in parts and "topo_summary" not in blocks:
        blocks = {**blocks, "topo_summary": summary_block(blocks["topo"])}
    F = np.concatenate([blocks[b][0] for b in parts], axis=1)
    names = sum((list(blocks[b][1]) for b in parts), [])
    fams = sum((list(blocks[b][2]) for b in parts), [])
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0), names, fams
