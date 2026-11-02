"""Multi-channel extension: a small network of coupled Jansen-Rit columns.

The single-channel generator (``neural_mass.simulate``) is one cortical column
driven by the STN-GPe loop. Here every run (virtual subject) is a small
*network* of ``n_sources`` Jansen-Rit columns, all driven by the same STN-GPe
loop, coupled to each other by delayed long-range excitatory connections
(Jansen & Rit 1995 used two coupled columns; David & Friston 2003 generalised
this to networks), and seen through a ``n_channels`` x ``n_sources`` lead
field (volume conduction), so cross-channel structure exists:

* **inter-column coupling.** Column ``i`` receives
  ``K * sum_j W_ij (r_j(t - D_ij) - rbar_j(t))`` pulses/s on top of its
  extrinsic input, where ``r_j = S(y1_j - y2_j)`` is the firing rate of the
  pyramidal population of column ``j``, ``rbar_j`` its slow running mean
  (0.5 s) and ``D_ij`` a cortico-cortical conduction delay. As for the
  basal-ganglia input of the single-channel model, only *fluctuations* are
  transmitted, so the coupling does not move the operating point of a
  column. ``W`` is row-normalised (sum_j W_ij = 1) and ring-shaped
  (neighbours strong, the opposite column weak).
* **common basal-ganglia drive.** One STN-GPe loop per run; its high-passed,
  delayed STN rate enters every column with a column-specific gain
  ``kappa * u_i``.
* **lead field.** Sources sit on a circle at 45, 135, 225 and 315 degrees,
  sensors at equally spaced angles on the same circle; the gain decays as a
  Gaussian of the chord distance, with a per-run multiplicative jitter
  (head geometry / electrode placement). Every sensor therefore sees a
  mixture of sources, like scalp EEG.

All parameter ranges are in ``regimes.NETWORK_REGIMES``; the per-column
class-defining ranges are the single-channel ones (``regimes.DEFAULT_REGIMES``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .neural_mass import (GPE_B, GPE_M, JR_NOISE_REF_DT, STN_B, STN_M, TAU_G, TAU_S,
                          holgado_sigmoid, jr_sigmoid)

# per-column arrays have shape (n_runs, n_sources)
COLUMN_KEYS = ("A", "B", "a", "b", "C", "p_mean", "p_std", "kappa")
# per-run scalars (STN-GPe loop and its route to the cortex)
LOOP_KEYS = ("w_sg", "w_gs", "w_gg", "w_cs", "ctx_drive", "ctx_std", "str_inh",
             "d_sg", "d_gs", "d_gg", "d_bc")


@dataclass
class NetworkParams:
    """Per-run parameters of the column network.

    Column parameters have shape (n_runs, n_sources); loop parameters and ``K``
    have shape (n_runs,); ``W`` and ``D`` have shape (n_runs, n_sources, n_sources).
    """

    A: np.ndarray
    B: np.ndarray
    a: np.ndarray
    b: np.ndarray
    C: np.ndarray
    p_mean: np.ndarray
    p_std: np.ndarray
    kappa: np.ndarray
    w_sg: np.ndarray
    w_gs: np.ndarray
    w_gg: np.ndarray
    w_cs: np.ndarray
    ctx_drive: np.ndarray
    ctx_std: np.ndarray
    str_inh: np.ndarray
    d_sg: np.ndarray
    d_gs: np.ndarray
    d_gg: np.ndarray
    d_bc: np.ndarray
    K: np.ndarray          # inter-column coupling gain (pulses/s per s^-1)
    W: np.ndarray          # row-normalised connection weights, zero diagonal
    D: np.ndarray          # inter-column delays (s)

    @property
    def n_runs(self) -> int:
        return int(np.asarray(self.A).shape[0])

    @property
    def n_sources(self) -> int:
        return int(np.asarray(self.A).shape[1])

    def subset(self, sl) -> "NetworkParams":
        return NetworkParams(**{k: np.asarray(v)[sl] for k, v in self.__dict__.items()})


def concat_network_params(parts: list[NetworkParams]) -> NetworkParams:
    keys = parts[0].__dict__.keys()
    return NetworkParams(**{k: np.concatenate([np.asarray(getattr(p, k)) for p in parts])
                            for k in keys})


def simulate_network(params: NetworkParams, duration: float, rng: np.random.Generator,
                     dt: float = 5e-4, fs_out: float = 250.0,
                     transient: float = 2.0) -> np.ndarray:
    """Integrate the column network (Euler-Maruyama), vectorised over runs and columns.

    Returns the pyramidal potential ``y1 - y2`` of every column,
    shape (n_runs, n_sources, n_samples), at ``fs_out``.
    """
    P = params.__dict__
    n, c = params.n_runs, params.n_sources
    fs_int = 1.0 / dt
    q = int(round(fs_int / fs_out))
    if abs(q * fs_out - fs_int) > 1e-6:
        raise ValueError("fs_out must divide the integration rate 1/dt")
    n_keep = int(round(duration * fs_out))
    n_steps = int(round((duration + transient) * fs_int))
    n_trans = int(round(transient * fs_out))

    k_sg = np.maximum(1, np.round(P["d_sg"] / dt).astype(int))
    k_gs = np.maximum(1, np.round(P["d_gs"] / dt).astype(int))
    k_gg = np.maximum(1, np.round(P["d_gg"] / dt).astype(int))
    k_bc = np.maximum(1, np.round(P["d_bc"] / dt).astype(int))
    k_cc = np.maximum(1, np.round(P["D"] / dt).astype(int))          # (n, c, c)
    hist_len = int(max(k_sg.max(), k_gs.max(), k_gg.max(), k_bc.max(), k_cc.max())) + 1
    idx = np.arange(n)
    ii = idx[:, None, None]
    jj = np.arange(c)[None, None, :]

    y = np.zeros((6, n, c))
    S = np.full(n, STN_B)
    G = np.full(n, GPE_B)
    S_hist = np.tile(S, (hist_len, 1))
    G_hist = np.tile(G, (hist_len, 1))
    r0 = jr_sigmoid(np.zeros((n, c)))
    R_hist = np.tile(r0, (hist_len, 1, 1))                           # (L, n, c)
    S_ema = S.copy()
    R_ema = r0.copy()
    ema_alpha = dt / 0.5

    A, B, a, b, C = P["A"], P["B"], P["a"], P["b"], P["C"]
    C1, C2, C3, C4 = C, 0.8 * C, 0.25 * C, 0.25 * C
    noise_scale = np.sqrt(JR_NOISE_REF_DT / dt)
    p_sd = P["p_std"] * noise_scale
    c_sd = P["ctx_std"] * noise_scale
    KW = P["K"][:, None, None] * P["W"]                               # (n, c, c)
    kappa = P["kappa"]

    out = np.empty((n_steps // q + 1, n, c))
    acc = np.zeros((n, c))
    j = 0
    for t in range(n_steps):
        h = t % hist_len
        S_d_sg = S_hist[(h - k_sg) % hist_len, idx]
        G_d_gs = G_hist[(h - k_gs) % hist_len, idx]
        G_d_gg = G_hist[(h - k_gg) % hist_len, idx]
        S_d_bc = S_hist[(h - k_bc) % hist_len, idx]
        R_d = R_hist[(h - k_cc) % hist_len, ii, jj]                   # (n, c, c): r_j(t - D_ij)

        ctx = P["ctx_drive"] + c_sd * rng.standard_normal(n)
        dS = (holgado_sigmoid(P["w_cs"] * ctx - P["w_gs"] * G_d_gs, STN_M, STN_B) - S) / TAU_S
        dG = (holgado_sigmoid(P["w_sg"] * S_d_sg - P["w_gg"] * G_d_gg - P["str_inh"],
                              GPE_M, GPE_B) - G) / TAU_G

        coupling = (KW * (R_d - R_ema[:, None, :])).sum(axis=2)       # (n, c)
        p = (P["p_mean"] + p_sd * rng.standard_normal((n, c))
             + kappa * (S_d_bc - S_ema)[:, None] + coupling)
        y0, y1, y2, y3, y4, y5 = y
        dy3 = A * a * jr_sigmoid(y1 - y2) - 2.0 * a * y3 - a * a * y0
        dy4 = A * a * (p + C2 * jr_sigmoid(C1 * y0)) - 2.0 * a * y4 - a * a * y1
        dy5 = B * b * C4 * jr_sigmoid(C3 * y0) - 2.0 * b * y5 - b * b * y2
        y[0] += dt * y3
        y[1] += dt * y4
        y[2] += dt * y5
        y[3] += dt * dy3
        y[4] += dt * dy4
        y[5] += dt * dy5
        S = S + dt * dS
        G = G + dt * dG
        S_ema += ema_alpha * (S - S_ema)
        v = y[1] - y[2]
        r = jr_sigmoid(v)
        R_ema += ema_alpha * (r - R_ema)
        h1 = (t + 1) % hist_len
        S_hist[h1] = S
        G_hist[h1] = G
        R_hist[h1] = r

        acc += v
        if (t + 1) % q == 0:
            out[j] = acc / q
            acc[:] = 0.0
            j += 1

    src = out[:j].transpose(1, 2, 0)[:, :, n_trans:n_trans + n_keep]
    return np.ascontiguousarray(src)


# --------------------------------------------------------------------------
# Geometry: ring connectivity and lead field
# --------------------------------------------------------------------------
def source_angles(n_sources: int) -> np.ndarray:
    """Sources on a circle, offset by half a step (45, 135, 225, 315 deg for 4)."""
    return 2 * np.pi * (np.arange(n_sources) + 0.5) / n_sources


def sensor_angles(n_channels: int) -> np.ndarray:
    return 2 * np.pi * np.arange(n_channels) / n_channels


def ring_weights(n_sources: int, rng: np.random.Generator, n_runs: int,
                 near=(0.5, 1.0), far=(0.0, 0.3)) -> np.ndarray:
    """Row-normalised ring connectivity: ring neighbours ~ U(near), others ~ U(far)."""
    W = np.zeros((n_runs, n_sources, n_sources))
    for i in range(n_sources):
        for k in range(n_sources):
            if i == k:
                continue
            d = min((i - k) % n_sources, (k - i) % n_sources)
            lo, hi = near if d == 1 else far
            W[:, i, k] = rng.uniform(lo, hi, size=n_runs)
    W /= W.sum(axis=2, keepdims=True) + 1e-12
    return W


def lead_field(n_channels: int, n_sources: int, width: float) -> np.ndarray:
    """Gaussian-of-chord-distance gains (n_channels, n_sources); max gain 1."""
    a_s = source_angles(n_sources)
    a_c = sensor_angles(n_channels)
    d = 2.0 * np.abs(np.sin((a_c[:, None] - a_s[None, :]) / 2.0))    # chord on unit circle
    return np.exp(-0.5 * (d / width) ** 2)
