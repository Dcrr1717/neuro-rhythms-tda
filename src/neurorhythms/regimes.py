"""Parameter regimes for the three classes (HEALTHY, EPILEPTIC, PARKINSONIAN).

Every run (virtual subject) draws its own parameters, so each class is a
*distribution* of dynamics, not a single trajectory. The class-defining
parameters are:

HEALTHY
    Jansen-Rit column in its standard alpha regime (A ~ 3.25 mV, p in the
    120-320 pulses/s band; Jansen & Rit 1995; Grimbert & Faugeras 2006) and a
    sub-threshold STN-GPe loop (loop gain below the Hopf bifurcation, so the
    basal ganglia only show noise-driven, damped beta fluctuations).
EPILEPTIC
    Increased excitatory synaptic gain A (raised excitation/inhibition ratio;
    Wendling et al. 2000, 2002) at lower extrinsic input, the region of the
    (A, p) plane where the column produces large-amplitude spike(-and-slow-
    wave)-like discharges at 2-6 Hz, from sporadic spikes near the boundary to
    sustained rhythmic spiking (Jansen & Rit 1995; Grimbert & Faugeras 2006;
    Touboul et al. 2011). Basal ganglia as in HEALTHY.
PARKINSONIAN
    Cortical column as in HEALTHY, but the reciprocal STN<->GPe coupling is
    strengthened above the Hopf threshold (dopamine depletion; Holgado et al.
    2010; Pavlides et al. 2015), producing a sustained 13-30 Hz beta rhythm
    that reaches the cortex via the BG-thalamo-cortical pathway (Brown 2001;
    Hammond et al. 2007; Litvak et al. 2011).

The Hopf threshold of the STN-GPe loop depends on each run's GPe self-
inhibition and delays, so the loop gain is drawn *relative* to the analytic
threshold returned by :func:`stn_gpe_hopf`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .neural_mass import GPE_B, GPE_M, STN_B, STN_M, TAU_G, TAU_S, RunParams

CLASSES = ("healthy", "epileptic", "parkinsonian")

# Slopes of the Holgado sigmoids at zero net input (the fixed point S=B_S, G=B_G
# by construction of the extrinsic offsets below): F'(0) = 4 (M - B) B / M^2.
DFS0 = 4.0 * (STN_M - STN_B) * STN_B / STN_M ** 2
DFG0 = 4.0 * (GPE_M - GPE_B) * GPE_B / GPE_M ** 2

# Uniform ranges [low, high] per class. Shared (nuisance) ranges are identical
# across classes; only the class-defining ranges differ.
DEFAULT_REGIMES: dict = {
    "shared": {
        "B": [21.5, 22.5],          # inhibitory gain (mV); JR: 22
        "a": [95.0, 105.0],         # excitatory rate constant (s^-1); JR: 100
        "b": [49.0, 55.0],          # inhibitory rate constant (s^-1); JR: 50 (b < ~48 -> spikes)
        "C": [130.0, 140.0],        # connectivity constant; JR: 135
        "p_std": [45.0, 70.0],      # input noise std at 1 ms (JR: U[120,320] has std 57.7)
        "w_gg": [1.5, 5.0],         # GPe self-inhibition
        "w_cs": [2.0, 2.0],         # cortex -> STN weight (scale absorbed in ctx noise)
        "ctx_std": [2.0, 8.0],      # std of cortical drive to STN at 1 ms
        "d_sg": [0.004, 0.007],     # STN -> GPe delay (s); Holgado: 6 ms
        "d_gs": [0.004, 0.007],     # GPe -> STN delay (s); Holgado: 6 ms
        "d_gg": [0.003, 0.005],     # GPe -> GPe delay (s); Holgado: 4 ms
        "kappa": [3.0, 10.0],       # BG -> cortex gain (pulses/s per spk/s)
        "d_bc": [0.008, 0.015],     # BG -> thalamus -> cortex delay (s)
    },
    "healthy": {
        "A": [3.10, 3.40],
        "p_mean": [190.0, 300.0],
        "bg_gain_rel": [0.40, 0.90],   # loop gain / Hopf threshold
    },
    "epileptic": {
        "A": [3.60, 4.20],
        "p_mean": [90.0, 160.0],
        "bg_gain_rel": [0.40, 0.90],
    },
    "parkinsonian": {
        "A": [3.10, 3.40],
        "p_mean": [190.0, 300.0],
        "bg_gain_rel": [1.05, 1.80],
    },
}


def stn_gpe_hopf(w_gg: float, d_sg: float, d_gs: float, d_gg: float,
                 f_max: float = 200.0) -> tuple[float, float]:
    """Hopf threshold of the linearised STN-GPe loop (Holgado et al. 2010).

    Linearising around the fixed point (S, G) = (B_S, B_G) gives the
    characteristic equation

        (1 + tau_S l)(1 + tau_G l + w_gg F_G' e^{-l d_gg}) + K e^{-l (d_sg + d_gs)} = 0,

    with loop gain K = w_sg w_gs F_S' F_G'. On the imaginary axis (l = i w)
    K(w) = -(1 + i w tau_S)(1 + i w tau_G + w_gg F_G' e^{-i w d_gg}) e^{i w D}
    must be real and positive; the smallest such K is the threshold.

    Returns
    -------
    g_crit : float
        Critical product w_sg * w_gs.
    f_hopf : float
        Oscillation frequency (Hz) at onset.
    """
    D = d_sg + d_gs

    def k_of(w):
        return -(1 + 1j * w * TAU_S) * (1 + 1j * w * TAU_G
                                         + w_gg * DFG0 * np.exp(-1j * w * d_gg)) * np.exp(1j * w * D)

    ws = 2 * np.pi * np.linspace(0.5, f_max, 4000)
    im = np.imag(k_of(ws))
    best = None
    for i in np.nonzero(np.sign(im[:-1]) != np.sign(im[1:]))[0]:
        w0 = brentq(lambda w: np.imag(k_of(w)), ws[i], ws[i + 1])
        K = np.real(k_of(w0))
        if K > 0 and (best is None or K < best[0]):
            best = (K, w0)
    if best is None:
        raise RuntimeError("no Hopf crossing found")
    K, w0 = best
    return float(K / (DFS0 * DFG0)), float(w0 / (2 * np.pi))


def sample_class_params(cls: str, n: int, rng: np.random.Generator,
                        regimes: dict | None = None) -> tuple[RunParams, dict]:
    """Draw parameters of ``n`` runs of class ``cls``.

    Returns the :class:`RunParams` and a dict of per-run metadata (e.g. the
    relative BG loop gain and the predicted beta frequency at onset).
    """
    if cls not in CLASSES:
        raise ValueError(f"unknown class {cls!r}")
    reg = regimes or DEFAULT_REGIMES
    spec = {**reg["shared"], **reg[cls]}

    def u(key):
        lo, hi = spec[key]
        return rng.uniform(lo, hi, size=n)

    A, B, a, b, C = u("A"), u("B"), u("a"), u("b"), u("C")
    p_mean, p_std = u("p_mean"), u("p_std")
    w_gg, w_cs, ctx_std = u("w_gg"), u("w_cs"), u("ctx_std")
    d_sg, d_gs, d_gg = u("d_sg"), u("d_gs"), u("d_gg")
    kappa, d_bc = u("kappa"), u("d_bc")
    rel = u("bg_gain_rel")

    g_crit = np.empty(n)
    f_hopf = np.empty(n)
    for i in range(n):
        g_crit[i], f_hopf[i] = stn_gpe_hopf(w_gg[i], d_sg[i], d_gs[i], d_gg[i])
    g = rel * g_crit
    # split the loop gain symmetrically between the two reciprocal pathways
    w_sg = np.sqrt(g)
    w_gs = np.sqrt(g)
    # extrinsic offsets that place the fixed point at the baseline rates
    # (STN 17 spk/s, GPe 75 spk/s): zero net input to both sigmoids.
    ctx_drive = w_gs * GPE_B / w_cs
    str_inh = w_sg * STN_B - w_gg * GPE_B   # net extrinsic input to GPe (may be <0 = excitation)

    params = RunParams(A=A, B=B, a=a, b=b, C=C, p_mean=p_mean, p_std=p_std,
                       w_sg=w_sg, w_gs=w_gs, w_gg=w_gg, w_cs=w_cs,
                       ctx_drive=ctx_drive, ctx_std=ctx_std, str_inh=str_inh,
                       d_sg=d_sg, d_gs=d_gs, d_gg=d_gg, kappa=kappa, d_bc=d_bc)
    meta = {"bg_gain_rel": rel, "bg_g_crit": g_crit, "bg_f_hopf": f_hopf}
    return params, meta


def concat_params(parts: list[RunParams]) -> RunParams:
    """Concatenate several RunParams along the run axis."""
    keys = parts[0].__dict__.keys()
    return RunParams(**{k: np.concatenate([np.asarray(getattr(p, k)) for p in parts]) for k in keys})
