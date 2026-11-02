"""Neural mass models used to synthesise EEG-like signals.

Two classic population-rate models are integrated *together*, vectorised over
independent simulation runs (one run = one virtual subject / recording):

* **Jansen-Rit (JR) cortical column** (Jansen & Rit, 1995): pyramidal cells,
  excitatory and inhibitory interneurons with second-order synaptic kernels.
  The EEG proxy is the mean membrane potential of the pyramidal population,
  ``y1 - y2``.
* **Wilson-Cowan-type STN-GPe loop** (Holgado, Terry & Bogacz, 2010): firing
  rates of the subthalamic nucleus (STN) and the external globus pallidus (GPe)
  with axonal delays. Strong reciprocal STN<->GPe coupling (the dopamine-depleted
  state) destabilises the fixed point through a delay-induced Hopf bifurcation
  and produces beta-band (13-30 Hz) oscillations.

Coupling: the STN rate, high-pass filtered and delayed (basal-ganglia ->
thalamus -> cortex), is added to the extrinsic input ``p(t)`` of the cortical
column. Every class (healthy, epileptic, parkinsonian) is simulated with the
*same* coupled model; only parameter regimes differ (see ``regimes.py``). This
avoids the trivial confound "class = different simulator".

Integration: Euler-Maruyama with a fixed step ``dt`` (default 0.5 ms), then
anti-aliased decimation to the output sampling rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

# --------------------------------------------------------------------------
# Fixed constants (not varied across runs)
# --------------------------------------------------------------------------
# Jansen-Rit sigmoid (Jansen & Rit 1995, Table 1)
JR_E0 = 2.5      # half of the maximum firing rate (s^-1)
JR_V0 = 6.0      # firing threshold (mV)
JR_R = 0.56      # sigmoid steepness (mV^-1)

# Holgado et al. (2010) sigmoids: F(x) = M / (1 + (M - B)/B * exp(-4 x / M))
STN_M, STN_B = 300.0, 17.0   # max / baseline firing rate of STN (spk/s)
GPE_M, GPE_B = 400.0, 75.0   # max / baseline firing rate of GPe (spk/s)
TAU_S = 0.0128               # STN membrane time constant (s)
TAU_G = 0.020                # GPe membrane time constant (s)

# Reference rate for the JR input noise: Jansen & Rit drew p(t) from a
# uniform distribution between 120 and 320 pulses/s at every (1 ms) step. We
# keep that noise *intensity* independent of dt by scaling the per-step
# standard deviation by sqrt(ref_dt / dt).
JR_NOISE_REF_DT = 1e-3


def jr_sigmoid(v: np.ndarray) -> np.ndarray:
    """Potential-to-rate sigmoid of the Jansen-Rit model (s^-1)."""
    return 2.0 * JR_E0 / (1.0 + np.exp(JR_R * (JR_V0 - v)))


def holgado_sigmoid(x: np.ndarray, m: float, b: float) -> np.ndarray:
    """Input-to-rate sigmoid of Holgado et al. (2010); F(0) = b, F(inf) = m."""
    z = np.clip(-4.0 * x / m, -50.0, 50.0)
    return m / (1.0 + (m - b) / b * np.exp(z))


@dataclass
class RunParams:
    """Per-run parameters; each attribute is an array of shape (n_runs,)."""

    # --- Jansen-Rit cortical column
    A: np.ndarray          # excitatory synaptic gain (mV)
    B: np.ndarray          # inhibitory synaptic gain (mV)
    a: np.ndarray          # excitatory rate constant (s^-1)
    b: np.ndarray          # inhibitory rate constant (s^-1)
    C: np.ndarray          # global connectivity constant
    p_mean: np.ndarray     # mean extrinsic input (pulses/s)
    p_std: np.ndarray      # std of extrinsic input at the 1 ms reference step
    # --- STN-GPe loop (Holgado et al. 2010)
    w_sg: np.ndarray       # STN -> GPe (excitatory)
    w_gs: np.ndarray       # GPe -> STN (inhibitory)
    w_gg: np.ndarray       # GPe -> GPe (inhibitory)
    w_cs: np.ndarray       # cortex -> STN (excitatory)
    ctx_drive: np.ndarray  # mean cortical drive to STN (spk/s)
    ctx_std: np.ndarray    # std of cortical drive at the 1 ms reference step
    str_inh: np.ndarray    # striatal inhibition of GPe (spk/s units)
    d_sg: np.ndarray       # STN -> GPe delay (s)
    d_gs: np.ndarray       # GPe -> STN delay (s)
    d_gg: np.ndarray       # GPe -> GPe delay (s)
    # --- basal ganglia -> cortex coupling
    kappa: np.ndarray      # gain from STN rate fluctuation to JR input (pulses/s per spk/s)
    d_bc: np.ndarray       # BG -> thalamus -> cortex delay (s)

    @property
    def n_runs(self) -> int:
        return int(np.asarray(self.A).shape[0])

    def as_dict(self) -> dict:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


def simulate(params: RunParams, duration: float, rng: np.random.Generator,
             dt: float = 5e-4, fs_out: float = 250.0, transient: float = 2.0,
             return_stn: bool = False):
    """Integrate the coupled JR + STN-GPe model for every run in parallel.

    Parameters
    ----------
    params : RunParams
        Arrays of shape (n_runs,).
    duration : float
        Length of the returned signal (s), after discarding ``transient``.
    rng : numpy Generator
    dt : float
        Integration step (s).
    fs_out : float
        Output sampling rate (Hz); must divide 1/dt.
    transient : float
        Initial segment discarded (s).
    return_stn : bool
        Also return the decimated STN firing rate.

    Returns
    -------
    eeg : (n_runs, n_samples) array, the JR pyramidal potential y1 - y2 (mV).
    stn : (n_runs, n_samples) array, only if ``return_stn``.
    """
    P = params.as_dict()
    n = params.n_runs
    fs_int = 1.0 / dt
    q = int(round(fs_int / fs_out))
    if abs(q * fs_out - fs_int) > 1e-6:
        raise ValueError("fs_out must divide the integration rate 1/dt")
    n_keep = int(round(duration * fs_out))
    n_steps = int(round((duration + transient) * fs_int))
    n_trans = int(round(transient * fs_out))

    # Delays in integration steps (>= 1)
    k_sg = np.maximum(1, np.round(P["d_sg"] / dt).astype(int))
    k_gs = np.maximum(1, np.round(P["d_gs"] / dt).astype(int))
    k_gg = np.maximum(1, np.round(P["d_gg"] / dt).astype(int))
    k_bc = np.maximum(1, np.round(P["d_bc"] / dt).astype(int))
    hist_len = int(max(k_sg.max(), k_gs.max(), k_gg.max(), k_bc.max())) + 1
    idx = np.arange(n)

    # State
    y = np.zeros((6, n))          # JR: y0,y1,y2 and derivatives y3,y4,y5
    S = np.full(n, STN_B)         # STN rate (spk/s)
    G = np.full(n, GPE_B)         # GPe rate (spk/s)
    S_hist = np.tile(S, (hist_len, 1))
    G_hist = np.tile(G, (hist_len, 1))
    S_ema = S.copy()              # slow running mean of S (high-pass for the BG->ctx input)
    ema_alpha = dt / 0.5          # 0.5 s time constant

    A, B, a, b, C = P["A"], P["B"], P["a"], P["b"], P["C"]
    C1, C2, C3, C4 = C, 0.8 * C, 0.25 * C, 0.25 * C
    noise_scale = np.sqrt(JR_NOISE_REF_DT / dt)
    p_sd = P["p_std"] * noise_scale
    c_sd = P["ctx_std"] * noise_scale

    out_eeg = np.empty((n_steps // q + 1, n))
    out_stn = np.empty_like(out_eeg) if return_stn else None
    acc_e = np.zeros(n)
    acc_s = np.zeros(n)
    j = 0

    for t in range(n_steps):
        h = t % hist_len
        # delayed values from ring buffers
        S_d_sg = S_hist[(h - k_sg) % hist_len, idx]
        G_d_gs = G_hist[(h - k_gs) % hist_len, idx]
        G_d_gg = G_hist[(h - k_gg) % hist_len, idx]
        S_d_bc = S_hist[(h - k_bc) % hist_len, idx]

        # ---- STN-GPe (Holgado et al. 2010)
        ctx = P["ctx_drive"] + c_sd * rng.standard_normal(n)
        dS = (holgado_sigmoid(P["w_cs"] * ctx - P["w_gs"] * G_d_gs, STN_M, STN_B) - S) / TAU_S
        dG = (holgado_sigmoid(P["w_sg"] * S_d_sg - P["w_gg"] * G_d_gg - P["str_inh"],
                              GPE_M, GPE_B) - G) / TAU_G

        # ---- Jansen-Rit cortical column
        p = (P["p_mean"] + p_sd * rng.standard_normal(n)
             + P["kappa"] * (S_d_bc - S_ema))
        y0, y1, y2, y3, y4, y5 = y
        dy0 = y3
        dy1 = y4
        dy2 = y5
        dy3 = A * a * jr_sigmoid(y1 - y2) - 2.0 * a * y3 - a * a * y0
        dy4 = A * a * (p + C2 * jr_sigmoid(C1 * y0)) - 2.0 * a * y4 - a * a * y1
        dy5 = B * b * C4 * jr_sigmoid(C3 * y0) - 2.0 * b * y5 - b * b * y2

        y[0] += dt * dy0
        y[1] += dt * dy1
        y[2] += dt * dy2
        y[3] += dt * dy3
        y[4] += dt * dy4
        y[5] += dt * dy5
        S = S + dt * dS
        G = G + dt * dG
        S_ema += ema_alpha * (S - S_ema)
        h1 = (t + 1) % hist_len
        S_hist[h1] = S
        G_hist[h1] = G

        # block-average then decimate (first anti-alias stage)
        acc_e += y[1] - y[2]
        if return_stn:
            acc_s += S
        if (t + 1) % q == 0:
            out_eeg[j] = acc_e / q
            acc_e[:] = 0.0
            if return_stn:
                out_stn[j] = acc_s / q
                acc_s[:] = 0.0
            j += 1

    eeg = out_eeg[:j].T[:, n_trans:n_trans + n_keep]
    if not return_stn:
        return np.ascontiguousarray(eeg)
    stn = out_stn[:j].T[:, n_trans:n_trans + n_keep]
    return np.ascontiguousarray(eeg), np.ascontiguousarray(stn)


def pink_noise(n_runs: int, n_samples: int, rng: np.random.Generator,
               exponent: float = 1.0) -> np.ndarray:
    """Unit-variance 1/f^exponent noise (background / measurement noise)."""
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]
    spec = (rng.standard_normal((n_runs, freqs.size))
            + 1j * rng.standard_normal((n_runs, freqs.size))) / freqs ** (exponent / 2.0)
    x = np.fft.irfft(spec, n=n_samples, axis=1)
    x -= x.mean(axis=1, keepdims=True)
    x /= x.std(axis=1, keepdims=True) + 1e-12
    return x


def highpass(x: np.ndarray, fs: float, fc: float = 0.5) -> np.ndarray:
    """Zero-phase Butterworth high-pass (removes DC offset / slow drift, like an EEG amplifier)."""
    sos = sps.butter(2, fc, btype="highpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x, axis=-1)
