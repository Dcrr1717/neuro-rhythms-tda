# The science behind the benchmark

*Written for a neuroscientist who wants to know what was simulated, why those
parameter regimes, what each feature measures physiologically, and how much
the result can and cannot say. Numbers quoted here come from
[`RESULTS.md`](RESULTS.md) (first benchmark, sections 1-7) and
[`SW1PERS_RESULTS.md`](SW1PERS_RESULTS.md) (second benchmark: SW1PerS,
multi-channel montage and Riemannian baseline, section 8), full
configurations.*

## 1. Question

Can a *topological* description of a single EEG-like trace (persistent
homology of its sublevel sets and of its delay embedding) tell a healthy
rhythm from an epileptic or a parkinsonian one **better than the spectral and
time-domain features clinical neurophysiology already uses** (band powers and
their ratios, spectral edge frequency, Hjorth parameters)? A small
convolutional network on the raw signal is the deep-learning reference.

We answer it on synthetic data where the ground truth is known by
construction, with the Go/No-Go protocol of the GEM benchmark
(`Dcrr1717/gem-geophysical-descriptor`): the topological view must beat the
baselines by at least **+0.05 accuracy**, with a paired 95 % confidence
interval that resamples whole simulation runs.

## 2. The generative model

Each *run* is one virtual recording: a cortical column (the EEG generator)
coupled to the subthalamic-pallidal loop of the basal ganglia. All three
classes use the **same** coupled model and differ only in parameter regimes,
so a classifier cannot win by recognising "which simulator produced this".

```
            cortical drive + noise
                  |
                  v
   +---------- STN  <---- inhibition ----+
   |  (tau_S = 12.8 ms)                  |
   | excitation, delay d_SG       GPe (tau_G = 20 ms) --+
   +----------------------------------> |  self-inhib.  |
                                         +---- d_GG <---+
   STN rate (high-passed), delay d_BC (BG -> thalamus -> cortex)
                  |
                  v
   p(t) = p_mean + noise + kappa * dSTN(t)  --> Jansen-Rit column --> y1 - y2 = "EEG"
```

### 2.1 Cortex: the Jansen-Rit column

The Jansen-Rit model (Jansen & Rit 1995), built on Lopes da Silva et al.
(1974), describes a cortical column with three interacting populations:
pyramidal cells, excitatory interneurons and inhibitory interneurons. Each
population converts incoming firing rate into a mean postsynaptic potential
through a second-order kernel `h(t) = G k t exp(-k t)` (excitatory gain
`A`, rate `a`; inhibitory gain `B`, rate `b`) and converts membrane potential
into firing rate through a sigmoid. The EEG proxy is the net postsynaptic
potential of the pyramidal cells, `y1 - y2`, which is what drives the
dendritic dipoles seen by scalp electrodes.

Its bifurcation structure is well mapped (Grimbert & Faugeras 2006; Touboul
et al. 2011). With the standard values (`A = 3.25 mV`, `B = 22 mV`,
`a = 100 /s`, `b = 50 /s`, `C = 135`) and an extrinsic input `p` in the
120-320 pulses/s range, the column oscillates at ~10 Hz: the **alpha
rhythm**. Raising the excitatory gain `A` (a raised excitation/inhibition
ratio) or lowering the input pushes the column through a saddle-node /
homoclinic boundary into a regime of large-amplitude, sharp, positive
**spikes followed by a slow inhibitory wave**, recurring at 2-6 Hz:
sporadic spikes near the boundary, rhythmic spike-and-wave-like discharges
further in (Jansen & Rit 1995; Wendling et al. 2000, 2002). This is the
standard neural-mass caricature of epileptiform activity.

A practical warning we met while setting the regimes: the alpha / spike
boundary is very sensitive to the inhibitory rate `b`. With `b` only 8 %
below 50 /s (slower GABA-A kinetics) the "healthy" column already spikes.
That is physiologically sensible (impaired inhibition is epileptogenic,
Wendling et al. 2002), but it means `b` cannot be jittered freely as a
nuisance parameter. We keep `b` in [49, 55] /s.

### 2.2 Basal ganglia: the STN-GPe loop

The subthalamic nucleus (STN, excitatory, glutamatergic) and the external
globus pallidus (GPe, inhibitory, GABAergic) form a reciprocally connected
loop that can act as a pacemaker (Plenz & Kitai 1999; Bevan et al. 2002).
We use the Wilson-Cowan-type firing-rate model of Holgado, Terry & Bogacz
(2010), with its time constants (`tau_S = 12.8 ms`, `tau_G = 20 ms`),
sigmoids (STN: max 300, baseline 17 spikes/s; GPe: max 400, baseline
75 spikes/s) and its axonal delays (STN->GPe and GPe->STN ~6 ms,
GPe->GPe ~4 ms; we draw them in 4-7 ms and 3-5 ms).

Linearising around the baseline rates gives the characteristic equation

```
(1 + tau_S l)(1 + tau_G l + w_GG F_G' e^{-l d_GG}) + w_SG w_GS F_S' F_G' e^{-l (d_SG + d_GS)} = 0 .
```

Because of the delays, the loop loses stability through a **Hopf
bifurcation** when the loop gain `w_SG * w_GS` exceeds a threshold; the
emerging oscillation is in the **beta band** (13-30 Hz). Dopamine depletion
strengthens the STN<->GPe coupling (Holgado et al. 2010; Pavlides et al.
2015; Mallet et al. 2008), which is the model's account of exaggerated beta
synchrony in Parkinson's disease (Brown et al. 2001; Kühn et al. 2006;
Hammond, Bergman & Brown 2007; Little & Brown 2014). The code
(`regimes.stn_gpe_hopf`) solves this equation numerically for every run, so
the loop gain is set *relative to that run's own threshold*. The analytic
threshold matches simulation (tested in `tests/test_neural_mass.py`), and
the predicted onset frequency of the parkinsonian runs of the full
configuration is 17-27 Hz, rising with GPe self-inhibition and falling with
longer delays.

### 2.3 Coupling and measurement

Basal-ganglia beta reaches the cortex through the thalamus, and cortical
beta is coherent with STN beta in patients (Williams et al. 2002; Litvak et
al. 2011). We model this pathway minimally: the STN rate, high-passed with a
0.5 s running mean and delayed by 8-15 ms, is added with gain `kappa` to the
extrinsic input of the column. `kappa` is drawn from the *same* range in
all classes; only the parkinsonian loop actually oscillates, so only there
does the coupling carry a sustained beta drive.

Each run's output is z-scored and mixed with 1/f background activity (std
0.2-0.8) and white sensor noise (std 0.05-0.2), high-passed at 0.5 Hz,
cut into 4 s windows, and **every window is z-scored again**. Absolute
amplitude, which differs by an order of magnitude between spiking and
alpha regimes in the raw model, is therefore *not available* to any
classifier: they must use waveform shape and spectrum.

## 3. The three classes

| class | cortex (Jansen-Rit) | basal ganglia (STN-GPe) | expected signature | references |
|---|---|---|---|---|
| **healthy** | `A` in [3.10, 3.40] mV, `p` in [190, 300] /s | loop gain 0.4-0.9 x Hopf threshold (damped, noise-driven) | ~10 Hz alpha, weak broadband beta | Jansen & Rit 1995; David & Friston 2003 |
| **epileptic** | `A` in [3.60, 4.20] mV, `p` in [90, 160] /s | as healthy | sharp positive spikes + slow wave at 2-6 Hz; sporadic to rhythmic | Wendling et al. 2000, 2002; Grimbert & Faugeras 2006; Touboul et al. 2011 |
| **parkinsonian** | as healthy | loop gain 1.05-1.8 x Hopf threshold (sustained oscillation) | alpha plus a beta peak (17-27 Hz in the full run) | Holgado et al. 2010; Pavlides et al. 2015; Brown et al. 2001; Litvak et al. 2011 |

Shared nuisance parameters drawn per run in every class: `B` 21.5-22.5 mV,
`a` 95-105 /s, `b` 49-55 /s, `C` 130-140, input noise std 45-70 (Jansen &
Rit's uniform 120-320 input has std 57.7), GPe self-inhibition 1.5-5,
cortical drive noise to STN, all BG delays, the BG->cortex gain `kappa` 3-10
and delay, and the two measurement-noise levels. Example windows, spectra,
delay embeddings and persistence diagrams of each class are in
[`figures/example_traces.png`](../figures/example_traces.png).

All regime ranges were fixed by inspecting simulated traces and spectra
(no classifier was involved), *before* any classification was run on the
full configuration. One iteration was needed: the first draft let `a` and `b`
vary by ±10 % and used a BG->cortex gain of 0.5-2, which produced spiking
"healthy" runs and parkinsonian runs without a visible beta peak; the
ranges were tightened until the signatures in the table held for the large
majority of runs. In the full run (noise-free model output, criteria in
`RESULTS.md`): 99 % of healthy runs are alpha-dominated, 90 % of epileptic
runs are dominated by slow, positively skewed spike activity and 86 % of
parkinsonian runs carry at least 10 % of their power in the beta band. The
remainder are borderline runs (epileptic columns close to the alpha/spike
boundary, parkinsonian loops with weak beta transfer to the cortex). They were
kept, because real recordings have such cases too, and they are where most
classification errors occur.

## 4. What each feature measures

### Baselines (non-topological)

* **Relative band powers** (delta 1-4, theta 4-8, alpha 8-13, beta 13-30,
  gamma 30-45 Hz) and **log band ratios** (theta/alpha, beta/alpha,
  slow/fast, ...): the workhorse of quantitative EEG. Beta/alpha is the
  obvious parkinsonian marker here; slow/fast the epileptic one.
* **Spectral edge frequencies** SEF50/90/95 (frequency below which 50/90/95 %
  of power lies; Rampil 1998), **peak frequency** and **spectral entropy**.
* **Hjorth mobility and complexity** (Hjorth 1970): mobility is the
  RMS-frequency of the signal, complexity its deviation from a pure sine.
  (Activity, the variance, is 1 after z-scoring.)
* **Strict extras** not named in the brief, added so that the baseline is
  genuinely strong: **skewness, kurtosis and line length** (Esteller et al.
  2001) are classic non-topological spike detectors, and the full **log PSD**
  (45 one-hertz bins) is the spectral half of the descriptor.
  `baselines_union` is all of the above (66 features).

### Topological view (GUDHI)

* **Sublevel-set persistence of the signal.** Sweep a threshold upward
  through the trace; each local minimum starts a connected component that
  dies when it merges with a deeper valley. The H0 diagram records
  (birth, death) of every valley; the superlevel version does the same for
  peaks, so **persistence = peak prominence**. A spike-and-wave trace gives
  a few very prominent peaks plus many tiny ones; alpha gives a regular
  comb of medium-prominence peaks.
* **Delay-embedding persistence.** The window is embedded as points
  `(x_t, x_{t+tau}, x_{t+2 tau})` (Takens 1981) at three fixed delays
  (8, 20, 40 ms) and filtered with the Alpha complex. A rhythm draws a loop;
  its **H1 persistence measures how large and how clean the loop is**
  (Perea & Harer 2015); two superimposed rhythms (alpha + beta) or the
  asymmetric spike-wave cycle give characteristic loop structures. H0
  measures clustering of the cloud (e.g. the slow segments between spikes).
* Each diagram is vectorised into 10 summary statistics (number of bars,
  total / maximal / mean persistence, persistent entropy, ...), a Betti
  curve and the first two persistence landscapes (Bubenik 2015) on fixed
  grids: 464 features. A compact variant with only the 80 summary
  statistics (`topo_summary`) is reported as a sensitivity analysis,
  declared before the full run.
* The **topological-spectral descriptor** is `psd + topo`.

### Deep model

A four-block 1-D CNN (about 50 k parameters; Conv-BatchNorm-ReLU-Pool,
global average pooling) trained for a fixed number of epochs on the raw
z-scored window, the standard end-to-end approach for EEG (Schirrmeister et
al. 2017; Roy et al. 2019).

## 5. Evaluation design (and why)

* **Grouped cross-validation by run.** The ten windows of one run share
  parameters and noise, so they are strongly correlated. Splitting them
  between training and test would let a model recognise the run rather than
  the class, a textbook leakage (Roberts et al. 2017; Kapoor & Narayanan
  2023). We use repeated StratifiedGroupKFold with the run as group; the
  sanity-check table in `RESULTS.md` shows how much ungrouped CV would have
  inflated the scores.
* **Paired group bootstrap.** All feature sets and the CNN are trained on
  identical folds; CIs resample runs with replacement (a cluster bootstrap;
  Field & Welsh 2007) and every comparison uses the same resamples for both
  sides, so the CI is on the *difference*.
* **Go/No-Go rule** (as in GEM): GO iff the margin over the reference is
  >= +0.05 accuracy. We report the margin against the brief-named
  baselines, against the strict baseline union, and the incremental gain of
  adding topology to the baselines; we also report *headroom* (1 minus the
  baseline accuracy), the largest margin that is arithmetically possible.
* **Leak detectors.** Permuting run labels must give chance accuracy.

## 6. What the benchmark found

Full configuration: 240 runs (80 per class), 2400 windows, 4 folds x 5
repeats grouped by run, 2000 group-bootstrap resamples. All numbers are
window-level accuracy (chance = 0.333); every one is in
[`RESULTS.md`](RESULTS.md).

1. **Verdict: NO-GO, under every reading of the rule.**
   * *As briefed*, the topological-spectral descriptor (PSD + topology,
     0.943) is slightly *below* the best band-power baseline (0.955):
     margin -0.012 [95 % CI -0.027, +0.002].
   * *Strict*: all non-topological features together reach 0.962, so the
     largest margin any feature set could reach is 0.038, less than the
     +0.05 GO margin. The best topology-containing set (baselines plus the
     80 persistence summaries) ties the baselines: -0.002 [-0.012, +0.008].
   * *Incremental*: adding the full 464-dimensional topological vector to
     the baselines *lowers* accuracy by -0.010 [-0.018, -0.004]; adding it
     to band power + Hjorth lowers it by -0.014 [-0.027, -0.004]. The extra
     dimensions bring variance but no new information.
2. **Topology does see the rhythms, just less completely than the
   spectrum.** On its own the topological view reaches 0.924 (0.931 with
   the 80 summary statistics only), far above chance, but 0.037 below the
   log-PSD alone (CI -0.058, -0.019). Persistence of the sublevel sets
   encodes peak prominences (spikes); persistence of the delay embedding
   encodes the size and regularity of the oscillation loop. Both are largely
   functions of the power spectrum and the amplitude distribution, which
   the baselines already measure.
3. **Where the errors are.** The band-power and union models both confuse
   6 % of epileptic windows with healthy ones and 4 % of parkinsonian
   windows with healthy ones. These are the borderline runs of section 3: epileptic
   columns near the alpha/spike boundary that spike only sporadically, and
   parkinsonian runs with a weak cortical beta peak.
4. **The 1D-CNN ties the best features** (0.966, +0.004 [-0.009, +0.018]
   vs all baselines) and has the best epileptic recall (0.97, vs 0.93-0.94
   for the band-power and union models).
   It catches isolated spikes that window-averaged spectral features dilute.
   It is, however, less robust to measurement noise (below).
5. **Noise does not rescue topology (exploratory, post hoc).** Multiplying
   the measurement noise by 2 and 4 hurts the topological view most
   (0.924 -> 0.889 -> 0.740) and the full baseline union least
   (0.962 -> 0.943 -> 0.903); the CNN falls to 0.885 at x4. The descriptor
   and the union stay 1-2 points *below* the baseline union at every noise
   level (CIs below 0 or touching it). At x4 the descriptor (0.882) is above
   the brief-named band-power + Hjorth baseline (0.861), but it shares its
   PSD block with the baseline union, which is higher still; the sweep did
   not evaluate the PSD alone, so that gap cannot be attributed to topology.
   Unlike the multi-scale filtration of the GEM study, our filtrations do no
   internal smoothing, a likely reason why noise hurts them more.
6. **Which features carry the signal.** Scrambling the band powers costs the
   union model 0.17 accuracy (both classifiers). A random forest loses only
   0.027 when the sublevel-set topology is scrambled and nothing when the
   delay-embedding families are scrambled, so the information they hold is
   redundant with the spectrum. (The logistic regression loses 0.07 per
   topological family, but that measures how much weight it spread over
   correlated features, not unique information.)
7. **The leak detectors pass.** With permuted run labels the union scores
   0.340 (chance). Ignoring the run grouping would have inflated accuracy
   (baseline union 0.964 -> 0.984, topology 0.925 -> 0.937), which is why
   every reported number uses grouped folds.

**Bottom line.** For single-channel rhythms whose defining features are
spectral (an alpha peak, a beta peak, slow spike-and-wave activity), a
topological description of the trace re-encodes information that a
well-built spectral feature set already has. It is a faithful
representation, but it is not a better one, and it degrades faster with
noise. This matches the GEM geophysical-flow benchmark. A fair next test
for topology would be a task where spectra are matched by construction, as
in GEM's IAAFT-surrogate design: for example, healthy vs epileptic windows
with identical power spectra and amplitude distributions but different
temporal structure.

## 7. What this can and cannot tell you

* **It is a statement about representations, not about patients.** The
  classes are regimes of neural-mass models; the benchmark asks which
  representation best recovers the regime from a 4 s single-channel trace.
  It says nothing about diagnostic accuracy on clinical EEG.
* **The parkinsonian class is a basal-ganglia beta signature routed to a
  cortical signal.** Clinically, beta is most prominent in STN local field
  potentials recorded from DBS electrodes; resting scalp EEG in Parkinson's
  disease is better known for generalised slowing (Soikkeli et al. 1991),
  and cortical beta is modest. Our "parkinsonian EEG" is closer to an
  ECoG/LFP-like mixture than to routine scalp EEG.
* **The epileptic class is a cortical-column caricature.** Jansen-Rit spikes
  have the right shape and rate, but true 3 Hz absence spike-wave
  discharges are thalamocortical (Destexhe 1998; Blumenfeld 2005;
  Breakspear et al. 2006; Taylor & Baier 2011). Ictal onset patterns
  (low-voltage fast activity) are not modelled.
* **Single channel, stationary windows.** No spatial information, no
  state transitions within a window, no artefacts beyond 1/f and white
  noise.
* **Classes are separable by design.** The regimes reproduce *known*
  signatures, and those signatures were discovered with spectral methods,
  so the benchmark starts with a structural advantage for spectral
  features. A topological advantage would have to come from waveform-shape
  information the spectrum misses (e.g. spike sharpness and asymmetry), and
  that is precisely what the strict baselines (skewness, kurtosis, line
  length) also capture.

## 8. Second benchmark: the topology made for signals, and the honest SOTA baseline

The first benchmark compared topology with band power on terms that favoured
band power. Sublevel-set persistence of a trace measures peak prominences,
and Alpha-complex persistence of a 3-D delay embedding at a few fixed lags
measures the size of the oscillation loop. Both are largely functions of the
amplitude distribution and the power spectrum, which band power already
summarises. The second benchmark (`python -m neurorhythms.sw1pers_bench`,
results in [`SW1PERS_RESULTS.md`](SW1PERS_RESULTS.md)) makes the fair test
and adds the baseline a reviewer would ask for.

### 8.1 SW1PerS: sliding windows and 1-persistence

Perea & Harer (2015) showed that the sliding-window embedding
`SW_{M,tau} f(t) = (f(t), f(t + tau), ..., f(t + M tau))` of a periodic signal
traces a closed curve, and that the persistence of the corresponding H1
class is controlled by the signal's Fourier coefficients and is largest when
the window `M tau` is close to one period (more precisely `P M / (M + 1)`).
After pointwise centring and normalisation the cloud lies on a sphere, so
the measure ignores amplitude, offset and slow amplitude modulation. It
scores *how periodic* the signal is, not how much power it has at a given
frequency. Two incommensurate rhythms (alpha plus an unrelated beta, say)
embed on a torus and leave a *second* persistent H1 class, so the second
largest persistence measures quasi-periodicity. SW1PerS (Perea, Deckard,
Haase & Harer 2015) is the practical pipeline: spline interpolation,
sliding-window embedding, centring and normalisation, mean-shift denoising,
Vietoris-Rips persistence over Z_p, and the periodicity score
`1 - mp / sqrt(3)`.

We apply it to every channel of every 4-s window with the literature
defaults, fixed before any classification: M = 12 (six harmonics), 160
sliding-window points, 8-nearest-neighbour mean-shift, Z_11 coefficients
(`ripser`). Rhythms differ across classes, so each window is scored at three
fixed periods (4, 10 and 20 Hz: spike-and-wave, alpha, beta) and at its own
dominant period, estimated without labels from the autocorrelation. Per
scale we keep five numbers: the largest and second-largest persistence, the
birth of the top class, the total persistence and the persistent entropy.
A CROCKER-style summary (Topaz, Ziegelmeier & Halverson 2015) adds how the
loop structure *evolves* over the window: Betti-1 on a fixed grid of Rips
scales for 1-s segments every 0.5 s, averaged over segments, with its
spread over segments, plus the mean, spread and minimum of the per-segment
persistence. This yields 43 numbers per channel. In `figures/sw1pers/sw1pers_clouds.png` a
pure sine gives one loop, the 10 + 21.3 Hz mixture gives two, and white noise
gives none.

### 8.2 A multi-channel montage

Covariance-based methods need channels, and real EEG has cross-channel
structure, so the second benchmark adds a multi-channel track. Each run is a
small network of four Jansen-Rit columns coupled by delayed long-range
excitatory connections (Jansen & Rit 1995 coupled two columns; David &
Friston 2003 generalised this to networks). Column `i` receives
`K sum_j W_ij (r_j(t - D_ij) - rbar_j)`, the delayed fluctuations of its
neighbours' pyramidal firing rates. The ring connectivity has strong
neighbours and a weak opposite column, the gain is K ~ U(10, 30) and the
delays are 10-25 ms. All four columns are driven by one STN-GPe loop with
column-specific gains. Six sensors see the four sources through a Gaussian
lead field (volume conduction) with per-run jitter. A fraction (20-50 %) of
the 1/f background is shared across channels. Every channel of every window
is z-scored, exactly as in the single-channel track.

The classes are defined *only* by the same parameters as before: in
epileptic runs one randomly placed focus column is in the spike regime
(focal epilepsy whose location varies across subjects); in parkinsonian runs
the shared loop is above its Hopf threshold. Coupling, delays, lead field
and noise are drawn from the same ranges in every class. The montage was not
tuned to favour any method. A pilot with a separate seed (999) ran only the
band-power reference, and only to confirm that it was not at ceiling (~0.91,
leaving room for a +0.05 margin). No candidate view was run on the pilot to
choose data or hyperparameters.

Cross-channel structure then arises from physiology rather than from the
labels. Delayed coupling entrains the columns to a common alpha rhythm. The
shared basal-ganglia loop makes parkinsonian beta *coherent* across channels.
Volume conduction spreads the large epileptic spikes over neighbouring
sensors (`figures/sw1pers/example_multichannel.png`).

### 8.3 The honest baseline: Riemannian geometry of covariance matrices

For multi-channel EEG the strongest classical approach is to describe a
window by the covariance matrix of its channels and to classify these
symmetric positive-definite (SPD) matrices with Riemannian geometry
(Barachant et al. 2012, 2013; reviews by Congedo, Barachant & Bhatia 2017 and
Lotte et al. 2018). We use the standard filter-bank form: band-pass the window in
the five bands of the band-power baseline, estimate one OAS-shrunk
covariance per band (Chen et al. 2010), and map each band's matrices to the
tangent space at their Riemannian mean (`pyriemann`), where a linear
classifier is appropriate. The reference mean is a statistic of the training
data, so it is fitted inside every CV fold.

This baseline largely *subsumes* the previous winner. With z-scored
channels, the diagonal entry of a band's covariance is that channel's share
of variance in the band, i.e. its relative band power (a unit test checks
this), and hence also the band ratios. The off-diagonal entries add
band-limited cross-channel coupling, which band power cannot see. Another
unit test builds two classes with identical spectra that differ only in
coupling: band power is at chance, the tangent space separates them. What
five bands do *not* carry is the finer spectral shape of the band-power view
(peak frequency, spectral edge frequencies, spectral entropy). A broadband
variant (`riemann_broadband`, one correlation matrix per window) isolates
the coupling part.

### 8.4 What the second benchmark found

Full configuration: 240 runs per track (80 per class), 2400 windows per
track, 4 folds x 5 repeats grouped by run, 2000 paired group-bootstrap
resamples. The single-channel track is the dataset of section 6 (it
reproduces band power 0.955, all baselines 0.962 and the CNN 0.966
exactly). All numbers are in [`SW1PERS_RESULTS.md`](SW1PERS_RESULTS.md).

**Q1. Does SW1PerS add value over band power? NO-GO.**

* On its own, SW1PerS is the *weakest* representation in both tracks:
  0.882 (single) and 0.819 (multi), 0.072 below band power in each
  (CIs -0.094 to -0.050). It is also below the sublevel/Alpha topology of
  the first benchmark (-0.042 single, -0.036 multi, CIs below 0). Dropping
  the CROCKER summary changes little (0.875 / 0.829).
* Added to band power, the pre-specified test: +0.008 [+0.001, +0.016]
  on the single channel (statistically detectable, P(margin <= 0) = 0.013,
  but a sixth of the +0.05 margin; band power is at 0.955, so a GO was
  arithmetically impossible there anyway), and -0.001 [-0.007, +0.003] on
  the montage, where a GO was possible (band power 0.891).
* Added to all non-topological features: -0.001 in both tracks.

**Q2. Does Riemannian covariance beat band power and the descriptor? Only
marginally, and not against both.**

* On the montage, the tangent-space view is the best single hand-crafted
  representation: 0.904 [0.870, 0.934]. Against band power (0.891) the
  margin is +0.013 [+0.000, +0.026] (P(margin <= 0) = 0.023): detectable,
  but a quarter of the GO margin. Against the topological-spectral
  descriptor (0.895) it is +0.010 [-0.002, +0.023], not detectable. It
  ties the full non-topological union (0.906; -0.002 [-0.013, +0.009]),
  which has the finer spectral shape that five band covariances lack.
* Its gain is where the physiology says it should be. Parkinsonian recall
  rises from 0.91 (band power) to 0.94, because the shared basal-ganglia
  loop makes beta *coherent* across channels, and coherence is exactly what
  the off-diagonal terms add. Epileptic recall is unchanged (0.84 vs 0.85).
* Coupling alone (`riemann_broadband`) reaches 0.670. That is far above
  chance, so cross-channel structure does carry class information, but
  much less than the per-channel spectra (-0.222 vs band power).

**Q3. Does adding topology to the Riemannian features help? NO.**
Riemann + SW1PerS: -0.008 [-0.018, +0.001]. Riemann + sublevel/Alpha
topology: -0.014 [-0.027, -0.003] (it hurts). Every view together vs every
non-topological view: -0.007 [-0.017, +0.001].

**The 1D-CNN ties again**: 0.966 single (+0.011 [-0.003, +0.027] vs band
power) and 0.896 multi (-0.008 [-0.025, +0.008] vs Riemann).

**Leak detectors pass**: permuted run labels give 0.332 (single) and 0.321 /
0.320 (multi, union / Riemann pipeline, which refits its tangent space in
every fold); ignoring the run grouping would have inflated multi-channel
band power from 0.885 to 0.946 and Riemann from 0.901 to 0.943.

### 8.5 Why nothing reaches the GO margin on the montage

On the montage the arithmetic headroom above band power (0.109) is larger
than the GO margin, yet no view gets close to it. The post hoc breakdown
in `SW1PERS_RESULTS.md` (exploratory, computed from the out-of-fold
predictions of the main run, not part of the verdicts) shows why. In the
noise-free source signals, 25 % of the epileptic focus columns never spike.
Those 20 epileptic runs are dynamically healthy. Together with 8
parkinsonian runs whose beta does not reach any column and 4 healthy runs,
they make 32 *borderline* runs of 240. The coupling to healthy neighbours
explains only a small part of this. A one-off check re-simulated the 80
epileptic networks with the inter-column coupling switched off (K = 0, 12 s,
seed 7): 21 % of the foci still did not spike, against 26 % with coupling.
Most non-spiking foci are parameter draws close to the alpha/spike boundary.
Under the same criterion, 90 % of the single-column epileptic runs spike;
that gap was not investigated further.

* On the 208 runs that show their class signature, every good view is near
  ceiling: band power 0.960, Riemann 0.970, all non-topological features 0.968,
  CNN 0.956 (SW1PerS 0.902).
* On the 32 borderline runs every view is at 0.44-0.51 (SW1PerS 0.28):
  with the spike regime absent from the signal, no representation can
  recover the label.

The remaining errors are therefore mostly a *data* ceiling, not a
representation ceiling. On typical runs the room above band power is about
0.04 accuracy, and Riemann takes a quarter of it. The single-channel track
shows the same pattern: on runs with their signature, band power is already
at 0.991. There, adding SW1PerS lifts the 20 borderline runs from 0.555 to
0.646, which is the source of its +0.008. That figure rests on 20 runs and is
exploratory. On the same runs the CNN reaches 0.691 and catches 74 % of the
windows of sporadically spiking epileptic runs (band power 38 %).

### 8.6 Interpretation

* **SW1PerS measures the wrong thing for this task.** It was designed to
  score *how periodic* a signal is, independently of its amplitude and,
  through the window matched to the dominant period, largely of its
  frequency. Here the classes differ in *which* rhythm is present (alpha,
  beta, 2-6 Hz spike-and-wave) and in waveform asymmetry, not in how
  regular the rhythm is. All three classes are noise-driven rhythms of
  similar regularity. The periodicity score is a faithful but nearly
  class-blind summary. This is the expected failure mode of an invariant
  descriptor applied to a task whose label lives in the quantity it is
  invariant to. It is not a defect of the method.
* **The montage confirms the single-channel verdict.** Neither the
  sublevel/Alpha persistence of the first benchmark nor SW1PerS adds
  information that band power, the full baseline union or the Riemannian
  view lack. Both are NO-GO on every pre-specified comparison, with CIs
  tight enough (about +/-0.01) to rule out any gain near +0.05.
* **Riemannian geometry is the right baseline, but not a decisive winner
  here.** It is the best or tied-best classical view on the montage, and its
  small gain is physiologically interpretable (coherent parkinsonian beta).
  Because the simulator's class information is mostly spectral and
  per-channel, coupling adds only an increment. On real multi-channel EEG,
  where coupling patterns (e.g. cortico-cortical beta coherence in
  Parkinson's disease, Silberstein et al. 2005) and spatial topographies
  matter more, the gap may be larger. This benchmark does not test that.
* **Publishable as a negative result**: two topological representations,
  one of them the method designed for signals, fail to add to band power
  on single- and multi-channel neural-mass EEG under grouped CV with
  paired CIs. The SOTA Riemannian baseline is marginally better than band
  power and ties the full spectral union. The remaining errors come from
  runs whose dynamics do not express their label.

### 8.7 Limits of the second benchmark

* SW1PerS hyperparameters (M = 12, 160 points, 8-NN mean-shift, three fixed
  scales plus the dominant period) were fixed from the literature and not
  tuned. Other choices (H2 for tori, multivariate sliding windows across
  channels, persistence images) were not tested. A different choice could
  move SW1PerS by a few points, but a +0.05 gain *over band power* would
  have to come from information the post hoc breakdown shows is not in the
  signal.
* The montage is a minimal network (4 columns, 6 sensors, Gaussian lead
  field), not a head model with realistic sources, conductivities and
  reference montages. Only 75 % of its epileptic foci actually spike (90 %
  of single-column epileptic runs do). That caps every method's accuracy and
  is the main reason no view can reach the GO margin on the montage.
* The fair topology test that remains is a task whose classes have matched
  spectra but different temporal organisation (e.g. a noisy limit cycle vs
  linear narrowband noise with the same power spectrum; IAAFT surrogates).
  There periodicity is the label, and SW1PerS should have an edge.
  Likewise, a task with matched spectra and different coupling would favour
  the Riemannian view (the unit test in `tests/test_riemann.py` is the
  minimal example).

## References

* Barachant A, Bonnet S, Congedo M, Jutten C (2012). Multiclass brain-computer interface classification by Riemannian geometry. *IEEE Trans Biomed Eng* 59:920-928.
* Barachant A, Bonnet S, Congedo M, Jutten C (2013). Classification of covariance matrices using a Riemannian-based kernel for BCI applications. *Neurocomputing* 112:172-178.
* Barachant A, et al. *pyRiemann: Biosignals classification with Riemannian geometry*. https://pyriemann.readthedocs.io
* Bauer U (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes. *J Appl Comput Topol* 5:391-423.
* Bevan MD, Magill PJ, Terman D, Bolam JP, Wilson CJ (2002). Move to the rhythm: oscillations in the subthalamic nucleus-external globus pallidus network. *Trends Neurosci* 25:525-531.
* Blumenfeld H (2005). Cellular and network mechanisms of spike-wave seizures. *Epilepsia* 46(Suppl 9):21-33.
* Breakspear M, Roberts JA, Terry JR, Rodrigues S, Mahant N, Robinson PA (2006). A unifying explanation of primary generalized seizures through nonlinear brain modeling and bifurcation analysis. *Cereb Cortex* 16:1296-1313.
* Brown P, Oliviero A, Mazzone P, Insola A, Tonali P, Di Lazzaro V (2001). Dopamine dependency of oscillations between subthalamic nucleus and pallidum in Parkinson's disease. *J Neurosci* 21:1033-1038.
* Bubenik P (2015). Statistical topological data analysis using persistence landscapes. *J Mach Learn Res* 16:77-102.
* Chen Y, Wiesel A, Eldar YC, Hero AO (2010). Shrinkage algorithms for MMSE covariance estimation. *IEEE Trans Signal Process* 58:5016-5029.
* Congedo M, Barachant A, Bhatia R (2017). Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review. *Brain-Computer Interfaces* 4:155-174.
* David O, Friston KJ (2003). A neural mass model for MEG/EEG: coupling and neuronal dynamics. *NeuroImage* 20:1743-1755.
* Destexhe A (1998). Spike-and-wave oscillations based on the properties of GABA-B receptors. *J Neurosci* 18:9099-9111.
* Edelsbrunner H, Harer J (2010). *Computational Topology: An Introduction*. American Mathematical Society.
* Esteller R, Echauz J, Tcheng T, Litt B, Pless B (2001). Line length: an efficient feature for seizure onset detection. *Proc. 23rd Annu. Int. Conf. IEEE EMBS*, 1707-1710.
* Field CA, Welsh AH (2007). Bootstrapping clustered data. *J R Stat Soc B* 69:369-390.
* Grimbert F, Faugeras O (2006). Bifurcation analysis of Jansen's neural mass model. *Neural Comput* 18:3052-3068.
* Hammond C, Bergman H, Brown P (2007). Pathological synchronization in Parkinson's disease: networks, models and treatments. *Trends Neurosci* 30:357-364.
* Hjorth B (1970). EEG analysis based on time domain properties. *Electroencephalogr Clin Neurophysiol* 29:306-310.
* Holgado AJN, Terry JR, Bogacz R (2010). Conditions for the generation of beta oscillations in the subthalamic nucleus-globus pallidus network. *J Neurosci* 30:12340-12352.
* Jansen BH, Rit VG (1995). Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. *Biol Cybern* 73:357-366.
* Kapoor S, Narayanan A (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4:100804.
* Kühn AA, Kupsch A, Schneider GH, Brown P (2006). Reduction in subthalamic 8-35 Hz oscillatory activity correlates with clinical improvement in Parkinson's disease. *Eur J Neurosci* 23:1956-1960.
* Little S, Brown P (2014). The functional role of beta oscillations in Parkinson's disease. *Parkinsonism Relat Disord* 20(Suppl 1):S44-S48.
* Litvak V, Jha A, Eusebio A, Oostenveld R, Foltynie T, Limousin P, Zrinzo L, Hariz MI, Friston K, Brown P (2011). Resting oscillatory cortico-subthalamic connectivity in patients with Parkinson's disease. *Brain* 134:359-374.
* Lopes da Silva FH, Hoeks A, Smits H, Zetterberg LH (1974). Model of brain rhythmic activity: the alpha-rhythm of the thalamus. *Kybernetik* 15:27-37.
* Lotte F, Bougrain L, Cichocki A, Clerc M, Congedo M, Rakotomamonjy A, Yger F (2018). A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update. *J Neural Eng* 15:031005.
* Mallet N, Pogosyan A, Márton LF, Bolam JP, Brown P, Magill PJ (2008). Parkinsonian beta oscillations in the external globus pallidus and their relationship with subthalamic nucleus activity. *J Neurosci* 28:14245-14258.
* Pavlides A, Hogan SJ, Bogacz R (2015). Computational models describing possible mechanisms for generation of excessive beta oscillations in Parkinson's disease. *PLoS Comput Biol* 11:e1004609.
* Perea JA, Deckard A, Haase SB, Harer J (2015). SW1PerS: Sliding windows and 1-persistence scoring; discovering periodicity in gene expression time series data. *BMC Bioinformatics* 16:257.
* Perea JA, Harer J (2015). Sliding windows and persistence: an application of topological methods to signal analysis. *Found Comput Math* 15:799-838. arXiv:1307.6188.
* Plenz D, Kitai ST (1999). A basal ganglia pacemaker formed by the subthalamic nucleus and external globus pallidus. *Nature* 400:677-682.
* Rampil IJ (1998). A primer for EEG signal processing in anesthesia. *Anesthesiology* 89:980-1002.
* Roberts DR, Bahn V, Ciuti S, et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40:913-929.
* Roy Y, Banville H, Albuquerque I, Gramfort A, Falk TH, Faubert J (2019). Deep learning-based electroencephalography analysis: a systematic review. *J Neural Eng* 16:051001.
* Schirrmeister RT, Springenberg JT, Fiederer LDJ, et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Hum Brain Mapp* 38:5391-5420.
* Silberstein P, Pogosyan A, Kühn AA, et al. (2005). Cortico-cortical coupling in Parkinson's disease and its modulation by therapy. *Brain* 128:1277-1291.
* Soikkeli R, Partanen J, Soininen H, Pääkkönen A, Riekkinen P Sr (1991). Slowing of EEG in Parkinson's disease. *Electroencephalogr Clin Neurophysiol* 79:159-165.
* Takens F (1981). Detecting strange attractors in turbulence. In: *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics 898:366-381.
* Taylor PN, Baier G (2011). A spatially extended model for macroscopic spike-wave discharges. *J Comput Neurosci* 31:679-684.
* The GUDHI Project (2015-). *GUDHI User and Reference Manual*. https://gudhi.inria.fr
* Topaz CM, Ziegelmeier L, Halverson T (2015). Topological data analysis of biological aggregation models. *PLoS ONE* 10:e0126383.
* Touboul J, Wendling F, Chauvel P, Faugeras O (2011). Neural mass activity, bifurcations, and epilepsy. *Neural Comput* 23:3232-3286.
* Tralie C, Saul N, Bar-On R (2018). Ripser.py: a lean persistent homology library for Python. *J Open Source Softw* 3(29):925.
* Tralie CJ, Perea JA (2018). (Quasi)periodicity quantification in video data, using topology. *SIAM J Imaging Sci* 11:1049-1077.
* Wendling F, Bartolomei F, Bellanger JJ, Chauvel P (2002). Epileptic fast activity can be explained by a model of impaired GABAergic dendritic inhibition. *Eur J Neurosci* 15:1499-1508.
* Wendling F, Bellanger JJ, Bartolomei F, Chauvel P (2000). Relevance of nonlinear lumped-parameter models in the analysis of depth-EEG epileptic signals. *Biol Cybern* 83:367-378.
* Williams D, Tijssen M, van Bruggen G, et al. (2002). Dopamine-dependent changes in the functional connectivity between basal ganglia and cerebral cortex in humans. *Brain* 125:1558-1569.
* Wilson HR, Cowan JD (1972). Excitatory and inhibitory interactions in localized populations of model neurons. *Biophys J* 12:1-24.
