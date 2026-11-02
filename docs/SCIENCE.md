# The science behind the benchmark

*Written for a neuroscientist who wants to know what was simulated, why those
parameter regimes, what each feature measures physiologically, and how much
the result can and cannot say. Numbers quoted here come from
[`RESULTS.md`](RESULTS.md) (full configuration).*

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
the predicted onset frequency of the parkinsonian runs is 14-30 Hz, rising
with GPe self-inhibition and falling with longer delays.

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
| **parkinsonian** | as healthy | loop gain 1.05-1.8 x Hopf threshold (sustained oscillation) | alpha plus a beta peak (14-30 Hz) | Holgado et al. 2010; Pavlides et al. 2015; Brown et al. 2001; Litvak et al. 2011 |

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
majority of runs.

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

See [`RESULTS.md`](RESULTS.md) for every number; the short version is
written there in the *Verdict* section and summarised in the README.

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

## References

* Bevan MD, Magill PJ, Terman D, Bolam JP, Wilson CJ (2002). Move to the rhythm: oscillations in the subthalamic nucleus-external globus pallidus network. *Trends Neurosci* 25:525-531.
* Blumenfeld H (2005). Cellular and network mechanisms of spike-wave seizures. *Epilepsia* 46(Suppl 9):21-33.
* Breakspear M, Roberts JA, Terry JR, Rodrigues S, Mahant N, Robinson PA (2006). A unifying explanation of primary generalized seizures through nonlinear brain modeling and bifurcation analysis. *Cereb Cortex* 16:1296-1313.
* Brown P, Oliviero A, Mazzone P, Insola A, Tonali P, Di Lazzaro V (2001). Dopamine dependency of oscillations between subthalamic nucleus and pallidum in Parkinson's disease. *J Neurosci* 21:1033-1038.
* Bubenik P (2015). Statistical topological data analysis using persistence landscapes. *J Mach Learn Res* 16:77-102.
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
* Mallet N, Pogosyan A, Márton LF, Bolam JP, Brown P, Magill PJ (2008). Parkinsonian beta oscillations in the external globus pallidus and their relationship with subthalamic nucleus activity. *J Neurosci* 28:14245-14258.
* Pavlides A, Hogan SJ, Bogacz R (2015). Computational models describing possible mechanisms for generation of excessive beta oscillations in Parkinson's disease. *PLoS Comput Biol* 11:e1004609.
* Perea JA, Harer J (2015). Sliding windows and persistence: an application of topological methods to signal analysis. *Found Comput Math* 15:799-838.
* Plenz D, Kitai ST (1999). A basal ganglia pacemaker formed by the subthalamic nucleus and external globus pallidus. *Nature* 400:677-682.
* Rampil IJ (1998). A primer for EEG signal processing in anesthesia. *Anesthesiology* 89:980-1002.
* Roberts DR, Bahn V, Ciuti S, et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40:913-929.
* Roy Y, Banville H, Albuquerque I, Gramfort A, Falk TH, Faubert J (2019). Deep learning-based electroencephalography analysis: a systematic review. *J Neural Eng* 16:051001.
* Schirrmeister RT, Springenberg JT, Fiederer LDJ, et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Hum Brain Mapp* 38:5391-5420.
* Soikkeli R, Partanen J, Soininen H, Pääkkönen A, Riekkinen P Sr (1991). Slowing of EEG in Parkinson's disease. *Electroencephalogr Clin Neurophysiol* 79:159-165.
* Takens F (1981). Detecting strange attractors in turbulence. In: *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics 898:366-381.
* Taylor PN, Baier G (2011). A spatially extended model for macroscopic spike-wave discharges. *J Comput Neurosci* 31:679-684.
* The GUDHI Project (2015-). *GUDHI User and Reference Manual*. https://gudhi.inria.fr
* Touboul J, Wendling F, Chauvel P, Faugeras O (2011). Neural mass activity, bifurcations, and epilepsy. *Neural Comput* 23:3232-3286.
* Wendling F, Bellanger JJ, Bartolomei F, Chauvel P (2000). Relevance of nonlinear lumped-parameter models in the analysis of depth-EEG epileptic signals. *Biol Cybern* 83:367-378.
* Wendling F, Bartolomei F, Bellanger JJ, Chauvel P (2002). Epileptic fast activity can be explained by a model of impaired GABAergic dendritic inhibition. *Eur J Neurosci* 15:1499-1508.
* Williams D, Tijssen M, van Bruggen G, et al. (2002). Dopamine-dependent changes in the functional connectivity between basal ganglia and cerebral cortex in humans. *Brain* 125:1558-1569.
* Wilson HR, Cowan JD (1972). Excitatory and inhibitory interactions in localized populations of model neurons. *Biophys J* 12:1-24.
