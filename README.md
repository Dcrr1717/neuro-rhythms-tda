# neuro-rhythms-tda

Classifying **healthy vs epileptic vs parkinsonian** brain rhythms simulated
with neural mass models, and asking honestly whether **topological data
analysis** adds value over the features EEG analysis already uses. Two
benchmarks, both under cross-validation grouped by simulation run with
paired bootstrap CIs:

1. **Persistent homology of the raw trace** (sublevel-set + Alpha-complex
   delay embedding, GUDHI) vs band power, strong baselines and a 1D-CNN.
2. **SW1PerS** (sliding-window embedding + 1-persistence, the topological
   method designed for signals; Perea & Harer 2015) on single-channel *and*
   a new **multi-channel montage** of coupled cortical columns, against the
   honest state-of-the-art baseline, **Riemannian tangent-space features of
   SPD channel covariances** (`pyriemann`).

Everything is synthetic and runs offline: no external data, no API keys.

## Result 2 (second benchmark): NO-GO for topology; Riemannian only marginally ahead

**SW1PerS does not add value over band power, and topology adds nothing on
top of Riemannian features.** Riemannian covariance is the best classical
view on the montage, but beats band power by only 1.3 points and is not
detectably better than the existing descriptor. Setup: 240 runs per track,
2400 four-second windows (1 or 6 channels), StratifiedGroupKFold (4 folds x
5 repeats) grouped by run, 2000 paired group-bootstrap resamples. GO iff
margin >= +0.05 accuracy.

| question | track | candidate | reference | margin [95% CI] | verdict |
|---|---|---|---|---|---|
| Q1. SW1PerS over band power | single | bandpower+sw1pers, 0.962 | bandpower, 0.955 | +0.008 [+0.001, +0.016] | **NO-GO** (detectable, far below margin) |
| Q1. SW1PerS over band power | multi | bandpower+sw1pers, 0.890 | bandpower, 0.891 | -0.001 [-0.007, +0.003] | **NO-GO** |
| Q2. Riemannian vs band power | multi | riemann, 0.904 | bandpower, 0.891 | +0.013 [+0.000, +0.026] | detectable, **below GO margin** |
| Q2. Riemannian vs descriptor | multi | riemann, 0.904 | descriptor, 0.895 | +0.010 [-0.002, +0.023] | no detectable difference |
| Q3. topology on top of Riemannian | multi | riemann+sw1pers, 0.896 | riemann, 0.904 | -0.008 [-0.018, +0.001] | **NO** |
| Q3. old topology on top of Riemannian | multi | riemann+topo, 0.890 | riemann, 0.904 | -0.014 [-0.027, -0.003] | worse |
| deep model | multi | cnn, 0.896 | riemann, 0.904 | -0.008 [-0.025, +0.008] | tie |

- **SW1PerS alone is the weakest view** (0.882 single, 0.819 multi), below
  band power and below the first benchmark's topology. It scores *how
  periodic* a rhythm is; here the classes differ in *which* rhythm is
  present.
- **Riemannian geometry's gain is where the physiology puts it**:
  parkinsonian recall 0.91 -> 0.94 (beta is coherent across channels because
  one basal-ganglia loop drives every column). Coupling alone reaches 0.670.
- **Why nothing reaches +0.05 on the montage** (exploratory, post hoc): 25 %
  of epileptic foci never spike, so those runs are dynamically healthy. On
  the 208 of 240 runs that show their class signature, band power already
  scores 0.960 and Riemann 0.970; on the 32 borderline runs every view is at
  0.44-0.51.
- **Leak detectors pass**: permuted run labels give 0.32-0.33 in both tracks.

Full tables: [`docs/SW1PERS_RESULTS.md`](docs/SW1PERS_RESULTS.md).
Interpretation: [`docs/SCIENCE.md`](docs/SCIENCE.md) section 8.

| accuracy of every view, both tracks | every pre-specified margin |
|---|---|
| ![accuracy](figures/sw1pers/accuracy_bars.png) | ![margins](figures/sw1pers/margins.png) |

![SW1PerS point clouds, H1 diagrams and CROCKER matrices: references and one window per class](figures/sw1pers/sw1pers_clouds.png)
![Multi-channel montage: one window per class and band-limited channel correlations](figures/sw1pers/example_multichannel.png)
![Confusion matrices, multi-channel track](figures/sw1pers/confusion_matrices_multi.png)

## Result 1 (first benchmark): NO-GO

**Topology does not add value over band-power baselines on this task.**
Setup: 240 simulation runs, 2400 four-second windows, StratifiedGroupKFold
(4 folds x 5 repeats) grouped by run, and paired group-bootstrap 95 % CIs.
The GO rule is the GEM rule: margin >= +0.05 accuracy.

| question | candidate | reference | margin [95% CI] | verdict |
|---|---|---|---|---|
| as briefed | topological-spectral descriptor, 0.943 | band power, 0.955 | -0.012 [-0.027, +0.002] | **NO-GO** |
| strict | best topology-containing set, 0.959 | all non-topological features, 0.962 | -0.002 [-0.012, +0.008] | **NO-GO** (headroom 0.038 < 0.05) |
| incremental | all features incl. topology, 0.951 | all non-topological features, 0.962 | -0.010 [-0.018, -0.004] | **NO-GO** (topology hurts slightly) |
| deep model | 1D-CNN on raw signal, 0.966 | all non-topological features, 0.962 | +0.004 [-0.009, +0.018] | tie |

- **Topology carries the information, but less of it than the spectrum.** The
  topological view alone reaches 0.924 (chance 0.333). That is 0.037 below
  the plain log-PSD.
- **It is the least noise-robust view.** In an exploratory noise sweep it
  falls to 0.740 at 4x measurement noise; the baseline union holds 0.903.
- **The leak detectors pass.** Permuted labels give 0.340. Ignoring the run
  grouping would have inflated accuracy by up to 2 points.
- **What is left is borderline cases.** About 10 % of epileptic and 14 % of
  parkinsonian runs sit near their bifurcation boundaries and look partly
  healthy. That is where the errors are.

Full tables: [`docs/RESULTS.md`](docs/RESULTS.md). Interpretation for
neuroscientists: [`docs/SCIENCE.md`](docs/SCIENCE.md) section 6.

![Example window, spectrum, delay embedding and persistence diagrams per class](figures/example_traces.png)

| accuracy of every view | margins vs baselines |
|---|---|
| ![accuracy](figures/accuracy_bars.png) | ![margins](figures/margins.png) |
| ![confusion](figures/confusion_matrix.png) | ![importance](figures/feature_importance.png) |

## How it works

```
 per class, per run:  draw parameters  ->  coupled Jansen-Rit column + STN-GPe loop  ->  "EEG"
                                            (Euler-Maruyama, dt = 0.5 ms, 250 Hz)
 measurement:         z-score + 1/f noise + white noise -> 0.5 Hz high-pass -> 4 s windows -> z-score
 features:            bandpower | hjorth | waveform | psd | topo (GUDHI)  -> feature sets and unions
 models:              LogReg, RandomForest (best of two per set) | 1D-CNN on raw windows
 evaluation:          repeated StratifiedGroupKFold (group = run), paired group-bootstrap CIs,
                      GO iff margin >= +0.05 accuracy (GEM rule)
```

Second benchmark (`python -m neurorhythms.sw1pers_bench`), on two tracks:

```
 single track:        the dataset above (same seed, same windows)
 multi track:         4 Jansen-Rit columns, delayed ring coupling, one shared STN-GPe loop
                      -> Gaussian lead field -> 6 sensors + 1/f (partly shared) + white noise
                      -> 0.5 Hz high-pass -> 4 s windows -> every channel z-scored
 features:            bandpower | baselines | topo | descriptor (per channel, as above)
                      sw1pers   per channel: spline -> sliding window (M = 12) -> centre/normalise
                                -> kNN mean-shift -> Rips H1 over Z_11 (ripser) at 4/10/20 Hz and
                                the dominant period, + CROCKER (Betti-1 over sub-segments x scale)
                      riemann   OAS covariance per band (5 bands) -> tangent space at the Riemannian
                                mean of the training fold (pyriemann, fitted inside every fold)
 models, evaluation:  as above; 1D-CNN takes all channels as input channels
```

| class | signature | how it is produced |
|---|---|---|
| healthy | ~10 Hz alpha | Jansen-Rit standard regime; basal-ganglia loop below its Hopf threshold |
| epileptic | 2-6 Hz spike-and-slow-wave | raised excitatory gain `A`, lower input (Wendling et al. 2000/2002) |
| parkinsonian | alpha + 14-30 Hz beta | STN-GPe loop gain above the analytic Hopf threshold (Holgado et al. 2010), routed to the column |

In the montage, an epileptic run has one randomly placed focus column in the
spike regime; a parkinsonian run's supercritical loop drives every column.
Coupling, delays, lead field and noise use the same ranges in every class.

Details, equations, parameter ranges and references: [`docs/SCIENCE.md`](docs/SCIENCE.md).

## Repository layout

```
src/neurorhythms/
  neural_mass.py      Jansen-Rit + STN-GPe model, vectorised over runs
  network.py          multi-channel montage: coupled Jansen-Rit columns, shared STN-GPe loop, lead field
  regimes.py          class parameter regimes, Hopf threshold, network ranges (NETWORK_REGIMES)
  dataset.py          simulation runs -> measurement model -> windows (+ run ids), single and multi-channel
  signatures.py       descriptive check that each class shows its signature
  features/
    baseline.py       band powers, ratios, SEF, Hjorth, waveform stats, log-PSD
    topology.py       GUDHI sublevel-set + delay-embedding (Alpha) persistence, vectorisations
    sw1pers.py        SW1PerS sliding-window 1-persistence (ripser) + CROCKER summary
    riemann.py        filter-bank SPD covariances + fold-local tangent space (pyriemann)
    __init__.py       feature sets of the first benchmark
  classical.py        grouped folds, LogReg / RandomForest CV (optional fold-local preprocessing)
  deep.py             1D-CNN (PyTorch), single- or multi-channel input
  stats.py            accuracy, group bootstrap, paired margins, Go/No-Go rule
  importance.py       feature-family permutation importance
  plots.py, report.py figures and report of the first benchmark
  pipeline.py         first benchmark CLI (python -m neurorhythms)
  sw1pers_bench.py    second benchmark CLI (python -m neurorhythms.sw1pers_bench)
  plots_sw1pers.py, report_sw1pers.py   figures and report of the second benchmark
configs/small.yaml, configs/full.yaml                 first benchmark: CI (~1.5 min) and reported run
configs/sw1pers_small.yaml, configs/sw1pers_full.yaml second benchmark: CI (~5-10 min) and reported run (~70 min)
docs/RESULTS.md, docs/SW1PERS_RESULTS.md              generated by the full runs (do not edit by hand)
docs/SCIENCE.md       background for neuroscientists, interpretation of both benchmarks, references
figures/, figures/sw1pers/                            figures of the full runs
results/full/, results/sw1pers_full/                  results.json of the full runs
tests/                pytest suite (offline)
```

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # for the CNN (optional)
pip install -e ".[dev]"
pytest -q                                          # 51 tests, offline, ~1 min
python -m neurorhythms                             # first benchmark, small config -> results/small/
python -m neurorhythms --config configs/full.yaml  # full run -> results/full/, figures/, docs/RESULTS.md
python -m neurorhythms --config configs/full.yaml --replot   # re-render figures + report only
python -m neurorhythms.sw1pers_bench               # second benchmark, small config -> results/sw1pers_small/
python -m neurorhythms.sw1pers_bench --config configs/sw1pers_full.yaml
                                                   # -> results/sw1pers_full/, figures/sw1pers/, docs/SW1PERS_RESULTS.md
python -m neurorhythms.sw1pers_bench --config configs/sw1pers_full.yaml --replot
```

Without PyTorch, pass `--no-cnn`. Simulations and features are cached in
`data/cache/` (keyed by a hash of the configuration); `--no-cache` disables it.
All randomness is seeded.

## License

MIT, Copyright (c) 2026 Cesar Daniel Reinoso. See [`LICENSE`](LICENSE).
