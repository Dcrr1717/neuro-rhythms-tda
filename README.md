# neuro-rhythms-tda

Classifying **healthy vs epileptic vs parkinsonian** brain rhythms from
neural-mass EEG with a **topological-spectral descriptor** (persistent
homology + power spectrum, GUDHI), strong non-topological baselines and a
small **1D-CNN**, under grouped cross-validation, with an honest Go/No-Go
verdict on whether topology adds value over band-power baselines.

> Results of the full run: see [`docs/RESULTS.md`](docs/RESULTS.md) (generated).
> Science and references: see [`docs/SCIENCE.md`](docs/SCIENCE.md).

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional, for the CNN
pip install -e ".[dev]"
pytest -q                                          # offline, ~1 min
python -m neurorhythms                             # small default config, ~1.5 min
python -m neurorhythms --config configs/full.yaml  # full run, ~1 h on 4 cores
```

## License
MIT, Copyright (c) 2026 Cesar Daniel Reinoso. See `LICENSE`.
