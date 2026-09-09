# Fourier Phase Space as a Candidate Code for Musical Pitch Structure

Reproducible code and data for the paper:

> **Frequency Range of Fourier Phase Space as a Candidate Code for Musical Pitch
> Structure: Evidence from Artificial Neural Network Models of Iranian Classical Music.**
> *Scientific Reports* (under review). DOI: `[INSERT-DOI]`

Artificial neural networks (ANNs) trained to solve classification problems from Iranian
classical music develop hidden-unit connection weights that are dominated by a single
**discrete Fourier phase space** — even though nothing in their training refers to the
Fourier transform. This repository contains the full simulation and analysis pipeline that
produces every number and figure in the paper.

---

## Key result at a glance

| | Western (12-tone) | Iranian (17-tone) |
|---|---|---|
| Pitch-class lattice | 12-tone (12-TET) | 24-tone quarter-tone |
| Fourier frequencies available | 1–6 | 1–12 |
| Frequencies actually recruited | **1–6** | **1–12** |
| Units at frequencies 7–11 (impossible for 12-tone) | 0 | **31%** |
| Fourier fit vs. task-label control | collapses to chance | collapses to chance |

The wider Iranian range is **not** merely a mechanical consequence of the larger lattice.
Two confounds are removed before the claim is made. The Nyquist phase space *k*=12 is
excluded, because on a quarter-tone lattice it collapses into a chromatic-versus-microtonal
binary partition rather than a periodic code. And the comparison is restricted to the
interval tasks, which both systems share, so that lattice cardinality is not confounded
with task type. What remains: 31% of Iranian units sit at frequencies a 12-tone lattice
cannot express, against none of the Western units (Fisher's exact test, *p* < 10⁻²⁸).

---

## Repository layout

```
iranian-anns-fourier/
├── README.md                  This file.
├── LICENSE                    MIT.
├── CITATION.cff               How to cite the software and paper.
├── pyproject.toml             Package metadata (pip-installable).
├── requirements.txt           Runtime dependencies (pip).
├── environment.yml            Conda environment specification.
├── Makefile                   Convenience targets (make help).
├── run_config.py              ← the one file you run (smoke / medium / full / sweep).
│
├── src/iranian_anns/          The Python package.
│   ├── pitch_systems.py       Pitch-class systems on their lattices; Fourier phase spaces.
│   ├── tasks.py               Stimulus encoding; interval-task builder.
│   ├── music_theory.py        Real Vaziri cents, 24 interval names, 7 Dang task rules.
│   ├── value_unit_net.py      Gaussian value-unit network + generalized delta rule.
│   ├── analysis.py            Phase-space fits, permutation null, baselines, concentration.
│   ├── run_experiment.py      Orchestration → unit_fits.csv.
│   └── analyze_results.py     Statistical tables, figures, results report.
│
├── paper/                     Manuscript sources.
│   ├── main.tex               Full LaTeX manuscript.
│   ├── references.bib         Bibliography.
│   ├── main.pdf               Compiled manuscript.
│   ├── make_fig6.py           Script that generates Figure 5 from the results.
│   └── figures/               Figure PDFs and the data behind Figure 5.
│
├── data/
│   ├── stimuli/               All eleven task stimulus sets, with readable labels.
│   ├── reference_results/     Reference outputs from the full run (for comparison).
│   │	└── figures/           Figures 1-3 of the paper.
│   │	└── tables/            Main outputs in csv file format.
│   ├── unit_fits.csv		   The output of run_experiment.py	
│   ├── README.md
│   ├── results_report.md
│
├── scripts/                   export_stimuli.py and other utilities.
│
├── tests/                     Fast reproducibility tests (pytest).
│
└── docs/
    ├── METHODS_OVERVIEW.md    How each part of the code maps onto the paper.
    └── DESIGN_DECISIONS.md    The three choices that shape the analysis, and why.
```

---

## Quick start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate    # optional but recommended
pip install -r requirements.txt
# optional: make `import iranian_anns` work from anywhere
pip install -e .
```

Requires Python ≥ 3.9. On Windows/Anaconda, `conda create -n music python=3.11` then
`pip install -r requirements.txt` works equally well.

### 2. Confirm the environment (≈ 2 minutes)

```bash
python run_config.py smoke
```

Expect the console to report `Iranian-17 [dft] range 1-12` and `Western-12 [dft] range 1-6`.

### 3. Reproduce the paper's numbers (hours)

```bash
python run_config.py full
```

This trains 25 networks per task across all nine Iranian tasks and the matched Western
tasks, then writes:

```
results_full/unit_fits.csv                 one row per hidden unit (raw results)
results_full/report/results_report.md      the narrative report
results_full/report/tables/*.csv           the numbers used in the manuscript
results_full/report/figures/*.png          figures
```

Intermediate profiles `medium` (≈ 20–40 min) and `sweep` (≈ 1 h, hidden-layer-size
robustness) are also available.

### 4. Or use the Makefile

```bash
make install    # dependencies + editable install
make test       # reproducibility tests
make stimuli    # export all stimulus sets
make smoke      # 2-minute sanity run
make full       # the paper's numbers
```

### 5. Verify reproducibility

```bash
pytest
```

The tests confirm the 24-tone lattice framing and that all nine tasks reproduce the exact
pattern and category counts reported in the paper (153/153/246/246/121/246/88/246/62
patterns; 24/13/36/16/15/12/11/8/7 categories).

---

## What the pipeline computes

For every hidden unit of every trained network, `analyze_results` reports:

- **Best-fitting phase space** and its multiple correlation `R` (ties resolved to the
  lowest frequency, so aliasing cannot inflate the recruited range).
- **Spectral concentration** — the share of a unit's explained spectral power in its single
  best frequency; a direct measure of the *one-phase-space* claim, reported against a
  per-system chance baseline.
- **Permutation null** with Benjamini–Hochberg FDR control across units.
- **Label-permutation and input-shuffle controls** establishing the structure is
  task-driven.
- **Non-Fourier baselines** (pitch-height ramp, random ceiling).
- All system-level comparisons made at the **network level** (the unit of replication).

See `data/sample_results/results_report.md` for a reference report from the full run.

---

## Reproducing Figure 5

```bash
python paper/make_fig6.py     # writes paper/figures/fig6_frequency_range.pdf
```

The script reads `paper/figures/fig5_data.csv` (the per-frequency significant-unit counts).
To regenerate that data from a fresh run, see the note at the top of `make_fig6.py`.

---

## Citing

If you use this code or data, please cite both the software (`CITATION.cff`) and the paper.

## License

MIT — see `LICENSE`.
