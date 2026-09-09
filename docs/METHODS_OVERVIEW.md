# Methods overview: how the code maps to the paper

This document connects each stage of the pipeline to the corresponding part of the
manuscript, so a reader can navigate from a claim in the paper to the code that produces it.

## 1. Pitch-class systems (`pitch_systems.py`)

The 17 Iranian pitch classes lie on a **24-tone quarter-tone lattice** (every position a
multiple of 50 cents), occupying lattice indices
`[0,2,3,4,6,7,8,10,12,13,14,16,17,18,20,22,23]`. The Western system is 12 notes on a
12-tone lattice.

- `iranian_system()`, `western_system()` build the `PitchSystem` objects.
- `dft_phase_bases(system)` builds phase space `k` as the 2-D span of
  `{cos(2πk·gₙ/G), sin(2πk·gₙ/G)}` at the active lattice indices `gₙ`, for
  `k = 1..⌊G/2⌋`. This gives **12 phase spaces for Iranian, 6 for Western** — the origin of
  the paper's "frequencies 1–12 vs 1–6."

## 2. Stimuli and tasks (`tasks.py`, `music_theory.py`)

- `music_theory.py` holds the authors' real values: the Vaziri cents, the **24 interval
  names** keyed by cents, and the **7 Dang task rules**. Even Dang tasks collapse dissonant
  Dangs into a shared category; odd tasks exclude them.
- `build_all_tasks(system)` produces all nine Iranian tasks with exactly the paper's
  pattern/category counts (verified in `tests/`).

## 3. Networks (`value_unit_net.py`)

Gaussian "value unit" activations `exp(-(net-μ)²)` with a trainable centre `μ` per unit,
trained by the generalized delta rule to a hit criterion (target-1 ≥ 0.9, target-0 ≤ 0.1).

## 4. Fits and inference (`analysis.py`)

- `PhaseBank` fits each hidden unit's weights to every phase space, returning the best
  frequency and its multiple correlation `R`. **Two safeguards**: degenerate (constant)
  bases are dropped, and aliasing ties are resolved to the lowest frequency so the recruited
  range cannot be spuriously inflated.
- `concentration()` = share of a unit's total spectral power in its best frequency (the
  spectral-concentration index, A1).
- `weight_shuffle_null()` builds the permutation null.
- `baseline_fits()` computes the pitch-height and random baselines.

## 5. Orchestration (`run_experiment.py`)

Trains the networks, applies the three control conditions (`none`, `label_shuffled`,
`input_shuffled`), and writes `unit_fits.csv`. Uses the paper's per-task architecture
(7 hidden units for interval tasks; 11/12/8/12/5/12/4 for Dang tasks 1–7).

## 6. Statistics and report (`analyze_results.py`)

- **FDR** (Benjamini–Hochberg) applied to permutation p-values within each system×control
  family; significance is defined on the q-value (A2).
- **7–12 band occupancy** table — the non-mechanical core of the cardinality claim (A3).
- **Concentration** table with per-system chance baseline (A1).
- **Network-level** comparisons throughout, because `system` is a between-network factor
  and hidden units are not independent.
- Assembles `results_report.md` and the figures.

## Design decision: no Arabic/Turkish simulation

The cardinality hypothesis is extended as a **falsifiable, sourced prediction** in the
Discussion rather than by simulating Arabic/Turkish systems, whose fine tunings are
contested in the primary sources. See `docs/DESIGN_DECISIONS.md`.
