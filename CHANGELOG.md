# Changelog

## 1.0.0 — 2026-07-12

First public release accompanying the manuscript submission.

- Full simulation pipeline (`iranian_anns`) reproducing the paper's nine Iranian tasks and
  matched Western tasks with exact pattern/category counts.
- 24-tone quarter-tone lattice framing; lattice-DFT phase spaces (frequencies 1–12 Iranian,
  1–6 Western).
- Permutation-null inference with Benjamini–Hochberg FDR control.
- Spectral-concentration index (with per-system chance baseline).
- 7–12 band-occupancy analysis (the non-mechanical core of the cardinality result).
- Label-permutation and input-shuffle controls; non-Fourier baselines.
- Network-level statistics throughout.
- Figure 5 generation script (seaborn).
- Reproducibility tests (`pytest`).
- LaTeX manuscript, bibliography, and compiled PDF.
