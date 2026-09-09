"""
iranian_anns -- Fourier phase-space analysis of artificial neural networks trained on
Iranian classical music classification tasks.

This package reproduces and extends the analysis in the accompanying manuscript,
"Frequency Range of Fourier Phase Space as a Candidate Code for Musical Pitch Structure."

Public API
----------
pitch_systems  : pitch-class systems on their lattices + Fourier phase-space construction
tasks          : stimulus encoding and interval-task builder
music_theory   : the real Vaziri cents, 24 interval names, and 7 Dang task rules
value_unit_net : Gaussian value-unit network with the generalized delta rule
analysis       : phase-space fits, permutation nulls, baselines, spectral concentration
run_experiment : orchestration -> unit_fits.csv
analyze_results: statistical tables, figures, and the results report

See README.md for a quick start, and run_config.py for ready-made run profiles.
"""

__version__ = "1.0.0"

from . import (  # noqa: F401
    pitch_systems,
    tasks,
    music_theory,
    value_unit_net,
    analysis,
)

__all__ = [
    "pitch_systems", "tasks", "music_theory", "value_unit_net",
    "analysis", "run_experiment", "analyze_results",
]
