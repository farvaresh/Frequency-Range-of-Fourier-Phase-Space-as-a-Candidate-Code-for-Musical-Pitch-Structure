"""
Reproducibility tests: verify that the pipeline reproduces the manuscript's exact
task structure and the 24-tone lattice framing. Run with:  pytest

These are fast, deterministic checks (no network training) that a reader can use to
confirm the environment reproduces the published pattern and category counts.
"""

import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if (_src / "iranian_anns").is_dir():
    sys.path.insert(0, str(_src))

from iranian_anns.pitch_systems import iranian_system, western_system
from iranian_anns.music_theory import build_all_tasks


def test_iranian_lattice_is_24_grid():
    iran = iranian_system()
    assert iran.grid == 24
    assert iran.N == 17
    assert iran.n_distinct_dft_freqs == 12
    assert list(iran.grid_index) == [0, 2, 3, 4, 6, 7, 8, 10, 12,
                                     13, 14, 16, 17, 18, 20, 22, 23]


def test_western_lattice_is_12_grid():
    west = western_system()
    assert west.grid == 12
    assert west.N == 12
    assert west.n_distinct_dft_freqs == 6


def test_all_nine_iranian_task_counts_match_paper():
    """Patterns and categories must match the manuscript exactly."""
    expected = {
        "interval_separate": (153, 24),
        "interval_collapsed": (153, 13),
        "dang1_all36": (246, 36),
        "dang2_zaed16": (246, 16),
        "dang3_nozaed15": (121, 15),
        "dang4_diss12": (246, 12),
        "dang5_nodiss11": (88, 11),
        "dang6_diss8": (246, 8),
        "dang7_nodiss7": (62, 7),
    }
    got = {name: (t.X.shape[0], t.n_outputs) for name, t in build_all_tasks(iranian_system())}
    for name, exp in expected.items():
        assert got[name] == exp, f"{name}: expected {exp}, got {got[name]}"


def test_lattice_dft_reproduces_authors_best_frequency():
    """The published example weight vector must yield best frequency 2 at R~0.5936."""
    import numpy as np
    from iranian_anns.pitch_systems import dft_phase_bases
    from iranian_anns.analysis import PhaseBank

    hid = [0.220886407, -1.683312654, -0.580427131, -0.277700385, 0.942225584,
           -0.806645758, 1.837973498, -0.113717677, -0.25987161, -1.19012549,
           -0.634296795, 0.557945417, 0.066122927, 2.333603528, -0.236773078,
           0.824133997, 0.221624858]
    bank = PhaseBank(dft_phase_bases(iranian_system()))
    k, R = bank.best(np.array(hid))
    assert k == 2
    assert abs(R - 0.5936) < 1e-3
