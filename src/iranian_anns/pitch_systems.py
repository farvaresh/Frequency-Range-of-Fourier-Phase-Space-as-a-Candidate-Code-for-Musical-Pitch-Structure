"""
Pitch-class systems and Fourier phase-space construction.

CORRECTED FRAMING (from the authors' original analysis code, `Final analysis codeVVV.py`):

The 17 Iranian pitch classes are NOT 17 arbitrary angular positions. In the modified
Vaziri temperament every note falls on a 24-tone quarter-tone lattice (every position is
a multiple of 50 cents = 1/24 octave). The 17 notes occupy 17 of the 24 equally spaced
positions:

    24-grid index : 0, 2, 3, 4, 6, 7, 8, 10, 12, 13, 14, 16, 17, 18, 20, 22, 23
    cents         : 0,100,150,200,300,350,400,500,600,650,700,800,850,900,1000,1100,1150

Consequences:
  * The natural Fourier basis is the length-24 DFT, whose distinct non-DC frequencies run
    1..12 (= 24/2). This is why the manuscript's phase-space frequencies 1..12 are correct
    and principled -- the Nyquist ceiling is set by the 24-point lattice, not by the count
    of active notes. (An earlier reading that used 1..8 = floor(17/2) was WRONG: it ignored
    the quarter-tone lattice.)
  * The Western 12-TET system lives on a 12-point lattice, ceiling 6. So the cross-system
    comparison is 12 (Iran, on a 24-grid) vs 6 (West, on a 12-grid): a lattice-cardinality
    comparison.
  * Because only 17 of 24 grid points are active, the phase-space templates are not perfectly
    orthogonal over the 17 active points. This is exactly why the permutation null and
    non-Fourier baselines (added in analysis.py) matter: they establish that the fits exceed
    chance despite the non-orthogonality.

The construction here evaluates phase space k on the actual grid positions of the active
notes, which reproduces the authors' method exactly (verified: same best frequency and R).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# The modified-Vaziri layout (from the authors' code)
# --------------------------------------------------------------------------- #

# Successive step sizes in cents between consecutive notes (closing the octave uses the
# implicit final 50c step to return to 1200). This is the authors' `cents` list.
IRANIAN_STEP_CENTS = [0, 100, 50, 50, 100, 50, 50, 100, 100, 50, 50, 100, 50, 50, 100, 100, 50]

IRANIAN_NAMES = ['c', 'dflat', 'dkoron', 'd', 'eflat', 'ekoron', 'e', 'f', 'fsori',
                 'fsharp', 'g', 'aflat', 'akoron', 'a', 'bflat', 'bkoron', 'b']

GRID = 24              # quarter-tone lattice size for the Iranian system
WESTERN_GRID = 12


def iranian_abs_cents() -> np.ndarray:
    """Absolute cents of the 17 pitch classes (cumulative sum of the step sizes)."""
    return np.cumsum(IRANIAN_STEP_CENTS).astype(float)


class PitchSystem:
    """A pitch-class system defined on an equally spaced lattice of size `grid`.

    `grid_index[n]` gives the lattice position (0..grid-1) of active note n.
    For 12-TET, grid = 12 and the 12 notes occupy all positions 0..11.
    For the Iranian system, grid = 24 and the 17 notes occupy 17 of 24 positions.
    """

    def __init__(self, names, grid_index, grid, label):
        self.names = list(names)
        self.grid_index = np.asarray(grid_index, dtype=int)
        self.grid = grid
        self.label = label
        self.N = len(self.names)
        assert self.grid_index.shape == (self.N,)
        # angular position of each active note on the lattice, in radians
        self.angles = 2.0 * np.pi * self.grid_index / self.grid

    @property
    def n_distinct_dft_freqs(self) -> int:
        """floor(grid/2): distinct non-DC DFT frequencies of the lattice."""
        return self.grid // 2

    def step_index(self, i: int, j: int) -> int:
        """Directed distance i->j in *lattice units* (0..grid-1)."""
        return int((self.grid_index[j] - self.grid_index[i]) % self.grid)

    def cents(self, i: int, j: int) -> float:
        """Directed interval i->j in cents on [0,1200)."""
        return 1200.0 * ((self.grid_index[j] - self.grid_index[i]) % self.grid) / self.grid


def iranian_system(step_cents=None, names=None) -> PitchSystem:
    """Build the 17-note Iranian system on the 24-tone lattice."""
    steps = IRANIAN_STEP_CENTS if step_cents is None else step_cents
    names = IRANIAN_NAMES if names is None else names
    abs_cents = np.cumsum(steps).astype(float)
    grid_index = np.round(abs_cents / (1200.0 / GRID)).astype(int)
    return PitchSystem(names, grid_index, GRID, label="Iranian-17")


def western_system() -> PitchSystem:
    """Build the 12-TET Western system (12 notes on a 12-point lattice)."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return PitchSystem(names, np.arange(12), WESTERN_GRID, label="Western-12")


# --------------------------------------------------------------------------- #
# Phase-space bases
# --------------------------------------------------------------------------- #

def dft_phase_bases(system: PitchSystem) -> dict:
    """
    Principled DFT phase spaces on the lattice. Phase space k is the 2-D span of
    {cos(2*pi*k*g_n/grid), sin(2*pi*k*g_n/grid)} evaluated at the active grid positions
    g_n, for k = 1..floor(grid/2). This reproduces the authors' construction exactly
    (their phase-offset search over 1200 steps == the {cos,sin} 2-D projection here).

    Iranian: grid=24 -> frequencies 1..12.   Western: grid=12 -> frequencies 1..6.
    """
    g = system.grid_index
    bases = {}
    for k in range(1, system.grid // 2 + 1):
        ang = 2.0 * np.pi * k * g / system.grid
        bases[k] = np.column_stack([np.cos(ang), np.sin(ang)])
    return bases


# Backwards-compatible aliases (older code imported these names).
def temperament_phase_bases(system: PitchSystem, max_freq=None) -> dict:
    """Alias: on the lattice framing the principled DFT already spans 1..grid/2, so the
    'template' basis and the DFT basis coincide. Kept so run_experiment's dual-basis loop
    still works; it will simply report identical numbers under both names."""
    return dft_phase_bases(system)


if __name__ == "__main__":
    iran, west = iranian_system(), western_system()
    print(f"{iran.label}: {iran.N} notes on a {iran.grid}-grid -> "
          f"{iran.n_distinct_dft_freqs} distinct DFT freqs")
    print(f"  grid positions: {list(iran.grid_index)}")
    print(f"  abs cents:      {[int(c) for c in iranian_abs_cents()]}")
    print(f"{west.label}: {west.N} notes on a {west.grid}-grid -> "
          f"{west.n_distinct_dft_freqs} distinct DFT freqs")
    print("C->Db cents:", round(iran.cents(0, 1), 1),
          " C->D-koron cents:", round(iran.cents(0, 2), 1))
