"""
Stimulus and task construction.

What is reconstructed exactly (runs out of the box):
  * The INTERVAL stimulus space. The manuscript reports 153 input patterns for the
    17-tone system; 153 = C(17+1, 2) = multisets of size 2 from 17 = 17 unisons + 136
    dyads. We generate exactly this set. (Western matched: C(12+1,2) = 78.)
  * The binary 17-d (or 12-d) encoding.
  * A default size-class labelling so every task trains end-to-end.

What YOU must plug in to reproduce the paper's exact category counts (24 / 13 for
intervals; 36/16/15/12/11/8/7 for the seven Dang tasks):
  * `label_interval(system, i, j)`  -> the named interval category.
  * `dang_catalog(system)`          -> the list of valid Dangs and their names,
    plus the dissonance-handling rule for each of the 7 Dang tasks.
These depend on the Iranian music-theory tables in your supplement and on domain
judgement, so they are exposed as overridable callables rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Callable

import numpy as np

from .pitch_systems import PitchSystem


# ----------------------------------------------------------------------------- #
# Encoding
# ----------------------------------------------------------------------------- #

def encode(pcs: tuple[int, ...], N: int) -> np.ndarray:
    """Binary indicator vector: 1 where a pitch class is present."""
    v = np.zeros(N, dtype=float)
    for p in pcs:
        v[p] = 1.0
    return v


@dataclass
class Task:
    name: str
    system: PitchSystem
    X: np.ndarray              # (n_patterns, N) inputs
    y: np.ndarray              # (n_patterns,) integer class ids
    class_names: list[str]
    stim_pcs: list[tuple]      # the pitch-class content behind each row (for inspection)

    @property
    def n_outputs(self) -> int:
        return len(self.class_names)

    @property
    def Y_onehot(self) -> np.ndarray:
        Y = np.zeros((len(self.y), self.n_outputs), dtype=float)
        Y[np.arange(len(self.y)), self.y] = 1.0
        return Y

    def summary(self) -> str:
        return (f"{self.name}: {self.X.shape[0]} patterns, "
                f"{self.n_outputs} output categories, N={self.system.N}")


# ----------------------------------------------------------------------------- #
# Default labellers (REPLACE with exact music-theory tables for final runs)
# ----------------------------------------------------------------------------- #

def default_interval_label(system: PitchSystem, i: int, j: int, collapse_inversions: bool) -> str:
    """
    Default: label by undirected step-class. Reproduces the STRUCTURE of the two
    interval tasks (separate vs collapsed inversions) but NOT the exact named-category
    counts, which require the Iranian interval-name map.
    """
    if i == j:
        return "unison"
    s = system.step_index(i, j)           # directed steps 1..N-1
    if collapse_inversions:
        s = min(s, system.N - s)          # fold inversion: s ~ N-s
        return f"ic{s}"                    # interval class
    return f"i{s}"                         # directed-ish interval (separate inversion)


# ----------------------------------------------------------------------------- #
# Interval tasks
# ----------------------------------------------------------------------------- #

def build_interval_task(system: PitchSystem,
                        collapse_inversions: bool = False,
                        labeller: Callable[[PitchSystem, int, int, bool], str] | None = None,
                        name: str | None = None) -> Task:
    """
    Interval stimulus space = all multisets {i <= j} of size 2 from the N pitch classes.
    For N=17 this is 153 patterns; for N=12, 78. Matches the manuscript's 153.
    """
    labeller = labeller or default_interval_label
    rows, labels, stim = [], [], []
    for i, j in combinations_with_replacement(range(system.N), 2):
        pcs = (i,) if i == j else (i, j)
        rows.append(encode(pcs, system.N))
        labels.append(labeller(system, i, j, collapse_inversions))
        stim.append(pcs)
    return _finalize(system, rows, labels, stim,
                     name or f"interval_{'collapsed' if collapse_inversions else 'separate'}_{system.label}")


# ----------------------------------------------------------------------------- #
# Dang tasks
# ----------------------------------------------------------------------------- #

def enumerate_dangs(system: PitchSystem,
                    fourth_cents: float = 498.0,
                    cents_tol: float = 12.0,
                    allowed_step_cents: tuple[float, ...] | None = None):
    """
    Enumerate candidate Dangs: 4 pitch classes built from 3 successive ascending
    intervals whose total spans a perfect fourth (~498 cents). Returns list of
    (root, (s1, s2, s3), pcs) where s* are step counts.

    `allowed_step_cents` optionally restricts which single-step intervals may be used
    (this is where the Iranian interval inventory enters and where exact pattern counts
    like 246 come from). If None, any positive step that keeps the total within
    tolerance of a fourth is allowed.
    """
    N = system.N
    # precompute directed cents for every step count 1..N-1 from a canonical root 0
    out = []
    seen = set()
    for root in range(N):
        # three ascending moves
        for s1 in range(1, N):
            p1 = (root + s1) % N
            c1 = system.cents(root, p1)
            for s2 in range(1, N):
                p2 = (p1 + s2) % N
                c2 = system.cents(p1, p2)
                for s3 in range(1, N):
                    p3 = (p2 + s3) % N
                    c3 = system.cents(p2, p3)
                    total = c1 + c2 + c3
                    if abs(total - fourth_cents) > cents_tol:
                        continue
                    if allowed_step_cents is not None:
                        if not all(any(abs(c - a) <= cents_tol for a in allowed_step_cents)
                                   for c in (c1, c2, c3)):
                            continue
                    pcs = (root, p1, p2, p3)
                    if len(set(pcs)) != 4:
                        continue
                    key = pcs
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((root, (s1, s2, s3), pcs))
    return out


def default_dang_label(intervals_cents: tuple[float, float, float]) -> str:
    """Default Dang name = its rounded interval signature. Replace with the named catalog."""
    return "-".join(str(int(round(c))) for c in intervals_cents)


def build_dang_task(system: PitchSystem,
                    dissonance_rule: Callable[[tuple], str | None] | None = None,
                    name: str = "dang",
                    **enum_kwargs) -> Task:
    """
    Build a Dang classification task. `dissonance_rule(intervals_cents)` returns either
    a replacement label (e.g. a shared 'dissonant' bucket) or None to keep the Dang's
    own name, or the sentinel '__EXCLUDE__' to drop it from the training set. The seven
    manuscript Dang tasks differ ONLY in this rule, so implement seven rules and reuse
    this builder.
    """
    rows, labels, stim = [], [], []
    for root, steps, pcs in enumerate_dangs(system, **enum_kwargs):
        ic = (system.cents(pcs[0], pcs[1]),
              system.cents(pcs[1], pcs[2]),
              system.cents(pcs[2], pcs[3]))
        lab = default_dang_label(ic)
        if dissonance_rule is not None:
            r = dissonance_rule(ic)
            if r == "__EXCLUDE__":
                continue
            if r is not None:
                lab = r
        rows.append(encode(pcs, system.N))
        labels.append(lab)
        stim.append(pcs)
    return _finalize(system, rows, labels, stim, name)


# ----------------------------------------------------------------------------- #
# Finalize
# ----------------------------------------------------------------------------- #

def _finalize(system, rows, labels, stim, name) -> Task:
    X = np.array(rows, dtype=float)
    uniq = sorted(set(labels))
    idx = {c: k for k, c in enumerate(uniq)}
    y = np.array([idx[c] for c in labels], dtype=int)
    return Task(name=name, system=system, X=X, y=y, class_names=uniq, stim_pcs=stim)


if __name__ == "__main__":
    from .pitch_systems import iranian_system, western_system
    iran, west = iranian_system(), western_system()

    t1 = build_interval_task(iran, collapse_inversions=False)
    t2 = build_interval_task(iran, collapse_inversions=True)
    tw = build_interval_task(west, collapse_inversions=False, name="interval_separate_Western-12")
    print(t1.summary())
    print(t2.summary())
    print(tw.summary())

    d = build_dang_task(iran, name="dang_demo")
    print(d.summary())
