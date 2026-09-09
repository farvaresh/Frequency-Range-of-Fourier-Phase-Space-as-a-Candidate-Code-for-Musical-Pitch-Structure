"""
Music-theory tables and task catalogue (the plug-in layer).

Everything a domain expert must supply to reproduce the paper's EXACT category counts
lives here, as editable tables and clearly-named rule functions. Nothing below is
guessed from the supplement; the defaults are runnable placeholders. Edit the four
marked blocks, then `build_all_tasks(system)` returns the 2 interval + 7 Dang tasks.

The networks, encodings, phase-space analysis and nulls do NOT depend on these labels
being exactly right — only the reported category counts and the specific learned
solutions do.
"""

from __future__ import annotations

import numpy as np

from .pitch_systems import PitchSystem
from .tasks import build_interval_task, build_dang_task, Task


# ============================================================================ #
# BLOCK 1 — exact cents profile (edit in pitch_systems.iranian_system(cents=...))
# Anchors known from the manuscript: C->Db = 100c (pi/6), C->D-koron = 150c (pi/4).
# ============================================================================ #


# ============================================================================ #
# BLOCK 2 — interval names.  Map each directed step count (1..N-1) to its Iranian
# interval name.  Two interval tasks are derived from this:
#   * "separate"  : every interval AND its inversion as distinct categories  (paper: 24)
#   * "collapsed" : an interval and its inversion share a category            (paper: 13)
# The default below names by step count and therefore will NOT reproduce 24/13; replace
# the values with your gamut's interval names (some step counts may share a name, and a
# single step count may split into more than one named interval by spelling).
# ============================================================================ #

def default_interval_name_by_step(N: int) -> dict[int, str]:
    return {s: f"step{s}" for s in range(1, N)}

# The authors' interval nomenclature, keyed by interval size in CENTS (from
# `create intervals(.net24Vaziri).py`). Reproduces the 24 named interval categories of
# the first interval task. The 13-category (collapsed) task folds each interval with its
# inversion (keys c and 1200-c share a category).
INTERVAL_NAME_BY_CENTS = {
    0: "hengam", 50: "zaed", 100: "baqieh", 150: "mojanab", 200: "tanini",
    250: "bishtanini", 300: "tanini+baqieh", 350: "tanini+mojanab", 400: "tanini+tanini",
    450: "tanini+bishtanini", 500: "dang", 550: "tanini+tanini+mojanab", 600: "dang+baqieh",
    650: "dang+mojanab", 700: "chireh", 750: "dang+mojanab+baqieh", 800: "chireh+baqieh",
    850: "chireh+mojanab", 900: "chireh+tanini", 950: "chireh+bishtanini",
    1000: "chireh+tanini+baqieh", 1050: "chireh+tanini+mojanab", 1100: "chireh+tanini+tanini",
    1150: "chireh+tanini+baqieh+mojanab",
}

INTERVAL_NAME_BY_STEP: dict[int, str] | None = None   # legacy, unused


def interval_label(system: PitchSystem, i: int, j: int, collapse_inversions: bool) -> str:
    c = int(round(system.cents(i, j)))
    if collapse_inversions:
        c = min(c, (1200 - c) % 1200)     # fold interval with its inversion
    return INTERVAL_NAME_BY_CENTS.get(c, f"c{c}")


# ============================================================================ #
# BLOCK 3 — Dang construction (reproduces DANG_51V_20V.py exactly).
# A Dang is 4 pitch classes whose outer interval is exactly a perfect fourth = 500 cents
# (= 10 steps on the 24-grid). For each such (start, end) pair we collect the interior
# notes and choose 2 of them, giving all 4-note Dangs. This yields 246 Dangs / 36 unique
# interval signatures, matching the manuscript.
# ============================================================================ #

FOURTH_CENTS = 500        # perfect fourth, exactly (authors use 500, not 498)


def enumerate_dangs_exact(system: PitchSystem):
    """Return list of (pcs4, sig3) where pcs4 is the 4 pitch-class indices and sig3 is the
    tuple of the three successive interval sizes in cents. Reproduces the authors' code."""
    N = system.N

    def cents(i, j):
        return int(round(system.cents(i, j)))

    dangs = {}
    for i in range(N):
        for j in range(N):
            if cents(i, j) != FOURTH_CENTS:
                continue
            # walk upward from i to j collecting the interior notes (wrapping)
            seq = [i]
            k = i
            while True:
                k = (k + 1) % N
                seq.append(k)
                if k == j:
                    break
            interior = seq[1:-1]
            for x in range(len(interior)):
                for y in range(x + 1, len(interior)):
                    pcs = (seq[0], interior[x], interior[y], seq[-1])
                    if len(set(pcs)) != 4:
                        continue
                    sig = (cents(pcs[0], pcs[1]), cents(pcs[1], pcs[2]), cents(pcs[2], pcs[3]))
                    dangs[pcs] = sig
    return [(pcs, sig) for pcs, sig in dangs.items()]


# ============================================================================ #
# BLOCK 4 — the SEVEN Dang classification tasks (exact rules from the manuscript
# and DANG_51V_20V.py). Even tasks (2,4,6) COLLAPSE dissonant Dangs into one shared
# category (keeping all 246 patterns); odd tasks (3,5,7) EXCLUDE them (fewer patterns).
# Interval-size codes in cents:  Zaed=50, Baqieh=100, Mojanab=150, BishTanini=250,
# Tanini+Baqieh=300.
# ============================================================================ #

def _has_step(sig, c):
    return any(s == c for s in sig)

def _has_adjacent(sig, c1, c2):
    """True if steps (sig[0],sig[1]) or (sig[1],sig[2]) equal (c1,c2) in order."""
    return (sig[0] == c1 and sig[1] == c2) or (sig[1] == c1 and sig[2] == c2)


def _is_dissonant_task4(sig):
    # Zaed anywhere, or a Baqieh-Baqieh or Baqieh-Mojanab adjacent sequence
    return (_has_step(sig, 50)
            or _has_adjacent(sig, 100, 100)
            or _has_adjacent(sig, 100, 150))


def _is_dissonant_task6(sig):
    # task4 dissonances, plus Tanini+Baqieh(300) anywhere, plus Mojanab-Baqieh and
    # Baqieh-BishTanini adjacent sequences
    return (_is_dissonant_task4(sig)
            or _has_step(sig, 300)
            or _has_adjacent(sig, 150, 100)
            or _has_adjacent(sig, 100, 250))


def dang_rule_1(sig):   # 36 categories: every unique signature is its own class
    return "-".join(map(str, sig))

def dang_rule_2(sig):   # 16: collapse any Dang containing a Zaed into one dissonance class
    return "DISSONANT" if _has_step(sig, 50) else "-".join(map(str, sig))

def dang_rule_3(sig):   # 15: EXCLUDE Zaed Dangs; classify the rest by signature
    return "__EXCLUDE__" if _has_step(sig, 50) else "-".join(map(str, sig))

def dang_rule_4(sig):   # 12: collapse task-4 dissonances into one class
    return "DISSONANT" if _is_dissonant_task4(sig) else "-".join(map(str, sig))

def dang_rule_5(sig):   # 11: EXCLUDE task-4 dissonances
    return "__EXCLUDE__" if _is_dissonant_task4(sig) else "-".join(map(str, sig))

def dang_rule_6(sig):   # 8: collapse task-6 dissonances into one class
    return "DISSONANT" if _is_dissonant_task6(sig) else "-".join(map(str, sig))

def dang_rule_7(sig):   # 7: EXCLUDE task-6 dissonances
    return "__EXCLUDE__" if _is_dissonant_task6(sig) else "-".join(map(str, sig))


DANG_RULES = [
    ("dang1_all36",       dang_rule_1),
    ("dang2_zaed16",      dang_rule_2),
    ("dang3_nozaed15",    dang_rule_3),
    ("dang4_diss12",      dang_rule_4),
    ("dang5_nodiss11",    dang_rule_5),
    ("dang6_diss8",       dang_rule_6),
    ("dang7_nodiss7",     dang_rule_7),
]


def _build_dang_task_exact(system: PitchSystem, rule, name: str) -> Task:
    """Build one Dang task from the exact enumeration + a signature->label rule.
    A rule returning '__EXCLUDE__' drops that Dang from the training set."""
    from .tasks import encode, _finalize
    rows, labels, stim = [], [], []
    for pcs, sig in enumerate_dangs_exact(system):
        lab = rule(sig)
        if lab == "__EXCLUDE__":
            continue
        rows.append(encode(pcs, system.N))
        labels.append(lab)
        stim.append(pcs)
    return _finalize(system, rows, labels, stim, name)


def build_all_tasks(system: PitchSystem) -> list[tuple[str, Task]]:
    """The full Iranian battery: 2 interval tasks + 7 Dang tasks (exact reproduction)."""
    tasks = [
        ("interval_separate",
         build_interval_task(system, collapse_inversions=False, labeller=interval_label,
                             name=f"interval_separate_{system.label}")),
        ("interval_collapsed",
         build_interval_task(system, collapse_inversions=True, labeller=interval_label,
                             name=f"interval_collapsed_{system.label}")),
    ]
    for name, rule in DANG_RULES:
        tasks.append((name, _build_dang_task_exact(system, rule, name)))
    return tasks


if __name__ == "__main__":
    from .pitch_systems import iranian_system
    iran = iranian_system()
    for name, t in build_all_tasks(iran):
        print(f"{name:26s} -> {t.X.shape[0]:4d} patterns, {t.n_outputs:3d} categories")
