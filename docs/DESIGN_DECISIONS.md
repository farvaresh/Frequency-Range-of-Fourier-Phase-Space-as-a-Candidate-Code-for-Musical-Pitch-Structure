# Design decisions

Three choices shape this analysis in ways that are not obvious from the code alone. Each
was made deliberately, and each is recorded here so that a reader can disagree with it on
the merits.

---

## 1. The Iranian pitch classes are placed on a 24-tone lattice, which fixes the frequency ceiling at 12

The 17 Iranian pitch classes of the modified Vaziri temperament are not 17 arbitrary
positions on the octave circle. Every one of them falls on a multiple of 50 cents, that is,
on a quarter-tone (24-point) lattice. In lattice indices they occupy

```
0, 2, 3, 4, 6, 7, 8, 10, 12, 13, 14, 16, 17, 18, 20, 22, 23
```

and the seven remaining positions are unused. This matters because it determines the
principled Fourier basis. A length-24 signal has `24/2 = 12` distinct non-DC frequencies,
so the Iranian system admits phase spaces `k = 1 … 12`. A 12-tone system sits on a
12-point lattice and admits `k = 1 … 6`.

A tempting alternative is to treat the 17 active notes as 17 equally spaced samples, which
would give `⌊17/2⌋ = 8` frequencies. That reading is wrong: it discards the quarter-tone
geometry that defines the system, and it does not reproduce the original analysis. The
lattice construction implemented here reproduces the authors' original phase-offset search
exactly, down to the same best-fitting frequency and correlation for a published example
weight vector (see `tests/test_reproducibility.py`).

One consequence deserves attention. Because only 17 of 24 lattice points are active, the
phase-space templates are **not** perfectly orthogonal over the active points. This is
precisely why the permutation null and the non-Fourier baselines are necessary rather than
decorative: they establish that observed fits exceed chance despite the non-orthogonality.

---

## 2. The Nyquist phase space (k = 12) is excluded from the central claim

On a 24-point lattice, phase space `k = 12` evaluates to `cos(π·g) = (−1)^g`. It is `+1` at
every even lattice index and `−1` at every odd index, and its sine component vanishes
identically. The even indices are exactly the twelve chromatic pitch classes; the odd
indices are exactly the five Koron/Sori microtonal inflections.

Phase space 12 is therefore mathematically identical to a binary chromatic-versus-microtonal
indicator, and it is one-dimensional rather than a genuine two-dimensional phase space. A
hidden unit that simply learns to separate chromatic from microtonal inputs will correlate
near-perfectly with it, without representing anything periodic.

This is quantitatively material. Of the 580 Iranian units in the nominal 7–12 band, 328 sit
at `k = 12` alone. We therefore report the band **7–11** as the confound-free claim, and
mark `k = 12` explicitly in the figures. The prevalence of `k = 12` units is itself
informative — it suggests that a salient part of what these networks learn about Iranian
pitch material is the chromatic/microtonal distinction — but it cannot be counted as
evidence for periodic coding.

`analyze_results.table_nyquist_confound()` reports the band with and without `k = 12`.

---

## 3. Arabic and Turkish systems are stated as predictions, not simulated

The cardinality hypothesis predicts a recruited-frequency ceiling of `⌊G/2⌋` for a lattice
of `G` equal steps: 6 for Western 12-TET, 12 for the Iranian quarter-tone lattice, 12 for
Arabic maqam on its 24-TET grid, and 26 for Turkish makam on the 53-comma model. Only the
first two are simulated here.

The reason is not that the Arabic and Turkish traditions are less worth studying, nor that
their theoretical models are uniquely contested. It is that simulating them would require
us to *select* one lattice from among competing proposals. Arabic 24-TET is explicitly a
notational convention that does not fix performed microtonal detail (Touma, 1996), and the
53-comma Arel–Ezgi–Uzdilek model is standard in theory but only partially realised in
practice (Bozkurt et al., 2009). Whichever lattice we picked, the resulting ceiling would
be an artefact of our choice rather than a finding.

The same caveat applies to our own case, and the manuscript records it: the modified Vaziri
temperament is itself a twentieth-century systematisation, and performed Persian music uses
flexible intonation that it does not fully capture (Farhat, 1990). What differs is that for
the Iranian case the specification was fixed independently of us — it is the system in which
the networks were originally trained and their stimuli generated. Stating the other ceilings
as predictions keyed to a declared lattice avoids making a musicological choice on those
traditions' behalf, and leaves the prediction falsifiable by whoever does make it.
