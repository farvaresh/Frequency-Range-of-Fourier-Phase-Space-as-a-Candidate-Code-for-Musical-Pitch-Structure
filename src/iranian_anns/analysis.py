"""
Phase-space analysis, permutation nulls, and non-Fourier baselines (vectorized).

Speed: each phase space k is reduced once to an orthonormal basis Q_k (N x 2) spanning
the mean-centred {cos(k.), sin(k.)} columns. Then for ANY weight vector w (centred wc):
    R_k^2 = || Q_k^T wc ||^2 / || wc ||^2
which is two dot products, no least-squares in the loop. The permutation null stacks
all permuted vectors into one (n_perm x N) matrix and computes every R_k with a single
matmul per k.

Fit measures:
  * R (2-D / phase-aware): sqrt of the variance of wc explained by phase space k.
  * cos-only: |Pearson r| of wc with the cosine column (literal manuscript reading).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _center_cols(M):
    return M - M.mean(axis=0, keepdims=True)


class PhaseBank:
    """Precomputed orthonormal bases for a set of phase spaces {k: (N,2) matrix}.

    Two numerical safeguards, both essential for correctness:

    1. DEGENERATE BASES ARE DROPPED. For an equally spaced system, template k = N gives
       cos(k*phi_n) = 1 for all n, which is a constant column; after mean-centering the
       whole basis is (numerically) zero. `np.linalg.qr` of a zero matrix still returns an
       orthonormal Q built from floating-point noise, which then fits ANY weight vector
       (observed R up to 0.9). Such bases carry no information and are excluded.

    2. ALIASING TIES ARE BROKEN TOWARD THE LOWEST FREQUENCY. On equally spaced angles,
       frequencies k and N-k span the identical subspace, so R_k == R_{N-k} in exact
       arithmetic. Floating-point noise (~1e-15) otherwise lets the aliased duplicate win
       at random, spuriously inflating the apparent frequency range. We therefore treat
       fits within `tie_tol` as tied and report the smallest such k.
    """

    def __init__(self, bases, rank_tol=1e-8, tie_tol=1e-9):
        self.tie_tol = tie_tol
        self.Q, self.cos, self.dropped = {}, {}, []
        for k, B in sorted(bases.items()):
            Bc = _center_cols(B)
            scale = np.abs(Bc).max()
            if scale < rank_tol:                      # basis vanishes after centering
                self.dropped.append(k)
                continue
            Q, _ = np.linalg.qr(Bc)
            # guard against a rank-1 basis (one column vanishing after centering)
            s = np.linalg.svd(Bc, compute_uv=False)
            if s[-1] / max(s[0], 1e-300) < rank_tol:
                Q = Q[:, :1]                          # keep the single informative direction
            self.Q[k] = Q
            self.cos[k] = Bc[:, 0]
        if not self.Q:
            raise ValueError("all phase-space bases were degenerate")
        self.freqs = sorted(self.Q.keys())
        self.N = next(iter(self.Q.values())).shape[0]

    def R_by_freq(self, w):
        wc = w - w.mean()
        ss = float(wc @ wc)
        if ss == 0:
            return {k: 0.0 for k in self.freqs}
        return {k: float(np.sqrt(max(0.0, float((Q.T @ wc) @ (Q.T @ wc)) / ss)))
                for k, Q in self.Q.items()}

    def _argmax_lowest(self, r: dict):
        """Best frequency, resolving near-ties (aliasing) toward the lowest k."""
        best_R = max(r.values())
        cands = [k for k in self.freqs if r[k] >= best_R - self.tie_tol]
        k = min(cands)
        return k, r[k]

    def best(self, w):
        return self._argmax_lowest(self.R_by_freq(w))

    def phase_of(self, w, k, raw_basis):
        wc = w - w.mean()
        Bc = _center_cols(raw_basis)
        if np.abs(Bc).max() < 1e-12:
            return float("nan")
        beta, *_ = np.linalg.lstsq(Bc, wc, rcond=None)
        return float(np.arctan2(beta[1], beta[0]))

    def cos_best(self, w):
        wc = w - w.mean()
        nw = np.linalg.norm(wc)
        rs = {}
        for k, c in self.cos.items():
            denom = nw * np.linalg.norm(c)
            rs[k] = 0.0 if denom == 0 else abs(float(wc @ c / denom))
        return self._argmax_lowest(rs)

    def best_R_batch(self, Wrows):
        Wc = Wrows - Wrows.mean(axis=1, keepdims=True)
        ss = np.einsum("pn,pn->p", Wc, Wc)
        ss = np.where(ss == 0, 1.0, ss)
        best = np.zeros(Wrows.shape[0])
        for Q in self.Q.values():
            proj = Wc @ Q
            r2 = np.einsum("pj,pj->p", proj, proj) / ss
            best = np.maximum(best, r2)
        return np.sqrt(np.clip(best, 0, 1))

    def concentration(self, w):
        """
        Spectral concentration index (A1): the share of the weight vector's total
        explained spectral power that falls in its single best-fitting phase space.
        Unlike max |R|, this directly quantifies the manuscript's actual claim -- that
        each hidden unit is dominated by ONE phase space. A unit split across two
        frequencies has a high max |R| but a LOW concentration.

        Returns concentration in (0, 1]: power(best freq) / sum_k power(freq k),
        where power(k) = R_k^2. 1.0 means a single frequency explains all the
        Fourier-explainable structure; ~1/K means power is spread evenly across the K
        phase spaces.
        """
        R = self.R_by_freq(w)
        powers = {k: R[k] ** 2 for k in R}
        total = sum(powers.values())
        if total <= 0:
            return 0.0
        return max(powers.values()) / total


@dataclass
class UnitFit:
    best_freq: int
    best_R: float
    best_phase: float
    best_freq_cos: int
    best_R_cos: float
    concentration: float = float("nan")   # A1: spectral concentration in the best freq


def fit_unit(w, bank, raw_bases):
    bk, R = bank.best(w)
    bkc, Rc = bank.cos_best(w)
    ph = bank.phase_of(w, bk, raw_bases[bk])
    conc = bank.concentration(w)
    return UnitFit(best_freq=bk, best_R=R, best_phase=ph, best_freq_cos=bkc,
                   best_R_cos=Rc, concentration=conc)


def weight_shuffle_null(w, bank, n_perm=2000, rng=None):
    rng = rng or np.random.default_rng()
    observed = bank.best(w)[1]
    idx = np.argsort(rng.random((n_perm, len(w))), axis=1)
    Wp = w[idx]
    null = bank.best_R_batch(Wp)
    p = (1 + np.sum(null >= observed)) / (n_perm + 1)
    return {"observed": float(observed), "null_mean": float(null.mean()),
            "null_p95": float(np.quantile(null, 0.95)), "p_value": float(p)}


def baseline_fits(w, system_angles, rng=None, n_random=200):
    rng = rng or np.random.default_rng()
    N = len(w)
    wc = w - w.mean()
    nw = np.linalg.norm(wc)
    ramp = np.arange(N, dtype=float) - (N - 1) / 2
    ph = 0.0 if nw == 0 else abs(float(wc @ ramp / (nw * np.linalg.norm(ramp))))
    V = rng.standard_normal((n_random, N))
    Vc = V - V.mean(axis=1, keepdims=True)
    num = Vc @ wc
    den = np.linalg.norm(Vc, axis=1) * nw
    rr = np.abs(np.where(den == 0, 0.0, num / den))
    return {"pitch_height_r": ph, "random_r_p95": float(np.quantile(rr, 0.95))}


def analyze_network(W_ih, bank, raw_bases):
    return [fit_unit(W_ih[h], bank, raw_bases) for h in range(W_ih.shape[0])]


if __name__ == "__main__":
    from .pitch_systems import iranian_system, temperament_phase_bases
    import time
    iran = iranian_system()
    rng = np.random.default_rng(0)
    raw = temperament_phase_bases(iran)
    bank = PhaseBank(raw)
    w = np.cos(3 * iran.angles) + 0.05 * rng.standard_normal(iran.N)
    uf = fit_unit(w, bank, raw)
    print("best freq:", uf.best_freq, "R=", round(uf.best_R, 3), "phase=", round(uf.best_phase, 3))
    t0 = time.time()
    nul = weight_shuffle_null(w, bank, n_perm=2000, rng=rng)
    print("null:", {k: round(v, 3) for k, v in nul.items()}, f"({time.time()-t0:.4f}s)")
    print("baseline:", {k: round(v, 3) for k, v in baseline_fits(w, iran.angles, rng).items()})
