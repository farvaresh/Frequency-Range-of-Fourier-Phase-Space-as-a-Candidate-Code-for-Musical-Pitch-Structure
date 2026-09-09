"""
Gaussian "value unit" network trained with the generalized delta rule.

Faithful to the manuscript's description of the Rumelhart environment (Dawson, 2008):
  * Gaussian activation on ALL hidden and output units:  a(net) = exp(-(net - mu)^2)
  * One trainable mean mu per unit, initialised to 0.
  * Connection weights initialised ~ Uniform(-0.1, 0.1).
  * Generalized delta rule (online), learning rate 0.005, SSE objective.
  * Pattern order re-randomised each epoch.
  * "Hit" criterion: for every pattern, target-1 unit activation >= 0.9 and every
    target-0 unit activation <= 0.1. Training stops when all patterns are hits.

Gradients are hand-derived (no autograd dependency); these nets are tiny so pure
NumPy is fast and exactly reproduces the delta-rule update.

  Activation:        a = exp(-(net - mu)^2)
  d a / d net  = -2 (net - mu) a
  d a / d mu   = +2 (net - mu) a = -(d a / d net)

  Output delta:  delta_o = (a_o - t_o) * (-2 (net_o - mu_o) a_o)
  Hidden delta:  delta_h = (sum_o delta_o W_ho[o,h]) * (-2 (net_h - mu_h) a_h)
  Weight grads:  dE/dW_ho[o,h] = delta_o a_h[h];   dE/dW_ih[h,i] = delta_h x[i]
  Mean grads:    dE/dmu_o = -delta_o;              dE/dmu_h = -delta_h
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gaussian(net, mu):
    return np.exp(-(net - mu) ** 2)


@dataclass
class TrainResult:
    converged: bool       # True iff strict hit criterion met for ALL patterns
    epochs: int
    W_ih: np.ndarray      # (H, In)  input -> hidden weights (the object analysed)
    mu_h: np.ndarray      # (H,)
    W_ho: np.ndarray      # (Out, H)
    mu_o: np.ndarray      # (Out,)
    final_max_err: float
    accuracy: float = float("nan")   # argmax classification accuracy of returned weights


class ValueUnitNet:
    def __init__(self, n_in, n_hidden, n_out, lr=0.005, seed=0,
                 hit_hi=0.9, hit_lo=0.1, weight_range=0.1, mu_break=0.0):
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.lr = lr
        self.hit_hi, self.hit_lo = hit_hi, hit_lo
        rng = np.random.default_rng(seed)
        self.W_ih = rng.uniform(-weight_range, weight_range, size=(n_hidden, n_in))
        self.W_ho = rng.uniform(-weight_range, weight_range, size=(n_out, n_hidden))
        # Means initialised to 0 (paper). A small symmetry-break (mu_break>0) can be
        # used to escape the all-at-peak saddle; set mu_break=0 for the literal paper init.
        self.mu_h = rng.uniform(-mu_break, mu_break, size=n_hidden)
        self.mu_o = rng.uniform(-mu_break, mu_break, size=n_out)

    # --- forward -----------------------------------------------------------
    def forward(self, x):
        net_h = self.W_ih @ x
        a_h = gaussian(net_h, self.mu_h)
        net_o = self.W_ho @ a_h
        a_o = gaussian(net_o, self.mu_o)
        return net_h, a_h, net_o, a_o

    def forward_batch(self, X):
        net_h = X @ self.W_ih.T              # (P, H)
        a_h = gaussian(net_h, self.mu_h)
        net_o = a_h @ self.W_ho.T            # (P, Out)
        a_o = gaussian(net_o, self.mu_o)
        return a_o, a_h

    # --- one online update -------------------------------------------------
    def _update(self, x, t):
        net_h, a_h, net_o, a_o = self.forward(x)
        # output layer
        delta_o = (a_o - t) * (-2.0 * (net_o - self.mu_o) * a_o)      # (Out,)
        dW_ho = np.outer(delta_o, a_h)                                # (Out, H)
        dmu_o = -delta_o
        # hidden layer
        back = self.W_ho.T @ delta_o                                  # (H,)
        delta_h = back * (-2.0 * (net_h - self.mu_h) * a_h)           # (H,)
        dW_ih = np.outer(delta_h, x)                                  # (H, In)
        dmu_h = -delta_h
        # apply
        self.W_ho -= self.lr * dW_ho
        self.mu_o -= self.lr * dmu_o
        self.W_ih -= self.lr * dW_ih
        self.mu_h -= self.lr * dmu_h

    # --- one vectorized full-batch update (fast; same generalized delta rule) --
    def _update_batch(self, X, Y):
        net_h = X @ self.W_ih.T                       # (P,H)
        a_h = gaussian(net_h, self.mu_h)
        net_o = a_h @ self.W_ho.T                     # (P,Out)
        a_o = gaussian(net_o, self.mu_o)
        P = X.shape[0]
        delta_o = (a_o - Y) * (-2.0 * (net_o - self.mu_o) * a_o)   # (P,Out)
        dW_ho = delta_o.T @ a_h / P
        dmu_o = -delta_o.mean(0)
        back = delta_o @ self.W_ho                                 # (P,H)
        delta_h = back * (-2.0 * (net_h - self.mu_h) * a_h)        # (P,H)
        dW_ih = delta_h.T @ X / P
        dmu_h = -delta_h.mean(0)
        self.W_ho -= self.lr * dW_ho
        self.mu_o -= self.lr * dmu_o
        self.W_ih -= self.lr * dW_ih
        self.mu_h -= self.lr * dmu_h

    # --- hit criterion -----------------------------------------------------
    def all_hits(self, X, Y):
        a_o, _ = self.forward_batch(X)
        hi_ok = np.where(Y == 1, a_o >= self.hit_hi, True)
        lo_ok = np.where(Y == 0, a_o <= self.hit_lo, True)
        return bool(np.all(hi_ok & lo_ok))

    def max_pattern_error(self, X, Y):
        a_o, _ = self.forward_batch(X)
        return float(np.max(np.abs(a_o - Y)))

    def mean_error(self, X, Y):
        a_o, _ = self.forward_batch(X)
        return float(np.mean(np.abs(a_o - Y)))

    def accuracy(self, X, y):
        a_o, _ = self.forward_batch(X)
        return float((a_o.argmax(1) == y).mean())

    def _snapshot(self, converged, epoch, X, Y, y):
        return TrainResult(converged, epoch, self.W_ih.copy(), self.mu_h.copy(),
                           self.W_ho.copy(), self.mu_o.copy(),
                           self.max_pattern_error(X, Y), self.accuracy(X, y))

    # --- train -------------------------------------------------------------
    def train(self, X, Y, y=None, max_epochs=20000, check_every=25, rng=None,
              anneal=True, mode="batch", verbose=False):
        """
        Train to the strict hit criterion. Returns the converged weights if the
        criterion is met; otherwise returns the BEST weights seen (lowest mean error),
        with converged=False, so downstream phase-space analysis always has a trained
        network to work on. `anneal` linearly decays lr over training.

        mode='batch'  : fast vectorized full-batch generalized delta rule (default).
        mode='online' : per-pattern updates with per-epoch reshuffling (paper-faithful,
                        much slower in Python; use for final confirmatory runs).
        """
        rng = rng or np.random.default_rng()
        if y is None:
            y = Y.argmax(1)
        P = X.shape[0]
        base_lr = self.lr
        best_mean = np.inf
        best = self._snapshot(False, 0, X, Y, y)
        for epoch in range(1, max_epochs + 1):
            if anneal:
                self.lr = base_lr * (1.0 - 0.9 * epoch / max_epochs)
            if mode == "online":
                for p in rng.permutation(P):
                    self._update(X[p], Y[p])
            elif mode == "hybrid":
                # batch for the first 70% of the budget, then online polish (which is
                # far better at driving individual hard patterns to the Gaussian tails)
                if epoch < int(0.7 * max_epochs):
                    self._update_batch(X, Y)
                else:
                    for p in rng.permutation(P):
                        self._update(X[p], Y[p])
            else:
                self._update_batch(X, Y)
            if epoch % check_every == 0 or epoch == 1:
                if self.all_hits(X, Y):
                    self.lr = base_lr
                    return self._snapshot(True, epoch, X, Y, y)
                m = self.mean_error(X, Y)
                if m < best_mean:
                    best_mean = m
                    best = self._snapshot(False, epoch, X, Y, y)
                if verbose:
                    print(f"  epoch {epoch:5d}  max|err|={self.max_pattern_error(X, Y):.3f} "
                          f"mean={m:.3f} acc={self.accuracy(X, y):.3f}")
        self.lr = base_lr
        return best


def train_one(task, n_hidden, seed, lr=0.5, max_epochs=15000,
              shuffle_pc_to_input=False, mu_break=0.3, anneal=True, mode="hybrid"):
    """
    Train a single network on a task. If shuffle_pc_to_input is True, the assignment
    of pitch classes to input dimensions is randomly permuted BEFORE training (the
    'shuffled-input' control for whether circular input coding scaffolds Fourier
    solutions). The permutation is returned so the analysis can either undo it or
    test fit on the shuffled coordinates.
    """
    rng = np.random.default_rng(seed)
    X = task.X.copy()
    perm = np.arange(task.system.N)
    if shuffle_pc_to_input:
        perm = rng.permutation(task.system.N)
        X = X[:, perm]
    Y = task.Y_onehot
    net = ValueUnitNet(task.system.N, n_hidden, task.n_outputs, lr=lr, seed=seed,
                       mu_break=mu_break)
    res = net.train(X, Y, y=task.y, max_epochs=max_epochs, rng=rng, anneal=anneal, mode=mode)
    return res, perm


if __name__ == "__main__":
    from .pitch_systems import iranian_system
    from .tasks import build_interval_task
    iran = iranian_system()
    task = build_interval_task(iran, collapse_inversions=True)
    print(task.summary())
    res, perm = train_one(task, n_hidden=7, seed=1, max_epochs=4000)
    print(f"converged={res.converged} epochs={res.epochs} max|err|={res.final_max_err:.3f}")
