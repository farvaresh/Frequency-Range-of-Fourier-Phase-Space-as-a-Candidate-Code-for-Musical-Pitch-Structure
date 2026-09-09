"""
Orchestration: run the full (or a quick) experiment and write a tidy, long-format
results table (one row per hidden unit) plus a summary figure.

Each row carries everything a Scientific Reports reanalysis needs:
  * network identity (network_seed) -> use as a RANDOM EFFECT in a mixed model,
    addressing the non-independence-of-units limitation;
  * basis in {'dft','template'} -> report the principled DFT comparison (freqs 1..floor(N/2))
    as primary and the temperament template (freqs 1..max_freq) as secondary;
  * best_freq under both the 2-D and literal cos-only measures;
  * weight-shuffle permutation p-value -> replaces the sorted-bar t-tests;
  * non-Fourier baselines (pitch-height, random ceiling);
  * shuffled_input flag -> the control for whether circular input coding scaffolds
    Fourier solutions.

Usage:
    python -m iranian_anns.run_experiment --quick      # small, ~minute
    python -m iranian_anns.run_experiment              # paper-scale defaults
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .pitch_systems import (iranian_system, western_system,
                            dft_phase_bases, temperament_phase_bases)
from .tasks import build_interval_task, build_dang_task
from .value_unit_net import train_one
from .analysis import PhaseBank, analyze_network, weight_shuffle_null, baseline_fits


def make_tasks(quick: bool, full_battery: bool = False):
    """
    Returns a list of (label, task) pairs.
      * quick=True            -> 2 Iranian + 2 Western interval tasks (fast smoke test).
      * full_battery=True     -> full Iranian battery (2 interval + 7 Dang) from
                                 music_theory.build_all_tasks, plus matched Western
                                 interval tasks (the within-study replication).
      * otherwise (default)   -> interval tasks for both systems + one Dang demo.
    """
    iran, west = iranian_system(), western_system()
    from .music_theory import interval_label as _iran_lbl
    if quick:
        return [
            ("iran_interval_separate", build_interval_task(iran, collapse_inversions=False, labeller=_iran_lbl)),
            ("iran_interval_collapsed", build_interval_task(iran, collapse_inversions=True, labeller=_iran_lbl)),
            ("west_interval_separate", build_interval_task(west, collapse_inversions=False,
                                                           name="interval_separate_Western-12")),
            ("west_interval_collapsed", build_interval_task(west, collapse_inversions=True,
                                                            name="interval_collapsed_Western-12")),
        ]
    if full_battery:
        from .music_theory import build_all_tasks
        tasks = [(f"iran_{n}", t) for n, t in build_all_tasks(iran)]
        tasks += [
            ("west_interval_separate", build_interval_task(west, collapse_inversions=False,
                                                           name="interval_separate_Western-12")),
            ("west_interval_collapsed", build_interval_task(west, collapse_inversions=True,
                                                            name="interval_collapsed_Western-12")),
        ]
        return tasks
    tasks = [
        ("iran_interval_separate", build_interval_task(iran, collapse_inversions=False, labeller=_iran_lbl)),
        ("iran_interval_collapsed", build_interval_task(iran, collapse_inversions=True, labeller=_iran_lbl)),
        ("west_interval_separate", build_interval_task(west, collapse_inversions=False,
                                                       name="interval_separate_Western-12")),
        ("west_interval_collapsed", build_interval_task(west, collapse_inversions=True,
                                                        name="interval_collapsed_Western-12")),
        ("iran_dang_demo", build_dang_task(iran, name="dang_demo")),
    ]
    return tasks


def _emit_rows(rows, label, task, banks, raw_bases, W, res, H, seed, n_perm, rng, control):
    for basis_name, bank in banks.items():
        fits = analyze_network(W, bank, raw_bases[basis_name])
        for h, uf in enumerate(fits):
            nul = weight_shuffle_null(W[h], bank, n_perm=n_perm, rng=rng)
            base = baseline_fits(W[h], task.system.angles, rng=rng, n_random=200)
            rows.append(dict(
                task=label, system=task.system.label, basis=basis_name,
                control=control, n_hidden=H, network_seed=seed,
                converged=res.converged, accuracy=round(res.accuracy, 4),
                hidden_unit=h,
                best_freq=int(uf.best_freq), best_R=round(uf.best_R, 4),
                best_phase=round(uf.best_phase, 4),
                best_freq_cos=int(uf.best_freq_cos), best_R_cos=round(uf.best_R_cos, 4),
                concentration=round(uf.concentration, 4),
                null_p=round(nul["p_value"], 5), null_mean=round(nul["null_mean"], 4),
                null_p95=round(nul["null_p95"], 4),
                pitch_height_r=round(base["pitch_height_r"], 4),
                random_r_p95=round(base["random_r_p95"], 4),
            ))


# Hidden-unit counts reported in the manuscript, per task. Used when
# hidden_units_per_task=True; otherwise the shared `hidden_units` sweep is used.
PAPER_HIDDEN_UNITS = {
    "iran_interval_separate": (7,),
    "iran_interval_collapsed": (7,),
    "iran_dang1_all36": (11,),
    "iran_dang2_zaed16": (12,),
    "iran_dang3_nozaed15": (8,),
    "iran_dang4_diss12": (12,),
    "iran_dang5_nodiss11": (5,),
    "iran_dang6_diss8": (12,),
    "iran_dang7_nodiss7": (4,),
    "west_interval_separate": (7,),
    "west_interval_collapsed": (7,),
}


def run(quick=False, out_dir="results", n_networks=25, hidden_units=(4, 6, 8, 10, 12),
        max_freq_template=12, n_perm=2000, max_epochs=15000,
        do_shuffled_input=True, do_label_null=True, seed0=1000, mode="hybrid",
        full_battery=False, hidden_units_per_task=False):
    """
    hidden_units_per_task: if True, use PAPER_HIDDEN_UNITS for each task (the
    minimal-sufficient architecture reported in the manuscript) instead of sweeping
    the shared `hidden_units` tuple. Tasks not listed fall back to `hidden_units`.
    """
    if quick:
        n_networks, hidden_units, n_perm, max_epochs, mode = 5, (6, 10), 400, 6000, "batch"

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed0)
    rows = []

    for label, task in make_tasks(quick, full_battery=full_battery):
        task_hidden = (PAPER_HIDDEN_UNITS.get(label, hidden_units)
                       if hidden_units_per_task else hidden_units)
        raw_bases = {
            "dft": dft_phase_bases(task.system),          # lattice DFT: Iran 1..12, West 1..6
        }
        banks = {name: PhaseBank(b) for name, b in raw_bases.items()}
        for bn, bk in banks.items():
            if bk.dropped:
                print(f"  [{label}|{bn}] dropped degenerate phase spaces: {bk.dropped} "
                      f"(constant after centering); kept {bk.freqs}")

        for H in task_hidden:
            for net_i in range(n_networks):
                seed = int(rng.integers(0, 2**31 - 1))
                # --- main, trained on the real task -----------------------
                res, perm = train_one(task, n_hidden=H, seed=seed, max_epochs=max_epochs,
                                      mode=mode, shuffle_pc_to_input=False)
                _emit_rows(rows, label, task, banks, raw_bases, res.W_ih, res, H, seed,
                           n_perm, rng, control="none")

                # --- input-shuffle robustness (weights un-permuted) -------
                if do_shuffled_input:
                    r2, perm2 = train_one(task, n_hidden=H, seed=seed,
                                          max_epochs=max_epochs, mode=mode,
                                          shuffle_pc_to_input=True)
                    W2 = r2.W_ih[:, np.argsort(perm2)]
                    _emit_rows(rows, label, task, banks, raw_bases, W2, r2, H, seed,
                               n_perm, rng, control="input_shuffled")

                # --- LABEL-permutation null (the strong control) ----------
                # Train on a random relabelling: if Fourier fits are just as strong,
                # the structure is an artefact of analysis flexibility, not the task.
                if do_label_null:
                    yperm = rng.permutation(task.y)
                    null_task = type(task)(name=task.name + "_LBLNULL", system=task.system,
                                           X=task.X, y=yperm, class_names=task.class_names,
                                           stim_pcs=task.stim_pcs)
                    r3, _ = train_one(null_task, n_hidden=H, seed=seed,
                                      max_epochs=max_epochs, mode=mode)
                    _emit_rows(rows, label, null_task, banks, raw_bases, r3.W_ih, r3, H,
                               seed, n_perm, rng, control="label_shuffled")
        print(f"[done] {label}: {task.summary()}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "unit_fits.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {len(df)} unit-rows -> {csv_path}")
    _summarize(df, out_dir)
    _figure(df, out_dir)
    return df


def _summarize(df: pd.DataFrame, out_dir: str):
    real = df[df.control == "none"]
    print("\n=== Frequency range used (best_freq), 2-D measure, permutation-significant units ===")
    sig = real[real.null_p < 0.05]
    for (system, basis), g in sig.groupby(["system", "basis"]):
        freqs = sorted(int(x) for x in g.best_freq.unique())
        denom = len(real[(real.system == system) & (real.basis == basis)])
        print(f"  {system:12s} [{basis:8s}] range {min(freqs)}-{max(freqs)} "
              f"| freqs: {freqs} | mean R={g.best_R.mean():.3f} | sig {len(g)}/{denom}")

    print("\n=== Controls (label-null vs real): mean best_R and fraction permutation-significant ===")
    for (system, ctrl), g in df[df.basis == "template"].groupby(["system", "control"]):
        print(f"  {system:12s} {ctrl:15s}: mean R={g.best_R.mean():.3f} "
              f"frac sig={np.mean(g.null_p < 0.05):.2f}  (n={len(g)})")

    print("\n=== Baseline comparison (real nets) ===")
    for system, g in real[real.basis == "template"].groupby("system"):
        print(f"  {system:12s}: best_R={g.best_R.mean():.3f}  "
              f"pitch_height_r={g.pitch_height_r.mean():.3f}  "
              f"random_ceiling={g.random_r_p95.mean():.3f}")


def _figure(df: pd.DataFrame, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = df[(df.null_p < 0.05) & (df.control == "none")]
    systems = sorted(sig.system.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, basis in zip(axes, ["dft", "template"]):
        for system in systems:
            g = sig[(sig.system == system) & (sig.basis == basis)]
            if len(g) == 0:
                continue
            kmax = int(df[df.basis == basis].best_freq.max())
            bins = np.arange(0.5, kmax + 1.5, 1)
            ax.hist(g.best_freq, bins=bins, alpha=0.55, label=system,
                    edgecolor="black", linewidth=0.4)
        ax.set_title(f"Best-fit phase-space frequency ({basis})")
        ax.set_xlabel("frequency k")
        ax.set_ylabel("# significant hidden units")
        ax.legend()
    fig.suptitle("Frequency range of recruited Fourier phase spaces (permutation-significant units)")
    path = os.path.join(out_dir, "frequency_range.png")
    fig.savefig(path, dpi=140)
    print(f"Wrote figure -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small fast demo run")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-shuffle-control", action="store_true")
    ap.add_argument("--no-label-null", action="store_true")
    ap.add_argument("--full-battery", action="store_true",
                    help="run the full Iranian 2-interval + 7-Dang battery")
    args = ap.parse_args()
    run(quick=args.quick, out_dir=args.out,
        do_shuffled_input=not args.no_shuffle_control,
        do_label_null=not args.no_label_null,
        full_battery=args.full_battery)
