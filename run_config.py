r"""
run_config.py -- the single file you edit and run.

This file sits at the repository root and finds the package automatically (whether or not
you have run `pip install -e .`). From the repository root:

    python run_config.py smoke      # ~2 min   : does everything work?
    python run_config.py medium     # ~20-40 m : does the pattern appear?
    python run_config.py full       # hours    : the numbers reported in the paper
    python run_config.py sweep      # ~1 h     : hidden-layer-size robustness check

Each profile writes to its own folder (results_smoke\, results_medium\, results_full\,
results_sweep\) and then runs the statistical analysis, producing:

    <out>\unit_fits.csv             one row per hidden unit (raw results)
    <out>\report\results_report.md  the narrative report -> read this
    <out>\report\tables\*.csv       numbers for the manuscript
    <out>\report\figures\*.png      figures

WHY PROFILES: the full configuration trains thousands of networks. Run `smoke` first to
confirm your environment works, then `medium` to confirm the scientific pattern is there,
and only then commit to `full`.
"""

import sys
import time
from pathlib import Path

# Make the package importable whether or not it has been pip-installed. If a local
# src/iranian_anns exists next to this file, add it to the path; otherwise rely on the
# installed package.
_here = Path(__file__).resolve().parent
_src = _here / "src"
if (_src / "iranian_anns").is_dir():
    sys.path.insert(0, str(_src))

from iranian_anns.run_experiment import run
from iranian_anns.analyze_results import analyze


# ---------------------------------------------------------------------------
# PROFILES
# ---------------------------------------------------------------------------
# n_networks   : independently initialised networks per task per hidden-size
# hidden_units : hidden-layer sizes to sweep over
# n_perm       : permutation-null resamples per hidden unit
# max_epochs   : training budget per network
# mode         : "batch"  = fast, approximate
#                "hybrid" = batch warm-up + online polish (reaches the hit criterion)
#                "online" = per-pattern updates, faithful to the original paper (slowest)
# full_battery : True  -> 9 Iranian tasks (2 interval + 7 Dang) + 2 Western tasks
#                False -> interval tasks only (2 Iranian + 2 Western) + 1 Dang demo
# ---------------------------------------------------------------------------

PROFILES = {
    # ~2 minutes. Purpose: prove the environment works. Ignore the science here.
    "smoke": dict(
        out_dir="results_smoke",
        n_networks=3, hidden_units=(6, 10), n_perm=300, max_epochs=4000,
        mode="batch", full_battery=False,
        do_shuffled_input=False, do_label_null=True,
    ),

    # ~20-40 minutes. Purpose: confirm the scientific pattern (DFT 1-8 vs 1-6;
    # label-null collapse) before committing to the long run.
    "medium": dict(
        out_dir="results_medium",
        n_networks=8, hidden_units=(7, 12), n_perm=1000, max_epochs=8000,
        mode="hybrid", full_battery=True,
        do_shuffled_input=False, do_label_null=True,
    ),

    # Hours. Purpose: the numbers that go into the manuscript.
    # Uses the paper's per-task hidden-unit counts (7 for intervals; 11/12/8/12/5/12/4
    # for Dangs 1-7). Set mode="online" for maximal fidelity to the original Rumelhart
    # training (roughly 4x slower again than "hybrid").
    "full": dict(
        out_dir="results_full",
        n_networks=25, hidden_units=(7,), n_perm=2000, max_epochs=15000,
        mode="hybrid", full_battery=True, hidden_units_per_task=True,
        do_shuffled_input=True, do_label_null=True,
    ),

    # Hours. The reviewer control for "did minimal-sufficient sizing bias the result?"
    # Same tasks, but deliberately over-parameterised hidden layers.
    "sweep": dict(
        out_dir="results_sweep",
        n_networks=10, hidden_units=(4, 8, 12, 16, 20), n_perm=1000, max_epochs=12000,
        mode="hybrid", full_battery=True,
        do_shuffled_input=False, do_label_null=False,
    ),
}


def estimate(cfg):
    """Rough training count and wall-clock estimate, so nothing runs for hours by surprise."""
    n_tasks = 11 if cfg["full_battery"] else 5
    n_conditions = 1 + int(cfg["do_shuffled_input"]) + int(cfg["do_label_null"])
    n_sizes = 1 if cfg.get("hidden_units_per_task") else len(cfg["hidden_units"])
    n_trainings = n_tasks * n_sizes * cfg["n_networks"] * n_conditions
    # measured on a reference machine, largest task, per training:
    sec_per = {"batch": 3.0, "hybrid": 13.0, "online": 55.0}[cfg["mode"]]
    # most tasks are smaller than the largest, so scale down a little
    secs = n_trainings * sec_per * 0.6
    return n_trainings, secs


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if name not in PROFILES:
        print(f"Unknown profile '{name}'. Choose one of: {', '.join(PROFILES)}")
        sys.exit(1)

    cfg = dict(PROFILES[name])
    out_dir = cfg["out_dir"]
    n_trainings, secs = estimate(cfg)

    print(f"\n=== Profile: {name} ===")
    print(f"  tasks          : {'full battery (11)' if cfg['full_battery'] else 'intervals + demo (5)'}")
    sizes = "per-task (paper)" if cfg.get("hidden_units_per_task") else str(cfg["hidden_units"])
    print(f"  networks/task  : {cfg['n_networks']}  x hidden sizes {sizes}")
    print(f"  trainer mode   : {cfg['mode']}   epochs {cfg['max_epochs']}")
    print(f"  controls       : label_null={cfg['do_label_null']}  input_shuffled={cfg['do_shuffled_input']}")
    print(f"  ~{n_trainings} network trainings, rough estimate {secs/60:.0f} min "
          f"({secs/3600:.1f} h)")
    print(f"  output folder  : {out_dir}\\\n")

    if secs > 900:
        ans = input("This will take a while. Type 'yes' to continue: ").strip().lower()
        if ans != "yes":
            print("Aborted."); sys.exit(0)

    t0 = time.time()
    run(quick=False, **cfg)
    print(f"\nTraining + fitting done in {(time.time()-t0)/60:.1f} min.")

    print("\nRunning statistical analysis...")
    analyze(f"{out_dir}/unit_fits.csv", f"{out_dir}/report")

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min.")
    print(f"READ THIS FIRST:  {out_dir}\\report\\results_report.md")


if __name__ == "__main__":
    main()
