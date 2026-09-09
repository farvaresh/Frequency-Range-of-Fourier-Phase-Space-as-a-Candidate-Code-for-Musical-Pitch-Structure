#!/usr/bin/env python3
"""
Diagnose a unit_fits.csv before running the analysis.

Usage:
    python check_unit_fits.py path\\to\\unit_fits.csv
"""
import sys
import pandas as pd

REQUIRED = {
    "task":         "task identifier",
    "system":       "Iranian-17 / Western-12",
    "control":      "none / label_shuffled / input_shuffled",
    "network_seed": "network identity (needed for network-level tests)",
    "basis":        "phase-space basis (dft)",
    "best_freq":    "best-fitting phase-space frequency",
    "best_R":       "fit strength (needed for logit-R and Wilcoxon)",
    "null_p":       "permutation-null p-value",
}
OPTIONAL = {
    "concentration":  "spectral concentration (section 1b)",
    "pitch_height_r": "non-Fourier baseline (section 4)",
    "random_r_p95":   "random ceiling baseline (section 4)",
    "accuracy":       "training accuracy",
}

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    df = pd.read_csv(path)
    print(f"File   : {path}")
    print(f"Rows   : {len(df):,}")
    print(f"Columns: {len(df.columns)}\n")

    missing = [c for c in REQUIRED if c not in df.columns]
    print("REQUIRED columns")
    for c, why in REQUIRED.items():
        mark = "ok  " if c in df.columns else "MISS"
        print(f"  [{mark}] {c:<14s} {why}")
    print("\nOPTIONAL columns")
    for c, why in OPTIONAL.items():
        mark = "ok  " if c in df.columns else "--  "
        print(f"  [{mark}] {c:<14s} {why}")

    print()
    if missing:
        print("RESULT: this file cannot be analysed. Missing:", ", ".join(missing))
        print()
        print("This looks like a trimmed export rather than the full pipeline output.")
        print("Use the unit_fits.csv written directly by run_config.py, e.g.")
        print(r"    results_full\unit_fits.csv")
        sys.exit(2)

    print("RESULT: all required columns present -- ready to analyse.")
    for col in ("system", "control", "basis"):
        print(f"  {col}: {sorted(df[col].unique())}")
    print(f"  networks: {df.network_seed.nunique()}")

if __name__ == "__main__":
    main()
