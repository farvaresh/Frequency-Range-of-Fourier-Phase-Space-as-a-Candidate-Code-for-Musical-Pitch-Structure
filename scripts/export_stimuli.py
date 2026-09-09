#!/usr/bin/env python3
"""Export every task's stimulus set and target labels to CSV.

Produces one file per task in data/stimuli/, so that the training sets can be inspected
or re-used without running the pipeline. Each row is one stimulus: the binary input
vector (one column per pitch class) plus the target category.

Usage:
    python scripts/export_stimuli.py [output_dir]
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_src = _root / "src"
if (_src / "iranian_anns").is_dir():
    sys.path.insert(0, str(_src))

import pandas as pd

from iranian_anns.pitch_systems import iranian_system, western_system
from iranian_anns.music_theory import build_all_tasks, interval_label
from iranian_anns.tasks import build_interval_task


def export(task, names, path):
    cols = {f"pc_{n}": task.X[:, i].astype(int) for i, n in enumerate(names)}
    df = pd.DataFrame(cols)
    df.insert(0, "stimulus_id", range(1, len(df) + 1))
    df["target_category"] = task.y
    df["target_label"] = [task.class_names[c] for c in task.y]
    df["pitch_classes"] = ["+".join(names[i] for i in pcs) for pcs in task.stim_pcs]
    df.to_csv(path, index=False)
    return len(df), task.n_outputs


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _root / "data" / "stimuli"
    out.mkdir(parents=True, exist_ok=True)

    iran, west = iranian_system(), western_system()
    manifest = []

    for name, task in build_all_tasks(iran):
        n, k = export(task, iran.names, out / f"iranian_{name}.csv")
        manifest.append(dict(file=f"iranian_{name}.csv", system="Iranian-17",
                             task=name, stimuli=n, categories=k))

    for coll, name in [(False, "interval_separate"), (True, "interval_collapsed")]:
        t = build_interval_task(west, collapse_inversions=coll)
        n, k = export(t, west.names, out / f"western_{name}.csv")
        manifest.append(dict(file=f"western_{name}.csv", system="Western-12",
                             task=name, stimuli=n, categories=k))

    pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)
    print(f"Wrote {len(manifest)} stimulus files to {out}")
    for m in manifest:
        print(f"  {m['file']:38s} {m['stimuli']:4d} stimuli, {m['categories']:3d} categories")


if __name__ == "__main__":
    main()
