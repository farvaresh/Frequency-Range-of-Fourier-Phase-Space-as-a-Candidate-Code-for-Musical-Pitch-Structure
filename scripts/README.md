# Scripts

| script | purpose |
|---|---|
| `check_unit_fits.py` | Checks that a `unit_fits.csv` has every column the analysis needs, and says which are missing. Run this first if `analyze_results` raises an error. |
| `export_stimuli.py` | Exports every task's stimulus set and labels to `data/stimuli/`. |
| `probe_project.py` | Scans one or more project folders and writes a small `probe_report.txt` inventorying spreadsheets and code files — used during development to locate the original music-theory definition tables. Not needed to run the pipeline. |

Usage:

```bash
python scripts/check_unit_fits.py results_full/unit_fits.csv
python scripts/export_stimuli.py
python scripts/probe_project.py "path/to/folder1" "path/to/folder2"
```
