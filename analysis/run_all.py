#!/usr/bin/env python3
# analysis/run_all.py
"""Run every analysis in order: rebuild the predictions, compute all statistics, write the tables."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    'load',                          # data/ -> output/cells.pkl, accuracy.csv, datasets.csv
    'kfold_bootstrap',               # Section 4.1: accuracy and rank intervals, family contrasts; Section 4.2 bootstraps
    'kfold_selection_bias',          # Section 4.1: split-half check of the best-of-family contrast
    'kfold_degeneracy_metrics',      # Sections 4.3 and 4.4: near no-skill, collapse, metric sensitivity
    'kfold_extra_metrics',           # balanced accuracy and Cohen's kappa per pipeline
    'kfold_panel',                   # Section 4.5: dataset-panel sensitivity
    'kfold_per_subject_selection',
    'kfold_topk_vote',
    'kfold_variance_decomposition',  # Section 4.2: standard deviation per source of variation
    'make_tables',                   # tables/*.tex
]

if __name__ == '__main__':
    for s in STEPS:
        t = time.time()
        print(f'== {s}', flush=True)
        subprocess.run([sys.executable, str(HERE / f'{s}.py')], check=True)
        print(f'   {time.time() - t:.0f} s', flush=True)
