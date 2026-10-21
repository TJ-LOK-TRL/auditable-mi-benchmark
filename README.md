# auditable-mi-benchmark

Per-trial predictions and statistical analysis for the paper
**A Statistically Auditable Benchmark of Motor Imagery EEG Decoders**.

29 pipelines (16 classical, 13 deep) on 20 MOABB motor imagery datasets, with
intra-subject 5-fold cross-validation. Every trial is tested exactly once,
and deep models have 10 training seeds. Starting from the predictions, this repository
recomputes every statistic and result table of the paper on a laptop, without a GPU and
without downloading EEG data.

## Quick start

```bash
pip install -r requirements.txt
python analysis/run_all.py
```

The statistics are written to `output/` as CSV files and the LaTeX tables to `tables/`.
The full run takes about 10 minutes.

## Contents

| Path | What it is |
|---|---|
| `data/predictions.npz` | Predicted class of every test trial, for every pipeline, subject and seed |
| `data/models.csv` | Pipelines per dataset, their family (classical or deep) and number of seeds |
| `data/datasets.json` | Per-dataset information (subjects, classes, channels, sampling rate, no-skill accuracy) |
| `data/grid_edge.csv` | Grid-edge check (SVM `C` extended to 1000), from separate runs |
| `analysis/` | The analysis scripts, run in order by `run_all.py` |

## Data format

`predictions.npz` is a NumPy archive with three kinds of keys:

- `<dataset>/classes`: class names; the integer labels below index into this array.
- `<dataset>/<subject>/y_true`: true label of each test trial of that subject.
- `<dataset>/<subject>/<pipeline>`: predicted labels, an int8 array of shape seeds x trials.
  Deep pipelines have 10 seeds (42 to 51) and classical pipelines have one (42).

All pipelines see the same trials in the same order, so the predictions can be compared
trial by trial. For example:

```python
import numpy as np
z = np.load('data/predictions.npz')
y = z['BCICompIV2a/1/y_true']
p = z['BCICompIV2a/1/IFNet']        # 10 x 576
print((p == y).mean(axis=1))        # accuracy of each seed on subject 1
```

## Scripts and paper sections

| Script | Paper |
|---|---|
| `kfold_bootstrap.py` | Section 4.1 (accuracy and rank intervals, deep vs classical) and 4.2 (pairs of deep models) |
| `kfold_selection_bias.py` | Section 4.1, split-half check of the best-of-family contrast |
| `kfold_degeneracy_metrics.py` | Sections 4.3 and 4.4 |
| `kfold_extra_metrics.py` | Balanced accuracy and Cohen's kappa |
| `kfold_panel.py` | Section 4.5 |
| `kfold_variance_decomposition.py` | Section 4.2, standard deviations per source of variation |
| `make_tables.py` | All result tables |

The tables that only describe the setup (datasets, hyperparameter grids, deep training
recipes, encoder settings) are not regenerated here.

## License

Code: MIT. Predictions in `data/`: CC BY 4.0. The EEG datasets keep their own licenses
and are available through [MOABB](https://moabb.neurotechx.com/).

## Citation

If you use these predictions or this analysis, please cite the paper (reference to be added).
