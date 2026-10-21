#!/usr/bin/env python3
# analysis/kfold_per_subject_selection.py
"""Does choosing a different model for each subject beat always using the best single model?

In-sample gap: mean over subjects of the subject's best model accuracy, minus the accuracy of
the best single model of the dataset, both chosen and scored on all trials.
Held-out gap (split-half): each subject's trials are split into halves A and B, stratified by
class; the per-subject best model and the best single model are both chosen on A and scored
on B. Repeated over N_SPLITS random splits. Deep models are scored by their per-trial
correctness averaged over the ten seeds. No retraining: every trial was a test trial of the
5-fold CV, so A and B only change which test trials pick and which score. The 95% interval
of the held-out gap resamples subjects (10,000 resamples).

Reads output/cells.pkl (run load.py first). Writes output/per_subject_selection.csv
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

OUT = Path(__file__).resolve().parent.parent / 'output'
N_SPLITS = 20


def dataset_gaps(models, corr, y_true):
    # corr[sid]: (n_models, n_trials) correctness averaged over seeds
    sids = sorted(corr)
    acc = np.array([corr[s].mean(1) for s in sids])          # (n_subjects, n_models)
    in_sample = acc.max(1).mean() - acc.mean(0).max()
    held, per_subj = [], []
    for split in range(N_SPLITS):
        a_acc, b_acc = [], []
        for s in sids:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=split)
            a, b = next(sss.split(np.zeros(len(y_true[s])), y_true[s]))
            a_acc.append(corr[s][:, a].mean(1))
            b_acc.append(corr[s][:, b].mean(1))
        a_acc, b_acc = np.array(a_acc), np.array(b_acc)
        diff = b_acc[np.arange(len(sids)), a_acc.argmax(1)] - b_acc[:, a_acc.mean(0).argmax()]
        per_subj.append(diff)
        held.append(diff.mean())
    # 95% interval of the held-out gap over resampled subjects (splits averaged per subject)
    d = np.mean(per_subj, 0)
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), (10_000, len(d)))].mean(1)
    best = models[int(acc.mean(0).argmax())]
    return dict(best_single=best, best_single_acc=acc.mean(0).max(), in_sample_gap=in_sample,
                held_out_gap=float(np.mean(held)), held_out_sd=float(np.std(held)),
                splits_positive=int(np.sum(np.array(held) > 0)),
                ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)))


def main():
    data = pickle.load(open(OUT / 'cells.pkl', 'rb'))
    cells, y_true, info = data['cells'], data['y_true'], data['datasets']
    rows = []
    for ds in data['panel']:
        models = sorted(m for (d, m) in cells if d == ds)
        corr = {sid: np.stack([cells[(ds, m)]['correct'][sid].mean(0) for m in models])
                for sid in y_true[ds]}
        rows.append(dict(dataset=ds, n_subjects=info[ds]['n_subjects'],
                         trials_per_subject=info[ds]['trials_per_subject_mean'],
                         **dataset_gaps(models, corr, y_true[ds])))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / 'per_subject_selection.csv', index=False)
    show = r.copy()
    for c in ('best_single_acc', 'in_sample_gap', 'held_out_gap', 'held_out_sd', 'ci_lo', 'ci_hi'):
        show[c] = (100 * show[c]).round(2)
    print(show.to_string(index=False))
    print(f'\nmean in-sample gap {100 * r.in_sample_gap.mean():.2f} pp, '
          f'mean held-out gap {100 * r.held_out_gap.mean():.2f} pp, '
          f'held-out > 0 in {(r.held_out_gap > 0).sum()}/{len(r)} datasets, '
          f'interval above 0 in {(r.ci_lo > 0).sum()}, below 0 in {(r.ci_hi < 0).sum()}')


if __name__ == '__main__':
    main()
