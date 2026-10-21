#!/usr/bin/env python3
# analysis/kfold_degeneracy_metrics.py
"""Near no-skill pipelines, collapsed cells, and ranking under balanced accuracy and kappa.

near no-skill: dataset accuracy within 5 points of the no-skill accuracy (mean
majority-class rate of the subjects' trials). collapsed cell: a (subject, seed) whose
predictions fall in one class for at least 95% of the trials.

Writes to output/: degeneracy.csv, metric_scores.csv, metric_sensitivity.csv
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

OUT = Path(__file__).resolve().parent.parent / 'output'
NEAR, COLLAPSE = 0.05, 0.95


def main():
    data = pickle.load(open(OUT / 'cells.pkl', 'rb'))
    cells, y_true, info = data['cells'], data['y_true'], data['datasets']
    rows = []
    for (ds, m), c in cells.items():
        acc, bacc, kap, col = [], [], [], []
        for sid, preds in c['pred'].items():
            yt = y_true[ds][sid]
            for yp in preds:
                acc.append((yp == yt).mean())
                bacc.append(balanced_accuracy_score(yt, yp))
                kap.append(cohen_kappa_score(yt, yp))
                col.append(np.unique(yp, return_counts=True)[1].max() / len(yp) >= COLLAPSE)
        n_seeds = len(next(iter(c['pred'].values())))
        mean_subj = lambda v: np.asarray(v).reshape(-1, n_seeds).mean(1).mean()
        rows.append(dict(dataset=ds, model=m, family=c['family'], acc=mean_subj(acc), bacc=mean_subj(bacc),
                         kappa=mean_subj(kap), collapsed_share=np.mean(col), noskill=info[ds]['noskill']))
    s = pd.DataFrame(rows)
    s['near_noskill'] = s.acc - s.noskill < NEAR
    s.to_csv(OUT / 'metric_scores.csv', index=False)
    s[['dataset', 'model', 'family', 'acc', 'noskill', 'near_noskill', 'collapsed_share']].to_csv(OUT / 'degeneracy.csv', index=False)
    sens = []
    for ds, g in s.groupby('dataset'):
        # equal scores can differ by ~1e-16 after summation; round so they tie
        r = {k: g[k].round(10).rank(ascending=False, method='min') for k in ('acc', 'bacc', 'kappa')}
        for metric in ('bacc', 'kappa'):
            shift = (r[metric] - r['acc']).abs()
            sens.append(dict(dataset=ds, metric=metric, top_same=g.loc[r[metric].idxmin(), 'model'] == g.loc[r['acc'].idxmin(), 'model'],
                             n_shifted=int((shift > 0).sum()), max_shift=int(shift.max())))
    pd.DataFrame(sens).to_csv(OUT / 'metric_sensitivity.csv', index=False)
    print(pd.DataFrame(sens).pivot(index='dataset', columns='metric', values='n_shifted').to_string())
    print(s[s.near_noskill].groupby(['dataset', 'family']).size().unstack(fill_value=0).to_string())


if __name__ == '__main__':
    main()
