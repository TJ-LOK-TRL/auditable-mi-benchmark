#!/usr/bin/env python3
# analysis/kfold_extra_metrics.py
"""Accuracy, balanced accuracy, macro F1, weighted F1 and Cohen's kappa next to chance levels.

Every metric is computed per (subject, seed) on the predictions of all 5 test folds pooled,
averaged over seeds, then over subjects (same aggregation as accuracy.csv). Chance levels:
chance = 1 / n_classes; noskill = mean majority-class rate of the subjects' trials (what a
model that always predicts the most frequent class would score).

Reads output/cells.pkl (run load.py first). Writes to output/:
  extra_metrics.csv          one row per (dataset, model)
  extra_metrics_best.csv     best classical and best deep pipeline per dataset (by accuracy)
  extra_metrics_overall.csv  per model, averaged over the datasets
  extra_metrics.md           the last two as markdown tables
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

OUT = Path(__file__).resolve().parent.parent / 'output'
METRICS = {
    'acc': lambda yt, yp: (yp == yt).mean(),
    'bacc': balanced_accuracy_score,
    'f1_macro': lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
    'f1_weighted': lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0),
    'kappa': cohen_kappa_score,
}
PCT = ['acc', 'bacc', 'f1_macro', 'f1_weighted', 'chance', 'noskill']


def scores(cells, y_true, info):
    rows = []
    for (ds, m), c in cells.items():
        per_subj = {k: [] for k in METRICS}
        for sid, preds in c['pred'].items():
            yt = y_true[ds][sid]
            for k, f in METRICS.items():
                per_subj[k].append(np.mean([f(yt, yp) for yp in preds]))
        rows.append(dict(dataset=ds, model=m, family=c['family'],
                         n_classes=info[ds]['n_classes'], n_subjects=info[ds]['n_subjects'],
                         chance=1 / info[ds]['n_classes'], noskill=info[ds]['noskill'],
                         **{k: float(np.mean(v)) for k, v in per_subj.items()}))
    return pd.DataFrame(rows).sort_values(['dataset', 'family', 'model'])


def best_per_dataset(s):
    rows = []
    for ds, g in s.groupby('dataset'):
        r = g.iloc[0][['dataset', 'n_classes', 'n_subjects', 'chance', 'noskill']].to_dict()
        for fam in ('classical', 'deep'):
            b = g[g.family == fam].sort_values('acc', ascending=False).iloc[0]
            r[f'{fam}_model'] = b.model
            r.update({f'{fam}_{k}': b[k] for k in METRICS})
        rows.append(r)
    return pd.DataFrame(rows)


def to_md(df, pct_cols):
    df = df.copy()
    for c in df.columns:
        if c in pct_cols:
            df[c] = (100 * df[c]).map('{:.1f}'.format)
        elif df[c].dtype.kind == 'f':
            df[c] = df[c].map('{:.3f}'.format)
    return df.to_markdown(index=False)


def main():
    data = pickle.load(open(OUT / 'cells.pkl', 'rb'))
    s = scores(data['cells'], data['y_true'], data['datasets'])
    s['acc_minus_chance'] = s.acc - s.chance
    s.to_csv(OUT / 'extra_metrics.csv', index=False)

    best = best_per_dataset(s)
    best.to_csv(OUT / 'extra_metrics_best.csv', index=False)

    overall = (s.groupby(['model', 'family'])[list(METRICS) + ['acc_minus_chance']].mean()
               .reset_index().sort_values('acc', ascending=False))
    overall.to_csv(OUT / 'extra_metrics_overall.csv', index=False)

    best_pct = PCT + [f'{f}_{k}' for f in ('classical', 'deep') for k in METRICS if k != 'kappa']
    with open(OUT / 'extra_metrics.md', 'w') as fh:
        fh.write(f'# Extra metrics, {s.dataset.nunique()} datasets, {s.model.nunique()} pipelines\n\n'
                 'Percentages except kappa. chance = 1/n_classes; noskill = always predicting the '
                 'majority class. Best pipeline per family chosen by accuracy.\n\n'
                 '## Best pipeline per dataset\n\n' + to_md(best, best_pct) + '\n\n'
                 '## Per pipeline, mean over datasets\n\n'
                 + to_md(overall, PCT + ['acc_minus_chance']) + '\n')
    print(to_md(best, best_pct))
    print()
    print(to_md(overall, PCT + ['acc_minus_chance']))


if __name__ == '__main__':
    main()
