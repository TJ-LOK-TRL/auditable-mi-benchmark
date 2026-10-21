#!/usr/bin/env python3
# analysis/load.py
"""Rebuild output/cells.pkl, accuracy.csv and datasets.csv from data/.

For every (dataset, model) cells.pkl stores the per-subject correctness matrix
(seeds x test trials) and the predicted labels; y_true holds the test labels of each
subject, identical for all models and seeds, so every comparison is paired by trial.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)


def main():
    info = json.load(open(DATA / 'datasets.json'))
    panel, ds_info = info['panel'], info['datasets']
    models = pd.read_csv(DATA / 'models.csv')
    z = np.load(DATA / 'predictions.npz')
    subjects = {}
    for k in z.files:
        ds, s, m = (k.split('/', 2) + [None])[:3]
        if m == 'y_true':
            subjects.setdefault(ds, []).append(int(s))

    cells, y_true, rows = {}, {}, []
    for ds in panel:
        classes = z[f'{ds}/classes']
        sids = sorted(subjects[ds])
        y_true[ds] = {s: classes[z[f'{ds}/{s}/y_true']] for s in sids}
        for r in models[models.dataset == ds].itertuples():
            corr, preds = {}, {}
            for s in sids:
                preds[s] = classes[z[f'{ds}/{s}/{r.model}']]
                assert preds[s].shape == (r.n_seeds, len(y_true[ds][s])), (ds, r.model, s)
                corr[s] = (preds[s] == y_true[ds][s][None]).astype(np.float64)
            cells[(ds, r.model)] = {'family': r.family, 'correct': corr, 'pred': preds}
            rows.append({'dataset': ds, 'model': r.model, 'family': r.family,
                         'acc': np.mean([corr[s].mean() for s in sids])})
    acc = pd.DataFrame(rows)
    acc.to_csv(OUT / 'accuracy.csv', index=False)
    pd.DataFrame(ds_info).T.rename_axis('dataset').to_csv(OUT / 'datasets.csv')
    pickle.dump({'cells': cells, 'y_true': y_true, 'datasets': ds_info, 'panel': panel},
                open(OUT / 'cells.pkl', 'wb'))
    (OUT / 'grid_edge.csv').write_bytes((DATA / 'grid_edge.csv').read_bytes())
    print(f'{len(panel)} datasets, {acc.model.nunique()} models, {len(cells)} (dataset, model) pairs')


if __name__ == '__main__':
    main()
