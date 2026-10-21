#!/usr/bin/env python3
# analysis/kfold_topk_vote.py
"""Majority vote of the k best models against the best single model, held out.

Split-half per subject (stratified by class, N_SPLITS random splits): the k best models and the
best single model of the dataset are ranked by their accuracy on half A (mean over subjects and
seeds) and scored on half B. Vote: plurality of the k predictions per trial, ties broken in
favour of the higher-ranked model. Each seed s uses the seed-s predictions of every deep model
(classical pipelines are deterministic); results are averaged over seeds and splits.
The 95% interval resamples subjects (10,000 resamples).

Reads output/cells.pkl (run load.py first). Writes output/topk_vote.csv
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

OUT = Path(__file__).resolve().parent.parent / 'output'
N_SPLITS, N_SEEDS, KS = 20, 10, (3, 5)


def vote(preds, classes):
    # preds: (k, n_trials) ordered best first; plurality, ties to the earliest (best) model
    counts = np.stack([(preds == c).sum(0) for c in classes])           # (n_classes, n_trials)
    first_rank = np.stack([np.where((preds == c).any(0), (preds == c).argmax(0), len(preds))
                           for c in classes])
    score = counts * (len(preds) + 1) - first_rank
    return np.asarray(classes)[score.argmax(0)]


def main():
    data = pickle.load(open(OUT / 'cells.pkl', 'rb'))
    cells, y_true = data['cells'], data['y_true']
    rows = []
    for ds in data['panel']:
        models = sorted(m for (d, m) in cells if d == ds)
        sids = sorted(y_true[ds])
        classes = np.unique(np.concatenate([y_true[ds][s] for s in sids]))
        pred = {s: np.stack([np.resize(cells[(ds, m)]['pred'][s], (N_SEEDS, len(y_true[ds][s])))
                             for m in models]) for s in sids}           # (n_models, n_seeds, n_trials)
        gain = {k: np.zeros((N_SPLITS, len(sids))) for k in KS}
        for split in range(N_SPLITS):
            halves = {}
            for s in sids:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=split)
                halves[s] = next(sss.split(np.zeros(len(y_true[ds][s])), y_true[ds][s]))
            a_acc = np.mean([(pred[s][:, :, halves[s][0]] == y_true[ds][s][halves[s][0]]).mean((1, 2))
                             for s in sids], 0)
            order = np.argsort(-a_acc, kind='stable')
            for j, s in enumerate(sids):
                b = halves[s][1]
                yb = y_true[ds][s][b]
                single = (pred[s][order[0], :, b.tolist()].T == yb).mean()
                for k in KS:
                    fused = np.mean([(vote(pred[s][order[:k], seed][:, b], classes) == yb).mean()
                                     for seed in range(N_SEEDS)])
                    gain[k][split, j] = fused - single
        row = dict(dataset=ds, n_subjects=len(sids))
        rng = np.random.default_rng(0)
        for k in KS:
            d = gain[k].mean(0)
            boot = d[rng.integers(0, len(d), (10_000, len(d)))].mean(1)
            row.update({f'k{k}_gain': d.mean(), f'k{k}_lo': np.percentile(boot, 2.5),
                        f'k{k}_hi': np.percentile(boot, 97.5)})
        rows.append(row)
        print(ds, {k: round(100 * row[f'k{k}_gain'], 2) for k in KS}, flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / 'topk_vote.csv', index=False)
    show = r.copy()
    num = [c for c in show.columns if c.startswith('k')]
    show[num] = (100 * show[num]).round(2)
    print(show.to_string(index=False))
    for k in KS:
        print(f'k={k}: mean held-out gain {100 * r[f"k{k}_gain"].mean():.2f} pp, '
              f'interval above 0 in {(r[f"k{k}_lo"] > 0).sum()}/20, below 0 in {(r[f"k{k}_hi"] < 0).sum()}/20')


if __name__ == '__main__':
    main()
