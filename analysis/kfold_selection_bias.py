#!/usr/bin/env python3
# analysis/kfold_selection_bias.py
"""Selection bias of the best-of-family contrast, estimated by split-half selection.

For every dataset and repetition, each subject's test trials are split into two
class-stratified halves. The best deep and the best classical pipeline are chosen on
half A and their difference is measured on half B (held out), and compared with the
difference measured on half B when the best pipelines are also chosen on half B
(in-sample, as in the main analysis). Writes output/selection_bias.csv.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / 'output'
REPS = 50
RNG = np.random.default_rng(20261014)


def main():
    data = pickle.load(open(OUT / 'cells.pkl', 'rb'))
    cells, y_true = data['cells'], data['y_true']
    rows = []
    for ds in data['panel']:
        models = sorted(m for (d, m) in cells if d == ds)
        deep = np.array([cells[(ds, m)]['family'] == 'deep' for m in models])
        sids = sorted(y_true[ds])
        C = {s: np.stack([cells[(ds, m)]['correct'][s].mean(0) for m in models]) for s in sids}
        for r in range(REPS):
            accA, accB = [], []
            for s in sids:
                y = y_true[ds][s]
                a = np.zeros(len(y), bool)
                for c in np.unique(y):
                    idx = RNG.permutation(np.flatnonzero(y == c))
                    a[idx[: len(idx) // 2]] = True
                accA.append(C[s][:, a].mean(1))
                accB.append(C[s][:, ~a].mean(1))
            accA, accB = np.mean(accA, 0), np.mean(accB, 0)
            held = accB[deep][accA[deep].argmax()] - accB[~deep][accA[~deep].argmax()]
            insample = accB[deep].max() - accB[~deep].max()
            rows.append(dict(dataset=ds, rep=r, held_out=held, in_sample=insample,
                             best_deep_same=accA[deep].argmax() == accB[deep].argmax(),
                             best_clas_same=accA[~deep].argmax() == accB[~deep].argmax()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'selection_bias.csv', index=False)
    g = df.groupby('dataset')[['in_sample', 'held_out']].mean()
    g['bias'] = g.in_sample - g.held_out
    print((100 * g).round(2).to_string())
    print('mean over datasets: in-sample %.2f pp, held-out %.2f pp, bias %.2f pp'
          % tuple(100 * g[['in_sample', 'held_out', 'bias']].mean()))


if __name__ == '__main__':
    main()
