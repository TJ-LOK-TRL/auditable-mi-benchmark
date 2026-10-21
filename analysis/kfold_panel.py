#!/usr/bin/env python3
# analysis/kfold_panel.py
"""Dataset-panel sensitivity: which conclusions a smaller panel would have reached.

The full panel (all datasets in output/accuracy.csv) is the reference. For every
panel size k, sub-panels are enumerated (k <= 3) or sampled, and on each one we recompute
the mean-rank leaderboard of all pipelines and the best deep - best classical contrast
(mean over the sub-panel's datasets). Point estimates only: within-dataset sampling noise
is covered by kfold_bootstrap.py.

Writes to output/: panel_k.csv, panel_named.csv, panel_leave_one_out.csv
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

OUT = Path(__file__).resolve().parent.parent / 'output'
N_SAMPLES = 5000
RNG = np.random.default_rng(20261014)
# panels used in the literature; BNCI2014004 is BCI Competition IV 2b, Schirrmeister2017 is HGD
NAMED = {
    'BCI IV 2a': ['BCICompIV2a'],
    'BCI IV 2a + 2b': ['BCICompIV2a', 'BNCI2014004'],
    'BCI IV 2a + 2b + HGD': ['BCICompIV2a', 'BNCI2014004', 'Schirrmeister2017'],
    'BCI IV 2a + PhysionetMI': ['BCICompIV2a', 'PhysionetMI'],
    'Cho2017 + Lee2019MI + PhysionetMI': ['Cho2017', 'Lee2019MI', 'PhysionetMI'],
    'MOABB suggested MI panel': ['BCICompIV2a', 'BNCI2014004', 'Schirrmeister2017', 'PhysionetMI'],
}


def main():
    acc = pd.read_csv(OUT / 'accuracy.csv')
    A = acc.pivot(index='dataset', columns='model', values='acc')
    fam = acc.drop_duplicates('model').set_index('model').family
    deep, clas = fam.index[fam == 'deep'], fam.index[fam == 'classical']
    # accuracies equal in exact arithmetic can differ by ~1e-16 after summation; round so they tie
    R = A.round(10).rank(axis=1, ascending=False)
    d_best = A[deep].max(1) - A[clas].max(1)
    full_rank = R.mean().sort_values()
    full_top, full_top_clas = full_rank.index[0], R[clas].mean().idxmin()
    full_sign = np.sign(d_best.mean())
    names = A.index.values

    def tied_min(series):
        """All pipelines sharing the lowest mean rank, so a tie at the top is reported as a tie."""
        return sorted(series.index[np.isclose(series, series.min())])

    def summary(panel):
        mr = R.loc[panel].mean()
        top, top_clas = tied_min(mr), tied_min(R.loc[panel, clas].mean())
        return dict(d_best=d_best[panel].mean(), top=' / '.join(top), top_family=fam[top[0]],
                    top_has_full=full_top in top, top_classical=' / '.join(top_clas),
                    top_classical_has_full=full_top_clas in top_clas,
                    top_classical_rank=mr.rank()[top_clas[0]],
                    tau=kendalltau(mr[full_rank.index], full_rank)[0])

    rows = []
    for k in range(1, len(names)):
        panels = (list(combinations(names, k)) if k <= 3 else
                  [RNG.choice(names, k, replace=False) for _ in range(N_SAMPLES)])
        s = pd.DataFrame([summary(list(p)) for p in panels])
        rows.append(dict(k=k, n_panels=len(panels),
                         p_verdict_flipped=np.mean(np.sign(s.d_best) != full_sign),
                         p_top_same=np.mean(s.top_has_full), p_top_classical=np.mean(s.top_family == 'classical'),
                         p_top_classical_same=np.mean(s.top_classical_has_full),
                         tau_median=s.tau.median(), d_best_q05=s.d_best.quantile(.05),
                         d_best_median=s.d_best.median(), d_best_q95=s.d_best.quantile(.95)))
    pd.DataFrame(rows).to_csv(OUT / 'panel_k.csv', index=False)

    named = [dict(panel='All datasets', n=len(names), **summary(list(names)))]
    named += [dict(panel=k, n=len(v), **summary(v)) for k, v in NAMED.items()]
    named.append(dict(panel='All but BCI IV 2a', n=len(names) - 1,
                      **summary([d for d in names if d != 'BCICompIV2a'])))
    pd.DataFrame(named).to_csv(OUT / 'panel_named.csv', index=False)

    loo = [dict(left_out=d, **summary([x for x in names if x != d])) for d in names]
    pd.DataFrame(loo).assign(d_best_dataset=d_best[names].values).to_csv(OUT / 'panel_leave_one_out.csv', index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(pd.DataFrame(named).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
