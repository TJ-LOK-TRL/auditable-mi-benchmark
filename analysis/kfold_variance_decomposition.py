#!/usr/bin/env python3
# analysis/kfold_variance_decomposition.py
"""Share of accuracy variance due to model, subject, their interaction and seed, per dataset.

Within each dataset: balanced two-way ANOVA, accuracy ~ model + subject + model x subject,
the ten seeds as replicates (classical pipelines are deterministic and enter as ten identical
replicates, so the residual reflects the deep models only). Each term is reported as its share
of the total sum of squares (eta squared).

noise_ceiling: interaction share that binomial test-trial noise alone would produce if every
(model, subject) accuracy were an independent draw around its observed value, summed over its
n_test trials (an upper bound, since all models share the same test trials).
model_share_skill: model share recomputed without the models within 5 points of the
no-skill accuracy (degeneracy.csv, from kfold_degeneracy_metrics.py).

Across datasets: two-way decomposition of the dataset-level kappa (chance-corrected, so
datasets with different class counts are comparable), kappa ~ dataset + model, one value per
(dataset, model), the residual being the dataset x model interaction.

Reads output/cells.pkl, degeneracy.csv, extra_metrics.csv. Writes to output/:
variance_decomposition.csv, variance_across_datasets.csv
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / 'output'
N_SEEDS = 10


def anova_shares(acc):
    # acc: (n_models, n_subjects, n_seeds)
    grand = acc.mean()
    cell = acc.mean(2)
    m_eff = cell.mean(1) - grand
    s_eff = cell.mean(0) - grand
    a, b, r = acc.shape
    ss = dict(model=b * r * (m_eff ** 2).sum(), subject=a * r * (s_eff ** 2).sum(),
              interaction=r * ((cell - grand - m_eff[:, None] - s_eff[None, :]) ** 2).sum(),
              residual=((acc - cell[..., None]) ** 2).sum())
    total = ((acc - grand) ** 2).sum()
    return {k: v / total for k, v in ss.items()}, total


def dataset_matrix(cells, ds, models, sids):
    acc = np.empty((len(models), len(sids), N_SEEDS))
    n_trials = np.empty(len(sids))
    for i, m in enumerate(models):
        for j, sid in enumerate(sids):
            per_seed = cells[(ds, m)]['correct'][sid].mean(1)
            acc[i, j] = np.resize(per_seed, N_SEEDS)
            n_trials[j] = cells[(ds, m)]['correct'][sid].shape[1]
    return acc, n_trials


def main():
    data = pickle.load(open(OUT / 'cells.pkl', 'rb'))
    cells, y_true = data['cells'], data['y_true']
    deg = pd.read_csv(OUT / 'degeneracy.csv').set_index(['dataset', 'model']).near_noskill
    rows = []
    for ds in data['panel']:
        models = sorted(m for (d, m) in cells if d == ds)
        sids = sorted(y_true[ds])
        acc, n_trials = dataset_matrix(cells, ds, models, sids)
        shares, total = anova_shares(acc)
        a, b, r = acc.shape
        p = acc.mean(2)
        noise_ss = r * (1 - 1 / a) * (1 - 1 / b) * (p * (1 - p) / n_trials[None, :]).sum()
        keep = [i for i, m in enumerate(models) if not deg[(ds, m)]]
        skill = anova_shares(acc[keep])[0]['model'] if len(keep) > 2 else np.nan
        rows.append(dict(dataset=ds, n_subjects=len(sids), n_models=len(models),
                         **{f'{k}_share': v for k, v in shares.items()},
                         noise_ceiling=noise_ss / total, model_share_skill=skill,
                         n_near_noskill=len(models) - len(keep)))
    r = pd.DataFrame(rows).sort_values('model_share', ascending=False)
    r['dominant'] = r[['model_share', 'subject_share', 'interaction_share']].idxmax(1).str.replace('_share', '')
    r['interaction_real'] = r.interaction_share > r.noise_ceiling
    r.to_csv(OUT / 'variance_decomposition.csv', index=False)

    e = pd.read_csv(OUT / 'extra_metrics.csv')
    k = e.pivot(index='dataset', columns='model', values='kappa').values
    g = k.mean()
    ss = dict(dataset=k.shape[1] * ((k.mean(1) - g) ** 2).sum(), model=k.shape[0] * ((k.mean(0) - g) ** 2).sum())
    tot = ((k - g) ** 2).sum()
    across = pd.DataFrame([{**{f'{n}_share': v / tot for n, v in ss.items()},
                            'interaction_share': 1 - sum(ss.values()) / tot}])
    across.to_csv(OUT / 'variance_across_datasets.csv', index=False)

    show = r.copy()
    pct = [c for c in show.columns if c.endswith('share') or c in ('noise_ceiling', 'model_share_skill')]
    show[pct] = (100 * show[pct]).round(1)
    print(show.to_string(index=False))
    print('\nmedian shares (%):', (100 * r[['model_share', 'subject_share', 'interaction_share',
                                            'residual_share']].median()).round(1).to_dict())
    print('dominant term:', r.dominant.value_counts().to_dict())
    print('\nacross datasets (kappa, %):', (100 * across.iloc[0]).round(1).to_dict())


if __name__ == '__main__':
    main()
