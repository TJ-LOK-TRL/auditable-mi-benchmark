#!/usr/bin/env python3
# analysis/kfold_bootstrap.py
"""Hierarchical paired bootstrap on the final 5-fold predictions.

Per dataset: resample subjects with replacement, then each drawn subject's test trials
with replacement (B replicates). Draws are shared by all models, so model differences
are paired. Correctness is first averaged over the training seeds.

Writes to output/:
  model_ci.csv         accuracy and rank with 95% intervals, plus the SD over seeds
  family_contrast.csv  best deep - best classical and mean deep - mean classical, per dataset
                       and averaged over datasets, with and without IFNet
  pairwise_deep.csv    every pair of deep models: Welch test over the 10 seed accuracies
                       vs the hierarchical and the trial-only (subjects fixed) bootstrap
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / 'output'
B = 1000
RNG = np.random.default_rng(20261014)
DOMINANT = 'IFNet'
SECOND = 'EEGEncoder'


def pct(a, axis=0):
    return np.percentile(a, [2.5, 97.5], axis=axis)


def dataset_replicates(cells, ds, models):
    """Return (hier B x M, trial-only B x M, point M, per-seed 10 x M) dataset accuracies."""
    sids = sorted(cells[(ds, models[0])]['correct'])
    C = {s: np.stack([cells[(ds, m)]['correct'][s].mean(0) for m in models]) for s in sids}   # M x T
    per_seed = []
    for m in models:
        c = cells[(ds, m)]['correct']
        k = c[sids[0]].shape[0]
        per_seed.append(np.stack([c[s].mean(1) for s in sids]).mean(0) if k > 1
                        else np.repeat(np.mean([c[s].mean() for s in sids]), 10))
    per_seed = np.stack(per_seed, 1)                                                           # 10 x M
    tb = np.stack([C[s][:, RNG.integers(0, C[s].shape[1], (B, C[s].shape[1]))].mean(2).T for s in sids], 1)  # B x S x M
    sidx = RNG.integers(0, len(sids), (B, len(sids)))
    hier = np.take_along_axis(tb, sidx[:, :, None], 1).mean(1)
    trial_only = tb.mean(1)
    point = np.stack([C[s].mean(1) for s in sids]).mean(0)
    return hier, trial_only, point, per_seed


def main():
    data = pickle.load(open(OUT / 'cells.pkl', 'rb'))
    cells, panel = data['cells'], data['panel']
    mrows, frows, prows = [], [], []
    fam_reps = {}
    for ds in panel:
        models = sorted(m for (d, m) in cells if d == ds)
        fam = np.array([cells[(ds, m)]['family'] for m in models])
        hier, trial_only, point, per_seed = dataset_replicates(cells, ds, models)
        # accuracies equal in exact arithmetic can differ by ~1e-16 after summation; round so they tie
        ranks = stats.rankdata(-np.round(hier, 10), axis=1)
        prank = stats.rankdata(-np.round(point, 10))
        lo, hi = pct(hier)
        rlo, rhi = pct(ranks)
        for j, m in enumerate(models):
            mrows.append(dict(dataset=ds, model=m, family=fam[j], acc=point[j], ci_lo=lo[j], ci_hi=hi[j],
                              rank=prank[j], rank_lo=rlo[j], rank_hi=rhi[j],
                              boot_sd=hier[:, j].std(), trial_only_sd=trial_only[:, j].std(),
                              seed_sd=per_seed[:, j].std(ddof=1)))
        deep, clas = fam == 'deep', fam == 'classical'
        no_dom = deep & (np.array(models) != DOMINANT)
        no_two = no_dom & (np.array(models) != SECOND)
        for name, dmask in (('all deep', deep), (f'deep without {DOMINANT}', no_dom),
                            (f'deep without {DOMINANT} and {SECOND}', no_two)):
            for kind, f in (('best', np.max), ('mean', np.mean)):
                rep = f(hier[:, dmask], 1) - f(hier[:, clas], 1)
                pt = f(point[dmask]) - f(point[clas])
                fam_reps.setdefault((name, kind), []).append(rep)
                l, h = pct(rep)
                frows.append(dict(dataset=ds, deep_set=name, contrast=kind, point=pt, ci_lo=l, ci_hi=h,
                                  excludes_zero=(l > 0) | (h < 0)))
        idx = np.flatnonzero(deep)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                d_h, d_t = hier[:, i] - hier[:, j], trial_only[:, i] - trial_only[:, j]
                (hl, hh), (tl, th) = pct(d_h), pct(d_t)
                prows.append(dict(dataset=ds, a=models[i], b=models[j], diff=point[i] - point[j],
                                  seed_p=stats.ttest_ind(per_seed[:, i], per_seed[:, j], equal_var=False).pvalue,
                                  hier_lo=hl, hier_hi=hh, trial_lo=tl, trial_hi=th))
        print(ds, 'done', flush=True)
    # panel-level contrast: mean over datasets, each dataset resampled independently
    for (name, kind), reps in fam_reps.items():
        pooled = np.mean(reps, 0)
        sub = [r for r in frows if r['deep_set'] == name and r['contrast'] == kind]
        l, h = pct(pooled)
        frows.append(dict(dataset='Mean over datasets', deep_set=name, contrast=kind,
                          point=np.mean([r['point'] for r in sub]), ci_lo=l, ci_hi=h, excludes_zero=(l > 0) | (h < 0)))
    pd.DataFrame(mrows).to_csv(OUT / 'model_ci.csv', index=False)
    pd.DataFrame(frows).to_csv(OUT / 'family_contrast.csv', index=False)
    p = pd.DataFrame(prows)
    p['seed_sig'] = p.seed_p < 0.05
    p['hier_sig'] = (p.hier_lo > 0) | (p.hier_hi < 0)
    p['trial_sig'] = (p.trial_lo > 0) | (p.trial_hi < 0)
    p.to_csv(OUT / 'pairwise_deep.csv', index=False)
    print(f"pairs {len(p)}: seed-significant {p.seed_sig.sum()}, of which not hierarchical-significant "
          f"{(p.seed_sig & ~p.hier_sig).sum()}, not trial-only-significant {(p.seed_sig & ~p.trial_sig).sum()}")


if __name__ == '__main__':
    main()
