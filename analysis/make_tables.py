#!/usr/bin/env python3
# analysis/make_tables.py
"""Generate the result tables of the paper in tables/ from output/*.csv (no hand-typed numbers)."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'output'
TAB = HERE.parent / 'tables'
TAB.mkdir(exist_ok=True)
DOMINANT = 'IFNet'
SECOND = 'EEGEncoder'


def minus(s):
    """Typeset a leading minus sign as a math minus instead of a hyphen."""
    return '$-$' + s[1:] if s.startswith('-') else s


def pp(x):
    return minus(f'{100 * x:+.1f}')


def bound(x):
    return minus(f'{100 * x:+.2f}') if abs(100 * x) < 0.05 else pp(x)


def ci(lo, hi, bold=False):
    s = f'[{bound(lo)}, {bound(hi)}]'
    return r'\textbf{' + s + '}' if bold else s


def write(fname, lines):
    (TAB / fname).write_text('\n'.join(lines) + '\n')


def table(fname, spec, header, body):
    write(fname, [rf'\begin{{tabular}}{{{spec}}}', r'\toprule', header + r' \\', r'\midrule', *body,
                  r'\bottomrule', r'\end{tabular}'])


def overall_by_dataset():
    m = pd.read_csv(OUT / 'model_ci.csv')
    f = pd.read_csv(OUT / 'family_contrast.csv')
    noskill = pd.read_csv(OUT / 'datasets.csv').set_index('dataset').noskill
    body = []
    for ds, g in m.groupby('dataset'):
        b = g.loc[g.acc.idxmax()]
        bc = g[g.family == 'classical'].loc[lambda x: x.acc.idxmax()]
        c = f[(f.dataset == ds) & (f.deep_set == 'all deep') & (f.contrast == 'best')].iloc[0]
        body.append(f'{ds} & {b.model} & {100 * b.acc:.1f} [{100 * b.ci_lo:.1f}, {100 * b.ci_hi:.1f}] & '
                    f'{pp(b.acc - noskill[ds])} & {bc.model} & {100 * bc.acc:.1f} & '
                    f'{pp(c.point)} {ci(c.ci_lo, c.ci_hi, c.excludes_zero)} \\\\')
    table('overall_by_dataset.tex', 'llcclcc',
          r'Dataset & Best pipeline & Acc. (95\% interval) & Over no-skill (pp) & Best classical & Acc. & Best deep $-$ best classical (pp)', body)


def overall_by_model():
    m = pd.read_csv(OUT / 'model_ci.csv')
    noskill = pd.read_csv(OUT / 'datasets.csv').set_index('dataset').noskill
    m['margin'] = m.acc - m.dataset.map(noskill)
    g = m.groupby(['model', 'family']).agg(mean_rank=('rank', 'mean'), wins=('rank', lambda r: (r == 1).sum()),
                                           top3=('rank', lambda r: (r <= 3).sum()),
                                           best_rank=('rank', 'min'), worst_rank=('rank', 'max'),
                                           margin=('margin', 'mean')).reset_index().sort_values('mean_rank')
    g.to_csv(OUT / 'overall_by_model.csv', index=False)
    body = [f'{r.model} & {"Deep" if r.family == "deep" else "Classical"} & {r.mean_rank:.2f} & '
            f'{r.best_rank:g} to {r.worst_rank:g} & {r.wins} & {r.top3} & {pp(r.margin)} \\\\' for r in g.itertuples()]
    table('overall_by_model.tex', 'llccccc',
          r'Pipeline & Family & Mean rank & Rank range & Wins & Top 3 & Over no-skill (pp)', body)


def family():
    f = pd.read_csv(OUT / 'family_contrast.csv')
    body = []
    for ds in [d for d in f.dataset.unique() if d != 'Mean over datasets'] + ['Mean over datasets']:
        if ds == 'Mean over datasets':
            body.append(r'\midrule')
        cells = []
        for deep_set, contrast in (('all deep', 'best'), ('all deep', 'mean'), (f'deep without {DOMINANT}', 'best'),
                                   (f'deep without {DOMINANT} and {SECOND}', 'best')):
            r = f[(f.dataset == ds) & (f.deep_set == deep_set) & (f.contrast == contrast)].iloc[0]
            cells.append(f'{pp(r.point)} {ci(r.ci_lo, r.ci_hi, r.excludes_zero)}')
        body.append(f'{ds} & ' + ' & '.join(cells) + r' \\')
    table('family_contrast.tex', 'lcccc',
          rf'Dataset & Best deep $-$ best classical & Mean deep $-$ mean classical & Without {DOMINANT} & Without {DOMINANT} and {SECOND}', body)


def uncertainty():
    m = pd.read_csv(OUT / 'model_ci.csv')
    m['w'] = m.rank_hi - m.rank_lo
    d = m[m.family == 'deep'].groupby('dataset')[['seed_sd', 'trial_only_sd', 'boot_sd']].median()
    w = m.groupby('dataset').w.agg(['median', 'max'])
    body = [f'{ds} & {100 * r.seed_sd:.2f} & {100 * r.trial_only_sd:.2f} & {100 * r.boot_sd:.2f} & '
            f'{w.loc[ds, "median"]:.1f} & {w.loc[ds, "max"]:.1f} \\\\' for ds, r in d.iterrows()]
    body += [r'\midrule', f'Median & {100 * d.seed_sd.median():.2f} & {100 * d.trial_only_sd.median():.2f} & '
             f'{100 * d.boot_sd.median():.2f} & {w["median"].median():.1f} & {w["max"].median():.1f} \\\\']
    table('uncertainty.tex', 'lccccc',
          r'Dataset & Seeds & Test trials & Subjects and trials & Rank interval width, median & Widest', body)


def pairwise():
    p = pd.read_csv(OUT / 'pairwise_deep.csv')
    small = p['diff'].abs() < 0.01
    body = []
    for label, col in (('Welch test over the 10 seed accuracies', 'seed_sig'),
                       ('Bootstrap over test trials, subjects fixed', 'trial_sig'),
                       ('Bootstrap over subjects and test trials', 'hier_sig')):
        body.append(f'{label} & {p[col].sum():,} & {(p[col] & small).sum()} \\\\')
    table('pairwise.tex', 'lcc', rf'Procedure & Significant (of {len(p):,} pairs) & Significant among the {small.sum()} pairs less than 1 pp apart', body)


def degeneracy():
    d = pd.read_csv(OUT / 'degeneracy.csv')
    bacc = pd.read_csv(OUT / 'metric_scores.csv').set_index(['dataset', 'model']).bacc
    n_classes = pd.read_csv(OUT / 'datasets.csv').set_index('dataset').n_classes
    body = []
    for ds, g in d.groupby('dataset'):
        near = g[g.near_noskill]
        coll = g[g.collapsed_share >= 0.5]
        near_bacc = sum(bacc[(ds, m)] - 1 / n_classes[ds] < 0.05 for m in g.model)
        if near.empty and coll.empty:
            continue
        body.append(f'{ds} & {100 * g.noskill.iloc[0]:.1f} & {len(near)} & {(near.family == "deep").sum()} & '
                    f'{(near.family == "classical").sum()} & {near_bacc} & {len(coll)} \\\\')
    table('degeneracy.tex', 'lcccccc',
          r'Dataset & No-skill (\%) & Near no-skill (of 29) & of which deep & of which classical & Near chance, bal.\ acc. & Collapsed in $\geq$50\% of cells', body)


def metrics():
    s = pd.read_csv(OUT / 'metric_sensitivity.csv')
    noskill = pd.read_csv(OUT / 'datasets.csv').set_index('dataset').noskill
    k = s[s.metric == 'kappa'].set_index('dataset')
    b = s[s.metric == 'bacc'].set_index('dataset')
    body = [f'{ds} & {100 * noskill[ds]:.1f} & {b.loc[ds, "n_shifted"]} & {b.loc[ds, "max_shift"]} & '
            f'{k.loc[ds, "n_shifted"]} & {k.loc[ds, "max_shift"]} & {"Yes" if k.loc[ds, "top_same"] else "No"} \\\\'
            for ds in k.index if k.loc[ds, 'n_shifted'] or b.loc[ds, 'n_shifted']]
    table('metric_sensitivity.tex', 'lcccccc',
          r'Dataset & No-skill (\%) & Moved (bal.\ acc.) & Max shift & Moved ($\kappa$) & Max shift & Same top ($\kappa$)', body)


def panel():
    k = pd.read_csv(OUT / 'panel_k.csv')
    flip = lambda p: '$<$0.1' if 0 < p < 0.001 else f'{100 * p:.1f}'
    body = [f'{r.k} & {r.n_panels} & {flip(r.p_verdict_flipped)} & {100 * r.p_top_same:.0f} & '
            f'{100 * r.p_top_classical_same:.0f} & {r.tau_median:.2f} & {pp(r.d_best_q05)} to {pp(r.d_best_q95)} \\\\'
            for r in k[k.k.isin([1, 2, 3, 4, 5, 6, 8, 10, 11, 15])].itertuples()]
    table('panel_k.tex', 'ccccccc',
          r'$k$ & Panels & Verdict flipped (\%) & Same top (\%) & Same top classical (\%) & Median $\tau$ & Best deep $-$ best classical, 90\% range (pp)', body)
    from kfold_panel import NAMED
    n = pd.read_csv(OUT / 'panel_named.csv')
    acc = pd.read_csv(OUT / 'accuracy.csv')
    A = acc.pivot(index='dataset', columns='model', values='acc')
    fam = acc.drop_duplicates('model').set_index('model').family
    deep = [m for m in A if fam[m] == 'deep' and m != SECOND]
    d_no2 = A[deep].max(1) - A[[m for m in A if fam[m] == 'classical']].max(1)
    members = dict(NAMED, **{'All datasets': list(A.index), 'All but BCI IV 2a': [x for x in A.index if x != 'BCICompIV2a']})
    label = {'MOABB suggested MI panel': 'MOABB suggested panel'}
    body = [f"{label.get(r.panel, r.panel).replace('BCI IV 2a', '2a')} & {r.n} & {pp(r.d_best)} & {pp(d_no2[members[r.panel]].mean())} & {r.top} & "
            f'{r.top_classical} ({r.top_classical_rank:g}) & {r.tau:.2f} \\\\' for r in n.itertuples()]
    table('panel_named.tex', 'lcccccc',
          rf'Panel & Datasets & Best deep $-$ best classical (pp) & Without {SECOND} & Top pipeline & Top classical (rank) & $\tau$ with full panel', body)


def grid_edge():
    e = OUT / 'grid_edge.csv'
    if e.exists():
        g = pd.read_csv(e)
        body = [f'{r.dataset} & {r.model} & {100 * r.acc_final:.1f} & {100 * r.acc_extended:.1f} & {minus(f"{r.diff_pp:+.1f}")} & '
                f'{r.subjects_up}/{r.subjects_down}/{r.n_subjects} \\\\' for r in g.itertuples()]
        (TAB / 'grid_edge_summary.tex').write_text(
            f'{minus(f"{g.diff_pp.min():+.1f}")} to {minus(f"{g.diff_pp.max():+.1f}")}~pp, with a mean change of {minus(f"{abs(g.diff_pp.mean()) if abs(g.diff_pp.mean()) < 0.05 else g.diff_pp.mean():.1f}")}~pp over the {len(g)} '
            f'dataset and pipeline pairs%\n')
        table('grid_edge.tex', 'llcccc',
              r'Dataset & Pipeline & Benchmark grid & $C$ up to 1000 & Diff.\ (pp) & Subjects up/down/total', body)


def full_results():
    """Appendix: accuracy mean +- SD over subjects for every (pipeline, dataset), and the top 3 per dataset."""
    data = pickle.load(open(OUT / 'cells.pkl', 'rb'))
    rows = []
    for (ds, m), c in data['cells'].items():
        subj = np.array([v.mean() for v in c['correct'].values()])   # seed-averaged subject accuracies
        rows.append(dict(dataset=ds, model=m, family=c['family'], acc=subj.mean(), sd=subj.std(ddof=1)))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / 'full_results.csv', index=False)
    order = pd.read_csv(OUT / 'overall_by_model.csv').model.tolist()
    dsets = sorted(r.dataset.unique())
    best = r.loc[r.groupby('dataset').acc.idxmax()].set_index('dataset').model
    acc = r.set_index(['model', 'dataset'])
    for part, cols in (('a', dsets[:10]), ('b', dsets[10:])):
        header = 'Pipeline & ' + ' & '.join(r'\rotatebox{90}{' + d + '}' for d in cols)
        body = []
        for m in order:
            cells = []
            for d in cols:
                x = acc.loc[(m, d)]
                s = f'{100 * x.acc:.1f} $\\pm$ {100 * x.sd:.1f}'
                cells.append(r'\textbf{' + s + '}' if best[d] == m else s)
            body.append(f'{m} & ' + ' & '.join(cells) + r' \\')
        table(f'full_results_{part}.tex', 'l' + 'c' * len(cols), header, body)
    body = []
    for d in dsets:
        top = r[r.dataset == d].sort_values('acc', ascending=False).head(3)
        body.append(f'{d} & ' + ' & '.join(f'{t.model} ({100 * t.acc:.1f})' for t in top.itertuples()) + r' \\')
    table('top3.tex', 'llll', r'Dataset & First & Second & Third', body)


if __name__ == '__main__':
    for fn in (overall_by_dataset, overall_by_model, family, uncertainty, pairwise, degeneracy,
               metrics, panel, grid_edge, full_results):
        fn()
    print('tables written to', TAB)
