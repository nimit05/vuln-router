#!/usr/bin/env python3
"""Two-stage precision filter on IRIS, with honest selection and the per-project
breakdown that decides whether the headline number is real.

Stage 1 ranks alerts within a project by one model's score and keeps the top 25%.
Stage 2 keeps only those a second model also calls vulnerable. The (ranker,
confirmer) pair is chosen leave-one-project-out, so the held-out project never
influences the choice.

The pooled precision looks strong (0.522 vs a 0.152 base rate). The per-project
table shows why that is Simpson's paradox: almost every true positive comes from
ESAPI, whose own base rate is 0.50.
"""
import numpy as np, pandas as pd
from common import iris, MODELS
rng = np.random.default_rng(1)
P, V, S, meta = iris()
y = meta.loc[P.index, 'gold_label'].values.astype(int)
proj = meta.loc[P.index, 'proj'].values
projs = np.unique(proj)


def two_stage(rank_m, conf_m, idx, frac=0.25):
    s = P[rank_m].values[idx]
    kk = max(1, int(round(frac * len(idx))))
    top = idx[np.argsort(-s)[:kk]]
    return top[V[conf_m].values[top].astype(bool)]


def lopo_pair(g):
    """Pick (ranker, confirmer) on every project except g."""
    best, bs = None, -1
    for rm in MODELS:
        for cm in MODELS:
            t2 = k2 = 0
            for h in projs[projs != g]:
                i = np.where(proj == h)[0]
                kept = two_stage(rm, cm, i)
                t2 += (y[kept] == 1).sum(); k2 += len(kept)
            sc = t2 / k2 if k2 >= 20 else 0
            if sc > bs:
                best, bs = (rm, cm), sc
    return best


rows = []
for g in projs:
    pair = lopo_pair(g)
    i = np.where(proj == g)[0]
    kept = two_stage(*pair, i)
    nulls = [(y[rng.choice(i, len(kept), replace=False)] == 1).mean()
             for _ in range(300)] if len(kept) else [0]
    rows.append(dict(project=g[:40], paths=len(i), base=y[i].mean(),
                     pair=f'{pair[0]}+{pair[1]}', kept=len(kept), tp=(y[kept] == 1).sum(),
                     prec=(y[kept] == 1).mean() if len(kept) else np.nan,
                     rand=np.mean(nulls)))
t = pd.DataFrame(rows)
print(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
K, TP = t.kept.sum(), t.tp.sum()
print(f'\npooled: kept {K}/{len(y)} ({100*K/len(y):.1f}%)  precision {TP/K:.3f}  '
      f'recall {TP/(y==1).sum():.3f} ({TP} of {(y==1).sum()})  base rate {y.mean():.3f}')
sub = t[t.kept > 0]
print(f'beats own base rate in {(sub.prec>sub.base).sum()}/{len(sub)} projects; '
      f'mean per-project excess over matched random {(sub.prec-sub.rand).mean():+.3f}')

print('\nbootstrap over projects, hindsight-best pair (qwen-7b rank + qwen-1.5b confirm):')
vals = []
for _ in range(500):
    gs = rng.choice(projs, len(projs), replace=True)
    tp = k = rtp = rk = 0
    for g in gs:
        i = np.where(proj == g)[0]
        kept = two_stage('qwen-7b', 'qwen-1.5b', i)
        tp += (y[kept] == 1).sum(); k += len(kept)
        if len(kept):
            r = rng.choice(i, len(kept), replace=False)
            rtp += (y[r] == 1).sum(); rk += len(r)
    vals.append((tp / k if k else 0) - (rtp / rk if rk else 0))
lo, hi = np.percentile(vals, [2.5, 97.5])
print(f'  excess over matched random {np.mean(vals):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]')

print('\nsingle-stage ranking, honest (LOPO) ranker selection at 10% keep/project:')
tp = k = 0
picks = []
for g in projs:
    best, bs = None, -1
    for m in MODELS:
        t2 = k2 = 0
        for h in projs[projs != g]:
            i = np.where(proj == h)[0]
            kk = max(1, int(round(0.10 * len(i))))
            sel = i[np.argsort(-P[m].values[i])[:kk]]
            t2 += (y[sel] == 1).sum(); k2 += len(sel)
        sc = t2 / k2 if k2 else 0
        if sc > bs:
            best, bs = m, sc
    picks.append(best)
    i = np.where(proj == g)[0]
    kk = max(1, int(round(0.10 * len(i))))
    sel = i[np.argsort(-P[best].values[i])[:kk]]
    tp += (y[sel] == 1).sum(); k += len(sel)
print(f'  precision {tp/k:.3f} vs base rate {y.mean():.3f}   picks {pd.Series(picks).value_counts().to_dict()}')
