"""Does the ranking gain survive honest model selection and per-project checks?

Three threats to the qwen-7b result:
  1. hindsight - "qwen-7b ranks best" was chosen after seeing all 14 projects
  2. concentration - one huge project (ff4j, 63% of paths) could carry it
  3. noise - 344 positives over 14 projects is small

Tests: leave-one-project-out model selection (pick the ranker on 13 projects,
apply to the held-out one), per-project precision, and a bootstrap over projects.
Then a two-stage precision filter: rank cheap, confirm with a second model.
"""
import numpy as np, pandas as pd
from common import iris, MODELS
rng = np.random.default_rng(0)
P, V, S, meta = iris()
y = meta.loc[P.index, 'gold_label'].values.astype(int)
proj = meta.loc[P.index, 'proj'].values
projs = np.unique(proj)
FRAC = 0.10

def topk_mask(score, idx, frac=FRAC):
    keep = np.zeros(len(idx), bool)
    kk = max(1, int(round(frac * len(idx))))
    keep[np.argsort(-score[idx])[:kk]] = True
    return keep

def prec_of(score, frac=FRAC):
    tp = k = 0
    for g in projs:
        i = np.where(proj == g)[0]
        m = topk_mask(score, i, frac)
        tp += (y[i][m] == 1).sum(); k += m.sum()
    return tp / k if k else 0.0

print('1. per-model ranking precision at 10% keep/project (all 14 projects)')
for m in MODELS:
    print(f'   {m:16s} {prec_of(P[m].values):.3f}')
print(f'   {"ensemble mean":16s} {prec_of(P[MODELS].mean(1).values):.3f}')
print(f'   base rate {y.mean():.3f}')

print('\n2. leave-one-project-out ranker selection (no hindsight)')
tp = k = 0; picks = []
for g in projs:
    tr = proj != g
    best, bs = None, -1
    for m in MODELS:
        s = P[m].values; t2 = k2 = 0
        for h in projs[projs != g]:
            i = np.where(proj == h)[0]
            msk = topk_mask(s, i)
            t2 += (y[i][msk] == 1).sum(); k2 += msk.sum()
        sc = t2 / k2 if k2 else 0
        if sc > bs: best, bs = m, sc
    picks.append(best)
    i = np.where(proj == g)[0]
    msk = topk_mask(P[best].values, i)
    tp += (y[i][msk] == 1).sum(); k += msk.sum()
print(f'   LOPO-selected ranker precision {tp/k:.3f}   picks: {pd.Series(picks).value_counts().to_dict()}')

print('\n3. per-project precision, qwen-7b ranking at 10% (projects with >=20 paths)')
print(f'   {"project":48s} {"paths":>6s} {"pos%":>6s} {"kept":>5s} {"prec":>6s}')
wins = 0; tot = 0
for g in projs:
    i = np.where(proj == g)[0]
    if len(i) < 20: continue
    m = topk_mask(P['qwen-7b'].values, i)
    pr = (y[i][m] == 1).mean()
    base = y[i].mean()
    wins += pr > base; tot += 1
    print(f'   {g[:48]:48s} {len(i):6d} {100*base:6.1f} {m.sum():5d} {pr:6.3f}')
print(f'   beats its own project base rate in {wins}/{tot} projects')

print('\n4. bootstrap over projects (resample the 14 projects, 500 draws)')
vals = []
for _ in range(500):
    gs = rng.choice(projs, len(projs), replace=True)
    tp = k = rtp = rk = 0
    for g in gs:
        i = np.where(proj == g)[0]
        m = topk_mask(P['qwen-7b'].values, i)
        tp += (y[i][m] == 1).sum(); k += m.sum()
        r = np.zeros(len(i), bool); r[rng.choice(len(i), m.sum(), replace=False)] = True
        rtp += (y[i][r] == 1).sum(); rk += r.sum()
    vals.append((tp/k if k else 0) - (rtp/rk if rk else 0))
lo, hi = np.percentile(vals, [2.5, 97.5])
print(f'   precision excess over matched random: {np.mean(vals):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]')

print('\n5. two-stage: rank by qwen-7b top 25%, then require a second model to agree')
i_all = np.concatenate([np.where(proj == g)[0][topk_mask(P['qwen-7b'].values, np.where(proj == g)[0], 0.25)]
                        for g in projs])
stage1 = np.zeros(len(y), bool); stage1[i_all] = True
print(f'   stage 1 alone: kept {stage1.sum():4d}  precision {(y[stage1]==1).mean():.3f}')
for m in MODELS:
    k2 = stage1 & V[m].values.astype(bool)
    if k2.sum():
        # null: keep the same number at random from stage-1 survivors
        nulls = []
        for _ in range(200):
            r = np.zeros(len(y), bool)
            idx = np.where(stage1)[0]
            r[rng.choice(idx, k2.sum(), replace=False)] = True
            nulls.append((y[r] == 1).mean())
        print(f'   + {m:16s} kept {k2.sum():4d}  precision {(y[k2]==1).mean():.3f}  '
              f'random-from-stage1 {np.mean(nulls):.3f}  excess {(y[k2]==1).mean()-np.mean(nulls):+.3f}')
