"""Precision-first evaluation on IRIS alert triage.

The deployment question: of the alerts we keep, what fraction are real? The
honest null is NOT "no filter" but "keep the same NUMBER of alerts at random",
because precision at a small keep-rate rises by chance alone on a per-project
metric (E3).

Filters compared, all on the same 2,257 paths / 14 projects:
  single model verdict            what the pool offers alone
  unanimous / majority vote       consensus among the 5
  ensemble mean score, top-k/proj ranking rather than thresholding
  best single model, top-k/proj   the same, one model
  random, matched keep count      the null every row is measured against
"""
import glob
import numpy as np, pandas as pd
from common import iris, MODELS
rng = np.random.default_rng(0)
P, V, S, meta = iris()
y = meta.loc[P.index, 'gold_label'].values.astype(int)
proj = meta.loc[P.index, 'proj'].values
n = len(y)
print(f'{n} paths, {len(np.unique(proj))} projects, {y.mean():.1%} are real vulnerabilities\n')

def micro(keep):
    """Pooled precision/recall/F1 over all kept paths."""
    k = keep.sum()
    tp = (keep & (y == 1)).sum()
    prec = tp / k if k else 0.0
    rec = tp / (y == 1).sum()
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, k

def random_match(keep, reps=200):
    """Same number kept per project, chosen at random."""
    out = []
    for _ in range(reps):
        r = np.zeros(n, bool)
        for g in np.unique(proj):
            i = np.where(proj == g)[0]
            kk = int(keep[i].sum())
            if kk:
                r[rng.choice(i, kk, replace=False)] = True
        out.append(micro(r)[0])
    return np.mean(out), np.std(out)

rows = []
def add(tag, keep):
    p, r, f, k = micro(keep)
    rp, rs = random_match(keep)
    rows.append(dict(filter=tag, kept=k, keep_pct=100*k/n, precision=p, recall=r, f1=f,
                     rand_prec=rp, excess=p-rp, z=(p-rp)/rs if rs > 0 else np.nan))

add('no filter (keep all)', np.ones(n, bool))
for m in MODELS:
    add(f'single: {m}', V[m].values.astype(bool))
votes = V[MODELS].values.astype(int).sum(1)
add('consensus: all 5 say vuln', votes == 5)
add('consensus: >=4 say vuln', votes >= 4)
add('majority: >=3 say vuln', votes >= 3)
# ranking-based: top-k per project by mean score / by best single model's score
mean_p = P[MODELS].mean(1).values
for tag, score in [('ensemble mean score', mean_p), ('qwen-7b score', P['qwen-7b'].values)]:
    for frac in [0.05, 0.10, 0.25]:
        keep = np.zeros(n, bool)
        for g in np.unique(proj):
            i = np.where(proj == g)[0]
            kk = max(1, int(round(frac * len(i))))
            keep[i[np.argsort(-score[i])[:kk]]] = True
        add(f'top {frac:.0%}/project by {tag}', keep)
t = pd.DataFrame(rows)
print(t.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\nexcess = precision minus the matched-count random baseline; z = excess in sd of that baseline')
