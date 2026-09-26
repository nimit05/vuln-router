"""Is the reasoning gain real skill, or just bias reduction you can get free?

Longer reasoning cuts the model's yes-rate from 0.72 to 0.53 while accuracy
barely moves. If the gain is bias, then thresholding the CHEAP mode's own
confidence to the same yes-rate should reproduce it at ~zero cost.

Decision rules compared at each budget:
  raw        the model's own verdict
  thresholded  p_vuln > t, with t fit on the OTHER half of the pairs (no peeking)
  paired     within each pair, call the higher-p side vulnerable (needs both
             versions, so it is an upper reference, not deployable)
"""
import glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)

def load(b):
    d = pd.DataFrame([json.loads(l) for l in open(f'budget_{b}.jsonl')]).drop_duplicates('unit_id')
    d['p'] = d.p_vuln.fillna(0.5)
    return d

def pair_both(df, pred_col):
    v = df.pivot_table(index='pair_id', columns='gold_label', values=pred_col).dropna()
    return float(((v[1] == 1) & (v[0] == 0)).mean()), len(v)

BUD = [0, 256, 512, 1024, 2048, 4096]
print(f'{"budget":>7s} {"tokens":>7s} {"raw":>6s} {"thresh":>7s} {"paired":>7s} {"acc_raw":>8s} {"acc_thr":>8s} {"auc":>6s} {"yes_raw":>8s}')
for b in BUD:
    d = load(b)
    y = d.gold_label.values.astype(int)
    raw_pb, _ = pair_both(d.assign(pred_raw=d.pred.fillna(0).astype(int)), 'pred_raw')
    # honest thresholding: split pairs in half, fit t on one half, score the other
    pairs = d.pair_id.unique(); rng.shuffle(pairs)
    half = set(pairs[:len(pairs)//2])
    A, B = d[d.pair_id.isin(half)], d[~d.pair_id.isin(half)]
    def best_t(x):
        ts = np.unique(x.p.values)
        best, bt = -1, 0.5
        for t in ts:
            s, _ = pair_both(x.assign(q=(x.p > t).astype(int)), 'q')
            if s > best:
                best, bt = s, t
        return bt
    tA, tB = best_t(A), best_t(B)
    thr_pb = np.mean([pair_both(B.assign(q=(B.p > tA).astype(int)), 'q')[0],
                      pair_both(A.assign(q=(A.p > tB).astype(int)), 'q')[0]])
    # paired rule
    w = d.pivot_table(index='pair_id', columns='gold_label', values='p').dropna()
    paired = float((w[1] > w[0]).mean() + 0.5*(w[1] == w[0]).mean())
    acc_raw = (d.pred.fillna(0).values.astype(int) == y).mean()
    thr_all = best_t(d)
    acc_thr = ((d.p.values > thr_all).astype(int) == y).mean()
    print(f'{b:7d} {d.tokens.mean():7.0f} {raw_pb:6.3f} {thr_pb:7.3f} {paired:7.3f} '
          f'{acc_raw:8.3f} {acc_thr:8.3f} {roc_auc_score(y, d.p.values):6.3f} '
          f'{d.pred.fillna(0).mean():8.3f}')
print('\nthresholded = free bias fix on the SAME cheap outputs (t fit on held-out pairs)')
print('paired = sees both versions of the pair, upper reference only')
