#!/usr/bin/env python3
"""Q6 -- re-score the IRIS filter stage under a metric that cannot be gamed.

E3 showed IRIS's per-project AvgF1 is satisfied by keeping 10 random paths per
project (0.236) against the published router's 0.244, because the metric rewards
keeping FEW alerts. Any claim built on it -- including "the LLM filter stage
contributes nothing" -- is therefore a claim about the metric as much as about
the filter.

This re-asks the deployment question under a fixed review budget: an analyst
reviews k alerts per project, and we ask what they find. Spending is now equal by
construction, so being stingy buys nothing and a filter has to be good at
ORDERING rather than at dropping.

Two readings of each model, because they are not the same claim:
  filter  -- the deployed semantics. The model says keep/drop; the analyst
             reviews kept alerts first (random order inside the kept set), then
             the dropped ones if budget remains.
  rank    -- the most generous reading. Order every path by the model's p_vuln
             and review the top k. Uses the score, not just the verdict.

Reported against a matched-budget random null (same k, random order), which is
the honest baseline per E3/E13, plus a per-project breakdown and a bootstrap
over projects. `detected` counts projects where the analyst sees at least one
real vulnerability -- IRIS's own #Detected, made budget-aware.
"""
import sys, os, json
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import iris, MODELS

rng = np.random.default_rng(0)
BUDGETS = [5, 10, 20]
NDRAW = 2000


def load():
    P, V, S, meta = iris()
    y = meta.loc[P.index, 'gold_label'].astype(int).values
    proj = meta.loc[P.index, 'proj'].values
    return P, V, y, proj


def score(order_by_proj, y, proj, k):
    """order_by_proj: dict proj -> array of positional indices, review order.
    Returns (#projects with a TP found, mean per-project recall over TP projects)."""
    det, recs = 0, []
    for p, order in order_by_proj.items():
        m = proj == p
        yy = y[m]
        if yy.sum() == 0:
            continue                      # no TP to find; excluded from recall
        seen = yy[order[:k]]
        det += int(seen.sum() > 0)
        recs.append(seen.sum() / yy.sum())
    return det, float(np.mean(recs))


def by_proj(proj, fn):
    return {p: fn(proj == p) for p in pd.unique(proj)}


def main():
    P, V, y, proj = load()
    projects = pd.unique(proj)
    n_tp_proj = sum(1 for p in projects if y[proj == p].sum() > 0)
    print(f"{len(y)} paths, {len(projects)} projects, {n_tp_proj} contain a true positive")
    sizes = {p: int((proj == p).sum()) for p in projects}

    rows = []
    for k in BUDGETS:
        sat = sum(1 for p in projects if sizes[p] <= k)
        # --- matched-budget random null
        d_, r_ = [], []
        for _ in range(NDRAW):
            o = by_proj(proj, lambda m: rng.permutation(int(m.sum())))
            a, b = score(o, y, proj, k)
            d_.append(a); r_.append(b)
        rnd = dict(det=np.mean(d_), det_sd=np.std(d_), rec=np.mean(r_), rec_sd=np.std(r_))
        rows.append(dict(k=k, config='random (matched budget)', det=rnd['det'], rec=rnd['rec'],
                         z_det=0.0, z_rec=0.0, saturated=sat))

        for mdl in MODELS:
            p_v = P[mdl].values
            # rank: review highest p_vuln first
            o = by_proj(proj, lambda m, s=p_v: np.argsort(-s[m], kind='stable'))
            a, b = score(o, y, proj, k)
            rows.append(dict(k=k, config=f'{mdl} rank', det=a, rec=b,
                             z_det=(a - rnd['det']) / (rnd['det_sd'] or 1),
                             z_rec=(b - rnd['rec']) / (rnd['rec_sd'] or 1), saturated=sat))
            # filter: kept first, random inside kept and inside dropped
            keep = V[mdl].values.astype(int)
            da, rb = [], []
            for _ in range(200):
                def ordf(m, kp=keep):
                    kk = kp[m]
                    idx = np.arange(len(kk))
                    a1 = rng.permutation(idx[kk == 1]); a0 = rng.permutation(idx[kk == 0])
                    return np.concatenate([a1, a0])
                o = by_proj(proj, ordf)
                x, z = score(o, y, proj, k); da.append(x); rb.append(z)
            rows.append(dict(k=k, config=f'{mdl} filter', det=np.mean(da), rec=np.mean(rb),
                             z_det=(np.mean(da) - rnd['det']) / (rnd['det_sd'] or 1),
                             z_rec=(np.mean(rb) - rnd['rec']) / (rnd['rec_sd'] or 1), saturated=sat))

        # oracle: true positives first (ceiling, not achievable)
        o = by_proj(proj, lambda m: np.argsort(-y[m], kind='stable'))
        a, b = score(o, y, proj, k)
        rows.append(dict(k=k, config='ORACLE (TPs first)', det=a, rec=b,
                         z_det=np.nan, z_rec=np.nan, saturated=sat))

    df = pd.DataFrame(rows)
    for k in BUDGETS:
        sub = df[df.k == k]
        print(f"\n=== review budget k = {k} alerts/project "
              f"({int(sub.saturated.iloc[0])} of {len(projects)} projects have <= {k} paths, so all rows tie there)")
        print(f"{'configuration':24s} {'detected':>10s} {'z':>7s} {'recall':>8s} {'z':>7s}")
        print('-' * 60)
        for _, x in sub.iterrows():
            zd = '   --' if pd.isna(x.z_det) else f'{x.z_det:+6.1f}'
            zr = '   --' if pd.isna(x.z_rec) else f'{x.z_rec:+6.1f}'
            print(f'{x.config:24s} {x.det:9.1f}/{n_tp_proj:<2d} {zd} {x.rec:8.3f} {zr}')

    # --- the control E13 demands: where does an apparent win actually come from?
    # mean per-project recall is dominated by SMALL projects, where a budget of k
    # reviews most of the project and recall approaches 1 for everyone. The
    # deployment question only has teeth where the budget actually binds.
    print("\n=== per-project breakdown of the only positive row (qwen-7b rank, k=10)")
    K, s_ = 10, P['qwen-7b'].values
    pr = []
    for p_ in projects:
        m = proj == p_; yy = y[m]
        if yy.sum() == 0:
            continue
        o = np.argsort(-s_[m], kind='stable')
        rec = yy[o[:K]].sum() / yy.sum()
        nul = np.mean([yy[rng.permutation(len(yy))[:K]].sum() / yy.sum() for _ in range(2000)])
        pr.append(dict(project=p_.split('__')[1][:26], paths=int(m.sum()), tps=int(yy.sum()),
                       rank=rec, random=nul, excess=rec - nul, binds=int(m.sum()) > 3 * K))
    pr = pd.DataFrame(pr).sort_values('paths', ascending=False)
    print(pr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    binding = pr[pr.binds]
    print(f"\nbeats matched random in {(pr.excess > 0).sum()} of {len(pr)} projects "
          f"(mean excess {pr.excess.mean():+.3f})")
    print(f"restricted to the {len(binding)} projects where the budget BINDS (> {3*K} paths): "
          f"beats random in {(binding.excess > 0).sum()}, mean excess {binding.excess.mean():+.3f}")
    print("The positive mean comes from projects small enough that k reviews most of them.")
    pr.to_csv('data/gr/q6_perproject.csv', index=False)

    out = 'data/gr/q6_budget_recall.csv'
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nz is standard deviations above the matched-budget random null over "
          f"{NDRAW} draws. A filter that carries no ordering signal scores z ~ 0.")


if __name__ == '__main__':
    main()
