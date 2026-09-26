#!/usr/bin/env python3
"""Thinking-budget frontier + self-consistency analysis.

Questions:
  1. How does quality rise with the reasoning budget, and where does it flatten?
  2. Is a per-unit budget router better than one uniform budget at equal spend?
     (compared against the shuffle null, as always)
  3. Does vote disagreement across k samples give the usable confidence signal
     that token log-probs failed to give?
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

D = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(0)


def load(path):
    d = pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates("unit_id")
    return d.set_index("unit_id")


def pair_both(d, col="pred"):
    v = d.reset_index().pivot_table(index="pair_id", columns="gold_label", values=col).dropna()
    return float(((v[1] == 1) & (v[0] == 0)).mean()), len(v)


def summarise(name, d):
    y = d.gold_label.values.astype(int)
    pred = d.pred.fillna(0).values.astype(int)
    pb, npair = pair_both(d.assign(pred=d.pred.fillna(0)))
    p = d.p_vuln.fillna(0.5).values
    return dict(name=name, n=len(d), pair_both=pb, npair=npair,
                acc=(pred == y).mean(), auc=roc_auc_score(y, p),
                says_vuln=pred.mean(), abstain=d.pred.isna().mean(),
                tokens=d.tokens.mean(), forced=d.n_forced.mean(),
                tok_p90_p10=d.tokens.quantile(.9) / max(d.tokens.quantile(.1), 1))


rows, byname = [], {}
for f in sorted(glob.glob(f"{D}/budget_*.jsonl"), key=lambda s: int(s.split("_")[-1][:-6])):
    b = int(f.split("_")[-1][:-6])
    d = load(f); byname[b] = d
    r = summarise(f"budget {b}", d); r["budget"] = b; rows.append(r)
if rows:
    t = pd.DataFrame(rows)
    print("== thinking-budget frontier (299 PrimeVul pairs, Qwen3-8B)")
    print(t[["name", "n", "pair_both", "acc", "auc", "says_vuln", "abstain",
             "tokens", "forced", "tok_p90_p10"]].to_string(index=False,
             float_format=lambda v: f"{v:.3f}"))

    # marginal value of tokens
    print("\n   marginal: pair-both-right gained per 1k extra output tokens")
    t = t.sort_values("budget")
    for i in range(1, len(t)):
        dq = t.pair_both.iloc[i] - t.pair_both.iloc[i-1]
        dc = (t.tokens.iloc[i] - t.tokens.iloc[i-1]) / 1000
        print(f"   {t.budget.iloc[i-1]:>5} -> {t.budget.iloc[i]:<5} "
              f"dq={dq:+.3f} dtok={dc*1000:7.0f}  q per 1k tok = {dq/dc if dc else float('nan'):+.4f}")

    # 2. per-unit budget routing vs one uniform budget, at equal token spend
    budgets = sorted(byname)
    ix = set.intersection(*[set(byname[b].index) for b in budgets])
    ix = sorted(ix)
    y = byname[budgets[0]].loc[ix].gold_label.values.astype(int)
    P = np.column_stack([byname[b].loc[ix].pred.fillna(0).values.astype(int) for b in budgets])
    T = np.column_stack([byname[b].loc[ix].tokens.values for b in budgets])
    correct = (P == y[:, None])
    # oracle: cheapest budget that gets this unit right
    first_ok = np.where(correct.any(1), correct.argmax(1), len(budgets) - 1)
    r = np.arange(len(ix))
    print(f"\n== per-unit budget routing on n={len(ix)}")
    for j, b in enumerate(budgets):
        print(f"   uniform budget {b:>5}: acc={correct[:, j].mean():.3f} "
              f"tokens={T[:, j].mean():7.0f}")
    print(f"   ORACLE cheapest-correct: acc={correct[r, first_ok].mean():.3f} "
          f"tokens={T[r, first_ok].mean():7.0f}")
    nulls = []
    for _ in range(200):
        Pn = np.column_stack([rng.permutation(P[:, j]) for j in range(P.shape[1])])
        cn = (Pn == y[:, None])
        fo = np.where(cn.any(1), cn.argmax(1), len(budgets) - 1)
        nulls.append(cn[r, fo].mean())
    print(f"   SHUFFLE-NULL oracle:     acc={np.mean(nulls):.3f}+-{np.std(nulls):.3f}")
    print(f"   >>> real headroom {correct[r, first_ok].mean()-max(correct.mean(0)):+.3f}, "
          f"null headroom {np.mean(nulls)-max(correct.mean(0)):+.3f}")

# 3. self-consistency
sc = f"{D}/sc5_1024.jsonl"
if os.path.exists(sc):
    d = load(sc)
    y = d.gold_label.values.astype(int)
    print(f"\n== self-consistency k=5 @1024 on n={len(d)}")
    r = summarise("sc5@1024", d)
    print(f"   majority vote: pair-both-right={r['pair_both']:.3f} acc={r['acc']:.3f} "
          f"auc={r['auc']:.3f} tokens={r['tokens']:.0f}")
    if 1024 in byname:
        b = byname[1024]
        ix = d.index.intersection(b.index)
        print(f"   single sample @1024: pair-both-right={pair_both(b.loc[ix])[0]:.3f} "
              f"acc={(b.loc[ix].pred.fillna(0).values.astype(int)==b.loc[ix].gold_label.values).mean():.3f} "
              f"tokens={b.loc[ix].tokens.mean():.0f}")
    # vote fraction as a score and as a confidence signal
    vf = d.vote_frac.fillna(0.5).values
    print(f"   vote fraction as a bug score: AUC={roc_auc_score(y, vf):.3f}")
    conf = np.abs(vf - 0.5)             # 0 = split vote
    pred = d.pred.fillna(0).values.astype(int)
    ok = (pred == y).astype(int)
    print(f"   vote agreement predicts its own correctness: AUC={roc_auc_score(ok, conf):.3f}")
    for thr in [0.6, 0.8, 1.0]:
        m = np.abs(vf - 0.5) >= (thr - 0.5)
        if m.sum():
            print(f"   units with >= {thr:.0%} agreement: {m.mean():.0%} of units, "
                  f"accuracy there {ok[m].mean():.3f} (vs {ok.mean():.3f} overall)")
