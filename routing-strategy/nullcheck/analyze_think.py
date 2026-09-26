#!/usr/bin/env python3
"""Think-vs-direct analysis: skill, shuffle-null headroom, and the cost spread.

Every claim here gets the same null test the model-routing claims got: if
shuffling a mode's answers reproduces the headroom, the headroom is luck.
"""
import glob, json, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

rng = np.random.default_rng(0)
D = sys.argv[1] if len(sys.argv) > 1 else "."


def load(name):
    f = f"{D}/{name}.jsonl"
    if not glob.glob(f):
        return None
    d = pd.DataFrame([json.loads(l) for l in open(f)]).drop_duplicates("unit_id")
    d["p"] = d.p_vuln.fillna(0.5)
    d["pred"] = d.pred.fillna(0)          # unparsed verdict = "not vulnerable", flagged below
    return d.set_index("unit_id")


def pair_rank(d):
    """How often the buggy version scores above its own fix. 0.5 = coin flip."""
    w = d.reset_index().pivot_table(index="pair_id", columns="gold_label", values="p")
    w = w.dropna()
    return float((w[1] > w[0]).mean() + 0.5 * (w[1] == w[0]).mean()), len(w)


def pair_both_right(d):
    """Fraction of pairs where BOTH versions are labelled correctly (PrimeVul's metric)."""
    v = d.reset_index().pivot_table(index="pair_id", columns="gold_label", values="pred").dropna()
    return float(((v[1] == 1) & (v[0] == 0)).mean()), len(v)


def report(name, d):
    y = d.gold_label.values.astype(int)
    print(f"\n== {name}: n={len(d)} pos={y.mean():.2f} unparsed={d.p_vuln.isna().mean():.1%} "
          f"truncated_code={d.truncated.mean():.1%} finish_length={(d.finish=='length').mean():.1%}")
    pr, npair = pair_rank(d)
    br, _ = pair_both_right(d)
    print(f"   F1={f1_score(y, d.pred.astype(int)):.3f}  AUC={roc_auc_score(y, d.p):.3f}  "
          f"pair-rank={pr:.3f} (n={npair})  pair-both-right={br:.3f}  says-vulnerable={d.pred.mean():.2f}")
    print(f"   cost: out_tokens mean={d.out_tokens.mean():.0f} p10={d.out_tokens.quantile(.1):.0f} "
          f"p90={d.out_tokens.quantile(.9):.0f} spread p90/p10={d.out_tokens.quantile(.9)/max(d.out_tokens.quantile(.1),1):.1f}x")
    if "think_tokens" in d and d.think_tokens.max() > 0:
        print(f"   thinking tokens: mean={d.think_tokens.mean():.0f} p90={d.think_tokens.quantile(.9):.0f}")
    return d


def two_mode(cheap, rich, cname, rname):
    """Is routing between two modes of ONE model real, or shuffle-level luck?"""
    ix = cheap.index.intersection(rich.index)
    c, r = cheap.loc[ix], rich.loc[ix]
    y = c.gold_label.values.astype(int)
    vc, vr = c.pred.values.astype(int), r.pred.values.astype(int)
    f_c, f_r = f1_score(y, vc), f1_score(y, vr)
    orac = np.where((vc == y) | (vr == y), y, 1 - y)
    nulls = []
    for _ in range(200):
        sc, sr = rng.permutation(vc), rng.permutation(vr)
        nulls.append(f1_score(y, np.where((sc == y) | (sr == y), y, 1 - y)))
    print(f"\n== routing {cname} <-> {rname} on n={len(ix)}")
    print(f"   {cname} F1={f_c:.3f}   {rname} F1={f_r:.3f}   oracle F1={f1_score(y, orac):.3f}   "
          f"SHUFFLE-NULL oracle F1={np.mean(nulls):.3f}+-{np.std(nulls):.3f}")
    print(f"   real headroom over best single mode: {f1_score(y, orac)-max(f_c,f_r):+.3f}  "
          f"null headroom: {np.mean(nulls)-max(f_c,f_r):+.3f}")
    agree = (vc == vr).mean()
    flips = ((vc != vr)).mean()
    gain = ((vr == y) & (vc != y)).mean()
    lose = ((vc == y) & (vr != y)).mean()
    print(f"   modes agree {agree:.2f}; {rname} fixes {gain:.1%} and breaks {lose:.1%} of units")
    # can the cheap mode's own confidence tell us when to escalate?
    esc = (np.abs(c.p.values - 0.5))         # low = unsure
    helped = ((vr == y) & (vc != y)).astype(int)
    if helped.sum() > 5 and helped.sum() < len(helped):
        print(f"   escalation signal: AUC of (1-|p-0.5|) for predicting 'rich mode fixes it' = "
              f"{roc_auc_score(helped, -esc):.3f}")
    # cost of always-rich vs routed at matched quality
    print(f"   cost: {cname} {c.out_tokens.sum():.0f} out-tokens, {rname} {r.out_tokens.sum():.0f} "
          f"({r.out_tokens.sum()/max(c.out_tokens.sum(),1):.1f}x)")


if __name__ == "__main__":
    sets = {n: load(n) for n in ["qwen3_nothink", "qwen3_think", "phi4_direct", "phi4_cot"]}
    for n, d in sets.items():
        if d is not None:
            report(n, d)
    if sets["qwen3_nothink"] is not None and sets["qwen3_think"] is not None:
        two_mode(sets["qwen3_nothink"], sets["qwen3_think"], "qwen3-nothink", "qwen3-think")
    if sets["phi4_direct"] is not None and sets["phi4_cot"] is not None:
        two_mode(sets["phi4_direct"], sets["phi4_cot"], "phi4-direct", "phi4-cot")
    # cross-model: does a bigger model beat a thinking smaller one?
    if sets["qwen3_think"] is not None and sets["phi4_cot"] is not None:
        two_mode(sets["qwen3_think"], sets["phi4_cot"], "qwen3-think", "phi4-cot")
