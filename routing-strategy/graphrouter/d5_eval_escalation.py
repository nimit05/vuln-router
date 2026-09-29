#!/usr/bin/env python3
"""D5 -- score a learned escalation policy (GraphRouter v3) as a cascade.

Reads the npz d2_graphrouter_v2.py writes. For each lambda, "escalated" = paths
the router sends to the large model. The analyst reviews escalated paths first,
ordered by the large model's score, then the rest by the small model's score --
the same ordering D4 used for the rule-based cascade -- and we report the fixed-
budget metric (E19) against two policies with the SAME number of escalations in
EVERY project:

  rule    escalate the small model's highest-scoring paths (D4's cascade)
  random  escalate paths at random (the null)

A learned router is only worth having if it beats `rule` -- the no-training
policy it has to justify itself against -- not just `random`.
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--small", default="qwen3-8b-nothink")
    ap.add_argument("--large", default="qwen3-32b-nothink")
    ap.add_argument("--slices", default=os.path.join(ROOT, "data/gr/slices.jsonl"))
    ap.add_argument("--lambdas", default="0,0.002,0.005,0.01,0.02,0.03,0.05")
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--variant", default="pred_v2")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    z = np.load(a.npz, allow_pickle=True)
    models = [str(m) for m in z["models"]]
    ids, y, P, S = z["ids"], z["y"], z["P"], z["S"]
    meta = {json.loads(l)["path_id"]: json.loads(l) for l in open(a.slices) if l.strip()}
    proj = np.array([meta[str(i)]["project"] for i in ids])
    iS, iL = models.index(a.small), models.index(a.large)
    ps, pl, cs, cl = P[:, iS], P[:, iL], S[:, iS], S[:, iL]
    Pj = sorted(set(proj)); idx = {p: np.where(proj == p)[0] for p in Pj}
    bugp = [p for p in Pj if y[idx[p]].sum() > 0]
    mean_sec = S.mean(0); cost_m = mean_sec / mean_sec.max()

    def budget(esc, k=10):
        s = np.where(esc, 1.0 + pl, ps)
        det, rec, recb = 0, [], []
        for p in bugp:
            ix = idx[p]; o = ix[np.argsort(-(s[ix] + rng.random(len(ix)) * 1e-9))[:k]]
            f = y[o].sum(); det += f > 0; r = f / y[ix].sum(); rec.append(r)
            if len(ix) > 3 * k:
                recb.append(r)
        return det, float(np.mean(rec)), float(np.mean(recb))

    def same_count(esc, how):
        e = np.zeros(len(y), bool)
        for p in Pj:
            ix = idx[p]; n = int(esc[ix].sum())
            if n:
                pick = ix[np.argsort(-ps[ix])[:n]] if how == "rule" else rng.choice(ix, n, replace=False)
                e[pick] = True
        return e

    print(f"{'policy':22} {'esc':>5} {'GPU-s':>6}  {'det':>5} {'rec@10':>7} {'binding':>7} | "
          f"{'rule rec':>8} {'binding':>7} | {'random rec':>14} {'z':>5}")
    for name, esc in (("small only", np.zeros(len(y), bool)), ("large on all", np.ones(len(y), bool))):
        d, r, rb = budget(esc)
        print(f"{name:22} {int(esc.sum()):5} {cs.sum() + cl[esc].sum():6.0f}  {d:2}/{len(bugp)} {r:7.3f} {rb:7.3f} |")
    pred = z[a.variant]
    for lam in [float(x) for x in a.lambdas.split(",")]:
        esc = (pred - lam * cost_m).argmax(1) == iL
        if esc.sum() in (0, len(y)):
            print(f"lambda={lam:<15g} {int(esc.sum()):5}  (degenerate)"); continue
        d, r, rb = budget(esc)
        _, rr, rrb = budget(same_count(esc, "rule"))
        nul = np.array([budget(same_count(esc, "random"))[1] for _ in range(a.draws)])
        print(f"lambda={lam:<15g} {int(esc.sum()):5} {cs.sum() + cl[esc].sum():6.0f}  {d:2}/{len(bugp)} "
              f"{r:7.3f} {rb:7.3f} | {rr:8.3f} {rrb:7.3f} | {nul.mean():.3f}+-{nul.std():.3f} "
              f"{(r - nul.mean()) / max(nul.std(), 1e-9):+5.1f}")


if __name__ == "__main__":
    main()
