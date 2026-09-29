#!/usr/bin/env python3
"""D4 -- get the most out of the skilled models without a learned router.

D2 found GraphRouter v2 ties a same-mix random router: with Qwen3-8B and 32B
failing on the same paths, per-path routing from a text embedding has nothing to
learn. Three cheaper ways to use the two skilled models, each against its null:

1. **Ensemble** -- mean of the two P(vulnerable). Cost = both models.
2. **Cascade** -- run the 8B on every path, send the paths it finds MOST
   suspicious (top fraction f per project) to the 32B, and rank the escalated
   paths by the 32B's score above the rest. This routes on a POST-call signal
   (the cheap model's own score), which the 8B has skill for (AUC 0.687), unlike
   GraphRouter's pre-call embedding. Null: escalate the same number per project
   at random.
3. **Floor for IRIS's table** -- 32B keep/drop plus the top-k paths per project
   by 32B score, so no project with a bug is dropped wholesale (always-32B loses
   2 of 10). Null: E3's random-k per project, and 32B verdicts + random-k floor.

Metrics: within-project AUC, fixed-budget det@10 / recall@10 (E19) with the
binding-budget subset, and IRIS AvgF1 via their own score_subset.metrics().
"""
import argparse, importlib.util, json, math, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def pv(r):
    if r.get("p_vuln") is not None:
        return float(r["p_vuln"])
    return 0.5 if r.get("pred_label") is None else float(r["pred_label"])


def load(path):
    out = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            if not r.get("error"):
                out[r["unit_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", default=os.path.join(ROOT, "data/gr/d1/qwen3-8b-nothink__units_paths.jsonl"))
    ap.add_argument("--large", default=os.path.join(ROOT, "data/gr/d1/qwen3-32b-nothink__units_paths.jsonl"))
    ap.add_argument("--small-name", default="qwen3-8b-nothink", help="cost key of --small")
    ap.add_argument("--large-name", default="qwen3-32b-nothink", help="cost key of --large")
    ap.add_argument("--extra", nargs="*", default=[], help="name=path, added to the ensemble table")
    ap.add_argument("--cost", default=os.path.join(ROOT, "data/gr/d1/cost_per_token.json"))
    ap.add_argument("--slices", default=os.path.join(ROOT, "data/gr/slices.jsonl"))
    ap.add_argument("--truth", default=os.path.join(ROOT, "data/gr/iris_truth.json"))
    ap.add_argument("--score-subset", default=os.path.join(ROOT, "../IRIS/reproduction/score_subset.py"))
    ap.add_argument("--draws", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    spec = importlib.util.spec_from_file_location("ss", a.score_subset)
    ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
    projects = sorted(json.load(open(a.truth)))
    sl = [json.loads(l) for l in open(a.slices) if l.strip()]
    Sm, Lg = load(a.small), load(a.large)
    ext = {s.split("=", 1)[0]: load(s.split("=", 1)[1]) for s in a.extra}
    ids = [s["path_id"] for s in sl if s["path_id"] in Sm and s["path_id"] in Lg
           and all(s["path_id"] in e for e in ext.values())]
    meta = {s["path_id"]: s for s in sl}
    y = np.array([int(meta[i]["label"]) for i in ids])
    proj = np.array([meta[i]["project"] for i in ids])
    ps = np.array([pv(Sm[i]) for i in ids]); pl = np.array([pv(Lg[i]) for i in ids])
    ks = np.array([Sm[i].get("pred_label") != 0 for i in ids])
    kl = np.array([Lg[i].get("pred_label") != 0 for i in ids])
    cpt = json.load(open(a.cost))
    def secs(rows, name):
        si, so = cpt[name]
        return np.array([rows[i]["in_tokens"] * si + rows[i]["out_tokens"] * so for i in ids])
    cs, cl = secs(Sm, a.small_name), secs(Lg, a.large_name)
    P = sorted(set(proj)); bugp = [p for p in P if y[proj == p].sum() > 0]
    idx = {p: np.where(proj == p)[0] for p in P}
    print(f"{len(ids)} paths, {y.sum()} TP, {len(bugp)} projects with a bug; "
          f"GPU-s all-8B {cs.sum():.0f}, all-32B {cl.sum():.0f}")

    def wauc(s):
        num = den = 0.0
        for p in bugp:
            ix = idx[p]; sp, sn = s[ix][y[ix] == 1], s[ix][y[ix] == 0]
            if len(sp) and len(sn):
                d = sp[:, None] - sn[None, :]
                num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
        return num / den

    def budget(s, k=10):
        det, rec, recb = 0, [], []
        for p in bugp:
            ix = idx[p]; o = ix[np.argsort(-(s[ix] + rng.random(len(ix)) * 1e-9))[:k]]
            f = y[o].sum(); det += f > 0; r = f / y[ix].sum(); rec.append(r)
            if len(ix) > 3 * k:
                recb.append(r)
        return det, float(np.mean(rec)), float(np.mean(recb))

    def show(name, s, cost):
        d, r, rb = budget(s)
        print(f"  {name:40} AUC {wauc(s):.3f}  det@10 {d:2}/{len(bugp)}  recall@10 {r:.3f}  "
              f"binding {rb:.3f}  GPU-s {cost:6.0f}")
        return r, rb

    print("\n== 1. single models and ensembles (fixed-budget metric)")
    show("random order", rng.random(len(y)), 0)
    show(a.small_name, ps, cs.sum())
    show(a.large_name, pl, cl.sum())
    show("ensemble mean(small, large)", (ps + pl) / 2, cs.sum() + cl.sum())
    for name, e in ext.items():
        pe = np.array([pv(e[i]) for i in ids])
        show(name, pe, 0)
        show(f"ensemble mean(32B, {name})", (pl + pe) / 2, cl.sum())
        show(f"ensemble mean(8B, 32B, {name})", (ps + pl + pe) / 3, cs.sum() + cl.sum())

    print(f"\n== 2. cascade: {a.small_name} on all, top-f per project by its score escalated to {a.large_name}")
    for f in (0.1, 0.2, 0.3, 0.5):
        esc = np.zeros(len(y), bool)
        for p in P:
            ix = idx[p]; n = max(1, int(round(f * len(ix))))
            esc[ix[np.argsort(-ps[ix])[:n]]] = True
        s = np.where(esc, 1.0 + pl, ps)            # escalated paths reviewed first, by 32B
        cost = cs.sum() + cl[esc].sum()
        r, rb = show(f"cascade f={f}", s, cost)
        null = []
        for _ in range(a.draws):
            e2 = np.zeros(len(y), bool)
            for p in P:
                ix = idx[p]; n = int(esc[ix].sum()); e2[rng.choice(ix, n, replace=False)] = True
            null.append(budget(np.where(e2, 1.0 + pl, ps))[1:])
        null = np.array(null)
        print(f"  {'':40} random escalation, same count: recall@10 {null[:,0].mean():.3f}"
              f"+-{null[:,0].std():.3f} (z {(r-null[:,0].mean())/max(null[:,0].std(),1e-9):+.1f})"
              f"  binding {null[:,1].mean():.3f}")

    print("\n== 3. IRIS table: 32B keep/drop + per-project floor of top-k by 32B score")
    allp = [dict(id=s["path_id"], project=s["project"], label=int(s["label"])) for s in sl]
    pos = {i: n for n, i in enumerate(ids)}
    def score(keep):
        per = defaultdict(lambda: [0, 0])
        for q in allp:
            if keep(q):
                per[q["project"]][0] += 1; per[q["project"]][1] += q["label"]
        return ss.metrics([{"paths": per[p][0], "tp_paths": per[p][1], "recall": per[p][1] > 0}
                           for p in projects])
    def fmt(m):
        return f"det {m['detected']:2}/16  FDR {m['avg_fdr']:6.2f}  AvgF1 {m['avg_f1']:.3f}"
    print(f"  {'no filter':40} {fmt(score(lambda q: True))}")
    print(f"  {'always-32B (verdict)':40} {fmt(score(lambda q: q['id'] not in pos or kl[pos[q['id']]]))}")
    for k in (1, 3, 5, 10):
        top = set()
        for p in P:
            ix = idx[p]; top |= {ids[j] for j in ix[np.argsort(-pl[ix])[:k]]}
        m = score(lambda q: q["id"] not in pos or kl[pos[q["id"]]] or q["id"] in top)
        mo = score(lambda q: q["id"] not in pos or q["id"] in top)
        rk, rf = [], []
        for _ in range(a.draws):
            rt = set()
            for p in P:
                ix = idx[p]; rt |= {ids[j] for j in rng.choice(ix, min(k, len(ix)), replace=False)}
            rk.append(score(lambda q: q["id"] not in pos or q["id"] in rt)["avg_f1"])
            rf.append(score(lambda q: q["id"] not in pos or kl[pos[q["id"]]] or q["id"] in rt)["avg_f1"])
        print(f"  32B verdict + top-{k:<2} floor              {fmt(m)}   | 32B verdict + random-{k} floor {np.mean(rf):.3f}+-{np.std(rf):.3f}")
        print(f"  32B top-{k:<2} only                        {fmt(mo)}   | random-{k} only (E3) {np.mean(rk):.3f}+-{np.std(rk):.3f}")
    # threshold on the 32B score chosen leave-one-project-out (no peeking at the held-out project)
    ths = np.linspace(0.05, 0.95, 19)
    keep_lopo = {}
    for p in P:
        tr = np.where(proj != p)[0]
        def f1_at(t):
            kk = pl[tr] >= t
            per = defaultdict(lambda: [0, 0])
            for j in tr[kk]:
                per[proj[j]][0] += 1; per[proj[j]][1] += y[j]
            recs = [{"paths": per[q][0], "tp_paths": per[q][1], "recall": per[q][1] > 0}
                    for q in projects if q != p]
            return ss.metrics(recs)["avg_f1"]
        t = max(ths, key=f1_at)
        for j in idx[p]:
            keep_lopo[ids[j]] = pl[j] >= t
    print(f"  {'32B, threshold picked leave-one-project-out':40} {fmt(score(lambda q: keep_lopo.get(q['id'], True)))}")


if __name__ == "__main__":
    main()
