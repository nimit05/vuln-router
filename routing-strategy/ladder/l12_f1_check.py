"""L12 -- the four scaled-IRIS pipelines under F1-style metrics, next to found@10.
  IRIS AvgF1  per project keep the 10 top-ranked paths; recall = 1 if one is real, precision = real
              kept / kept; F1 averaged over all 80 projects (no path at all = 0). Random order = 20 shuffles.
  path F1     each model's own yes/no verdict on every scored path (projects with a real path), pooled.
Run on gpu7 (reads /tmp/nimit/scaled)."""
import json, numpy as np
R = "/tmp/nimit/scaled"
projects = open(f"{R}/projects.txt").read().split()
def load(run, m):
    U = [json.loads(l) for l in open(f"{R}/data/{run}/units_paths.jsonl")]
    d = {}
    for l in open(f"{R}/probe/{m}__{run}.jsonl"):
        x = json.loads(l); d[x["unit_id"]] = (x["p_vuln"], x["pred_label"])
    y = np.array([u["gold_label"] for u in U]); g = np.array([u["project"] for u in U])
    s = np.array([0.5 if d.get(u["unit_id"], (None,))[0] is None else d[u["unit_id"]][0] for u in U])
    v = np.array([-1 if d.get(u["unit_id"], (None, None))[1] is None else d[u["unit_id"]][1] for u in U])
    return y, g, s, v
def avgf1_top10(y, g, s, rand=False, k=10, reps=20):
    rng = np.random.default_rng(0); tot = 0.0
    for p in projects:
        m = np.where(g == p)[0]
        if not len(m): continue
        f = 0.0
        for _ in range(reps if rand else 1):
            sc = rng.random(len(m)) if rand else s[m] + rng.random(len(m)) * 1e-9
            top = m[np.argsort(-sc)[:k]]
            tp = y[top].sum()
            if tp:
                prec = tp / len(top); f += 2 * prec / (prec + 1)
        tot += f / (reps if rand else 1)
    return tot / len(projects)
def found10(y, g, s, rand=False, k=10, reps=20):
    rng = np.random.default_rng(0); n = 0.0
    for p in projects:
        m = np.where(g == p)[0]
        if not len(m) or y[m].sum() == 0: continue
        h = 0.0
        for _ in range(reps):
            sc = rng.random(len(m)) if rand else s[m] + rng.random(len(m)) * 1e-9
            h += float(y[m[np.argsort(-sc)[:k]]].sum() > 0)
        n += h / reps
    return n
print(f"{'labelling':10s} {'ranking':8s} {'found@10':>8s} {'IRIS AvgF1 (top 10 kept)':>24s} {'path F1 of yes/no':>17s} {'precision':>9s} {'recall':>7s}")
for run, lab in (("s_q8_full", "8B"), ("s_q32_full", "32B")):
    y, g, s8, v8 = load(run, "qwen3-8b-nothink"); _, _, s32, v32 = load(run, "qwen3-32b-nothink")
    print(f"{lab:10s} {'random':8s} {found10(y,g,None,True):8.1f} {avgf1_top10(y,g,None,True):24.3f} {'':>17s}")
    for rk, s, v in (("8B", s8, v8), ("32B", s32, v32)):
        ok = v >= 0
        tp = ((v == 1) & (y == 1) & ok).sum(); fp = ((v == 1) & (y == 0) & ok).sum(); fn = ((v != 1) & (y == 1) & ok).sum()
        P = tp / max(tp + fp, 1); Rc = tp / max(tp + fn, 1); F = 2 * P * Rc / max(P + Rc, 1e-9)
        print(f"{lab:10s} {rk:8s} {found10(y,g,s):8.1f} {avgf1_top10(y,g,s):24.3f} {F:17.3f} {P:9.3f} {Rc:7.3f}")
