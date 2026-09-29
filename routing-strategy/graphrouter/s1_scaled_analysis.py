#!/usr/bin/env python3
"""S1 -- the paper table: does each benchmark's score rank models by real skill?

For every model on every triage unit, three numbers:
  skill      within-group AUC: over (real bug, false alarm) pairs from the SAME
             project (IRIS) or category (OWASP), how often the bug scores higher.
             0.5 = coin flip. Uses the model's P(vulnerable), not its verdict.
  official   IRIS: AvgF1 via IRIS's own score_subset.metrics (keep = model says
             vulnerable; every project in the denominator).
             OWASP: the Benchmark score, mean over categories of TPR - FPR.
  budget     fixed-budget recall@10: review each project's top 10 by P(vulnerable),
             fraction of its real bugs found (projects with >= 1 bug).

Then Kendall tau between skill and each score across models, with a 95% interval
from resampling projects (cluster bootstrap), and a model-pair table: how many
pairs does the official score order the wrong way when skill differs clearly?

Also, on IRIS only: the 32B on the projects that are NEW in the scaled run (it was
never looked at there), names-hidden AUC, and the 8B->32B cascade.
"""
import argparse, collections, glob, importlib.util, json, os
import numpy as np

SCALE = os.path.expanduser("~/nimit/scale")


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


def kendall(a, b):
    c = d = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            c += s > 0; d += s < 0
    return (c - d) / max(c + d, 1)


class Bench:
    """Per-group sufficient statistics so a project-level bootstrap is cheap."""

    def __init__(self, units, probes, groups_all, keep_unprobed=True):
        self.units, self.probes, self.groups_all = units, probes, groups_all
        self.by = collections.defaultdict(list)
        for u in units:
            self.by[u["group"]].append(u)

    def stats(self, model, rng):
        """-> {group: (auc_num, auc_den, kept, kept_tp, n_tp, rec10_found, tp_pos, fp_pos, n_neg)}"""
        R = self.probes[model]; out = {}
        for g in self.groups_all:
            us = self.by.get(g, [])
            s = np.array([pv(R[u["id"]]) if u["id"] in R else 0.5 for u in us])
            y = np.array([u["label"] for u in us])
            k = np.array([(R[u["id"]].get("pred_label") != 0) if u["id"] in R else True for u in us])
            sp, sn = s[y == 1], s[y == 0]
            if len(sp) and len(sn):
                dd = sp[:, None] - sn[None, :]
                num, den = (dd > 0).sum() + 0.5 * (dd == 0).sum(), dd.size
            else:
                num = den = 0
            found = 0
            if y.sum():
                top = np.argsort(-(s + rng.random(len(s)) * 1e-9))[:10]
                found = y[top].sum()
            out[g] = (num, den, int(k.sum()), int((k & (y == 1)).sum()), int(y.sum()), int(found),
                      int((k & (y == 1)).sum()), int((k & (y == 0)).sum()), int((y == 0).sum()))
        return out


def scores(st, groups, iris_metrics=None):
    num = sum(st[g][0] for g in groups); den = sum(st[g][1] for g in groups)
    auc = num / den if den else float("nan")
    rec = [st[g][5] / st[g][4] for g in groups if st[g][4]]
    out = {"auc": auc, "recall10": float(np.mean(rec)) if rec else float("nan")}
    if iris_metrics:
        recs = [{"paths": st[g][2], "tp_paths": st[g][3], "recall": st[g][3] > 0} for g in groups]
        m = iris_metrics(recs); out.update(avgf1=m["avg_f1"], det=m["detected"], fdr=m["avg_fdr"])
    else:                                   # OWASP Benchmark score: mean(TPR - FPR)
        j = [st[g][6] / st[g][4] - st[g][7] / st[g][8] for g in groups if st[g][4] and st[g][8]]
        out["official"] = float(np.mean(j))
    return out


def report(name, bench, models, official_key, iris_metrics, boot, rng):
    groups = bench.groups_all
    ST = {m: bench.stats(m, rng) for m in models}
    S = {m: scores(ST[m], groups, iris_metrics) for m in models}
    print(f"\n== {name}: {len(bench.units)} units, {sum(u['label'] for u in bench.units)} real, "
          f"{sum(1 for g in groups if any(u['label'] for u in bench.by.get(g, [])))} groups with a real bug, "
          f"{len(groups)} groups")
    hdr = f"   {'model':20} {'skill AUC':>9} {'official':>9} {'recall@10':>9}"
    if iris_metrics:
        hdr += f" {'det':>6} {'FDR%':>6}"
    print(hdr)
    for m in sorted(models, key=lambda m: -S[m]["auc"]):
        line = f"   {m:20} {S[m]['auc']:9.3f} {S[m][official_key]:9.3f} {S[m]['recall10']:9.3f}"
        if iris_metrics:
            line += f" {S[m]['det']:3}/{len(groups):<3} {S[m]['fdr']:6.2f}"
        print(line)
    if iris_metrics:
        nf = iris_metrics([{"paths": len(bench.by.get(g, [])), "tp_paths": sum(u["label"] for u in bench.by.get(g, [])),
                            "recall": any(u["label"] for u in bench.by.get(g, []))} for g in groups])
        print(f"   {'no filter':20} {'':9} {nf['avg_f1']:9.3f}")
    a = [S[m]["auc"] for m in models]
    for key in (official_key, "recall10"):
        b = [S[m][key] for m in models]
        taus = []
        for _ in range(boot):
            gs = list(rng.choice(groups, len(groups)))
            Sb = {m: scores(ST[m], gs, iris_metrics) for m in models}
            taus.append(kendall([Sb[m]["auc"] for m in models], [Sb[m][key] for m in models]))
        lo, hi = np.nanpercentile(taus, [2.5, 97.5])
        print(f"   Kendall tau(skill, {key}): {kendall(a, b):+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")
    # pairs the official score orders the wrong way, among pairs with a clear skill gap
    wrong = total = 0
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            if abs(S[m1]["auc"] - S[m2]["auc"]) >= 0.10:
                total += 1
                wrong += np.sign(S[m1]["auc"] - S[m2]["auc"]) != np.sign(S[m1][official_key] - S[m2][official_key])
    print(f"   model pairs with skill gap >= 0.10: {total}; official score orders {wrong} of them the wrong way")
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default=SCALE)
    ap.add_argument("--score-subset", default=os.path.expanduser("~/nimit/grv2/iris/score_subset.py"))
    ap.add_argument("--old-projects", default=os.path.expanduser("~/nimit/grv2/data/gr/iris_truth.json"))
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    spec = importlib.util.spec_from_file_location("ss", a.score_subset)
    ss = importlib.util.module_from_spec(spec); spec.loader.exec_module(ss)
    D = os.path.join(a.scale, "data")

    # ---- IRIS, scaled
    truth = json.load(open(os.path.join(D, "iris_truth.json")))
    sl = [json.loads(l) for l in open(os.path.join(D, "slices.jsonl")) if l.strip()]
    units = [dict(id=s["path_id"], group=s["project"], label=int(s["label"])) for s in sl]
    probes = {}
    for f in sorted(glob.glob(os.path.join(a.scale, "probe", "*__orig.jsonl"))):
        probes[os.path.basename(f).split("-nothink__")[0].split("-verbose__")[0].split("-low__")[0]] = load(f)
    models = sorted(probes)
    S = report("IRIS / CWE-Bench-Java, scaled", Bench(units, probes, sorted(truth)), models,
               "avgf1", ss.metrics, a.boot, rng)

    old = set(json.load(open(a.old_projects)))
    new_groups = [g for g in sorted(truth) if g not in old]
    if "qwen3-32b" in probes:
        b = Bench(units, probes, new_groups)
        st = b.stats("qwen3-32b", rng); sc = scores(st, new_groups, ss.metrics)
        print(f"\n   Qwen3-32B on the {len(new_groups)} NEW projects only: AUC {sc['auc']:.3f}, "
              f"recall@10 {sc['recall10']:.3f}")
        for v in ("anonA", "anonB"):
            f = os.path.join(a.scale, "probe", f"qwen3-32b-nothink__{v}.jsonl")
            if os.path.exists(f):
                pb = {"x": load(f)}
                stv = Bench(units, pb, sorted(truth)).stats("x", rng)
                print(f"   Qwen3-32B names hidden ({v}), all projects: AUC {scores(stv, sorted(truth), ss.metrics)['auc']:.3f}")
    # cascade 8B -> 32B, top 30% per project by the 8B
    if {"qwen3-8b", "qwen3-32b"} <= set(probes):
        P8, P32 = probes["qwen3-8b"], probes["qwen3-32b"]
        by = collections.defaultdict(list)
        for u in units:
            if u["id"] in P8 and u["id"] in P32:
                by[u["group"]].append(u)
        rec, rec_rand = [], []
        for g, us in by.items():
            y = np.array([u["label"] for u in us])
            if not y.sum():
                continue
            s8 = np.array([pv(P8[u["id"]]) for u in us]); s32 = np.array([pv(P32[u["id"]]) for u in us])
            n = max(1, int(round(0.3 * len(us))))
            esc = np.zeros(len(us), bool); esc[np.argsort(-s8)[:n]] = True
            top = np.argsort(-np.where(esc, 1 + s32, s8))[:10]; rec.append(y[top].sum() / y.sum())
            rr = []
            for _ in range(200):
                e2 = np.zeros(len(us), bool); e2[rng.choice(len(us), n, replace=False)] = True
                t2 = np.argsort(-np.where(e2, 1 + s32, s8))[:10]; rr.append(y[t2].sum() / y.sum())
            rec_rand.append(np.mean(rr))
        print(f"   cascade 8B->32B (top 30%/project): recall@10 {np.mean(rec):.3f}; "
              f"random escalation {np.mean(rec_rand):.3f}; 32B alone {S['qwen3-32b']['recall10']:.3f}")

    # ---- OWASP Benchmark
    fo = os.path.join(D, "units_owasp.jsonl")
    ow = sorted(glob.glob(os.path.join(a.scale, "probe_owasp", "*__owasp.jsonl")))
    if os.path.exists(fo) and ow:
        ou = [json.loads(l) for l in open(fo) if l.strip()]
        units_o = [dict(id=u["unit_id"], group=u["project"], label=int(u["gold_label"])) for u in ou]
        po = {os.path.basename(f).split("-nothink__")[0].split("-verbose__")[0].split("-low__")[0]: load(f) for f in ow}
        report("OWASP Benchmark Java (categories as groups)", Bench(units_o, po, sorted({u["group"] for u in units_o})),
               sorted(po), "official", None, a.boot, rng)


if __name__ == "__main__":
    main()
