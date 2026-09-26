#!/usr/bin/env python3
"""C3 -- precision / recall / F1 for every configuration, in two senses.

IRIS's headline table reports AvgFDR and AvgF1, which hides precision and recall
behind an average and a complement. This restates the same runs as P/R/F1, and
it reports them TWO ways because the two disagree sharply and only one of them
is comparable to the published numbers.

**Per-project (IRIS's own definitions, Sec 3.6).** This is what their table
means, and it is already a macro-average over projects:

    Prec(P) = kept true-positive paths / kept paths     (undefined if 0 kept)
    Rec(P)  = 1 if ANY kept path is a true positive, else 0      <- BINARY
    F1(P)   = 2*Prec*Rec / (Prec+Rec)
    macro   = mean over ALL projects

Note what Rec(P) is: a project counts as fully recalled the moment ONE true path
survives. It says nothing about the other true paths. AvgFDR is 1 - mean Prec.

**Per-path (ordinary classification).** Treats each of the 2,259 paths as one
prediction, keep = positive:

    precision = TP / (TP+FP),  recall = TP / (TP+FN),  F1 = harmonic mean

Micro pools every path; macro averages the per-project values. This is the
honest view of how many real findings survive filtering, and it is much harsher
than IRIS's binary recall.

The paper rows can only be scored the first way. Their published CSVs give paths,
true-positive paths and recall per project, but not how many true paths existed
before their filter ran, so per-path recall has no denominator for them.
"""
from __future__ import annotations
import argparse, csv, glob, json, os
from collections import defaultdict


def f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="data/gr/slices.jsonl")
    ap.add_argument("--truth", default="data/gr/iris_truth.json")
    ap.add_argument("--probe-dir", default="data/gr/probe")
    ap.add_argument("--suffix", default="units_paths")
    ap.add_argument("--routes", default="data/gr/routes4.json")
    ap.add_argument("--iris-repo", default="../IRIS")
    ap.add_argument("--project-info", default="data/bench/project_info.csv")
    a = ap.parse_args()

    projects = sorted(json.load(open(a.truth)))
    paths = []
    for line in open(a.slices):
        if not line.strip():
            continue
        r = json.loads(line)
        paths.append((r["path_id"], r["project"], r["label"], r.get("sliceable", 1)))

    probes = {}
    for f in sorted(glob.glob(os.path.join(a.probe_dir, f"*__{a.suffix}.jsonl"))):
        m = os.path.basename(f).split("__")[0]
        d = {}
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("in_tokens"):
                d[r["unit_id"]] = r["pred_label"]
        probes[m] = d
    routes = json.load(open(a.routes)) if os.path.exists(a.routes) else {}

    def keeps(model, pid, sl):
        if not sl:
            return True                     # never seen by a model -> kept
        return probes.get(model, {}).get(pid, 1) == 1

    def score(fn, name):
        per = defaultdict(lambda: {"kept": 0, "ktp": 0, "tp": 0})
        for pid, proj, lab, sl in paths:
            per[proj]["tp"] += lab
            if fn(pid, sl):
                per[proj]["kept"] += 1
                per[proj]["ktp"] += lab
        mp, mr, mf = [], [], []             # IRIS per-project
        pp, pr, pf = [], [], []             # per-path, macro over projects
        TP = FP = FN = 0
        for proj in projects:
            d = per.get(proj, {"kept": 0, "ktp": 0, "tp": 0})
            prec = (d["ktp"] / d["kept"]) if d["kept"] else 0.0
            rec = 1.0 if d["ktp"] > 0 else 0.0
            mp.append(prec); mr.append(rec); mf.append(f1(prec, rec))
            tp_, fp_, fn_ = d["ktp"], d["kept"] - d["ktp"], d["tp"] - d["ktp"]
            TP += tp_; FP += fp_; FN += fn_
            p2 = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
            r2 = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
            pp.append(p2); pr.append(r2); pf.append(f1(p2, r2))
        n = len(projects)
        micro_p = TP / (TP + FP) if (TP + FP) else 0.0
        micro_r = TP / (TP + FN) if (TP + FN) else 0.0
        return dict(name=name,
                    ip=sum(mp)/n, ir=sum(mr)/n, if1=sum(mf)/n,
                    pp=sum(pp)/n, pr=sum(pr)/n, pf=sum(pf)/n,
                    mp=micro_p, mr=micro_r, mf=f1(micro_p, micro_r))

    rows = [score(lambda p, s: True, "ours: no filter")]
    for m in sorted(probes):
        rows.append(score(lambda p, s, m=m: keeps(m, p, s), f"ours: always-{m}"))
    if routes:
        rows.append(score(lambda p, s: keeps(routes.get(p, ""), p, s),
                          "ours: GraphRouter (routed)"))

    def oracle(pid, sl, lab_of={p: l for p, _, l, _ in paths}):
        if not sl:
            return True
        want = lab_of[pid] == 1
        ms = sorted(probes)
        return want if any(keeps(m, pid, sl) == want for m in ms) else keeps(ms[0], pid, sl)
    rows.append(score(oracle, "ours: per-path oracle"))

    # paper rows: IRIS-style only
    info = {r["project_slug"]: r for r in csv.DictReader(open(a.project_info))}
    keys = {(info[s]["cve_id"], info[s]["github_tag"]) for s in projects if s in info}
    paper = []
    for name in ("CodeQL.csv", "IRIS+DeepSeekCoder-7B.csv", "IRIS+GPT-4.csv"):
        path = os.path.join(a.iris_repo, "results", name)
        if not os.path.exists(path):
            continue
        mp, mr, mf = [], [], []
        for r in csv.DictReader(open(path)):
            if (r["CVE"], r["Tag"]) not in keys:
                continue
            def num(x):
                try: return int(float(x))
                except (TypeError, ValueError): return 0
            pa, tp = num(r["Paths"]), num(r["TP Paths"])
            prec = (tp / pa) if pa else 0.0
            rec = 1.0 if r["Recall"] == "1" else 0.0
            mp.append(prec); mr.append(rec); mf.append(f1(prec, rec))
        if mp:
            k = len(mp)
            paper.append(dict(name=f"paper: {name[:-4]}",
                              ip=sum(mp)/k, ir=sum(mr)/k, if1=sum(mf)/k))

    print(f"\nPER-PROJECT, IRIS's own definitions (recall is BINARY per project)")
    print(f"{'configuration':30} {'Prec':>7} {'Recall':>8} {'F1-macro':>10}")
    print("-" * 58)
    for r in paper + rows:
        print(f"{r['name']:30} {r['ip']:7.3f} {r['ir']:8.3f} {r['if1']:10.3f}")

    print(f"\nPER-PATH classification (all {len(paths)} paths; paper rows not scorable)")
    print(f"{'configuration':30} {'Prec':>7} {'Recall':>8} {'F1-macro':>10} {'F1-micro':>10}")
    print("-" * 69)
    for r in rows:
        print(f"{r['name']:30} {r['pp']:7.3f} {r['pr']:8.3f} {r['pf']:10.3f} {r['mf']:10.3f}")


if __name__ == "__main__":
    main()
