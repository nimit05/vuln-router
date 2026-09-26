#!/usr/bin/env python3
"""C2 -- why the router underperforms its own oracle, measured rather than asserted.

The routed row of the table lands below several always-<model> rows. That is not
undertraining: at 20 epochs it scored AvgF1 0.177 and at 1,000 epochs 0.165. This
script shows the actual cause, which is a mismatch between what GraphRouter
optimises and what IRIS's table rewards.

GraphRouter's supervision is, per query,

    label = eye(n_llms)[argmax(effect)]

i.e. "which model is right about THIS path". That is per-path correctness. IRIS's
#Detected is per PROJECT and needs only ONE true-positive path to survive
filtering in each project. The two objectives pull in opposite directions once
the base rate is skewed, and here 85% of paths are false alerts.

Two measurements make the case:

1. **Per-path accuracy is anti-correlated with true-positive retention.** Ordering
   configurations by accuracy orders them almost exactly inversely by how many of
   the 344 true-positive paths they keep. A filter that drops nearly everything
   is right about most paths, because most paths are false, and destroys
   detection doing it.

2. **The argmax label excludes the high-recall model.** granite-8b has the
   HIGHEST effect on true vulnerabilities and the LOWEST mean effect overall,
   because the negative majority dominates the mean. It therefore wins the
   training label on a small minority of paths, and the router learns to almost
   never select it -- discarding the one model whose behaviour the deployment
   metric actually rewards.

This is the same hazard that failed gate G1 on the PrimeVul track: with a
lopsided base rate, prediction bias imitates skill. Here it is the objective, not
a model, that is fooled by it.
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import Counter, defaultdict


def load_paths(slices: str) -> dict[str, tuple[str, int, int]]:
    out = {}
    for line in open(slices):
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["path_id"]] = (r["project"], r["label"], r.get("sliceable", 1))
    return out


def load_probes(probe_dir: str, suffix: str) -> dict[str, dict[str, int]]:
    out = {}
    for f in sorted(glob.glob(os.path.join(probe_dir, f"*__{suffix}.jsonl"))):
        m = os.path.basename(f).split("__")[0]
        d = {}
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("in_tokens"):          # a failed request is not a verdict
                d[r["unit_id"]] = r["pred_label"]
        out[m] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="data/gr/slices.jsonl")
    ap.add_argument("--probe-dir", default="data/gr/probe")
    ap.add_argument("--suffix", default="units_paths")
    ap.add_argument("--routes", default="data/gr/routes.json")
    ap.add_argument("--router-data", default=None,
                    help="router_data.csv, to report per-model effect and how "
                         "often each model wins the argmax label")
    ap.add_argument("--llm-json", default=None)
    a = ap.parse_args()

    paths = load_paths(a.slices)
    probes = load_probes(a.probe_dir, a.suffix)
    routes = json.load(open(a.routes)) if os.path.exists(a.routes) else {}
    n_tp = sum(1 for _, lab, _ in paths.values() if lab == 1)

    def verdict(model: str, pid: str, sliceable: int) -> int:
        if not sliceable:
            return 1                        # never seen by a model -> kept
        return probes.get(model, {}).get(pid, 1)

    rows = []
    def measure(fn, name):
        ok = tot = tp_kept = kept = 0
        for pid, (_, lab, sl) in paths.items():
            v = fn(pid, sl)
            tot += 1
            ok += int(v == lab)
            kept += v
            if lab == 1:
                tp_kept += v
        rows.append((name, 100 * ok / tot, tp_kept, 100 * kept / tot))

    for m in sorted(probes):
        measure(lambda p, s, m=m: verdict(m, p, s), f"always-{m}")
    if routes:
        measure(lambda p, s: verdict(routes.get(p, ""), p, s), "GraphRouter (routed)")

    print("1. Per-path accuracy vs true-positive retention")
    print(f"{'configuration':26} {'per-path acc':>13} {'TP kept':>12} {'keeps':>8}")
    print("-" * 62)
    for name, acc, tpk, keep in sorted(rows, key=lambda r: r[1]):
        print(f"{name:26} {acc:12.1f}% {tpk:>8}/{n_tp} {keep:7.1f}%")
    print("\n   Read down the accuracy column and up the TP column: they invert.")
    print("   IRIS #Detected depends on TP retention, so maximising per-path")
    print("   accuracy actively costs detection.")

    if routes:
        by_label = defaultdict(Counter)
        for pid, (_, lab, _) in paths.items():
            if pid in routes:
                by_label[lab][routes[pid]] += 1
        print("\n2. What the router selects, split by ground truth")
        for lab, what in ((1, "TRUE vulnerabilities"), (0, "false alerts")):
            tot = sum(by_label[lab].values())
            if not tot:
                continue
            share = {k: f"{100*v/tot:.1f}%" for k, v in by_label[lab].most_common()}
            print(f"   on {what:22} ({tot:4} paths): {share}")

    if a.router_data and a.llm_json and os.path.exists(a.router_data):
        import csv as _csv
        desc = json.load(open(a.llm_json))
        order = [spec["short"] for spec in desc.values()]
        eff = defaultdict(list); eff_pos = defaultdict(list); eff_neg = defaultdict(list)
        per_query = defaultdict(dict)
        with open(a.router_data, newline="") as fh:
            for r in _csv.DictReader(fh):
                short = {k: v["short"] for k, v in desc.items()}[r["llm"]]
                e = float(r["effect"])
                eff[short].append(e)
                (eff_pos if r["ground_truth"] == "1" else eff_neg)[short].append(e)
                per_query[r["query_id"]][short] = e
        wins = Counter(max(d, key=d.get) for d in per_query.values() if d)
        nq = len(per_query)
        print("\n3. Effect per model, and how often each wins the argmax label")
        print(f"{'model':15} {'mean':>8} {'on TP':>8} {'on FP':>8} {'wins argmax':>13}")
        print("-" * 56)
        for m in order:
            if not eff[m]:
                continue
            print(f"{m:15} {sum(eff[m])/len(eff[m]):8.3f} "
                  f"{sum(eff_pos[m])/max(len(eff_pos[m]),1):8.3f} "
                  f"{sum(eff_neg[m])/max(len(eff_neg[m]),1):8.3f} "
                  f"{100*wins[m]/max(nq,1):12.1f}%")
        print("\n   A model can be best on true vulnerabilities and worst on")
        print("   average. The argmax label sees only the average, so that model")
        print("   is rarely the training target and the router rarely picks it.")


if __name__ == "__main__":
    main()
