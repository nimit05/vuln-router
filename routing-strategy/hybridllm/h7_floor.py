#!/usr/bin/env python3
"""H6 -- the project floor: buy back #Detected that a per-path router sells.

The router in h5 reaches AvgF1 0.231 against always-granite's 0.207 while
spending 11% fewer GPU-seconds, and pays for it by dropping to 9/16 projects
detected. That trade is invisible to the objective: Hybrid LLM's label is per
path, IRIS's #Detected is per PROJECT and needs only ONE true-positive path to
survive in each. A router can therefore be right about nearly every path in a
project and still lose the project, which is the same shape of failure
docs/06-objective-misalignment.md diagnosed in GraphRouter, one level down.

The floor is the cheapest possible repair and it needs no retraining:

    in each project, the k paths the SMALL model called most likely vulnerable
    go to the large model regardless of r(x)

k is a second knob beside tau. It costs at most k * (c_large - c_small) per
project -- 16 projects * k paths, against 2,257 paths total -- so even k = 10
is a rounding error on the bill, and it is aimed exactly where #Detected is
won or lost. Ranking by the small model's own p_vuln is deliberate: the floor
must be computable at inference, and p_vuln is the only per-path risk estimate
available before the large model has been called.

Reported as a grid over (tau, k) so the reader can see the floor's price, not
just its best cell. Scoring is IRIS's own `metrics()`, same conventions as h3:
every scored project stays in the denominator, and an unsliceable path is kept
because no model ever saw it.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from collections import defaultdict


def load_metrics(path: str):
    spec = importlib.util.spec_from_file_location("score_subset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.metrics


def score(paths, keep_fn, metrics, all_projects):
    per = defaultdict(lambda: {"paths": 0, "tp_paths": 0})
    for p in paths:
        if keep_fn(p):
            per[p["project"]]["paths"] += 1
            per[p["project"]]["tp_paths"] += p["label"]
    return metrics([{"paths": per.get(q, {"paths": 0})["paths"],
                     "tp_paths": per.get(q, {"tp_paths": 0})["tp_paths"],
                     "recall": per.get(q, {"tp_paths": 0})["tp_paths"] > 0}
                    for q in all_projects])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="h5 (or h2) output")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--slices", default="data/gr/slices.jsonl")
    ap.add_argument("--truth", default="data/gr/iris_truth.json")
    ap.add_argument("--score-subset", default="../IRIS/reproduction/score_subset.py")
    ap.add_argument("--taus", default="0.80,0.85,0.90,0.95,1.00")
    ap.add_argument("--ks", default="0,1,2,3,5,10")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    metrics = load_metrics(a.score_subset)
    all_projects = sorted(json.load(open(a.truth)))
    sc = json.load(open(a.scores))
    small, large = sc["small"], sc["large"]
    p_small = sc["p_small_suffices"]
    pairs = {r["id"]: r for r in
             (json.loads(l) for l in open(a.pairs or sc["pairs"]) if l.strip())}

    paths = [{"unit_id": r["path_id"], "project": r["project"],
              "label": r["label"], "sliceable": r.get("sliceable", 1)}
             for r in (json.loads(l) for l in open(a.slices) if l.strip())]

    # per project, path ids ordered by the SMALL model's risk estimate, highest
    # first. Computed once: it does not depend on tau or k.
    by_proj = defaultdict(list)
    for uid, rec in pairs.items():
        by_proj[rec["project"]].append((rec["p_small"], uid))
    for v in by_proj.values():
        v.sort(reverse=True)

    def run(tau: float, k: int):
        forced = {uid for v in by_proj.values() for _, uid in v[:k]}
        routes, secs, n_large = {}, 0.0, 0
        for uid, rec in pairs.items():
            use_small = (p_small.get(uid, 0.0) >= tau) and uid not in forced
            routes[uid] = use_small
            secs += rec["sec_small"] if use_small else rec["sec_large"]
            n_large += 0 if use_small else 1

        def keep(p):
            rec = pairs.get(p["unit_id"])
            if rec is None:
                return True                       # unsliceable -> kept, as in h3
            key = "pred_small" if routes[p["unit_id"]] else "pred_large"
            return rec[key] == 1

        m = score(paths, keep, metrics, all_projects)
        m["large_pct"] = 100.0 * n_large / max(len(routes), 1)
        m["gpu_s"] = secs
        m["forced"] = len(forced)
        return m

    ref = {}
    for key, name in (("pred_small", small), ("pred_large", large)):
        m = score(paths, lambda p, k=key: (pairs[p["unit_id"]][k] == 1
                                           if p["unit_id"] in pairs else True),
                  metrics, all_projects)
        sk = "sec_small" if key == "pred_small" else "sec_large"
        m["gpu_s"] = sum(r[sk] for r in pairs.values())
        ref[f"always-{name}"] = m

    hdr = (f"{'configuration':30} {'forced':>7} {'%large':>7} {'GPU-s':>9} "
           f"{'#Det':>7} {'AvgFDR%':>8} {'AvgF1':>7}")
    print("=" * len(hdr))
    print(f"project floor -- small={small}, large={large}, "
          f"router={sc['backbone']}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    lines = ["| Configuration | forced | %large | GPU-s | #Det | AvgFDR% | AvgF1 |",
             "|---|---|---|---|---|---|---|"]

    def emit(name, m, forced="--"):
        print(f"{name:30} {str(forced):>7} {m['large_pct'] if 'large_pct' in m else 0.0:7.1f} "
              f"{m['gpu_s']:9.1f} {m['detected']:>3}/{m['n']:<3} "
              f"{m['avg_fdr']:>8.2f} {m['avg_f1']:>7.3f}")
        lines.append(f"| {name} | {forced} | "
                     f"{m.get('large_pct', 0.0):.1f} | {m['gpu_s']:.1f} | "
                     f"{m['detected']}/{m['n']} | {m['avg_fdr']:.2f} | "
                     f"{m['avg_f1']:.3f} |")

    ref["always-" + small]["large_pct"] = 0.0
    ref["always-" + large]["large_pct"] = 100.0
    for name, m in ref.items():
        emit(name, m)
    print("-" * len(hdr))
    lines.append("| | | | | | | |")

    best = None
    for tau in [float(t) for t in a.taus.split(",")]:
        for k in [int(x) for x in a.ks.split(",")]:
            m = run(tau, k)
            emit(f"tau={tau:.2f} k={k}", m, m["forced"])
            # "wins" = beats always-large on BOTH axes, detection not sacrificed
            big = ref["always-" + large]
            if (m["avg_f1"] > big["avg_f1"] and m["gpu_s"] < big["gpu_s"]
                    and m["detected"] >= big["detected"]):
                if best is None or m["avg_f1"] > best[1]["avg_f1"]:
                    best = ((tau, k), m)
        lines.append("| | | | | | | |")
    print("-" * len(hdr))
    if best:
        (tau, k), m = best
        print(f"dominates always-{large} on quality, cost AND detection: "
              f"tau={tau:.2f} k={k}  AvgF1={m['avg_f1']:.3f} "
              f"({big['avg_f1']:.3f})  GPU-s={m['gpu_s']:.1f} "
              f"({big['gpu_s']:.1f})  #Det={m['detected']}/{m['n']}")
    else:
        print(f"no (tau, k) beats always-{large} on quality, cost and "
              f"detection at once")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(
            f"# Project floor on the Hybrid LLM router\n\n"
            f"small = `{small}`, large = `{large}`, router = `{sc['backbone']}`, "
            f"loss `{sc['loss_type']}` t={sc['match_t']:g}, "
            f"leave-one-project-out.\n`forced` is the number of paths sent to "
            f"the large model by the floor regardless of the router "
            f"(top-k per project by the small model's p_vuln).\nMetrics are "
            f"IRIS Sec 3.6 via their own `score_subset.metrics`.\n\n"
            + "\n".join(lines) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
