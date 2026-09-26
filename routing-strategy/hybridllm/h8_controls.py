#!/usr/bin/env python3
"""H7 -- is the ROUTER earning the win, or is the floor and the budget?

h6 finds a cell (tau=0.85, k=10) that beats always-granite-8b on quality, cost
and detection at once. Two cheaper explanations have to die before that can be
called a routing result, because the LOPO AUC evidence (h5: 0.46-0.57 across
pairs, no config consistently ahead) says the router head is weak:

  1. **The floor alone.** The floor sends the k riskiest paths per project to
     the large model, ranked by the small model's p_vuln, and #Detected only
     needs one true path per project. Maybe the floor does all the work and the
     router contributes nothing. Control: `floor-only`, always-small everywhere
     except the floor.

  2. **The budget alone.** Sending 47% of paths to a better model buys quality
     whoever picks them. Control: `random`, the SAME number of large-model
     calls and the SAME floor, with the non-floor calls drawn uniformly at
     random, averaged over `--reps` draws with the spread reported.

A router that does not clear both controls is a budget knob with a classifier
bolted on. Reported side by side per cell, so the margin over each control is
readable directly rather than inferred across tables.

Scoring is IRIS's own `metrics()`, via h6's helpers, so no arithmetic is
duplicated here.
"""
from __future__ import annotations
import argparse, json, os, random, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h7_floor import load_metrics, score       # noqa: E402
from collections import defaultdict            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--slices", default="data/gr/slices.jsonl")
    ap.add_argument("--truth", default="data/gr/iris_truth.json")
    ap.add_argument("--score-subset", default="../IRIS/reproduction/score_subset.py")
    ap.add_argument("--cells", default="0.85:10,0.90:5,0.95:5,0.70:10",
                    help="tau:k cells to control, comma separated")
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
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

    by_proj = defaultdict(list)
    for uid, rec in pairs.items():
        by_proj[rec["project"]].append((rec["p_small"], uid))
    for v in by_proj.values():
        v.sort(reverse=True)

    def evaluate(to_large: set[str]):
        """Score one routing. `to_large` is the set of ids sent to the large
        model; everything else keeps the small model's verdict."""
        def keep(p):
            rec = pairs.get(p["unit_id"])
            if rec is None:
                return True                    # unsliceable -> kept, as in h3/h6
            k = "pred_large" if p["unit_id"] in to_large else "pred_small"
            return rec[k] == 1
        m = score(paths, keep, metrics, all_projects)
        m["gpu_s"] = sum(r["sec_large"] if u in to_large else r["sec_small"]
                         for u, r in pairs.items())
        m["large_pct"] = 100.0 * len(to_large) / max(len(pairs), 1)
        return m

    ref = {}
    for name, sel in ((f"always-{small}", set()),
                      (f"always-{large}", set(pairs))):
        ref[name] = evaluate(sel)

    hdr = (f"{'configuration':34} {'%large':>7} {'GPU-s':>9} {'#Det':>7} "
           f"{'AvgFDR%':>8} {'AvgF1':>7}")
    print("=" * len(hdr))
    print(f"controls -- small={small}, large={large}, router={sc['backbone']}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    lines = ["| Configuration | %large | GPU-s | #Det | AvgFDR% | AvgF1 |",
             "|---|---|---|---|---|---|"]

    def emit(name, m, f1_note=""):
        f1 = f"{m['avg_f1']:.3f}" if isinstance(m["avg_f1"], float) else m["avg_f1"]
        print(f"{name:34} {m['large_pct']:7.1f} {m['gpu_s']:9.1f} "
              f"{m['detected']:>3}/{m['n']:<3} {m['avg_fdr']:>8.2f} "
              f"{f1:>7}{f1_note}")
        lines.append(f"| {name} | {m['large_pct']:.1f} | {m['gpu_s']:.1f} | "
                     f"{m['detected']}/{m['n']} | {m['avg_fdr']:.2f} | "
                     f"{f1}{f1_note} |")

    for name, m in ref.items():
        emit(name, m)
    print("-" * len(hdr))
    lines.append("| | | | | | |")

    rng = random.Random(a.seed)
    for cell in a.cells.split(","):
        tau, k = cell.split(":")
        tau, k = float(tau), int(k)
        forced = {uid for v in by_proj.values() for _, uid in v[:k]}

        routed = {u for u in pairs if p_small.get(u, 0.0) < tau} | forced
        m_r = evaluate(routed)
        emit(f"router+floor  tau={tau:.2f} k={k}", m_r)

        m_f = evaluate(set(forced))
        emit(f"  floor-only (always-small) k={k}", m_f)

        pool = [u for u in pairs if u not in forced]
        n_extra = max(len(routed) - len(forced), 0)
        f1s, dets, gpus = [], [], []
        for _ in range(a.reps):
            pick = set(rng.sample(pool, n_extra)) | forced
            mm = evaluate(pick)
            f1s.append(mm["avg_f1"])
            dets.append(mm["detected"])
            gpus.append(mm["gpu_s"])
        sd = statistics.pstdev(f1s) if len(f1s) > 1 else 0.0
        m_rand = {"large_pct": 100.0 * (n_extra + len(forced)) / len(pairs),
                  "gpu_s": statistics.mean(gpus),
                  "detected": f"{statistics.mean(dets):.1f}",
                  "n": ref[f"always-{large}"]["n"],
                  "avg_fdr": float("nan"),
                  "avg_f1": f"{statistics.mean(f1s):.3f}+-{sd:.3f}"}
        print(f"{'  random, matched budget':34} {m_rand['large_pct']:7.1f} "
              f"{m_rand['gpu_s']:9.1f} {m_rand['detected']:>3}/{m_rand['n']:<3} "
              f"{'--':>8} {m_rand['avg_f1']:>7}  ({a.reps} draws)")
        lines.append(f"|   random, matched budget | {m_rand['large_pct']:.1f} | "
                     f"{m_rand['gpu_s']:.1f} | {m_rand['detected']}/{m_rand['n']} "
                     f"| -- | {m_rand['avg_f1']} |")

        beat_floor = m_r["avg_f1"] - m_f["avg_f1"]
        beat_rand = m_r["avg_f1"] - statistics.mean(f1s)
        z = beat_rand / sd if sd > 1e-9 else float("nan")
        print(f"{'  -> router margin':34} over floor-only "
              f"{beat_floor:+.3f}, over random {beat_rand:+.3f} "
              f"({z:+.1f} sd)")
        lines.append(f"| **margin over floor-only {beat_floor:+.3f}, over "
                     f"random {beat_rand:+.3f} ({z:+.1f} sd)** | | | | | |")
        print("-" * len(hdr))
        lines.append("| | | | | | |")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(
            f"# Controls for the Hybrid LLM router + project floor\n\n"
            f"small = `{small}`, large = `{large}`, router = `{sc['backbone']}`, "
            f"loss `{sc['loss_type']}` t={sc['match_t']:g}, "
            f"leave-one-project-out.\n\n"
            f"`floor-only` keeps the small model everywhere except the k "
            f"riskiest paths per project.\n`random, matched budget` spends the "
            f"same number of large-model calls, floor included, on paths drawn "
            f"uniformly at random, over {a.reps} draws (mean +- sd).\n"
            f"Metrics are IRIS Sec 3.6 via their own `score_subset.metrics`.\n\n"
            + "\n".join(lines) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
