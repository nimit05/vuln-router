#!/usr/bin/env python3
"""THE GATE. Reconcile our path set against IRIS's own accounting, per project.

Nothing downstream -- B3 slicing, B6 probing, B7 training, the final table --
means anything unless the candidate stream we route over is the SAME stream IRIS
scored. This compares, per project:

    paths   ours (rows in paths_labelled.jsonl)  vs  num_paths
    tp      ours (label == 1)                    vs  num_tp_paths_method
    recall  ours (any tp)                        vs  recall_method

Truth comes from `b0_iris_truth.py`, which runs IRIS's OWN evaluator over the
current SARIFs -- NOT from the cached `results.json` next to them. Those caches
are stale: they describe an earlier, smaller query run and disagree with their
own SARIFs on 6 of 16 projects. Reconciling against them would mark a correct
extractor as broken and would put the IRIS baseline rows of the final table on a
different alert set from the router rows.

A project IRIS scored with zero paths is a match when we emit zero rows. It still
belongs in the denominator: AvgF1 is a mean over ALL projects, so dropping the
empty ones would silently inflate every configuration in the table.

Exit code 0 only when every project matches on all three. A non-zero exit is a
hard stop, not a warning.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default=os.path.expanduser(
        "~/nimit/graphrouter/data/paths_labelled.jsonl"))
    ap.add_argument("--truth", default=os.path.expanduser(
        "~/nimit/graphrouter/data/iris_truth.json"))
    a = ap.parse_args()

    if not os.path.exists(a.truth):
        print(f"missing {a.truth} -- run b0_iris_truth.py first", file=sys.stderr)
        sys.exit(2)
    truth = json.load(open(a.truth))

    mine = defaultdict(lambda: {"paths": 0, "tp": 0})
    for line in open(a.labelled):
        if not line.strip():
            continue
        r = json.loads(line)
        mine[r["project"]]["paths"] += 1
        mine[r["project"]]["tp"] += r.get("label", 0)

    hdr = (f"{'project':46} {'paths ours/IRIS':>19} {'tp ours/IRIS':>16} "
           f"{'recall':>14}  ok")
    print(hdr)
    print("-" * len(hdr))
    bad = 0
    for p in sorted(set(truth) | set(mine)):
        t = truth.get(p)
        if t is None:
            print(f"{p[:46]:46} {'-- not scored by IRIS --':>19}")
            bad += 1
            continue
        m = mine.get(p, {"paths": 0, "tp": 0})     # zero-path projects are real
        r_ours, r_iris = bool(m["tp"]), bool(t["recall_method"])
        ok = (m["paths"] == t["num_paths"]
              and m["tp"] == t["num_tp_paths_method"]
              and r_ours == r_iris)
        bad += 0 if ok else 1
        print(f"{p[:46]:46} {m['paths']:>8}/{t['num_paths']:<10} "
              f"{m['tp']:>7}/{t['num_tp_paths_method']:<8} "
              f"{str(r_ours):>6}/{str(r_iris):<7} {'OK' if ok else 'MISMATCH'}")

    print(f"\ntotals: paths {sum(v['paths'] for v in mine.values())}"
          f"/{sum(t['num_paths'] for t in truth.values())}   "
          f"tp {sum(v['tp'] for v in mine.values())}"
          f"/{sum(t['num_tp_paths_method'] for t in truth.values())}   "
          f"projects {len(truth)}")
    if bad:
        print(f"\nGATE FAILED: {bad} project(s) do not reconcile. Do not run B3.",
              file=sys.stderr)
        sys.exit(1)
    print("\nGATE PASSED: our path set is IRIS's path set, project by project.")


if __name__ == "__main__":
    main()
