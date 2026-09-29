#!/usr/bin/env python3
"""D1 gate -- can any model rank a real bug above a false alarm from the SAME project?

Within-project AUC: over every (true bug, false alarm) pair drawn from one
project, how often the model's P(vulnerable) is higher on the bug (ties 1/2).
Pooled over projects by pair count. Within-project is the number that matters:
a pooled AUC can be won by knowing which PROJECT has more bugs (ESAPI's base
rate is 0.50, ff4j's 0.13), which no router can use on a project it has not
seen (E13, E14).

Gate, fixed before the result: within-project AUC >= 0.65 and its 95% CI
(bootstrap over paths, stratified by project) excluding 0.5. Pass -> the full
think run is worth ~3 GPU-hours. Fail -> stop, no router can route to skill
that is not there.
"""
import argparse, json, math, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2_graphrouter_v2 import p_vuln, load_probe_file            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def within_auc(groups):
    """groups: list of (scores_pos, scores_neg) per project."""
    num = den = 0.0
    for sp, sn in groups:
        if len(sp) == 0 or len(sn) == 0:
            continue
        d = sp[:, None] - sn[None, :]
        num += (d > 0).sum() + 0.5 * (d == 0).sum()
        den += d.size
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=os.path.join(ROOT, "data/gr/d1_gate_ids.json"))
    ap.add_argument("--probe-dir", default=os.path.join(ROOT, "data/gr/probe5"))
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    ids = set(json.load(open(a.ids)))
    files = {os.path.basename(f).split("__")[0]: f for f in
             sorted(__import__("glob").glob(os.path.join(a.probe_dir, "*__units_paths.jsonl")))}
    for s in a.extra:
        n, f = s.split("=", 1); files[n] = f
    rng = np.random.default_rng(0)
    print(f"{'model':20} {'n':>4} {'yes%':>5} {'acc':>5} {'pooledAUC':>9} "
          f"{'withinAUC':>9} {'95% CI':>15} {'TPkept':>7} {'FPkept':>7} {'unparsed':>8}")
    for name, f in files.items():
        rows = {k: v for k, v in load_probe_file(f).items() if k in ids}
        if not rows:
            continue
        by = defaultdict(lambda: ([], []))
        allp, ally, yes, acc, un = [], [], 0, 0, 0
        for r in rows.values():
            p = p_vuln(r); g = int(r["gold_label"])
            by[r.get("project") or r.get("commit")][1 - g].append(p)
            allp.append(p); ally.append(g)
            yes += r.get("pred_label") == 1
            acc += r.get("pred_label") == g
            un += r.get("pred_label") is None
        groups = [(np.array(pos), np.array(neg)) for pos, neg in by.values()]
        w = within_auc(groups)
        bs = []
        for _ in range(a.boot):
            g2 = [(sp[rng.integers(0, len(sp), len(sp))] if len(sp) else sp,
                   sn[rng.integers(0, len(sn), len(sn))] if len(sn) else sn) for sp, sn in groups]
            bs.append(within_auc(g2))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        allp, ally = np.array(allp), np.array(ally)
        pooled = within_auc([(allp[ally == 1], allp[ally == 0])])
        n = len(rows)
        tpk = sum(1 for r in rows.values() if r["gold_label"] == 1 and r.get("pred_label") != 0)
        fpk = sum(1 for r in rows.values() if r["gold_label"] == 0 and r.get("pred_label") != 0)
        verdict = "PASS" if (w >= 0.65 and lo > 0.5) else ""
        print(f"{name:20} {n:4d} {100*yes/n:5.0f} {acc/n:5.2f} {pooled:9.3f} {w:9.3f} "
              f"[{lo:.3f},{hi:.3f}] {tpk:4d}/{int(ally.sum())} {fpk:4d}/{int((1-ally).sum())} "
              f"{un:8d} {verdict}")


if __name__ == "__main__":
    main()
