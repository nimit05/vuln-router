#!/usr/bin/env python3
"""H3 -- sweep Hybrid LLM's threshold and score every point with IRIS's metrics.

This is the deliverable. Hybrid LLM does not produce a routing, it produces a
FRONTIER: the router emits P(small suffices) per path and a threshold turns that
into a routing, so quality is traded for cost at test time without retraining
(the paper's central claim, ICLR 2024 sec 4).

    route(x) = small   if  P(small suffices | x) >= tau
               large   otherwise

tau = 0 is always-small, tau = 1 is always-large, and the interesting question
is whether anything in between beats both. Cost is GPU-seconds actually
measured by the probe, not a price sheet (docs/01-problem.md 1.4).

Scoring is IRIS's own `metrics()` imported from their `score_subset.py`, on the
same 2,259 paths `b1_validate.py` reconciled against their evaluator, with
c1_score_table's two conventions preserved exactly:

  * every scored project stays in the denominator, including the 6 with no
    true-positive path and the 2 with no paths;
  * an unsliceable path is KEPT, because no model ever saw it.

One reference row is not in the paper and matters more than the sweep:
**label-following**, the routing you get from Hybrid LLM's own `det_2cls`
target computed from the GROUND-TRUTH path labels instead of predicted. It is
an oracle -- it reads the answer and is not achievable at inference -- and it is
what a perfect router on this objective would achieve. Comparing it against
the sweep separates the two ways a router can fail -- an objective that points
somewhere useless, versus an objective that points somewhere real that the
predictor cannot find. GraphRouter failed the first way
(docs/06-objective-misalignment.md); this row is how we test the second.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_labels import det_label     # noqa: E402


def load_metrics(path: str):
    spec = importlib.util.spec_from_file_location("score_subset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.metrics


def score(paths, keep_fn, metrics, all_projects):
    per = defaultdict(lambda: {"paths": 0, "tp_paths": 0})
    for p in paths:
        if not keep_fn(p):
            continue
        per[p["project"]]["paths"] += 1
        per[p["project"]]["tp_paths"] += p["label"]
    recs = []
    for proj in all_projects:
        d = per.get(proj, {"paths": 0, "tp_paths": 0})
        recs.append({"paths": d["paths"], "tp_paths": d["tp_paths"],
                     "recall": d["tp_paths"] > 0})
    return metrics(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="h2 output")
    ap.add_argument("--pairs", default=None, help="defaults to the one in --scores")
    ap.add_argument("--slices", default="data/gr/slices.jsonl")
    ap.add_argument("--truth", default="data/gr/iris_truth.json")
    ap.add_argument("--score-subset", default="../IRIS/reproduction/score_subset.py")
    ap.add_argument("--steps", type=int, default=21)
    ap.add_argument("--out", default=None)
    ap.add_argument("--routes-at", type=float, default=None,
                    help="also write a c1-compatible routes.json at this tau")
    a = ap.parse_args()

    metrics = load_metrics(a.score_subset)
    all_projects = sorted(json.load(open(a.truth)))

    sc = json.load(open(a.scores))
    small, large = sc["small"], sc["large"]
    p_small = sc["p_small_suffices"]
    pairs = {r["id"]: r for r in
             (json.loads(l) for l in open(a.pairs or sc["pairs"]) if l.strip())}

    paths = []
    for line in open(a.slices):
        if not line.strip():
            continue
        r = json.loads(line)
        paths.append({"unit_id": r["path_id"], "project": r["project"],
                      "label": r["label"], "sliceable": r.get("sliceable", 1)})

    def verdict(model_key: str, p) -> bool:
        """Keep the path iff the chosen model called it vulnerable."""
        rec = pairs.get(p["unit_id"])
        if rec is None:
            return True                       # unsliceable / unprobed -> kept
        return rec[model_key] == 1

    def cost_of(model_key: str, p) -> float:
        rec = pairs.get(p["unit_id"])
        return 0.0 if rec is None else rec[model_key]

    def at(tau: float):
        """keep-decision, share routed large, and GPU-seconds spent."""
        n_large = 0.0
        secs = 0.0
        routes = {}
        for p in paths:
            rec = pairs.get(p["unit_id"])
            if rec is None:
                continue
            use_small = p_small.get(p["unit_id"], 0.0) >= tau
            routes[p["unit_id"]] = small if use_small else large
            n_large += 0.0 if use_small else 1.0
            secs += rec["sec_small"] if use_small else rec["sec_large"]
        m = score(paths, lambda p: verdict(
            "pred_small" if routes.get(p["unit_id"]) == small else "pred_large", p),
            metrics, all_projects)
        m["large_pct"] = 100.0 * n_large / max(len(routes), 1)
        m["gpu_s"] = secs
        return m, routes

    ref = []
    ref.append(("no filter (all paths)",
                score(paths, lambda p: True, metrics, all_projects), 0.0, 0.0))
    for key, name in (("pred_small", small), ("pred_large", large)):
        m = score(paths, lambda p, k=key: verdict(k, p), metrics, all_projects)
        sk = "sec_small" if key == "pred_small" else "sec_large"
        ref.append((f"always-{name}", m,
                    0.0 if key == "pred_small" else 100.0,
                    sum(cost_of(sk, p) for p in paths)))

    lab = {}
    for uid, rec in pairs.items():
        lab[uid] = det_label(rec["candidates"][0]["scores"]["q"],
                             rec["candidates"][1]["scores"]["q"], sc["match_t"])
    def follow(p):
        rec = pairs.get(p["unit_id"])
        if rec is None:
            return True
        return rec["pred_small" if lab[p["unit_id"]] == 0 else "pred_large"] == 1
    n_lg = sum(1 for v in lab.values() if v == 1)
    ref.append((f"label-following (perfect router)",
                score(paths, follow, metrics, all_projects),
                100.0 * n_lg / max(len(lab), 1),
                sum(pairs[u]["sec_small" if v == 0 else "sec_large"]
                    for u, v in lab.items())))

    def oracle(p):
        rec = pairs.get(p["unit_id"])
        if rec is None:
            return True
        want = p["label"] == 1
        if (rec["pred_small"] == 1) == want or (rec["pred_large"] == 1) == want:
            return want
        return rec["pred_small"] == 1
    ref.append(("pair oracle (upper bound)",
                score(paths, oracle, metrics, all_projects), float("nan"),
                float("nan")))

    taus = [i / (a.steps - 1) for i in range(a.steps)]
    sweep = [(t, *at(t)) for t in taus]

    hdr = (f"{'configuration':34} {'%large':>7} {'GPU-s':>9} "
           f"{'#Det':>7} {'AvgFDR%':>8} {'AvgF1':>7}")
    print("\n" + "=" * len(hdr))
    print(f"Hybrid LLM ({sc['loss_type']}, t={sc['match_t']:g}) -- "
          f"small={small}, large={large}")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    lines = ["| Configuration | %large | GPU-s | #Det | AvgFDR% | AvgF1 |",
             "|---|---|---|---|---|---|"]

    def emit(name, m, pl, gs):
        pls = "  --  " if pl != pl else f"{pl:6.1f}"
        gss = "   --   " if gs != gs else f"{gs:9.1f}"
        print(f"{name:34} {pls:>7} {gss:>9} {m['detected']:>3}/{m['n']:<3} "
              f"{m['avg_fdr']:>8.2f} {m['avg_f1']:>7.3f}")
        lines.append(f"| {name} | {pls.strip()} | {gss.strip()} | "
                     f"{m['detected']}/{m['n']} | {m['avg_fdr']:.2f} "
                     f"| {m['avg_f1']:.3f} |")

    for name, m, pl, gs in ref:
        emit(name, m, pl, gs)
    print("-" * len(hdr))
    lines.append("| | | | | | |")
    best = None
    for t, m, _ in sweep:
        emit(f"hybrid tau={t:.2f}", m, m["large_pct"], m["gpu_s"])
        if best is None or m["avg_f1"] > best[1]["avg_f1"]:
            best = (t, m)
    print("-" * len(hdr))
    print(f"best AvgF1 on the sweep: tau={best[0]:.2f}  "
          f"AvgF1={best[1]['avg_f1']:.3f}  #Det={best[1]['detected']}  "
          f"{best[1]['large_pct']:.1f}% routed large")

    if a.routes_at is not None:
        _, routes = at(a.routes_at)
        rp = os.path.splitext(a.scores)[0] + f".routes_tau{a.routes_at:g}.json"
        json.dump(routes, open(rp, "w"))
        print(f"wrote {rp} (feed to graphrouter/c1_score_table.py --routes)")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        open(a.out, "w").write(
            f"# Hybrid LLM on IRIS / CWE-Bench-Java\n\n"
            f"Router: `{sc['loss_type']}`, quality-gap tolerance "
            f"t={sc['match_t']:g}, backbone `{sc['backbone']}`, "
            f"leave-one-project-out.\nsmall = `{small}`, large = `{large}`. "
            f"Metrics are IRIS Sec 3.6 via their own `score_subset.metrics`.\n\n"
            + "\n".join(lines) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
