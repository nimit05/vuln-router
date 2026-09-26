#!/usr/bin/env python3
"""H4 -- every (small, large) pair, with the ceiling separated from the fit.

Hybrid LLM is a two-model method and the pool has five, so which pair is chosen
is a result, not a setup detail. For each ordered pair (small strictly cheaper
in measured GPU-seconds) this reports four numbers that answer four different
questions:

  always-small / always-large   is either endpoint already good enough?
  label-following               where would a PERFECT router on this objective
                                land? -- the ceiling the objective permits.
                                An ORACLE: computed from ground truth, never
                                achievable at inference, reported only as a
                                ceiling
  pair oracle                   where would a perfect router on ANY objective
                                land? -- the ceiling the model pool permits
  LOPO AUC                      can the objective's label actually be predicted
                                from a held-out project?

Splitting the ceiling from the fit is the whole point. A method can fail because
its target is worthless (GraphRouter, docs/06-objective-misalignment.md) or
because its target is valuable and unreachable. Those need different fixes and
the usual single routed row cannot tell them apart.

The AUC is reported twice, leave-one-project-out and a random in-project split.
The second is deliberately leaky and is here as a control: if it is high while
LOPO is at chance, the router is identifying the repository, not the path, and
any published number from an alert-level split is measuring that (protocol
section 4).
"""
from __future__ import annotations
import argparse, importlib.util, itertools, json, os, subprocess, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_labels import det_label, match_prob          # noqa: E402
from h2_train_router import featurize, fit_predict   # noqa: E402


def load_metrics(path: str):
    spec = importlib.util.spec_from_file_location("score_subset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.metrics


def score(paths, keep, metrics, all_projects):
    per = defaultdict(lambda: {"paths": 0, "tp_paths": 0})
    for p in paths:
        if keep(p):
            per[p["project"]]["paths"] += 1
            per[p["project"]]["tp_paths"] += p["label"]
    return metrics([{"paths": per.get(q, {"paths": 0})["paths"],
                     "tp_paths": per.get(q, {"tp_paths": 0})["tp_paths"],
                     "recall": per.get(q, {"tp_paths": 0})["tp_paths"] > 0}
                    for q in all_projects])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="data/gr/slices.jsonl")
    ap.add_argument("--truth", default="data/gr/iris_truth.json")
    ap.add_argument("--probe-dir", default="data/gr/probe5")
    ap.add_argument("--score-subset", default="../IRIS/reproduction/score_subset.py")
    ap.add_argument("--rho", type=float, default=10.0)
    ap.add_argument("--loss-type", default="det_2cls",
                    choices=["det_2cls", "prob_2cls"])
    ap.add_argument("--match-t", type=float, default=0.0)
    ap.add_argument("--models", default="qwen-1.5b,phi-3.8b,qwen-7b,"
                                        "granite-8b,deepseek-6.7b")
    ap.add_argument("--no-auc", action="store_true",
                    help="skip the two fits; ceilings only, seconds not minutes")
    ap.add_argument("--out", default="docs/RESULTS_HYBRIDLLM.md")
    a = ap.parse_args()

    metrics = load_metrics(a.score_subset)
    all_projects = sorted(json.load(open(a.truth)))
    paths = [{"unit_id": r["path_id"], "project": r["project"],
              "label": r["label"], "sliceable": r.get("sliceable", 1)}
             for r in (json.loads(l) for l in open(a.slices) if l.strip())]

    models = a.models.split(",")
    here = os.path.dirname(os.path.abspath(__file__))

    rows = []
    for m1, m2 in itertools.permutations(models, 2):
        tag = f"data/hl/pairs__{m1}__{m2}__rho{a.rho:g}.jsonl"
        if not os.path.exists(tag):
            subprocess.run([sys.executable, os.path.join(here, "h1_build_pairs.py"),
                            "--small", m1, "--large", m2, "--rho", str(a.rho),
                            "--probe-dir", a.probe_dir, "--slices", a.slices],
                           check=True, stdout=subprocess.DEVNULL)
        recs = {r["id"]: r for r in
                (json.loads(l) for l in open(tag) if l.strip())}
        mean_s = np.mean([r["sec_small"] for r in recs.values()])
        mean_l = np.mean([r["sec_large"] for r in recs.values()])
        if mean_s >= mean_l:
            continue                       # "small" must be the cheaper one

        def keep_with(key):
            return lambda p: (recs[p["unit_id"]][key] == 1
                              if p["unit_id"] in recs else True)

        lab = {u: det_label(r["candidates"][0]["scores"]["q"],
                            r["candidates"][1]["scores"]["q"], a.match_t)
               for u, r in recs.items()}

        def follow(p):
            if p["unit_id"] not in recs:
                return True
            k = "pred_small" if lab[p["unit_id"]] == 0 else "pred_large"
            return recs[p["unit_id"]][k] == 1

        def oracle(p):
            if p["unit_id"] not in recs:
                return True
            r, want = recs[p["unit_id"]], p["label"] == 1
            if (r["pred_small"] == 1) == want or (r["pred_large"] == 1) == want:
                return want
            return r["pred_small"] == 1

        m_small = score(paths, keep_with("pred_small"), metrics, all_projects)
        m_large = score(paths, keep_with("pred_large"), metrics, all_projects)
        m_foll = score(paths, follow, metrics, all_projects)
        m_orac = score(paths, oracle, metrics, all_projects)
        gpu_foll = sum(recs[u]["sec_small" if v == 0 else "sec_large"]
                       for u, v in lab.items())
        gpu_large = sum(r["sec_large"] for r in recs.values())

        auc_lopo = auc_rand = float("nan")
        if not a.no_auc:
            rr = list(recs.values())
            if a.loss_type == "det_2cls":
                y = np.array([1.0 - lab[r["id"]] for r in rr])
            else:
                y = np.array([match_prob(r["candidates"][0]["scores"]["q"],
                                         r["candidates"][1]["scores"]["q"],
                                         a.match_t) for r in rr])
            yb = (y >= 0.5).astype(int)
            if len(set(yb)) > 1:
                grp = np.array([r["project"] for r in rr])
                for name, sp in (("lopo", GroupKFold(
                                      n_splits=len(set(grp))).split(rr, y, grp)),
                                 ("rand", StratifiedKFold(
                                      5, shuffle=True, random_state=0).split(rr, yb))):
                    pr = np.zeros(len(rr))
                    for tr, te in sp:
                        Xtr, Xte = featurize([rr[i] for i in tr],
                                             [rr[i] for i in te], 20000)
                        pr[te] = fit_predict(Xtr, y[tr], Xte, 1.0, 0)
                    if name == "lopo":
                        auc_lopo = roc_auc_score(yb, pr)
                    else:
                        auc_rand = roc_auc_score(yb, pr)

        rows.append({
            "small": m1, "large": m2,
            "route_large_pct": 100.0 * sum(lab.values()) / len(lab),
            "gpu_saved_pct": 100.0 * (1 - gpu_foll / gpu_large),
            "f1_small": m_small["avg_f1"], "f1_large": m_large["avg_f1"],
            "f1_follow": m_foll["avg_f1"], "f1_oracle": m_orac["avg_f1"],
            "det_small": m_small["detected"], "det_large": m_large["detected"],
            "det_follow": m_foll["detected"], "n": m_foll["n"],
            "auc_lopo": auc_lopo, "auc_rand": auc_rand})
        r = rows[-1]
        print(f"{m1:14}->{m2:14} follow F1={r['f1_follow']:.3f} "
              f"(small {r['f1_small']:.3f} / large {r['f1_large']:.3f} / "
              f"oracle {r['f1_oracle']:.3f})  "
              f"AUC lopo={r['auc_lopo']:.3f} rand={r['auc_rand']:.3f}")

    rows.sort(key=lambda r: -r["f1_follow"])
    hdr = ("| small | large | %routed large | GPU-s saved | AvgF1 small | "
           "AvgF1 large | AvgF1 label-following | AvgF1 pair-oracle | "
           "AUC LOPO | AUC random (leaky) |")
    lines = [hdr, "|" + "---|" * 10]
    for r in rows:
        lines.append(
            f"| {r['small']} | {r['large']} | {r['route_large_pct']:.1f}% | "
            f"{r['gpu_saved_pct']:.1f}% | {r['f1_small']:.3f} | "
            f"{r['f1_large']:.3f} | **{r['f1_follow']:.3f}** | "
            f"{r['f1_oracle']:.3f} | {r['auc_lopo']:.3f} | {r['auc_rand']:.3f} |")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(
        f"# Hybrid LLM on IRIS / CWE-Bench-Java -- every model pair\n\n"
        f"Quality is the asymmetric security loss of docs/01-problem.md 1.5 at "
        f"rho = c_FN/c_FP = {a.rho:g}; label is upstream's `{a.loss_type}` at "
        f"tolerance t = {a.match_t:g}.\nMetrics are IRIS Sec 3.6 via their own "
        f"`score_subset.metrics`, over {rows[0]['n'] if rows else 0} projects.\n\n"
        f"`label-following` is a perfect router on Hybrid LLM's own target: the "
        f"ceiling the OBJECTIVE allows.\n`pair-oracle` is a perfect router on "
        f"any target: the ceiling the MODEL POOL allows.\n`AUC random` splits "
        f"inside a project and is leaky by construction; it is the control for "
        f"`AUC LOPO`.\n\n" + "\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
