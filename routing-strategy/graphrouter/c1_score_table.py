#!/usr/bin/env python3
"""C1 -- the deliverable. One table, our router against IRIS, their metrics.

The framing: our router REPLACES IRIS's posthoc LLM filter. Both stages consume
the same CodeQL(+LLM sources/sinks) alerts and both emit a subset of them, so a
configuration here is nothing more than a keep/drop decision over the 2,259
dataflow paths that `b1_validate.py` proved are IRIS's own paths.

    keep(path) = the chosen model answered "vulnerable"

Metrics are IRIS's, and the arithmetic is literally theirs: `metrics()` is
imported from `reproduction/score_subset.py` rather than reimplemented, so no
rounding or averaging convention can drift between our rows and their rows.

    Rec(P)    = 1 if any KEPT path in project P is a true positive
    Prec(P)   = kept true positives / kept paths        (undefined when 0 kept)
    AvgFDR    = mean over P with kept > 0 of (1 - Prec)
    AvgF1     = mean over ALL P of 2*Prec*Rec/(Prec+Rec)

Two conventions worth stating because both move numbers:

* **Every scored project stays in the denominator**, including the 6 with no
  true-positive path and the 2 with no paths at all. AvgF1 is a mean over ALL
  projects; dropping the empty ones would inflate every row equally and make the
  comparison against the published table meaningless.
* **A path with no slice is kept.** Two paths could not be sliced, so no model
  ever saw them. Keeping them is what a filter that never ran would do; dropping
  them would hand the router two free true-negatives it never earned.

Rows produced:
  no filter          every path kept -- IRIS's alerts before any LLM stage
  always-<model>     one model decides every path (5 rows)
  routed             the router picks one model per path
  oracle             per-path best case: any model correct on that path wins
  paper: <name>      the published per-project CSVs, restricted to OUR projects
"""
from __future__ import annotations
import argparse, csv, glob, importlib.util, json, os, sys
from collections import defaultdict


def load_metrics(score_subset_py: str):
    """Import IRIS's own metrics() so the arithmetic cannot drift from theirs."""
    spec = importlib.util.spec_from_file_location("score_subset", score_subset_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.metrics


def load_probes(probe_dir: str, suffix: str) -> dict[str, dict[str, dict]]:
    """model -> unit_id -> row. Only rows with in_tokens > 0 count: a failed
    request is still written to disk (deliberately, gate G2 needs the failures)
    and would otherwise read as a confident 'not vulnerable'."""
    out = {}
    for path in sorted(glob.glob(os.path.join(probe_dir, f"*__{suffix}.jsonl"))):
        model = os.path.basename(path).split("__")[0]
        rows, dropped = {}, 0
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("in_tokens"):
                dropped += 1
                continue
            rows[r["unit_id"]] = r
        out[model] = rows
        print(f"  {model:14} {len(rows):>5} valid rows"
              + (f"  ({dropped} failed requests ignored)" if dropped else ""))
    return out


def score(paths: list[dict], keep_fn, metrics) -> dict:
    """paths: every path IRIS counted, each with project + label."""
    per = defaultdict(lambda: {"paths": 0, "tp_paths": 0})
    for p in paths:
        if not keep_fn(p):
            continue
        per[p["project"]]["paths"] += 1
        per[p["project"]]["tp_paths"] += p["label"]
    recs = []
    for proj in ALL_PROJECTS:                    # empty projects stay counted
        d = per.get(proj, {"paths": 0, "tp_paths": 0})
        recs.append({"paths": d["paths"], "tp_paths": d["tp_paths"],
                     "recall": d["tp_paths"] > 0})
    return metrics(recs)


def paper_rows(iris_repo: str, project_info: str, projects: list[str],
               metrics, names: list[str]) -> list[tuple[str, dict]]:
    """The published per-project CSVs, restricted to the projects we scored."""
    info = {r["project_slug"]: r for r in csv.DictReader(open(project_info))}
    keys = {(info[s]["cve_id"], info[s]["github_tag"]) for s in projects
            if s in info}
    out = []
    for name in names:
        path = os.path.join(iris_repo, "results", name)
        if not os.path.exists(path):
            continue
        recs = []
        for r in csv.DictReader(open(path)):
            if (r["CVE"], r["Tag"]) not in keys:
                continue
            def num(x):
                try:
                    return int(float(x))
                except (TypeError, ValueError):
                    return 0
            recs.append({"paths": num(r["Paths"]), "tp_paths": num(r["TP Paths"]),
                         "recall": r["Recall"] == "1"})
        if recs:
            out.append((f"paper: {name[:-4]}", metrics(recs)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="data/gr/slices.jsonl",
                    help="B3 output: every path, sliceable or not")
    ap.add_argument("--truth", default="data/gr/iris_truth.json")
    ap.add_argument("--probe-dir", default="data/gr/probe")
    ap.add_argument("--suffix", default="units_paths",
                    help="probe files are <model>__<suffix>.jsonl")
    ap.add_argument("--routes", default=None,
                    help="JSON {unit_id: model} from the trained router")
    ap.add_argument("--routes-label", default="GraphRouter (routed)",
                    help="row name for --routes; any router emitting this file "
                         "format scores here, so the row must say which one")
    ap.add_argument("--iris-repo", default="../IRIS")
    ap.add_argument("--project-info", default="data/bench/project_info.csv")
    ap.add_argument("--score-subset", default="../IRIS/reproduction/score_subset.py")
    ap.add_argument("--out", default="docs/RESULTS_TABLE.md")
    a = ap.parse_args()

    metrics = load_metrics(a.score_subset)

    global ALL_PROJECTS
    truth = json.load(open(a.truth))
    ALL_PROJECTS = sorted(truth)

    paths = []
    for line in open(a.slices):
        if not line.strip():
            continue
        r = json.loads(line)
        paths.append({"unit_id": r["path_id"], "project": r["project"],
                      "label": r["label"], "sliceable": r.get("sliceable", 1)})
    print(f"{len(paths)} paths over {len(ALL_PROJECTS)} projects, "
          f"{sum(p['label'] for p in paths)} true positives\n")

    print("probe files:")
    probes = load_probes(a.probe_dir, a.suffix)
    models = sorted(probes)
    if not models:
        print(f"\nno probe files matched {a.probe_dir}/*__{a.suffix}.jsonl",
              file=sys.stderr)
        sys.exit(1)

    def says_vuln(model: str, p: dict) -> bool:
        if not p["sliceable"]:
            return True                       # never seen by a model -> kept
        r = probes[model].get(p["unit_id"])
        if r is None:
            return True                       # unprobed -> kept, same reason
        return r["pred_label"] == 1

    rows = [("ours: no filter (all paths)", score(paths, lambda p: True, metrics))]
    for m in models:
        rows.append((f"ours: always-{m}",
                     score(paths, lambda p, m=m: says_vuln(m, p), metrics)))

    if a.routes and os.path.exists(a.routes):
        routes = json.load(open(a.routes))
        def routed(p):
            m = routes.get(p["unit_id"])
            return says_vuln(m, p) if m in probes else True
        rows.append((f"ours: {a.routes_label}", score(paths, routed, metrics)))
    else:
        print("\n[note] no --routes file yet; the routed row is omitted.")

    def oracle(p):
        """Best case for per-path routing: if ANY model gets this path right,
        a perfect router would have picked it. Upper bound, not a result."""
        if not p["sliceable"]:
            return True
        want = p["label"] == 1
        if any(says_vuln(m, p) == want for m in models):
            return want
        return says_vuln(models[0], p)
    rows.append(("ours: per-path oracle (upper bound)", score(paths, oracle, metrics)))

    rows += paper_rows(a.iris_repo, a.project_info, ALL_PROJECTS, metrics,
                       ["CodeQL.csv", "IRIS+DeepSeekCoder-7B.csv", "IRIS+GPT-4.csv"])

    n = len(ALL_PROJECTS)
    hdr = f"{'configuration':38} {'#Det':>7} {'rate%':>7} {'AvgFDR%':>9} {'AvgF1':>7}"
    print("\n" + "=" * len(hdr))
    print(f"GraphRouter vs IRIS -- {n} projects, IRIS's metrics, same paths")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    lines = ["| Configuration | #Det | rate% | AvgFDR% | AvgF1 |",
             "|---|---|---|---|---|"]
    for name, m in rows:
        print(f"{name:38} {m['detected']:>3}/{m['n']:<3} {m['rate']:>7.2f} "
              f"{m['avg_fdr']:>9.2f} {m['avg_f1']:>7.3f}")
        lines.append(f"| {name} | {m['detected']}/{m['n']} | {m['rate']:.2f} "
                     f"| {m['avg_fdr']:.2f} | {m['avg_f1']:.3f} |")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write(
        f"# GraphRouter vs IRIS on CWE-Bench-Java\n\n"
        f"{n} projects, {len(paths)} dataflow paths, "
        f"{sum(p['label'] for p in paths)} true-positive paths.\n"
        f"Metrics are IRIS Sec 3.6, computed by their own `score_subset.metrics`.\n"
        f"Paper rows are the published per-project CSVs restricted to these "
        f"same {n} projects.\n\n" + "\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
