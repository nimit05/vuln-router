#!/usr/bin/env python3
"""Score an IRIS run on a project subset using the paper's own metric definitions,
and compare against the paper's published per-project CSVs restricted to the SAME
subset (so the comparison is like-for-like).

Paper metrics (Sec 3.6):
  Rec(P)      = 1 if any detected path hits a known-vulnerable location
  #Detected   = sum of Rec(P)
  Prec(P)     = #VulPath(P) / |Paths(P)|
  AvgFDR      = mean over P with |Paths(P)|>0 of (1 - Prec(P))
  AvgF1       = mean over ALL P of 2*Prec*Rec/(Prec+Rec)
"""
import argparse, csv, json, os, sys

IRIS = os.path.expanduser("~/nimit/vuln-pred-results/IRIS")
CWES = ["022", "078", "079", "094"]


def metrics(records):
    """records: list of dicts with paths, tp_paths, recall."""
    n = len(records)
    detected = sum(1 for r in records if r["recall"])
    fdrs, f1s = [], []
    for r in records:
        paths, tp = r["paths"], r["tp_paths"]
        prec = (tp / paths) if paths > 0 else None
        rec = 1.0 if r["recall"] else 0.0
        if prec is not None:
            fdrs.append(1 - prec)
        p = prec if prec is not None else 0.0
        f1s.append(0.0 if (p + rec) == 0 else 2 * p * rec / (p + rec))
    return {
        "n": n,
        "detected": detected,
        "rate": 100.0 * detected / n if n else 0.0,
        "avg_fdr": 100.0 * sum(fdrs) / len(fdrs) if fdrs else float("nan"),
        "avg_f1": sum(f1s) / len(f1s) if f1s else float("nan"),
    }


def load_ours(run_id, slugs, cwe_of):
    """Two on-disk layouts:
      CodeQL baseline : output/<slug>/common/cwe-XXXwCodeQL/results.json      (flat)
      IRIS LLM run    : output/<slug>/<run_id>/cwe-XXXwLLM-final/results.json (nested)
    """
    out, missing = [], []
    for slug in slugs:
        cwe = cwe_of[slug].replace("CWE-", "")
        cands = [
            os.path.join(IRIS, "output", slug, run_id, f"cwe-{cwe}wLLM-final", "results.json"),
            os.path.join(IRIS, "output", slug, "common", f"cwe-{cwe}wCodeQL", "results.json"),
        ]
        found = next((c for c in cands if os.path.exists(c)), None)
        if not found:
            missing.append(slug)
            continue
        d = json.load(open(found))
        r = d.get("posthoc_filter_result") or d.get("vanilla_result") or d
        out.append({
            "slug": slug,
            "alerts": r.get("num_results", 0),
            "paths": r.get("num_paths", 0),
            "tp_paths": r.get("num_tp_paths_method", 0),
            "recall": bool(r.get("recall_method", False)),
        })
    return out, missing


def load_paper(csv_name, keys):
    recs = []
    for r in csv.DictReader(open(os.path.join(IRIS, "results", csv_name))):
        key = (r["CVE"], r["Tag"])
        if key not in keys:
            continue
        def num(x):
            try: return int(float(x))
            except: return 0
        recs.append({
            "slug": key,
            "paths": num(r["Paths"]),
            "tp_paths": num(r["TP Paths"]),
            "recall": r["Recall"] == "1",
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--subset", default="/tmp/subset.txt")
    ap.add_argument("--label", default="ours")
    ap.add_argument("--compare", nargs="*",
                    default=["CodeQL.csv", "IRIS+DeepSeekCoder-7B.csv", "IRIS+GPT-4.csv"])
    a = ap.parse_args()

    slugs = [l.strip() for l in open(a.subset) if l.strip()]
    info = {r["project_slug"]: r for r in
            csv.DictReader(open(f"{IRIS}/data/cwe-bench-java/data/project_info.csv"))}
    cwe_of = {s: info[s]["cwe_id"] for s in slugs if s in info}
    keys = {(info[s]["cve_id"], info[s]["github_tag"]) for s in slugs if s in info}

    ours, missing = load_ours(a.run_id, [s for s in slugs if s in cwe_of], cwe_of)
    if missing:
        print(f"[warn] no results for {len(missing)} project(s): {', '.join(missing[:6])}"
              + (" ..." if len(missing) > 6 else ""), file=sys.stderr)
    if not ours:
        print("No results found. Did the run complete?", file=sys.stderr)
        sys.exit(1)

    scored = [(a.label + f" (run={a.run_id})", metrics(ours))]
    # restrict paper numbers to the projects WE actually scored
    ours_keys = {(info[r["slug"]]["cve_id"], info[r["slug"]]["github_tag"]) for r in ours}
    for c in a.compare:
        path = os.path.join(IRIS, "results", c)
        if os.path.exists(path):
            scored.append((f"paper: {c[:-4]}", metrics(load_paper(c, ours_keys))))

    print(f"\n=== IRIS subset reproduction — {len(ours)} projects "
          f"(same projects for every row) ===")
    print(f"{'configuration':38} {'#Det':>6} {'rate%':>7} {'AvgFDR%':>9} {'AvgF1':>7}")
    print("-" * 72)
    for name, m in scored:
        print(f"{name:38} {m['detected']:>3}/{m['n']:<2} {m['rate']:>7.2f} "
              f"{m['avg_fdr']:>9.2f} {m['avg_f1']:>7.3f}")

    print("\n--- per-project (ours) ---")
    print(f"{'project':58} {'CWE':8} {'alerts':>7} {'paths':>6} {'tp':>4} {'det':>5}")
    for r in sorted(ours, key=lambda x: x["slug"]):
        print(f"{r['slug'][:58]:58} {cwe_of[r['slug']]:8} {r.get('alerts',0):>7} "
              f"{r['paths']:>6} {r['tp_paths']:>4} {str(r['recall']):>5}")


if __name__ == "__main__":
    main()
