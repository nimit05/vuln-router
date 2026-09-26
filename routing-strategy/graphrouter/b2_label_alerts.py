#!/usr/bin/env python3
"""B2 -- label each alert against the CVE fix commit.

An alert is a TRUE POSITIVE when its dataflow touches the method the CVE fix
actually changed. That is CWE-Bench-Java's own ground truth (`fix_info.csv`
records file, method and line range per fixing commit) and it is the rule IRIS
scores against, so the labels here are comparable to published numbers.

Two decisions worth stating, because both change the number:

* **Any step counts, not just the sink.** A dataflow path that runs through the
  vulnerable method is reporting the right defect even when CodeQL names a sink
  further downstream. Scoring sink-only would mark correct detections as false.
* **Method granularity, not line.** The fix commit's line numbers refer to the
  PATCHED tree; the database is built from the buggy tag, so lines drift. The
  enclosing method range is the finest granularity that survives that drift.

The true-positive rate is the headline output. Each project carries one CVE, so
if it comes back near zero the candidate-LLM edges are almost constant and
nothing downstream can learn from them -- that is risk 1 in CHECKPOINTS.md and
this script is what tests it.
"""
from __future__ import annotations
import argparse, csv, json, os
from collections import Counter, defaultdict


def load_fixes(path: str) -> dict[str, list[dict]]:
    fixes = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                start = int(row["method_start"]); end = int(row["method_end"])
            except (ValueError, KeyError, TypeError):
                continue
            fixes[row["project_slug"]].append(
                dict(file=row["file"], method=row.get("method"),
                     start=start, end=end,
                     cls=row.get("class"), cve=row.get("cve_id")))
    return fixes


def load_project_cwe(path: str) -> dict[str, str]:
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            cwe = (row.get("cwe_id") or "").strip()
            if cwe.upper().startswith("CWE-"):
                out[row["project_slug"]] = f"CWE-{int(cwe.split('-')[1])}"
    return out


def same_file(a: str, b: str) -> bool:
    """SARIF uris and fix_info paths are both repo-relative, but one may carry a
    leading module prefix the other lacks; compare on the longest common tail."""
    if a == b:
        return True
    pa, pb = a.strip("/").split("/"), b.strip("/").split("/")
    n = min(len(pa), len(pb))
    return n > 0 and pa[-n:] == pb[-n:]


def label(alert: dict, fixes: list[dict]) -> tuple[int, dict | None]:
    locs = [(alert["sink_file"], alert["sink_line"])]
    locs += [(s["file"], s["line"]) for s in alert.get("flow", [])]
    for f in fixes:
        for fl, ln in locs:
            if ln is None or not same_file(fl, f["file"]):
                continue
            if f["start"] <= ln <= f["end"]:
                return 1, f
    return 0, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default=os.path.expanduser(
        "~/nimit/graphrouter/data/alerts.jsonl"))
    ap.add_argument("--bench", default=os.path.expanduser(
        "~/nimit/vuln-pred-results/IRIS/data/cwe-bench-java/data"))
    ap.add_argument("--out", default=os.path.expanduser(
        "~/nimit/graphrouter/data/alerts_labelled.jsonl"))
    a = ap.parse_args()

    fixes = load_fixes(os.path.join(a.bench, "fix_info.csv"))
    pcwe = load_project_cwe(os.path.join(a.bench, "project_info.csv"))
    alerts = [json.loads(l) for l in open(a.alerts) if l.strip()]

    per = defaultdict(lambda: [0, 0])
    rows = []
    for al in alerts:
        y, hit = label(al, fixes.get(al["project"], []))
        al["label"] = y
        al["gold_cwe"] = pcwe.get(al["project"])       # the project's CVE class
        al["fix_method"] = hit["method"] if hit else None
        per[al["project"]][y] += 1
        rows.append(al)

    n = len(rows); pos = sum(r["label"] for r in rows)
    print(f"labelled {n} alerts: {pos} true positives ({pos/max(n,1):.2%})")
    print(f"projects with no fix_info: "
          f"{sorted({r['project'] for r in rows if not fixes.get(r['project'])})}")
    print("\nper project (tp/total):")
    for p, (neg, posn) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"  {p[:54]:54s} {posn:4d}/{neg+posn:5d}")
    print("\nlabel by query CWE:",
          dict(Counter((r["query_cwe"], r["label"]) for r in rows)))

    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {a.out}")
