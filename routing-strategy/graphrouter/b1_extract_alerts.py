#!/usr/bin/env python3
"""B1 -- turn IRIS's CodeQL SARIF into one candidate row per alert.

Phase B step 1: "Run CodeQL over your training repos -> alerts, each carrying a
CWE tag and a dataflow path."

The databases and the query runs already exist from the IRIS reproduction, so
this reads their output rather than re-running CodeQL: 24 databases and ~4.5k
LLM-augmented alerts are on disk already, and rebuilding them costs days of
Maven/Gradle builds for an identical result.

Which query variant to read matters. Vanilla CodeQL (`cwe-*wCodeQL`) yields 29
alerts across 18 projects -- too few to train anything. IRIS's LLM-augmented
runs (`cwe-*wLLM`) yield ~4.5k, because inferring extra sources and sinks is the
whole point of that paper. The augmented set is the candidate stream; the
vanilla set is kept as a column so the two can be compared later.

Output: one JSON row per alert with the CWE tag, the sink, and the full dataflow
path -- exactly the three things the slicer in B3 needs.
"""
from __future__ import annotations
import argparse, json, glob, os, re, hashlib
from collections import Counter

CWE_DIR = re.compile(r"cwe-(\d+)w(LLM|CodeQL)")


def norm_cwe(raw: str) -> str:
    """'094' -> 'CWE-94'. CWE ids are not zero-padded in the taxonomy."""
    return f"CWE-{int(raw)}"


def flow_steps(result: dict) -> list[dict]:
    """The dataflow path, source first. Empty when CodeQL reported no codeFlow."""
    cfs = result.get("codeFlows") or []
    if not cfs:
        return []
    tfs = cfs[0].get("threadFlows") or []
    if not tfs:
        return []
    out = []
    for loc in tfs[0].get("locations", []):
        p = (loc.get("location") or {}).get("physicalLocation") or {}
        art = (p.get("artifactLocation") or {}).get("uri")
        reg = p.get("region") or {}
        if art is None:
            continue
        out.append({"file": art,
                    "line": reg.get("startLine"),
                    "end_line": reg.get("endLine"),
                    "message": ((loc.get("location") or {}).get("message")
                                or {}).get("text")})
    return out


def alert_id(project: str, rule: str, sink_file: str, sink_line, steps) -> str:
    """Stable id. Two runs of the same query must produce the same id, or the
    probe table cannot be joined to the graph across re-extractions."""
    key = json.dumps([project, rule, sink_file, sink_line,
                      [(s["file"], s["line"]) for s in steps]], sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def extract(iris_root: str, variant: str, kind: str) -> list[dict]:
    pat = os.path.join(iris_root, "output", "*", variant, f"cwe-*w{kind}",
                       "results.sarif")
    if kind == "CodeQL":                      # vanilla runs live under common/
        pat = os.path.join(iris_root, "output", "*", "common", f"cwe-*w{kind}",
                           "results.sarif")
    rows, seen = [], set()
    for path in sorted(glob.glob(pat)):
        parts = path.split(os.sep)
        project = parts[parts.index("output") + 1]
        m = CWE_DIR.search(path)
        cwe = norm_cwe(m.group(1)) if m else None
        try:
            sarif = json.load(open(path))
        except Exception as e:
            print(f"  ! unreadable {path}: {e}")
            continue
        for run in sarif.get("runs", []):
            for res in run.get("results", []):
                locs = res.get("locations") or []
                if not locs:
                    continue
                p = locs[0].get("physicalLocation") or {}
                sink_file = (p.get("artifactLocation") or {}).get("uri")
                sink_line = (p.get("region") or {}).get("startLine")
                if sink_file is None:
                    continue
                steps = flow_steps(res)
                aid = alert_id(project, res.get("ruleId", ""), sink_file,
                               sink_line, steps)
                if aid in seen:               # the same finding can be emitted
                    continue                  # by several query files
                seen.add(aid)
                rows.append(dict(
                    alert_id=aid,
                    project=project,
                    query_cwe=cwe,
                    rule=res.get("ruleId"),
                    message=((res.get("message") or {}).get("text") or "")[:500],
                    sink_file=sink_file,
                    sink_line=sink_line,
                    source_file=steps[0]["file"] if steps else sink_file,
                    source_line=steps[0]["line"] if steps else sink_line,
                    n_steps=len(steps),
                    cross_file=len({s["file"] for s in steps}) > 1 if steps else False,
                    flow=steps,
                    variant=variant if kind == "LLM" else "common",
                    detector=kind,
                ))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iris", default=os.path.expanduser(
        "~/nimit/vuln-pred-results/IRIS"))
    ap.add_argument("--variant", default="v15_full17",
                    help="IRIS run directory holding the augmented queries")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/nimit/graphrouter/data/alerts.jsonl"))
    a = ap.parse_args()

    rows = extract(a.iris, a.variant, "LLM")
    print(f"extracted {len(rows)} augmented alerts from variant {a.variant}")

    with_flow = sum(bool(r["flow"]) for r in rows)
    xfile = sum(r["cross_file"] for r in rows)
    print(f"  with a dataflow path : {with_flow} ({with_flow/max(len(rows),1):.1%})")
    print(f"  cross-file flows     : {xfile} ({xfile/max(len(rows),1):.1%})")
    print(f"  projects             : {len({r['project'] for r in rows})}")
    print("  by CWE:", dict(Counter(r["query_cwe"] for r in rows).most_common(8)))
    steps = sorted(r["n_steps"] for r in rows)
    if steps:
        print(f"  flow length: p50={steps[len(steps)//2]} "
              f"p95={steps[int(0.95*len(steps))]} max={steps[-1]}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {a.out}")
