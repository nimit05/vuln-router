#!/usr/bin/env python3
"""B1 (rebuilt) -- one candidate row per DATAFLOW PATH, matching IRIS's accounting.

Supersedes b1_extract_alerts.py, which emitted one row per SARIF *result*, kept
only `codeFlows[0]`, and deduplicated across query files. That undercounted by
roughly 2-3x against IRIS's own numbers (ff4j: 550 rows vs their 1229 paths) and
dropped true positives with the paths it dropped, so nothing built on it was
comparable to the published table.

IRIS counts a path exactly as `EvaluationPipeline.evaluate_sarif_result` does
(src/modules/evaluation_pipeline.py):

  * `iter_code_flows` walks `runs[0]["results"]` and yields EVERY `codeFlow` of
    every result -- a single result routinely carries several.
  * `ignore_code_flow` then drops a flow whose FIRST or LAST step message
    mentions `toString` or `println`. These are formatting hops, not taint.
  * `num_paths` is the count of what survives that filter.

So the unit here is `(result_index, flow_index)` within one SARIF, and the row
count per project must equal their `num_paths`. `b1_validate.py` is the gate.

Only projects IRIS actually scored are emitted: a project counts as scored when
`output/<proj>/<variant>/cwe-*wLLM-final/results.json` exists. rocketmq has 1620
alerts on disk but no such file, so IRIS never scored it -- including it made it
66% of the candidate pool and skewed the CWE mix to 89.6% CWE-94.
"""
from __future__ import annotations
import argparse, json, glob, os, re
from collections import Counter, defaultdict

CWE_DIR = re.compile(r"cwe-(\d+)w(?:LLM|CodeQL)")

# IRIS's own filter, verbatim in effect. Their version indexes blindly; ours
# treats a malformed step as "not ignorable" and counts it, which is the same
# decision their code reaches when the keys are present.
IGNORE_TOKENS = ("toString", "println")


def norm_cwe(raw: str) -> str:
    """'094' -> 'CWE-94'. CWE ids are not zero-padded in the taxonomy."""
    return f"CWE-{int(raw)}"


def _msg(loc: dict) -> str:
    return ((loc.get("location") or {}).get("message") or {}).get("text") or ""


def _locations(code_flow: dict) -> list[dict]:
    tfs = code_flow.get("threadFlows") or []
    if not tfs:
        return []
    return tfs[0].get("locations") or []


def ignore_code_flow(code_flow: dict) -> bool:
    """True when IRIS would skip this flow. Mirrors their `ignore_code_flow`."""
    locs = _locations(code_flow)
    if not locs:
        return False
    first, last = _msg(locs[0]), _msg(locs[-1])
    return any(t in first for t in IGNORE_TOKENS) or \
           any(t in last for t in IGNORE_TOKENS)


def flow_steps(code_flow: dict) -> list[dict]:
    """The path, source first. One entry per threadFlow location."""
    out = []
    for loc in _locations(code_flow):
        p = ((loc.get("location") or {}).get("physicalLocation")) or {}
        art = (p.get("artifactLocation") or {}).get("uri")
        if art is None:
            continue
        reg = p.get("region") or {}
        out.append({"file": art,
                    "line": reg.get("startLine"),
                    "end_line": reg.get("endLine"),
                    "message": _msg(loc)})
    return out


def scored_projects(iris_root: str, variant: str) -> dict[str, str]:
    """project -> cwe id ('094'), for projects IRIS produced a final score for."""
    out = {}
    pat = os.path.join(iris_root, "output", "*", variant, "cwe-*wLLM-final",
                       "results.json")
    for path in sorted(glob.glob(pat)):
        parts = path.split(os.sep)
        project = parts[parts.index("output") + 1]
        m = CWE_DIR.search(path)
        if m:
            out[project] = m.group(1)
    return out


def extract(iris_root: str, variant: str) -> tuple[list[dict], dict]:
    projects = scored_projects(iris_root, variant)
    rows, per_project = [], defaultdict(lambda: {"paths": 0, "ignored": 0,
                                                 "results": 0})
    for project, cwe_raw in sorted(projects.items()):
        sarif = os.path.join(iris_root, "output", project, variant,
                             f"cwe-{cwe_raw}wLLM", "results.sarif")
        if not os.path.exists(sarif):
            print(f"  ! no SARIF for scored project {project}: {sarif}")
            continue
        try:
            doc = json.load(open(sarif))
        except Exception as e:
            print(f"  ! unreadable {sarif}: {e}")
            continue
        runs = doc.get("runs") or []
        if not runs:
            continue
        results = runs[0].get("results") or []          # IRIS reads runs[0] only
        per_project[project]["results"] = len(results)
        for result_id, res in enumerate(results):
            if "codeFlows" not in res:
                continue
            locs = res.get("locations") or []
            p = (locs[0].get("physicalLocation") or {}) if locs else {}
            sink_file = (p.get("artifactLocation") or {}).get("uri")
            sink_line = (p.get("region") or {}).get("startLine")
            for flow_id, cf in enumerate(res["codeFlows"]):
                if ignore_code_flow(cf):
                    per_project[project]["ignored"] += 1
                    continue
                steps = flow_steps(cf)
                per_project[project]["paths"] += 1
                rows.append(dict(
                    path_id=f"{project}:{cwe_raw}:{result_id}:{flow_id}",
                    project=project,
                    result_id=result_id,
                    flow_id=flow_id,
                    query_cwe=norm_cwe(cwe_raw),
                    rule=res.get("ruleId"),
                    message=((res.get("message") or {}).get("text") or "")[:500],
                    sink_file=sink_file,
                    sink_line=sink_line,
                    source_file=steps[0]["file"] if steps else sink_file,
                    source_line=steps[0]["line"] if steps else sink_line,
                    n_steps=len(steps),
                    cross_file=len({s["file"] for s in steps}) > 1 if steps else False,
                    flow=steps,
                    variant=variant,
                ))
    return rows, dict(per_project)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iris", default=os.path.expanduser(
        "~/nimit/vuln-pred-results/IRIS"))
    ap.add_argument("--variant", default="v15_full17")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/nimit/graphrouter/data/paths.jsonl"))
    a = ap.parse_args()

    rows, per = extract(a.iris, a.variant)
    print(f"extracted {len(rows)} paths from variant {a.variant}")
    print(f"  projects            : {len(per)}")
    print("  by CWE:", dict(Counter(r['query_cwe'] for r in rows).most_common()))
    steps = sorted(r["n_steps"] for r in rows)
    if steps:
        print(f"  path length: p50={steps[len(steps)//2]} "
              f"p95={steps[int(0.95*len(steps))]} max={steps[-1]}")
    print("\nper project (results / paths kept / flows ignored):")
    for p, c in sorted(per.items()):
        print(f"  {p[:54]:54s} {c['results']:>6} {c['paths']:>6} {c['ignored']:>6}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {a.out}")
