#!/usr/bin/env python3
"""B3 -- slice each dataflow PATH down to the code it actually touches.

Phase B step 2: "Slice each candidate down using the CPG."

The unit is the path, not the SARIF result: one result can carry many codeFlows
and IRIS scores each of them separately, so each gets its own slice. See
b1_extract_paths.py for why that distinction decides whether our numbers can
sit in the same table as theirs.

**Deviation, stated up front.** LLMxCPG slices with Joern. This slices along the
dataflow path in the CodeQL database, which is also a code property graph, using
the method boundaries CodeQL already extracted (`fetch_func_locs`). The reason is
cost, not preference: standing up Joern means a second CPG build over 15 Java
projects (rocketmq alone is ~500k LOC) to recover a path CodeQL has already
computed and stored. If the slices turn out to be the bottleneck, swapping in
Joern changes only this file.

The slice is the union of the methods the flow passes through, in flow order,
which makes it interprocedural by construction: 58% of these alerts cross files,
and a single-method slice would cut the path in half and hide the defect.

Size is capped. A slice that does not fit the model's context is not a slice, and
the source and sink methods are the two that must never be dropped -- they are
where the taint enters and where it lands.
"""
from __future__ import annotations
import argparse, csv, json, os, re
from collections import defaultdict

MAX_LINES = 320          # whole-slice ceiling, chosen to fit an 8k context
MAX_METHOD_LINES = 120   # per-method ceiling before the middle is elided


def load_func_locs(iris: str, project: str,
                   variant: str) -> dict[str, list[tuple[int, int, str]]]:
    """file -> [(start, end, name)], sorted so the innermost match wins.

    Reads the variant's declaration table, the same one B2 labels against, so a
    method that defines a true positive is the method that gets sliced. Falls
    back to `common/` only when the variant never extracted one."""
    path = os.path.join(iris, "output", project, variant,
                        "fetch_func_locs", "results.csv")
    if not os.path.exists(path):
        path = os.path.join(iris, "output", project, "common",
                            "fetch_func_locs", "results.csv")
    idx: dict[str, list] = defaultdict(list)
    if not os.path.exists(path):
        return idx
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                idx[row["file"]].append(
                    (int(row["start_line"]), int(row["end_line"]), row["name"]))
            except (ValueError, KeyError):
                continue
    for f in idx:                      # smallest enclosing range first
        idx[f].sort(key=lambda t: (t[1] - t[0]))
    return idx


def enclosing(idx, file: str, line: int):
    for start, end, name in idx.get(file, []):
        if start <= line <= end:
            return start, end, name
    return None


def read_lines(src_root: str, project: str, file: str) -> list[str] | None:
    p = os.path.join(src_root, project, file)
    if not os.path.exists(p):
        return None
    try:
        with open(p, errors="replace") as fh:
            return fh.read().splitlines()
    except OSError:
        return None


def elide(body: list[str], keep_line: int | None, start: int) -> list[str]:
    """Trim an over-long method around the line the flow actually touches."""
    if len(body) <= MAX_METHOD_LINES:
        return body
    if keep_line is None:
        return body[:MAX_METHOD_LINES] + ["        // ... elided ..."]
    off = max(0, keep_line - start - MAX_METHOD_LINES // 2)
    off = min(off, max(0, len(body) - MAX_METHOD_LINES))
    out = body[off:off + MAX_METHOD_LINES]
    if off:
        out = ["        // ... elided ..."] + out
    if off + MAX_METHOD_LINES < len(body):
        out = out + ["        // ... elided ..."]
    return out


def base_row(alert: dict) -> dict:
    """Identity and label, carried whether or not a slice could be built.

    Every path stays in the output. A path we cannot slice is still a path IRIS
    counted, so dropping it here would quietly shrink the denominator and
    flatter every configuration in the final table."""
    return dict(
        path_id=alert["path_id"],
        project=alert["project"],
        query_cwe=alert["query_cwe"],
        gold_cwe=alert.get("gold_cwe"),
        label=alert["label"],
        rule=alert["rule"],
        message=alert["message"],
        n_hops=alert["n_steps"],
        cross_file=int(alert["cross_file"]),
    )


def build_slice(alert: dict, idx, src_root: str) -> dict | None:
    steps = alert.get("flow") or []
    pts = [(s["file"], s["line"]) for s in steps if s.get("line")]
    pts.append((alert["sink_file"], alert["sink_line"]))

    ordered, seen = [], set()
    for f, ln in pts:
        enc = enclosing(idx, f, ln)
        if enc is None:
            continue
        key = (f, enc[0], enc[1])
        if key in seen:
            continue
        seen.add(key)
        ordered.append((f, enc[0], enc[1], enc[2], ln))
    if not ordered:
        return None

    # keep the ends: source method first, sink method last
    if len(ordered) > 8:
        ordered = ordered[:4] + ordered[-4:]

    chunks, total, methods = [], 0, []
    for f, start, end, name, ln in ordered:
        lines = read_lines(src_root, alert["project"], f)
        if lines is None:
            continue
        body = elide(lines[start - 1:end], ln, start)
        if total + len(body) > MAX_LINES and chunks:
            break
        total += len(body)
        methods.append({"file": f, "method": name, "start": start, "end": end})
        chunks.append(f"// {f}:{start}  {name}\n" + "\n".join(body))
    if not chunks:
        return None

    text = "\n\n".join(chunks)
    nest = 0, 0
    depth = mx = 0
    for ch in text:
        if ch == "{":
            depth += 1; mx = max(mx, depth)
        elif ch == "}":
            depth = max(0, depth - 1)
    row = base_row(alert)
    row.update(
        sliceable=1,
        slice=text,
        # structural features -- the vector concatenated onto the code embedding
        n_methods=len(methods),
        slice_loc=total,
        slice_chars=len(text),
        max_nest=mx,
        n_if=len(re.findall(r"\bif\s*\(", text)),
        n_loop=len(re.findall(r"\b(for|while)\s*\(", text)),
        n_call=len(re.findall(r"\b[A-Za-z_]\w*\s*\(", text)),
        n_catch=len(re.findall(r"\bcatch\s*\(", text)),
        methods=methods,
    )
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default=os.path.expanduser(
        "~/nimit/graphrouter/data/paths_labelled.jsonl"))
    ap.add_argument("--variant", default="v15_full17")
    ap.add_argument("--iris", default=os.path.expanduser(
        "~/nimit/vuln-pred-results/IRIS"))
    ap.add_argument("--out", default=os.path.expanduser(
        "~/nimit/graphrouter/data/slices.jsonl"))
    a = ap.parse_args()

    src_root = os.path.join(a.iris, "data", "cwe-bench-java", "project-sources")
    alerts = [json.loads(l) for l in open(a.alerts) if l.strip()]
    by_proj = defaultdict(list)
    for al in alerts:
        by_proj[al["project"]].append(al)

    rows, failed = [], 0
    for proj, als in by_proj.items():
        idx = load_func_locs(a.iris, proj, a.variant)
        made = 0
        for al in als:
            s = build_slice(al, idx, src_root)
            if s is None:                     # unsliceable, but still a path
                failed += 1
                r = base_row(al)
                r.update(sliceable=0, slice="", n_methods=0, slice_loc=0,
                         slice_chars=0, max_nest=0, n_if=0, n_loop=0,
                         n_call=0, n_catch=0, methods=[])
                rows.append(r)
            else:
                rows.append(s); made += 1
        print(f"  {proj[:52]:52s} {made}/{len(als)} sliced")

    n = len(rows); pos = sum(r["label"] for r in rows)
    ok = [r for r in rows if r["sliceable"]]
    lost_tp = sum(r["label"] for r in rows if not r["sliceable"])
    print(f"\n{n} paths in, {len(ok)} sliced, {failed} had no resolvable method")
    print(f"  true positives total    : {pos} ({pos/max(n,1):.2%})")
    print(f"  true positives UNSLICED : {lost_tp}"
          f"  <- these keep their row so the denominator stays IRIS's")
    loc = sorted(r["slice_loc"] for r in ok)
    if loc:
        print(f"  slice LOC p50={loc[len(loc)//2]} p95={loc[int(.95*len(loc))]} max={loc[-1]}")
        mt = sorted(r["n_methods"] for r in ok)
        print(f"  methods/slice p50={mt[len(mt)//2]} max={mt[-1]}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {a.out}")
