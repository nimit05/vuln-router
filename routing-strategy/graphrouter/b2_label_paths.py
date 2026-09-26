#!/usr/bin/env python3
"""B2 (rebuilt) -- label each dataflow PATH true/false the way IRIS does.

Supersedes b2_label_alerts.py. That version compared a SARIF line number against
the `method_start`/`method_end` range in `fix_info.csv`. IRIS does something
different, and the difference is most of the count gap: they resolve every step
of a path to its enclosing `file:class:method` using CodeQL's OWN extracted
declaration tables, then intersect that set with the fixed methods.

Mirrors `EvaluationPipeline.evaluate_sarif_result` (recall@method branch):

  fixed_methods = { "<file>:<class>:<method>" from fix_info.csv, dropping any
                    row whose file contains "src/test" }

  for each step of the path:
      enclosing class  = the class  decl in the same file with the LARGEST
                         start_line such that start_line <= L <= end_line
      enclosing method = likewise from the method decl table
      (a step with no enclosing class OR no enclosing method is skipped)
      -> "<file>:<class>:<method>"

  path is TP  <=>  that set intersects fixed_methods

Why method granularity and not line: `fix_info.csv` line numbers refer to the
PATCHED tree while the CodeQL database is built at the buggy tag, so lines drift.
The enclosing method is the finest unit that survives the drift -- and it is what
the published `num_tp_paths_method` column counts, which is the only reason our
numbers can sit in the same table as theirs.

Per project this must reproduce `vanilla_result.num_tp_paths_method` exactly.
`b1_validate.py` is the gate; do not run B3 until it passes.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import Counter, defaultdict


def load_decls(path: str) -> dict[str, list[tuple[int, int, str]]]:
    """file -> [(start_line, end_line, name)], from fetch_func_locs /
    fetch_class_locs. Sorted by start_line DESCENDING so the first enclosing
    match is the innermost one, which is IRIS's sort order."""
    by_file = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                s, e = int(row["start_line"]), int(row["end_line"])
            except (KeyError, TypeError, ValueError):
                continue
            by_file[row["file"]].append((s, e, row["name"]))
    for f in by_file:
        by_file[f].sort(key=lambda t: -t[0])
    return dict(by_file)


def enclosing(decls: dict, file_name: str, line: int) -> str | None:
    for s, e, name in decls.get(file_name, ()):
        if s <= line <= e:
            return name              # list is start_line-descending => innermost
    return None


def passing_methods(path_row: dict, classes: dict, methods: dict) -> set[str]:
    out = set()
    for step in path_row.get("flow", []):
        f, ln = step.get("file"), step.get("line")
        if f is None or ln is None:
            continue
        c = enclosing(classes, f, ln)
        if c is None:
            continue
        m = enclosing(methods, f, ln)
        if m is None:
            continue
        out.add(f"{f}:{c}:{m}")
    return out


def load_fixed_methods(path: str) -> dict[str, set[str]]:
    """project -> {"file:class:method"}. IRIS drops test files here, not later."""
    out = defaultdict(set)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            f = row.get("file") or ""
            if "src/test" in f:
                continue
            out[row["project_slug"]].add(f"{f}:{row.get('class')}:{row.get('method')}")
    return dict(out)


def load_project_cwe(path: str) -> dict[str, str]:
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            cwe = (row.get("cwe_id") or "").strip()
            if cwe.upper().startswith("CWE-"):
                out[row["project_slug"]] = f"CWE-{int(cwe.split('-')[1])}"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default=os.path.expanduser(
        "~/nimit/graphrouter/data/paths.jsonl"))
    ap.add_argument("--iris", default=os.path.expanduser(
        "~/nimit/vuln-pred-results/IRIS"))
    ap.add_argument("--variant", default="v15_full17")
    ap.add_argument("--decl-root", default=None,
                    help="where fetch_func_locs lives; default <iris>/output/"
                         "<proj>/<variant>. Pass 'common' if the declaration "
                         "tables were extracted under output/<proj>/common.")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/nimit/graphrouter/data/paths_labelled.jsonl"))
    a = ap.parse_args()

    bench = os.path.join(a.iris, "data", "cwe-bench-java", "data")
    fixed = load_fixed_methods(os.path.join(bench, "fix_info.csv"))
    pcwe = load_project_cwe(os.path.join(bench, "project_info.csv"))
    rows = [json.loads(l) for l in open(a.paths) if l.strip()]

    sub = a.decl_root or a.variant
    cache, missing_decls = {}, set()

    def decls_for(project: str):
        if project not in cache:
            base = os.path.join(a.iris, "output", project, sub)
            c = os.path.join(base, "fetch_class_locs", "results.csv")
            m = os.path.join(base, "fetch_func_locs", "results.csv")
            if not (os.path.exists(c) and os.path.exists(m)):
                missing_decls.add(project)
                cache[project] = ({}, {})
            else:
                cache[project] = (load_decls(c), load_decls(m))
        return cache[project]

    per = defaultdict(lambda: [0, 0])          # project -> [total, tp]
    for r in rows:
        classes, methods = decls_for(r["project"])
        pm = passing_methods(r, classes, methods)
        hit = pm & fixed.get(r["project"], set())
        r["label"] = 1 if hit else 0
        r["gold_cwe"] = pcwe.get(r["project"])
        r["fix_method"] = sorted(hit)[0] if hit else None
        r["n_passing_methods"] = len(pm)
        per[r["project"]][0] += 1
        per[r["project"]][1] += r["label"]

    if missing_decls:
        print(f"[FATAL] no fetch_class_locs/fetch_func_locs under "
              f"output/<proj>/{sub} for: {sorted(missing_decls)}", file=sys.stderr)
        print("        every path in those projects is forced to label 0 -- "
              "find the right --decl-root before trusting any number.",
              file=sys.stderr)

    n = len(rows); pos = sum(r["label"] for r in rows)
    print(f"labelled {n} paths: {pos} true positives ({pos/max(n,1):.2%})")
    print("\nper project (tp/paths, recall@method):")
    for p, (tot, tp) in sorted(per.items()):
        print(f"  {p[:54]:54s} {tp:>5}/{tot:<6} {'YES' if tp else 'no'}")
    print(f"\nprojects with recall: {sum(1 for v in per.values() if v[1])}/{len(per)}")
    print("label by query CWE:",
          dict(Counter((r["query_cwe"], r["label"]) for r in rows)))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {a.out}")
