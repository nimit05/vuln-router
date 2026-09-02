"""Offline router simulation over existing LLMDFA logs.

Extends score_split.py's parser with the per-case STAGE SIGNALS that LLMDFA
already prints but the scorer discards. Those signals are what a cascade
router escalates on, so they can be replayed without touching a GPU.

Signals per case (counted between the .java line and the result dict):
    Applying path checker with solver...        -> solver_checks
    Constructing solving program...             -> z3_construct
    Refining solving program...                 -> z3_refine
    Solver-aided path checker succeeded...      -> z3_ok
    Solver-aided path checker failed...         -> z3_fail     <-- escalation trigger
    Applying path reachability check with LLM.. -> llm_fallback
    start to summarize...                       -> n_summaries (pre-LLM complexity proxy)
"""
import ast, os, re, sys, collections, itertools, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
SCRIPTS = "/Users/nimitwadhwa/Documents/projects/vuln-pred-results/LLMDFA/scripts"
sys.path.insert(0, SCRIPTS)
from score_split import metrics, split_variants, SPLIT_SEED  # noqa

CASE = re.compile(r"^\{'input_token_cost'.*\}\s*$", re.M)
JAVA = re.compile(r"^(/\S+\.java)\s*$", re.M)
HEAD = re.compile(r"model=(\S+)\s+bug=(\S+)")

MARKERS = {
    "solver_checks": "Applying path checker with solver...",
    "z3_construct":  "Constructing solving program...",
    "z3_refine":     "Refining solving program...",
    "z3_ok":         "Solver-aided path checker succeeded...",
    "z3_fail":       "Solver-aided path checker failed...",
    "llm_fallback":  "Applying path reachability check with LLM...",
    "n_summaries":   "start to summarize...",
    "n_callees":     "Processing callees...",
}


def parse(path):
    text = open(path, errors="replace").read()
    head = HEAD.search(text)
    if not head:
        return None
    model, bug = head.group(1).split("/")[-1], head.group(2)
    events = [(m.start(), "f", m.group(1), m.end()) for m in JAVA.finditer(text)]
    events += [(m.start(), "c", m.group(0), m.end()) for m in CASE.finditer(text)]
    events.sort()

    rows, pending, pend_end = [], None, None
    for start, kind, val, end in events:
        if kind == "f":
            pending, pend_end = val, end
        elif pending is not None:
            try:
                d = ast.literal_eval(val)
            except Exception:
                pending = None
                continue
            body = text[pend_end:start]
            for k, marker in MARKERS.items():
                d[k] = body.count(marker)
            stem = pending.split("/")[-1][:-5]
            form, variant = "?", "?"
            if "__" in stem:
                tail = stem.split("__", 1)[1]
                form, _, variant = tail.rpartition("_")
            d["file"], d["form"], d["variant"] = stem, form, variant
            d["model"], d["bug"] = model, bug
            rows.append(d)
            pending = None
    return {"model": model, "bug": bug, "rows": rows} if rows else None


def load(logdirs):
    merged = collections.defaultdict(dict)   # (model,bug) -> file -> row
    for d in logdirs:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".out"):
                continue
            r = parse(os.path.join(d, fn))
            if not r:
                continue
            for row in r["rows"]:
                merged[(r["model"], r["bug"])][row["file"]] = row
    return merged


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reproduction")
    merged = load([f"{base}/logs", f"{base}/logs_router"])

    print("=== inventory (model x bug) ===")
    for (m, b), files in sorted(merged.items()):
        rows = list(files.values())
        mm = metrics(rows)
        fails = sum(r["z3_fail"] for r in rows)
        checks = sum(r["solver_checks"] for r in rows)
        ncases_with_fail = sum(1 for r in rows if r["z3_fail"] > 0)
        print(f"{m:<34} {b:<5} n={len(rows):<5} F1={mm['f1']:.3f} "
              f"s/case={mm['sec_per_case']:6.1f}  solver={checks:<5} "
              f"z3_fail={fails:<5} cases_with_fail={ncases_with_fail}")

    print("\n=== matched sets per bug ===")
    for bug in ("xss", "osci", "dbz"):
        models = [m for (m, b) in merged if b == bug]
        if not models:
            continue
        sets = [set(merged[(m, bug)]) for m in models]
        common = set.intersection(*sets)
        print(f"{bug:<5} models={len(models)} matched_cases={len(common)}  "
              f"({', '.join(sorted(models))})")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- diagnostics
def case_err(r):
    """Net error on one case: missed TPs + false positives."""
    tp = min(r["analysis_result"]["TPs"], r["ground_truth"]["TPs"])
    miss = r["ground_truth"]["TPs"] - tp
    return miss + r["analysis_result"]["FPs"]


def trigger_diagnostic(merged):
    print("\n=== Is `Solver-aided path checker failed` predictive of error? ===")
    print(f"{'model':<34}{'bug':<6}{'trig%':>7}{'err|trig':>10}{'err|no':>9}{'lift':>7}"
          f"{'FP|trig':>9}{'FP|no':>8}")
    for (m, b), files in sorted(merged.items()):
        rows = list(files.values())
        trig = [r for r in rows if r["z3_fail"] > 0]
        notr = [r for r in rows if r["z3_fail"] == 0]
        if not trig or not notr:
            print(f"{m:<34}{b:<6}{100*len(trig)/len(rows):>6.1f}%       -        -      -")
            continue
        et = sum(case_err(r) for r in trig) / len(trig)
        en = sum(case_err(r) for r in notr) / len(notr)
        ft = sum(r["analysis_result"]["FPs"] for r in trig) / len(trig)
        fn = sum(r["analysis_result"]["FPs"] for r in notr) / len(notr)
        print(f"{m:<34}{b:<6}{100*len(trig)/len(rows):>6.1f}%{et:>10.2f}{en:>9.2f}"
              f"{(et/en if en else float('inf')):>7.1f}{ft:>9.2f}{fn:>8.2f}")


if __name__ == "__main__" and "--diag" in sys.argv:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reproduction")
    trigger_diagnostic(load([f"{base}/logs", f"{base}/logs_router"]))
