"""Score LLMDFA runs with a GROUPED train/test split over Juliet variant codes.

Why this exists
---------------
score_llmdfa.py reports one number per (model, bug type). That is enough to
reproduce the paper, but not enough to select a router honestly: choosing
"cheapest model within tau of the best F1" is a decision made by looking at the
scores, so reporting that same F1 is optimistically biased.

Why the split is by VARIANT and not random
------------------------------------------
Juliet is a complete Cartesian grid: every source/sink FORM appears with every
control-flow VARIANT.

    CWE80_XSS__CWE182_Servlet_File_01.java
    ^bug class   ^source/sink form   ^variant

    XSS  = 18 forms x 37 variants = 666
    OSCI = 12 forms x 37 variants = 444

A random case-level split therefore puts Servlet_File_01 in train and
Servlet_File_02 in test - near-identical programs. It scores well and proves
nothing.

The paper settles which axis carries the difficulty (Sec 4.2): it evaluates its
per-phase numbers on "37 programs for each bug type ... as other programs only
differ from the selected ones in terms of sources and sinks." So the variant is
the real axis of variation and the form is interchangeable. We split on variant.

This also lines up with the paper's own error analysis (Appendix A.4.2), where
three named variants break the Z3 stage in three different ways:

    _09   branch condition uses a library function   (Math.abs interpreted as And, not Or)
    _12   branch condition uses a nondeterministic call (read as a constant)
    _22a  branch condition uses a cross-file global   (value not tracked)

Those three are reported separately as a standing diagnostic, whichever side of
the split they land on.

Usage
-----
    python3 score_split.py <log>...                  # summary + split + diagnostics
    python3 score_split.py --by-variant <log>...     # full per-variant table
    python3 score_split.py --holdout 12 <log>...     # size of the held-out set
"""

import argparse
import ast
import collections
import random
import re
import sys

CASE = re.compile(r"^\{'input_token_cost'.*\}\s*$", re.M)
JAVA = re.compile(r"^(/\S+\.java)\s*$", re.M)
HEAD = re.compile(r"model=(\S+)\s+bug=(\S+)")

# Pre-registered: fixed seed, fixed rule, decided before any new model was scored.
SPLIT_SEED = 20260826
# Variants the paper documents as breaking path validation (Appendix A.4.2).
PAPER_HARD = ("09", "12", "22a")


def parse(path):
    """Pair each case-result dict with the .java file logged just above it."""
    text = open(path, errors="replace").read()
    head = HEAD.search(text)
    if not head:
        return None
    model, bug = head.group(1).split("/")[-1], head.group(2)

    events = [(m.start(), "f", m.group(1)) for m in JAVA.finditer(text)]
    events += [(m.start(), "c", m.group(0)) for m in CASE.finditer(text)]
    events.sort()

    rows, pending = [], None
    for _, kind, val in events:
        if kind == "f":
            pending = val
        elif pending is not None:
            try:
                d = ast.literal_eval(val)
            except Exception:
                pending = None
                continue
            stem = pending.split("/")[-1][:-5]          # drop .java
            form, variant = "?", "?"
            if "__" in stem:
                tail = stem.split("__", 1)[1]
                form, _, variant = tail.rpartition("_")
            d["file"], d["form"], d["variant"] = stem, form, variant
            rows.append(d)
            pending = None
    return {"model": model, "bug": bug, "rows": rows, "log": path} if rows else None


def metrics(rows):
    """LLMDFA counts reported TRACES, so cap true positives per case at that
    case's ground truth - otherwise several paths to one labelled sink push
    recall above 100%. Same convention as score_llmdfa.py."""
    tp = sum(min(r["analysis_result"]["TPs"], r["ground_truth"]["TPs"]) for r in rows)
    fp = sum(r["analysis_result"]["FPs"] for r in rows)
    gt = sum(r["ground_truth"]["TPs"] for r in rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / gt if gt else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    t = [r["single time cost"] for r in rows]
    return {
        "n": len(rows), "tp": tp, "fp": fp,
        "precision": p * 100, "recall": rc * 100, "f1": f1,
        "sec_per_case": sum(t) / len(t) if t else 0.0,
        "in_per_case": sum(r["input_token_cost"] for r in rows) // max(len(rows), 1),
        "out_per_case": sum(r["output_token_cost"] for r in rows) // max(len(rows), 1),
    }


def split_variants(variants, holdout_n):
    """Deterministic grouped split. Same variants held out for every model, so
    rows stay comparable."""
    ordered = sorted(variants)
    rng = random.Random(SPLIT_SEED)
    held = set(rng.sample(ordered, min(holdout_n, len(ordered))))
    return sorted(set(ordered) - held), sorted(held)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--holdout", type=int, default=12,
                    help="variants held out for reporting (default 12 of 37)")
    ap.add_argument("--by-variant", action="store_true",
                    help="print the full per-variant F1 table")
    args = ap.parse_args(argv)

    runs = [r for r in (parse(p) for p in args.logs) if r]
    if not runs:
        print("no parseable runs found")
        return 2

    merged = {}
    for r in runs:                                  # shards concatenate
        merged.setdefault((r["model"], r["bug"]), []).extend(r["rows"])

    for (model, bug), rows in sorted(merged.items()):
        by_file = {r["file"]: r for r in rows}      # de-dup re-run shards
        rows = list(by_file.values())
        variants = {r["variant"] for r in rows}
        sel, held = split_variants(variants, args.holdout)
        sel_rows = [r for r in rows if r["variant"] in sel]
        held_rows = [r for r in rows if r["variant"] in held]

        print(f"\n{'='*70}\n{model}  /  {bug.upper()}   "
              f"{len(rows)} cases, {len(variants)} variants, "
              f"{len({r['form'] for r in rows})} forms\n{'='*70}")
        print(f"{'set':<22} {'n':>5} {'prec':>8} {'recall':>8} {'F1':>7} {'s/case':>8}")
        for name, sub in (("ALL", rows),
                          (f"select ({len(sel)} var)", sel_rows),
                          (f"HELD OUT ({len(held)} var)", held_rows)):
            if not sub:
                continue
            m = metrics(sub)
            print(f"{name:<22} {m['n']:>5} {m['precision']:>7.2f}% "
                  f"{m['recall']:>7.2f}% {m['f1']:>7.3f} {m['sec_per_case']:>8.1f}")

        hard = [r for r in rows if r["variant"] in PAPER_HARD]
        if hard:
            print(f"\n  paper-documented hard variants {PAPER_HARD} "
                  f"(Appendix A.4.2):")
            for v in PAPER_HARD:
                sub = [r for r in rows if r["variant"] == v]
                if sub:
                    m = metrics(sub)
                    print(f"    _{v:<4} n={m['n']:<4} prec {m['precision']:6.2f}%  "
                          f"recall {m['recall']:6.2f}%  F1 {m['f1']:.3f}")

        if args.by_variant:
            print("\n  per-variant:")
            stats = []
            for v in sorted(variants):
                sub = [r for r in rows if r["variant"] == v]
                stats.append((metrics(sub)["f1"], v, metrics(sub)))
            for f1, v, m in sorted(stats):
                mark = "HELD" if v in held else "    "
                print(f"    {mark} _{v:<4} n={m['n']:<4} prec {m['precision']:6.2f}%  "
                      f"recall {m['recall']:6.2f}%  F1 {f1:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
