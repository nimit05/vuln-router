"""Score LLMDFA runs and tabulate them against the paper's Table 1.

Reads the per-case summary dicts LLMDFA prints to stdout, so it works off the
SLURM .out files with no rerun needed. All metrics are regenerable from these
logs without a GPU.

  python score_llmdfa.py ~/cluster/18*.out
"""

import ast
import re
import sys
from collections import defaultdict

# LLMDFA paper (NeurIPS 2024), Table 1, "Detection" rows: precision %, recall %, F1
PAPER = {
    "dbz":  {"gpt-3.5": (73.75, 92.16, 0.82), "gpt-4": (81.38, 95.75, 0.87),
             "gemini-1.0": (66.57, 74.21, 0.70), "claude-3": (76.91, 82.67, 0.80)},
    "xss":  {"gpt-3.5": (100.00, 92.31, 0.96), "gpt-4": (100.00, 98.64, 0.99),
             "gemini-1.0": (100.00, 94.60, 0.97), "claude-3": (100.00, 86.49, 0.93)},
    "osci": {"gpt-3.5": (100.00, 78.38, 0.88), "gpt-4": (100.00, 89.19, 0.94),
             "gemini-1.0": (100.00, 94.59, 0.97), "claude-3": (100.00, 97.30, 0.99)},
}

HEADER = re.compile(r"model=(\S+)\s+bug=(\S+)\s+mode=(\S+).*?cases=(\S+)\s+seed=(\S+)"
                    r"(?:\s+shard=(\S+))?")
# the dict has NESTED braces, so this must be greedy and line-anchored
# nested braces => greedy; LLMDFA prints a trailing space => \s*$ not $
CASE = re.compile(r"^\{'input_token_cost'.*\}\s*$", re.MULTILINE)


def parse(path):
    text = open(path, errors="replace").read()
    h = HEADER.search(text)
    if not h:
        return None
    model, bug, mode, limit, seed, shard = h.groups()
    rows = []
    for m in CASE.finditer(text):
        try:
            rows.append(ast.literal_eval(m.group(0)))
        except Exception:
            pass
    if not rows:
        return None
    return {
        "model": model.split("/")[-1], "bug": bug, "mode": mode,
        "limit": limit, "seed": seed, "shard": shard or "0+all",
        "rows": rows, "log": path,
    }


def score(run):
    """LLMDFA's TPs counts reported TRACES, not distinct bugs: several paths can
    reach the same labelled sink, so a case with one ground-truth bug can report
    TPs=2. Summing raw traces against per-case ground truth yields recall > 100%,
    which is how this was caught. We therefore cap true positives per case at that
    case's ground truth, and report the raw trace counts alongside."""
    rows = run["rows"]
    tp_raw = sum(r["analysis_result"]["TPs"] for r in rows)
    fp = sum(r["analysis_result"]["FPs"] for r in rows)
    gtp = sum(r["ground_truth"]["TPs"] for r in rows)
    tp = sum(min(r["analysis_result"]["TPs"], r["ground_truth"]["TPs"]) for r in rows)
    dup = tp_raw - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / gtp if gtp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    assert rec <= 1.0 + 1e-9, "recall above 100% - capping failed"
    t = [r["single time cost"] for r in rows]
    return {
        "cases": len(rows), "tp": tp, "fp": fp, "gtp": gtp,
        "tp_raw": tp_raw, "dup": dup,
        "precision": prec * 100, "recall": rec * 100, "f1": f1,
        "in_tok": sum(r["input_token_cost"] for r in rows),
        "out_tok": sum(r["output_token_cost"] for r in rows),
        "time_s": sum(t), "time_per_case": sum(t) / len(t),
    }


def main(paths):
    runs = [r for r in (parse(p) for p in paths) if r]
    if not runs:
        print("no parseable runs found")
        return
    # A long bug type may be split across shards (LLMDFA_CASE_OFFSET/COUNT), which are
    # disjoint by construction, so shards of one (model,bug,mode) must be CONCATENATED.
    # Keying without the shard would silently keep one shard and drop the rest.
    # A crashed shard leaves a short log, and its re-run leaves a full one under the
    # SAME key. Keep whichever holds more cases, so argument order cannot decide the
    # result (a failed 6-case run must never shadow its 500-case replacement).
    best = {}
    for r in runs:
        k = (r["model"], r["bug"], r["mode"], r["limit"], r["seed"], r["shard"])
        if k not in best or len(r["rows"]) > len(best[k]["rows"]):
            best[k] = r

    merged = {}
    for (model, bug, mode, limit, seed, shard), r in best.items():
        key = (model, bug, mode, limit, seed)
        if key in merged:
            merged[key]["rows"].extend(r["rows"])
            merged[key]["shards"].append(shard)
        else:
            m = dict(r)
            m["rows"] = list(r["rows"])
            m["shards"] = [shard]
            merged[key] = m

    by_model = defaultdict(dict)
    for (model, bug, _, _, _), r in merged.items():
        by_model[model][bug] = (r, score(r))

    for model, bugs in sorted(by_model.items()):
        print("\n## %s\n" % model)
        print("| bug | cases | TP | FP | dup traces | precision | recall | F1 | paper gpt-3.5 (full set) | dF1 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for bug in ("dbz", "xss", "osci"):
            if bug not in bugs:
                continue
            run, s = bugs[bug]
            ref = PAPER[bug]["gpt-3.5"]
            print("| %s | %d%s | %d | %d | %d | %.2f%% | %.2f%% | %.3f | %.2f%% / %.2f%% / %.2f | %+.3f |"
                  % (bug.upper(), s["cases"],
                     "" if run["mode"] == "all" else " (sample)",
                     s["tp"], s["fp"], s["dup"], s["precision"], s["recall"], s["f1"],
                     ref[0], ref[1], ref[2], s["f1"] - ref[2]))
        print("\n| bug | tokens in/case | tokens out/case | s/case | total |")
        print("|---|---|---|---|---|")
        for bug in ("dbz", "xss", "osci"):
            if bug not in bugs:
                continue
            run, s = bugs[bug]
            print("| %s | %d | %d | %.1f | %.1f min |"
                  % (bug.upper(), s["in_tok"] // s["cases"], s["out_tok"] // s["cases"],
                     s["time_per_case"], s["time_s"] / 60))
        for bug in ("dbz", "xss", "osci"):
            if bug in bugs and len(bugs[bug][0]["shards"]) > 1:
                print("\n%s merged from %d shards: %s"
                      % (bug.upper(), len(bugs[bug][0]["shards"]),
                         ", ".join(sorted(bugs[bug][0]["shards"]))))
        seeds = {r["seed"] for r, _ in bugs.values()}
        modes = {r["mode"] for r, _ in bugs.values()}
        print("\nsampling: mode=%s seed=%s" % (",".join(modes), ",".join(seeds)))
        print("NOTE: 'sample' rows are a seeded random subset, so they are comparable to the")
        print("paper in protocol but not in composition. Paper columns are its full benchmark.")


if __name__ == "__main__":
    main(sys.argv[1:])
