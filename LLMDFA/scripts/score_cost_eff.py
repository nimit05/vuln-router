import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ast
import re

from score_split import parse, split_variants  # noqa: E402

# Shard jobs (run_llmdfa_x2.sbatch) stream both shards to one %j.out, where they
# INTERLEAVE. score_split.parse pairs each result with the .java line above it,
# which mis-attributes across shards - measured at 7-8 pairs per 300-case DBZ
# job. The shard id is recoverable from the path (.../LLMDFA_0/, .../LLMDFA_1/),
# so pair per shard and DROP pairs that stay ambiguous rather than guess.
# score_split.parse is left untouched: it produced the published numbers.
_CASE = re.compile(r"^\{'input_token_cost'.*\}\s*$", re.M)
_JAVA = re.compile(r"^(/\S+\.java)\s*$", re.M)
_HEAD = re.compile(r"model=(\S+)\s+bug=(\S+)")
_SHARD = re.compile(r"/LLMDFA_(\d+)/")


def parse_sharded(path):
    """Shard-aware variant of score_split.parse. Identical output on
    single-shard logs; drops ambiguous pairs on interleaved ones."""
    text = open(path, errors="replace").read()
    head = _HEAD.search(text)
    if not head:
        return None
    model, bug = head.group(1).split("/")[-1], head.group(2)

    events = sorted([(m.start(), "f", m.group(1)) for m in _JAVA.finditer(text)] +
                    [(m.start(), "c", m.group(0)) for m in _CASE.finditer(text)])

    rows, pending, dropped = [], {}, 0
    for _, kind, val in events:
        if kind == "f":
            s = _SHARD.search(val)
            pending[s.group(1) if s else "0"] = val
        elif len(pending) == 1:
            java = pending.popitem()[1]
            try:
                d = ast.literal_eval(val)
            except Exception:
                continue
            stem = java.split("/")[-1][:-5]
            form, variant = "?", "?"
            if "__" in stem:
                form, _, variant = stem.split("__", 1)[1].rpartition("_")
            d["file"], d["form"], d["variant"] = stem, form, variant
            rows.append(d)
        else:                       # 0 or >1 shards pending: cannot attribute
            dropped += 1
            pending.clear()
    if dropped:
        print(f"  {path.split('/')[-1]}: dropped {dropped} ambiguous "
              f"interleaved pairs, kept {len(rows)}", file=sys.stderr)
    return {"model": model, "bug": bug, "rows": rows, "log": path} if rows else None

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(REPO, "LLMDFA", "reproduction", "logs")


def load(logs):
    """Parse logs into {(model, bug): [row]}.

    Shards of one job concatenate; a file re-run across shards is de-duplicated
    by name, same as score_split.main."""
    merged = {}
    for run in (parse_sharded(p) for p in logs):
        if run:
            merged.setdefault((run["model"], run["bug"]), []).extend(run["rows"])
    return {k: list({r["file"]: r for r in rows}.values())
            for k, rows in merged.items()}


def to_frame(merged):
    """One row per analysed case, for grouping with pandas.

    tp is CAPPED at the case's ground truth before it lands in the frame.
    LLMDFA counts traces, so several paths to one labelled sink would push
    recall above 100%. metrics() applies the same cap; aggregating raw TPs
    here instead would silently disagree with the published numbers."""
    return pd.DataFrame([
        {
            "model": model,
            "bug": bug,
            "file": r["file"],
            "form": r["form"],
            "variant": r["variant"],
            "tp": min(r["analysis_result"]["TPs"], r["ground_truth"]["TPs"]),
            "fp": r["analysis_result"]["FPs"],
            "gt": r["ground_truth"]["TPs"],
            "sec": r["single time cost"],
            "tok_in": r["input_token_cost"],
            "tok_out": r["output_token_cost"],
        }
        for (model, bug), rows in merged.items()
        for r in rows
    ])


def matched(df):
    """Restrict to the files EVERY model completed, per bug type.

    Qwen's DBZ run is 1,851 cases and the others are the 300-case sorted
    prefix; comparing raw totals compares different benchmarks."""
    keep = df.groupby(["bug", "file"])["model"].nunique() == df["model"].nunique()
    return df[df.set_index(["bug", "file"]).index.map(keep).fillna(False)]


def add_split(df, holdout_n=12):
    """Tag each row select/held using the pre-registered seeded split.

    Done per bug type: DBZ has 38 variants (26/12), XSS and OSCI have 37
    (25/12), so the select count is not a constant."""
    df = df.copy()
    df["split"] = ""
    for bug, sub in df.groupby("bug"):
        _, held = split_variants(set(sub["variant"]), holdout_n)
        df.loc[sub.index, "split"] = sub["variant"].isin(held).map(
            {True: "held", False: "select"})
    return df


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(sub):
    """Precision / recall / F1 / n / total GPU-seconds for a set of cases.

    tp is already capped per case in to_frame(), matching score_split.metrics.
    There are no true negatives in these logs - a clean program that is
    correctly left alone is never counted - so accuracy is undefined and is
    deliberately not reported."""
    tp, fp, gt = int(sub["tp"].sum()), int(sub["fp"].sum()), int(sub["gt"].sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / gt if gt else 0.0
    return {
        "n": len(sub),
        "precision": p * 100,
        "recall": r * 100,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "sec": float(sub["sec"].sum()),
    }


def cells(df):
    """{(bug, model, split): score}. This is the whole fitting surface:
    3 bug types x 4 models x 2 splits."""
    return {k: score(sub) for k, sub in df.groupby(["bug", "model", "split"])}


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def choose(bug, cell, tau, models, split="select"):
    """Cheapest model whose F1 is within tau of the best, on `split`.

    tau=0 collapses to plain argmax, which is the Sec 5.2 router - so that case
    is the regression test for this function.

    Ties break on (sec, model) so the fitted table is reproducible; anything
    order-dependent would break the pre-registration."""
    avail = [m for m in models if (bug, m, split) in cell]
    if not avail:
        return None
    best = max(cell[(bug, m, split)]["f1"] for m in avail)
    near = [m for m in avail if cell[(bug, m, split)]["f1"] >= best - tau]
    return min(near, key=lambda m: (cell[(bug, m, split)]["sec"], m))


def fit(cell, bugs, models, tau):
    """FIT on select only, then FREEZE. Held-out is never consulted here."""
    return {b: choose(b, cell, tau, models, "select") for b in bugs}


def evaluate(table, cell, split):
    """Score a frozen table. Avg F1 is the unweighted mean over bug types -
    weighting by case count would let XSS and OSCI swamp DBZ, which is the row
    that actually decides the outcome."""
    f1s, sec = [], 0.0
    for bug, model in table.items():
        if model is None or (bug, model, split) not in cell:
            return None, None
        m = cell[(bug, model, split)]
        f1s.append(m["f1"])
        sec += m["sec"]
    return sum(f1s) / len(f1s), sec


def baselines(cell, bugs, models, split):
    """always-<model> for each model, plus the oracle upper bound.

    The oracle picks per bug type using `split`'s own numbers. That is exactly
    the Sec 5.1 error, which is why it is labelled an upper bound and never a
    strategy - it is here to show how much headroom the router failed to take."""
    out = {}
    for m in models:
        f1, sec = evaluate({b: m for b in bugs}, cell, split)
        if f1 is not None:
            out[f"always {m}"] = (f1, sec)
    oracle = {b: max((m for m in models if (b, m, split) in cell),
                     key=lambda m: cell[(b, m, split)]["f1"], default=None)
              for b in bugs}
    f1, sec = evaluate(oracle, cell, split)
    if f1 is not None:
        out["ORACLE (upper bound)"] = (f1, sec)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_cells(cell, bugs, models, split):
    print(f"\n--- per bug type, {split} ---")
    print(f"{'bug':<6} {'model':<34} {'n':>5} {'prec':>8} {'recall':>8} "
          f"{'F1':>7} {'GPU-sec':>10}")
    for b in bugs:
        for m in models:
            c = cell.get((b, m, split))
            if c:
                print(f"{b.upper():<6} {m:<34} {c['n']:>5} {c['precision']:>7.2f}% "
                      f"{c['recall']:>7.2f}% {c['f1']:>7.3f} {c['sec']:>10.0f}")


def print_ranking(rows, title):
    print(f"\n--- {title} ---")
    print(f"{'strategy':<40} {'avg F1':>8} {'GPU-sec':>10}")
    for name, (f1, sec) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:<40} {f1:>8.3f} {sec:>10.0f}")


def print_generalisation(cell, bugs, models):
    """The Sec 5.3 table: select vs held-out F1 per model, per bug type.

    Whatever the router does, this is the finding worth reprinting - the models
    that win on select are not the ones that hold up on unseen code shapes."""
    print("\n--- select -> held-out F1 drop ---")
    for b in bugs:
        print(f"\n  {b.upper()}")
        print(f"    {'model':<34} {'select':>8} {'held':>8} {'drop':>8}")
        for m in models:
            s, h = cell.get((b, m, "select")), cell.get((b, m, "held"))
            if s and h:
                print(f"    {m:<34} {s['f1']:>8.3f} {h['f1']:>8.3f} "
                      f"{h['f1'] - s['f1']:>+8.3f}")


# ---------------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser(
        description="Cost-aware model selection: cheapest model within tau F1 "
                    "of the best. Fitted on select variants, scored on held out.")
    ap.add_argument("logs", nargs="*", help=f"default: {LOG_DIR}/*.out")
    ap.add_argument("--holdout", type=int, default=12,
                    help="variants held out (default 12)")
    ap.add_argument("--tau", type=float, default=0.02,
                    help="F1 tolerance; 0 reproduces the Sec 5.2 argmax router")
    ap.add_argument("--sweep", action="store_true",
                    help="print the tau frontier (diagnostic, not a selection basis)")
    args = ap.parse_args(argv)

    import glob
    logs = args.logs or sorted(glob.glob(os.path.join(LOG_DIR, "*.out")))
    merged = load(logs)
    if not merged:
        print("no parseable runs found")
        return 2

    df = add_split(matched(to_frame(merged)), args.holdout)
    cell = cells(df)
    bugs = sorted(df["bug"].unique())
    models = sorted(df["model"].unique())

    print(f"{len(df)} cases, {len(models)} models, {len(bugs)} bug types")
    for b in bugs:
        sub = df[df["bug"] == b]
        sel = sub[sub["split"] == "select"]["variant"].nunique()
        held = sub[sub["split"] == "held"]["variant"].nunique()
        print(f"  {b.upper():<6} select {sel} var, held out {held} var, "
              f"{sub['file'].nunique()} files")

    print_cells(cell, bugs, models, "select")

    if args.sweep:
        print("\n--- tau frontier ---")
        print("held-out numbers here are DIAGNOSTIC. Picking the tau that "
              "maximises them\nis the Sec 5.1 error with extra steps.")
        print(f"\n{'tau':>6} {'fitted table':<52} {'held F1':>8} {'GPU-sec':>10}")
        for tau in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 1.0]:
            t = fit(cell, bugs, models, tau)
            f1, sec = evaluate(t, cell, "held")
            desc = " ".join(f"{b}->{(m or '?').split('-')[0]}" for b, m in t.items())
            if f1 is not None:
                print(f"{tau:>6.3f} {desc:<52} {f1:>8.3f} {sec:>10.0f}")
        return 0

    table = fit(cell, bugs, models, args.tau)
    print(f"\n--- fitted table (tau={args.tau}, select variants only) ---")
    for b in bugs:
        m = table[b]
        c = cell.get((b, m, "select")) if m else None
        best = max((cell[(b, x, "select")]["f1"] for x in models
                    if (b, x, "select") in cell), default=0.0)
        if c:
            print(f"  {b.upper():<6} -> {m:<34} select F1 {c['f1']:.3f} "
                  f"(best {best:.3f}, gap {best - c['f1']:+.3f})  "
                  f"{c['sec']:.0f} GPU-sec")

    sel_f1, sel_sec = evaluate(table, cell, "select")
    print(f"\n  table on select: avg F1 {sel_f1:.3f}, {sel_sec:.0f} GPU-sec "
          f"(in-sample, not a result)")

    rows = baselines(cell, bugs, models, "held")
    f1, sec = evaluate(table, cell, "held")
    if f1 is not None:
        rows[f"ROUTER (tau={args.tau}, fitted on select)"] = (f1, sec)
    print_ranking(rows, "HELD OUT - the only reportable numbers")

    print_cells(cell, bugs, models, "held")
    print_generalisation(cell, bugs, models)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
