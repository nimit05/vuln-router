#!/usr/bin/env python3
"""Branch-count router for LLMDFA-style pipelines, plus the pre-flight test
that says whether routing is worth attempting on a new corpus at all.

Four commands
--------------
  route      Decide, per program, which model to call. No LLM in the decision.
  preflight  Given measurements of >=2 models on the same cases, answer GO /
             NO-GO before anyone builds a router.
  export     LLMDFA .out logs -> the JSONL preflight reads.
  selftest   Reproduce the three published held-out probe numbers.

The rule
--------
Route the branchiest fraction q of the corpus to the SMALL model, the rest to
the LARGE model. q = 0.409, taken from the fitted float_* DBZ corpus and never
re-fitted. Relative rather than absolute (`n_if >= 12`) because the absolute cut
is tied to one corpus's branchiness: on family 2 the median n_if falls 9 -> 3
and the fixed cut lands in the wrong place. The relative form is never worse and
lifts the weakest matched-count z from +2.4 to +3.0.

Why the small model on the branchy side -- it looks backwards and is not:
on branchy programs the 3.8B emits ~half the false alarms of the 12B
(1.42 vs 2.27 per case at n_if >= 12) because it concedes less when Z3 repair
fails. On flat programs the ordering reverses (0.57 vs 0.05).

Scope, stated plainly: validated on LLMDFA divide-by-zero over Juliet Java, two
held-out form families, two inference engines. It does NOT transfer to IRIS.
Run `preflight` before assuming it applies anywhere else.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict

Q_FITTED = 0.409          # fraction of the fitted corpus with n_if >= 12
ABS_THRESHOLD = 12        # fallback when the corpus is too small to rank
DISPERSION_FLOOR = 5.0    # precision points; below this, gate 1 fails
DRAWS = 2000

IF_RE = re.compile(r"\bif\s*\(")
COND_RE = re.compile(r"\bif\s*\(([^\{]*)\)")


# --------------------------------------------------------------- feature side

MULTIPART_RE = re.compile(r"(_\d+)[a-g]$")   # Juliet _51a.java / _51b.java


def canonical_stem(stem):
    """Juliet splits one variant across _51a.java and _51b.java. Both halves are
    one program, so they collapse to the same key. Matching on the trailing
    letter alone would also mangle ordinary names like `AbstractTestCase`, and
    naive stem matching drops exactly the cross-file (i.e. hard) cases."""
    return MULTIPART_RE.sub(r"\1", stem)


def program_index(root):
    """canonical stem -> [paths], multi-part variants merged."""
    idx = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith(".java"):
                continue
            idx.setdefault(canonical_stem(fn[:-5]), []).append(
                os.path.join(dirpath, fn))
    return idx


def branch_count(paths):
    """n_if over the whole program, multi-file variants concatenated."""
    if isinstance(paths, str):
        paths = [paths]
    text = "\n".join(open(p, errors="replace").read() for p in sorted(set(paths)))
    return len(IF_RE.findall(text)), len(COND_RE.findall(text))


def relative_cut(values, q):
    """Smallest n_if cut whose selected set is closest to the branchiest q.

    Cases tied on n_if route together. Splitting a tie by filename would let a
    naming convention decide which model sees a program, which is not a rule
    anyone could defend or reproduce on a new corpus.
    """
    if not values:
        return ABS_THRESHOLD
    target = q * len(values)
    best, best_gap = max(values), None
    for t in sorted(set(values), reverse=True):
        gap = abs(sum(1 for v in values if v >= t) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = t, gap
    return best


# ------------------------------------------------------------------- metrics

def f1_of(tp, fp, gt):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / gt if gt else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def aggregate(rows):
    """rows: iterable of (tp, fp, gt) -> dict of corpus-level metrics."""
    rows = list(rows)
    tp = sum(r[0] for r in rows)
    fp = sum(r[1] for r in rows)
    gt = sum(r[2] for r in rows)
    return {
        "n": len(rows), "tp": tp, "fp": fp, "gt": gt,
        # A model that makes no positive prediction has UNDEFINED precision,
        # not 0%. Reporting it as 0 fakes dispersion: on PrimeVul one silent
        # model inflated the spread from 8.9 to 24.1 points and turned a
        # NO-GO into a PASS.
        "precision": 100 * tp / (tp + fp) if tp + fp else None,
        "recall": 100 * tp / gt if gt else None,
        "f1": f1_of(tp, fp, gt),
        "predicts": tp + fp,
    }


def case_error(tp, fp, gt):
    """Net error on one case: missed true positives plus false alarms."""
    return (gt - min(tp, gt)) + fp


# ------------------------------------------------------------------ commands

def cmd_route(args):
    if os.path.isdir(args.target):
        idx = program_index(args.target)
        programs = sorted(idx.items())
    else:
        programs = [(canonical_stem(os.path.basename(args.target)[:-5]), [args.target])]

    if not programs:
        sys.exit(f"no .java files found under {args.target}")

    feats = {stem: branch_count(paths) for stem, paths in programs}
    values = [n_if for n_if, _ in feats.values()]

    if args.threshold is not None:
        cut, mode = args.threshold, "absolute (forced)"
    elif len(values) < args.min_corpus:
        cut, mode = ABS_THRESHOLD, f"absolute (corpus < {args.min_corpus}, cannot rank)"
    else:
        cut, mode = relative_cut(values, args.q), f"relative (branchiest q={args.q})"

    assignments = []
    for stem, (n_if, n_cond) in sorted(feats.items()):
        model = args.small if n_if >= cut else args.large
        assignments.append({"program": stem, "n_if": n_if, "n_cond": n_cond, "model": model})

    n_small = sum(1 for a in assignments if a["model"] == args.small)
    if args.json:
        print(json.dumps({"cut": cut, "mode": mode, "assignments": assignments}, indent=2))
    else:
        print(f"# cut: n_if >= {cut}  [{mode}]")
        print(f"# {len(assignments)} programs -> {n_small} small ({100*n_small/len(assignments):.1f}%), "
              f"{len(assignments)-n_small} large")
        print(f"{'program':<58}{'n_if':>6}  model")
        for a in assignments:
            print(f"{a['program']:<58}{a['n_if']:>6}  {a['model']}")


def load_measurements(path):
    """JSONL, one record per (case, model):
         {"case": "...", "model": "...", "tp": 1, "fp": 0, "gt": 1, "sec": 12.3}
       Also accepts a JSON array of the same records."""
    text = open(path).read().strip()
    records = json.loads(text) if text.startswith("[") else \
        [json.loads(l) for l in text.splitlines() if l.strip()]
    by_model, extra = defaultdict(dict), {}
    for r in records:
        by_model[r["model"]][r["case"]] = (
            int(r["tp"]), int(r["fp"]), int(r["gt"]), float(r.get("sec", 0.0)))
        extra.setdefault(r["case"], r)
    return by_model, extra


def cmd_preflight(args):
    by_model, extra = load_measurements(args.measurements)
    models = sorted(by_model)
    if len(models) < 2:
        sys.exit("need at least 2 models measured on the same cases")

    common = sorted(set.intersection(*(set(by_model[m]) for m in models)))
    if not common:
        sys.exit("no cases are shared by all models")

    print(f"models   : {', '.join(models)}")
    print(f"cases    : {len(common)} shared by all\n")

    # --- gate 1: do the models disperse on the metric you are routing for? --
    per_model = {m: aggregate(by_model[m][c][:3] for c in common) for m in models}
    fmt = lambda v, u="%": "   n/a" if v is None else f"{v:>5.1f}{u}"
    print(f"{'model':<28}{'precision':>11}{'recall':>9}{'F1':>8}{'predicts':>10}")
    for m in models:
        st = per_model[m]
        print(f"{m:<28}{fmt(st['precision']):>11}{fmt(st['recall']):>9}"
              f"{st['f1']:>8.3f}{st['predicts']:>10}")

    # A precision computed from a handful of positive predictions is noise, and
    # a model that never predicts positive has no precision at all. Either one
    # silently inflates the spread: on PrimeVul, deepseek-6.7b answers
    # "vulnerable" 4 times in 5,820 and its 0.0% was setting the floor.
    silent = [m for m in models
              if per_model[m]["precision"] is None
              or per_model[m]["predicts"] < args.min_predictions]
    if silent:
        print(f"\n        excluded from the spread (fewer than"
              f" {args.min_predictions} positive predictions, so precision is"
              f" undefined or pure noise): {', '.join(silent)}")
    live = [m for m in models if m not in silent]
    if len(live) < 2:
        print("\nGATE 1  fewer than 2 models make any positive prediction. STOP.\n")
        print("VERDICT: NO-GO")
        return

    spreads = {}
    for key in ("precision", "recall", "f1"):
        vals = [per_model[m][key] for m in live if per_model[m][key] is not None]
        scale = 1.0 if key == "f1" else 1.0
        spreads[key] = (max(vals) - min(vals)) * (100 if key == "f1" else scale)
    print(f"\nGATE 1  spread across the {len(live)} live models:"
          f"  precision {spreads['precision']:.1f} pts"
          f" | recall {spreads['recall']:.1f} pts"
          f" | F1 {spreads['f1']:.1f} pts")
    if args.metric == "precision" and spreads["recall"] > 2 * spreads["precision"]:
        print(f"        CONFOUNDED: recall spans {spreads['recall']:.1f} points across"
              f" these models, so they sit at\n        very different operating points"
              f" and their precisions are not comparable.\n        Precision bought by"
              f" answering less is not skill. Gate 2 is the decisive one.")

    spread = spreads[args.metric]
    gate1 = spread >= DISPERSION_FLOOR
    have_feature = bool(args.bench or args.feature)
    print(f"        routing for {args.metric}: {spread:.1f} points "
          f"({'PASS' if gate1 else 'FAIL'}, floor {DISPERSION_FLOOR})")
    if not gate1:
        if max(spreads.values()) >= DISPERSION_FLOOR:
            best = max(spreads, key=spreads.__getitem__)
            print(f"        NOTE: they DO disperse on {best}"
                  f" ({spreads[best]:.1f} pts) -- rerun with --metric {best}"
                  f" if that is what you are routing for.")
        # Gate 1 compares models MARGINALLY. Routing needs them to differ
        # CONDITIONALLY -- two models equally good overall can still be good on
        # different cases, which is the routable case. So a gate-1 failure only
        # decides when there is no candidate feature to test.
        if not have_feature:
            print("        Every live model scores the same, and no pre-call feature")
            print("        was supplied to test. Nothing left to check. STOP.\n")
            print("VERDICT: NO-GO")
            return
        print("        Marginally equal -- but equal-overall models can still be good")
        print("        on DIFFERENT cases, which is exactly the routable case.")
        print("        Gate 2 decides.")

    # --- gate 2: does a pre-call rule beat a split of the same size? --------
    # This is the gate that decides, NOT the shuffled oracle below. A rule can
    # work while the oracle headroom is fake -- DBZ is exactly that case -- so
    # testing the oracle would reject the one corpus where routing pays.
    if not have_feature:
        print("\nGATE 2  skipped: needs a pre-call feature. Either --bench <java dir>"
              "\n        (computes n_if from source) or --feature <numeric jsonl field>.")
        print("\nVERDICT: INCOMPLETE -- gate 1 passed, but the gate that decides did not run.")
        return
    if args.small not in by_model or args.large not in by_model:
        sys.exit(f"--small/--large must name measured models: {', '.join(models)}")

    feat, missing, fname = {}, 0, args.feature or "n_if"
    if args.feature:
        bad = 0
        for c in common:
            raw = extra.get(c, {}).get(args.feature)
            try:
                feat[c] = float(raw)
            except (TypeError, ValueError):
                feat[c], bad = 0.0, bad + 1
        if bad == len(common):
            sys.exit(f"--feature {args.feature!r} is absent or non-numeric on every case. "
                     f"Gate 2 needs a number known BEFORE any model is called.")
        missing = bad
    else:
        idx = program_index(args.bench)
        for c in common:
            paths = idx.get(canonical_stem(c)) or idx.get(c)
            if paths:
                feat[c] = branch_count(paths)[0]
            else:
                feat[c], missing = 0, missing + 1
    if missing:
        print(f"\n        warning: {missing}/{len(common)} cases have no {fname}"
              f" -- treated as 0")
    if len(set(feat.values())) < 2:
        print(f"\nGATE 2  {fname} is constant across all cases: it cannot split anything.")
        print("\nVERDICT: NO-GO")
        return

    cut = args.threshold if args.threshold is not None else relative_cut(
        [feat[c] for c in common], args.q)
    if not any(feat[c] < cut for c in common):
        print(f"\nGATE 2  every case is at or above the cut ({fname} >= {cut});"
              f" the rule degenerates to always-{args.small}.")
        print("\nVERDICT: NO-GO")
        return
    to_small = [c for c in common if feat[c] >= cut]

    def score(small_set):
        small_set = set(small_set)
        return aggregate(
            (by_model[args.small] if c in small_set else by_model[args.large])[c][:3]
            for c in common)

    rng = random.Random(args.seed)
    null = [score(rng.sample(common, len(to_small)))["f1"] for _ in range(args.draws)]
    mu, sd = statistics.mean(null), statistics.pstdev(null) or 1e-9

    def evaluate(small_model, large_model):
        a, b = args.small, args.large
        args.small, args.large = small_model, large_model
        st = score(to_small)
        args.small, args.large = a, b
        return st, (st["f1"] - mu) / sd

    routed, z = evaluate(args.small, args.large)
    rev, z_rev = evaluate(args.large, args.small)
    beaten = sum(1 for v in null if v >= routed["f1"]) / len(null)
    gate2 = z >= args.z_floor

    # precision at the nearest-recall fixed model -- the honest comparison
    cands = [m for m in live if per_model[m]["recall"] is not None]
    near = min(cands, key=lambda m: abs(per_model[m]["recall"] - routed["recall"]))

    print(f"\nGATE 2  rule: {fname} >= {cut:g} -> {args.small}, else {args.large}"
          f"  ({len(to_small)}/{len(common)} cases to small)")
    print(f"        routed                 : F1 {routed['f1']:.3f}  "
          f"prec {fmt(routed['precision'])}  recall {fmt(routed['recall'])}")
    print(f"        matched-count random   : F1 {mu:.3f} +/- {sd:.3f}  "
          f"({args.draws} draws)")
    print(f"        z                      : {z:+.1f}   beaten by {100*beaten:.1f}% of splits"
          f"  ({'PASS' if gate2 else 'FAIL'}, floor {args.z_floor})")
    print(f"        reversed direction     : F1 {rev['f1']:.3f}   z {z_rev:+.1f}"
          f"   ({'PASS' if z_rev >= args.z_floor else 'FAIL'})")
    if max(z, z_rev) >= args.z_floor:
        print("        NOTE: the direction must be FIXED BEFORE you see this table.")
        print("              Both are printed so that picking the winner here is a")
        print("              visible choice, not an invisible one. Selecting the")
        print("              passing direction post hoc fits your test set, and the")
        print("              result will not replicate on a held-out family.")
    print(f"        nearest-recall model   : {near} "
          f"prec {per_model[near]['precision']:.1f}% recall {per_model[near]['recall']:.1f}%"
          f"  -> {(routed['precision'] or 0) - per_model[near]['precision']:+.1f} precision pts "
          f"at {(routed['recall'] or 0) - per_model[near]['recall']:+.1f} recall")

    # --- information only: is the ORACLE worth chasing? ---------------------
    def oracle_f1(tables):
        return aggregate(
            tables[min(models, key=lambda m: case_error(*tables[m][c][:3]))][c][:3]
            for c in common)["f1"]

    real = oracle_f1(by_model)
    shuf = []
    for _ in range(min(args.draws, 500)):
        perm = {}
        for m in models:
            rows = [by_model[m][c][:3] for c in common]
            rng.shuffle(rows)
            perm[m] = dict(zip(common, rows))
        shuf.append(oracle_f1(perm))
    print(f"\nINFO    oracle {real:.3f} vs shuffled oracle "
          f"{statistics.mean(shuf):.3f} +/- {statistics.pstdev(shuf):.3f}")
    if real <= statistics.mean(shuf):
        print("        The oracle gap is shared blind spots, not opportunity.")
        print("        Do not quote it as headroom or as a target. It does NOT")
        print("        mean a rule cannot work -- gate 2 decides that.")

    print()
    if gate2:
        print("VERDICT: GO -- this rule carries information on this corpus,")
        print("         PROVIDED its direction was fixed before this run.")
        print("         Report precision at MATCHED RECALL against a matched-count")
        print("         random split, never F1 against the best fixed model alone.")
    else:
        if z <= -args.z_floor:
            print(f"VERDICT: NO-GO -- and note the rule is reliably WORSE than an")
            print(f"         arbitrary split (z {z:+.1f}), which means {fname} does carry")
            print(f"         signal, pointed the other way. That is a hypothesis to test")
            print(f"         on held-out data, NOT a result to claim from this table.")
        else:
            print("VERDICT: NO-GO -- the rule does no better than an arbitrary split")
            print("         of the same size. Try another pre-call feature, or stop.")


# ------------------------------------------------------------------ selftest

CASE_RE = re.compile(r"^\{'input_token_cost'.*\}\s*$", re.M)
JAVA_RE = re.compile(r"^(/\S+\.java)\s*$", re.M)
HEAD_RE = re.compile(r"model=(\S+)\s+bug=(\S+)")
SHORT = {"Phi-4-mini-instruct": "Phi", "Mistral-Nemo-Instruct-2407": "Nemo",
         "Qwen2.5-Coder-7B-Instruct": "Qwen", "granite-3.1-8b-instruct": "Granite"}


def parse_log(path):
    """-> (model_short, bug, {case: (tp, fp, gt, sec)})"""
    text = open(path, errors="replace").read()
    head = HEAD_RE.search(text)
    if not head:
        return None, None, {}
    raw = head.group(1).split("/")[-1]
    model, bug = SHORT.get(raw, raw), head.group(2)
    events = [(m.start(), "f", m.group(1)) for m in JAVA_RE.finditer(text)]
    events += [(m.start(), "c", m.group(0)) for m in CASE_RE.finditer(text)]
    events.sort()
    out, pending = {}, None
    for _, kind, val in events:
        if kind == "f":
            pending = val.split("/")[-1][:-5]
        elif pending is not None:
            try:
                d = ast.literal_eval(val)
            except Exception:
                pending = None
                continue
            tp = min(d["analysis_result"]["TPs"], d["ground_truth"]["TPs"])
            out[pending] = (tp, d["analysis_result"]["FPs"], d["ground_truth"]["TPs"],
                            d["single time cost"])
            pending = None
    return model, bug, out


PUBLISHED = {   # probe -> (Phi, Nemo, router_abs12, router_rel)
    "off815":  (0.575, 0.639, 0.699, 0.699),
    "off1481": (0.594, 0.649, 0.691, 0.701),
    "fam2":    (0.640, 0.698, 0.714, 0.723),
}


def cmd_selftest(args):
    logdir, bench = args.logs, args.bench
    idx = program_index(bench)
    ok = True
    print(f"{'probe':<9}{'series':<14}{'F1':>8}{'published':>11}{'delta':>8}  ")
    for probe, (p_phi, p_nemo, p_abs, p_rel) in PUBLISHED.items():
        tables = {}
        for fn in sorted(os.listdir(logdir)):
            if not fn.endswith(".out") or probe not in fn:
                continue
            model, _bug, rows = parse_log(os.path.join(logdir, fn))
            if model in ("Phi", "Nemo") and rows:
                tables.setdefault(model, {}).update(rows)
        if len(tables) < 2:
            print(f"{probe:<9}  (logs not found, skipped)")
            continue
        common = sorted(set(tables["Phi"]) & set(tables["Nemo"]))
        feats = {}
        for c in common:
            paths = idx.get(canonical_stem(c)) or idx.get(c)
            feats[c] = branch_count(paths)[0] if paths else 0
        cut_rel = relative_cut([feats[c] for c in common], Q_FITTED)

        series = {
            "always Phi":  [tables["Phi"][c][:3] for c in common],
            "always Nemo": [tables["Nemo"][c][:3] for c in common],
            f"router abs{ABS_THRESHOLD}":
                [(tables["Phi"] if feats[c] >= ABS_THRESHOLD else tables["Nemo"])[c][:3]
                 for c in common],
            f"router rel{cut_rel}":
                [(tables["Phi"] if feats[c] >= cut_rel else tables["Nemo"])[c][:3]
                 for c in common],
        }
        for (name, rows), pub in zip(series.items(), (p_phi, p_nemo, p_abs, p_rel)):
            got = aggregate(rows)["f1"]
            delta = got - pub
            flag = "ok" if abs(delta) <= args.tol else "MISMATCH"
            ok &= abs(delta) <= args.tol
            print(f"{probe:<9}{name:<14}{got:>8.3f}{pub:>11.3f}{delta:>+8.3f}  {flag}")
    print("\nselftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_export(args):
    """LLMDFA .out logs -> the JSONL `preflight` consumes."""
    by_model = defaultdict(dict)
    for logdir in args.logs:
        for fn in sorted(os.listdir(logdir)):
            if not fn.endswith(".out"):
                continue
            model, bug, rows = parse_log(os.path.join(logdir, fn))
            if not model or not rows:
                continue
            if args.bug and bug != args.bug:
                continue
            by_model[model].update(rows)
    if not by_model:
        sys.exit("no logs matched")
    out = open(args.out, "w") if args.out else sys.stdout
    n = 0
    for model, rows in sorted(by_model.items()):
        for case, (tp, fp, gt, sec) in sorted(rows.items()):
            out.write(json.dumps({"case": case, "model": model, "tp": tp,
                                  "fp": fp, "gt": gt, "sec": sec}) + "\n")
            n += 1
    if args.out:
        out.close()
        print(f"wrote {n} records over {len(by_model)} models -> {args.out}")


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("route", help="assign a model to each program")
    r.add_argument("target", help="a .java file or a directory of them")
    r.add_argument("--small", default="Phi-4-mini-instruct")
    r.add_argument("--large", default="Mistral-Nemo-Instruct-2407")
    r.add_argument("--q", type=float, default=Q_FITTED)
    r.add_argument("--threshold", type=int, default=None,
                   help=f"force an absolute n_if cut instead of the relative rule")
    r.add_argument("--min-corpus", type=int, default=30,
                   help="below this many programs, fall back to the absolute cut")
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=cmd_route)

    p = sub.add_parser("preflight", help="GO / NO-GO before building a router")
    p.add_argument("measurements", help="JSONL of {case, model, tp, fp, gt, sec}")
    p.add_argument("--bench", default=None, help="java dir, to compute n_if per case")
    p.add_argument("--feature", default=None,
                   help="numeric JSONL field to threshold on instead of n_if; "
                        "must be known BEFORE any model is called")
    p.add_argument("--metric", default="precision",
                   choices=("precision", "recall", "f1"),
                   help="which metric gate 1 tests dispersion on")
    p.add_argument("--small", default="Phi")
    p.add_argument("--large", default="Nemo")
    p.add_argument("--q", type=float, default=Q_FITTED)
    p.add_argument("--threshold", type=int, default=None, help="force an absolute cut")
    p.add_argument("--z-floor", type=float, default=2.0)
    p.add_argument("--min-predictions", type=int, default=30,
                   help="a model needs this many positive predictions before its "
                        "precision counts toward the gate-1 spread")
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_preflight)

    s = sub.add_parser("selftest", help="reproduce the published probe numbers")
    s.add_argument("--logs", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "LLMDFA", "reproduction", "logs_probe"))
    s.add_argument("--bench", default=os.environ.get("LLMDFA_BENCH", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "LLMDFA", "benchmark")))
    s.add_argument("--tol", type=float, default=0.002)
    s.set_defaults(func=cmd_selftest)

    e = sub.add_parser("export", help="LLMDFA .out logs -> measurement JSONL")
    e.add_argument("logs", nargs="+", help="one or more directories of .out logs")
    e.add_argument("--bug", default=None, help="keep only this bug type (xss / osci / dbz)")
    e.add_argument("--out", default=None, help="output path (default: stdout)")
    e.set_defaults(func=cmd_export)

    args = ap.parse_args()
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
