#!/usr/bin/env python3
"""H1 -- turn IRIS dataflow paths into Hybrid LLM's two-candidate router table.

Hybrid LLM (ICLR 2024) routes between exactly two models, a cheap SMALL and an
expensive LARGE, using a quality score per (query, model). Upstream's quality is
BARTScore against a reference answer. There is no reference answer here: the
model emits a keep/drop verdict on a CodeQL dataflow path, and what a wrong
verdict costs is not symmetric.

So quality is defined from the deployment loss of docs/01-problem.md 1.5,

    l(m,x) = c_FN * FN(m,x) + c_FP * FP(m,x),      rho = c_FN / c_FP

    q(m,x) = 1 - l(m,x) / c_FN   =   1            correct verdict
                                     0            false negative (a real
                                                  vulnerability is dropped)
                                     1 - 1/rho    false positive (an alert
                                                  survives that should not)

`rho` is the one knob. At rho = 1 quality IS accuracy and this reproduces the
paper's symmetric setting exactly. As rho grows, dropping a true path costs more
than keeping a false one, which is the regime IRIS is actually deployed in.
docs/06-objective-misalignment.md is the argument for why that matters: on a
population that is 85% negative, an objective that cannot tell "drops
everything" from "discriminates" picks the model that drops everything.

**Sampled quality.** `prob_2cls` needs a DISTRIBUTION of quality per model, which
upstream gets by generating several responses per query. The probe recorded one
greedy verdict per (path, model) plus `decision_logprob`, the log-probability of
the verdict token. That yields the same distribution more cheaply:

    p_vuln = exp(logprob)      if pred_label == 1
             1 - exp(logprob)  otherwise

and `--n-samples` Bernoulli(p_vuln) draws are laid down deterministically -- the
first round(p*n) samples say "vulnerable" -- so the file is reproducible and
`_match_prob` sees the same shape upstream feeds it. This substitution is the
one methodological liberty taken here and it is recorded in
docs/07-hybrid-llm.md rather than buried.

Output is upstream's own JSONL schema (instruction / input / output /
candidates[{text, scores, cost}]), so `third_party/HybridLLM/train_router.py`
can be pointed straight at it on a GPU box, plus our own flat fields for the
CPU backbone in h2.
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_labels import det_label, match_prob     # noqa: E402

INSTRUCTION = ("Decide whether the following CodeQL dataflow path is a true "
               "vulnerability of the stated CWE.\n")


def load_probe(probe_dir: str, model: str, suffix: str) -> dict[str, dict]:
    """unit_id -> row. Same validity rule as c1_score_table: a request that
    never reached the model has in_tokens 0 and must not read as a verdict."""
    hits = glob.glob(os.path.join(probe_dir, f"{model}__{suffix}.jsonl"))
    if not hits:
        raise SystemExit(f"no probe file {probe_dir}/{model}__{suffix}.jsonl")
    rows = {}
    for line in open(hits[0]):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("in_tokens"):
            continue
        rows[r["unit_id"]] = r
    return rows


def p_vuln(row: dict) -> float:
    """P(this model calls the path vulnerable), from the verdict token.

    A row with no verdict at all (every parse attempt failed) is the silent-zero
    defect of docs/05-probe-defects.md 1. It is NOT 'confidently benign', so it
    is reported at p = 0.5 and flagged, never allowed to look decisive."""
    lp = row.get("decision_logprob")
    if not row.get("sample_verdicts"):
        return 0.5
    if lp is None:
        return 1.0 if row["pred_label"] == 1 else 0.0
    p = math.exp(lp)
    p = min(max(p, 0.0), 1.0)
    return p if row["pred_label"] == 1 else 1.0 - p


def quality(pred: int, gold: int, rho: float) -> float:
    """q(m,x) in [0,1]. See the module docstring."""
    if pred == gold:
        return 1.0
    if gold == 1:                 # false negative: a real vulnerability dropped
        return 0.0
    return 1.0 - 1.0 / rho        # false positive: a junk alert kept


def samples(pv: float, gold: int, rho: float, n: int) -> list[float]:
    """n deterministic Bernoulli(pv) draws, scored. Deterministic so the table
    is byte-reproducible; the paper's sampling noise is not the object here."""
    k = int(round(pv * n))
    return [quality(1, gold, rho)] * k + [quality(0, gold, rho)] * (n - k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="data/gr/slices.jsonl")
    ap.add_argument("--probe-dir", default="data/gr/probe5",
                    help="probe5 is the only 5-model snapshot free of the "
                         "truncation defect; see docs/05-probe-defects.md")
    ap.add_argument("--suffix", default="units_paths")
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--rho", type=float, default=1.0,
                    help="c_FN / c_FP. 1.0 reproduces the paper's symmetric "
                         "quality; >1 prices a missed vulnerability higher")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--out-dir", default="data/hl")
    a = ap.parse_args()

    if a.rho < 1.0:
        raise SystemExit("--rho < 1 prices a false alarm above a missed "
                         "vulnerability; not a setting this task has")

    paths = [json.loads(l) for l in open(a.slices) if l.strip()]
    small = load_probe(a.probe_dir, a.small, a.suffix)
    large = load_probe(a.probe_dir, a.large, a.suffix)

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"pairs__{a.small}__{a.large}__rho{a.rho:g}.jsonl")

    n_written = n_unsliceable = n_unprobed = n_noverdict = 0
    agree = 0
    with open(out, "w") as fh:
        for p in paths:
            uid = p["path_id"]
            if not p.get("sliceable", 1):
                n_unsliceable += 1        # no model ever saw it; c1 keeps it
                continue
            rs, rl = small.get(uid), large.get(uid)
            if rs is None or rl is None:
                n_unprobed += 1
                continue
            gold = int(p["label"])
            for r in (rs, rl):
                if not r.get("sample_verdicts"):
                    n_noverdict += 1
            ps, pl = p_vuln(rs), p_vuln(rl)
            qs = samples(ps, gold, a.rho, a.n_samples)
            ql = samples(pl, gold, a.rho, a.n_samples)
            if det_label(qs, ql, 0.0) == 0:
                agree += 1

            rec = {
                # ---- upstream's schema, consumed by their train_router.py ----
                "id": uid,
                "instruction": INSTRUCTION,
                "input": f"CWE: {p['query_cwe']}\nFinding: {p['message']}\n\n"
                         f"{p['slice']}",
                "output": "vulnerable" if gold else "not vulnerable",
                "candidates": [
                    {"text": "", "model": a.small,
                     "scores": {"q": qs}, "cost": rs["seconds"]},
                    {"text": "", "model": a.large,
                     "scores": {"q": ql}, "cost": rl["seconds"]},
                ],
                # ---- ours: flat, for the CPU backbone and for h3 scoring ----
                "project": p["project"], "gold": gold, "cwe": p["query_cwe"],
                "pred_small": rs["pred_label"], "pred_large": rl["pred_label"],
                "sec_small": rs["seconds"], "sec_large": rl["seconds"],
                "p_small": ps, "p_large": pl,
                "q_small": float(sum(qs) / len(qs)),
                "q_large": float(sum(ql) / len(ql)),
                "feat": {k: p.get(k, 0) for k in
                         ("n_hops", "cross_file", "n_methods", "slice_loc",
                          "slice_chars", "max_nest", "n_if", "n_loop",
                          "n_call", "n_catch")},
            }
            fh.write(json.dumps(rec) + "\n")
            n_written += 1

    print(f"{a.small} (small) vs {a.large} (large), rho={a.rho:g}")
    print(f"  {n_written} routable paths written to {out}")
    print(f"  {n_unsliceable} unsliceable (kept by the filter, never routed), "
          f"{n_unprobed} unprobed")
    if n_noverdict:
        print(f"  [warn] {n_noverdict} (path,model) rows have NO verdict at all "
              f"-- scored at p=0.5, not as a confident 'benign'")
    print(f"  det_2cls at t=0: small suffices on {agree}/{n_written} "
          f"({100.0*agree/max(n_written,1):.1f}%)")


if __name__ == "__main__":
    main()
