#!/usr/bin/env python3
"""H5 -- add the token counts upstream's loader requires to the pair rows.

`hybrid_llm/pair_ranker/data.py:format_scores_costs` reads
`token_num_prompt` and `token_num_responses` off every candidate,
unconditionally. `h1_build_pairs.py` never emitted them: the linear backbone
reads `scores.q` and `cost` and nothing else, so the omission was invisible
until upstream's own loader ran.

They are not invented here. Both are measured quantities sitting in the probe
(`in_tokens`, `out_tokens`), joined on `unit_id`, which is the pair row's `id`.

What this does NOT change: `cost`. Upstream would overwrite it from
`model2prompt_cost`, but that table is keyed on gpt/phi/mistral/llama names and
ours are qwen/granite, so no prefix matches and the GPU-seconds this project
uses as its cost axis survive untouched. That is the intended outcome, not an
accident -- verified by `--check`.

    python3 hybridllm/h5_add_tokens.py --pairs data/hl/pairs__<s>__<l>__rho<r>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROBE_DIR = "data/gr/probe5"


def load_probe(short: str) -> dict[str, tuple[int, int]]:
    """unit_id -> (in_tokens, out_tokens) for one model."""
    path = os.path.join(PROBE_DIR, f"{short}__units_paths.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no probe for {short} at {path}")
    out: dict[str, tuple[int, int]] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["unit_id"]] = (int(r["in_tokens"] or 0),
                                 int(r["out_tokens"] or 0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default=None,
                    help="default: rewrite --pairs in place")
    ap.add_argument("--check", action="store_true",
                    help="report cost-table collisions and exit non-zero on one")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.pairs) if l.strip()]
    if not rows:
        raise SystemExit(f"{a.pairs} is empty")

    shorts = [c["model"] for c in rows[0]["candidates"]]
    probes = {s: load_probe(s) for s in shorts}
    print(f"{len(rows)} pair rows, candidates = {shorts}")

    # Upstream's price table is keyed by model-name prefix. A collision would
    # silently replace GPU-seconds with a dollar figure, so it is checked.
    from_upstream = ("gpt-4o", "gpt-35-turbo", "phi-3-mini", "phi-3-medium",
                     "mistral-7b", "mistral-8x7b", "llama-31-8b", "codestral-22b")
    clashes = [s for s in shorts if any(s.startswith(k) for k in from_upstream)]
    if clashes:
        print(f"  COST TABLE COLLISION: {clashes} -- upstream would overwrite "
              f"`cost`, and this project's cost axis is GPU-seconds")
        if a.check:
            sys.exit(1)
    else:
        print(f"  no cost-table collision: `cost` stays GPU-seconds")
    if a.check:
        sys.exit(0)

    missing = 0
    for r in rows:
        for c in r["candidates"]:
            hit = probes[c["model"]].get(r["id"])
            if hit is None:
                missing += 1
                c["token_num_prompt"] = 0
                c["token_num_responses"] = 0
            else:
                c["token_num_prompt"], c["token_num_responses"] = hit

    if missing:
        print(f"  WARNING {missing} (row, model) pairs had no probe entry -> 0")

    out = a.out or a.pairs
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
