#!/usr/bin/env python3
"""B3b -- turn slices into probe units for B6.

The model is shown what a triager is shown: the alert (rule, CWE, message)
followed by the dataflow slice. Giving it the code alone would be a different
task from the one the router is meant to route -- deciding whether *this alert*
is real, not whether some code is vulnerable in the abstract.

`commit` carries the PROJECT, because table.load() groups on it and the split
must be by project (paths within a repo share sources, sinks and idioms).

Only sliceable paths become units -- a model cannot be asked about a slice that
does not exist. The unsliceable ones are NOT forgotten: they stay in
slices.jsonl with `sliceable: 0`, and the scoring stage counts them as kept, the
same thing a filter that never saw them would do. Skipping them here and
dropping them there would shrink IRIS's denominator and inflate every row.
"""
from __future__ import annotations
import argparse, json, os

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default=os.path.expanduser(
        "~/nimit/graphrouter/data/slices.jsonl"))
    ap.add_argument("--out", default=os.path.expanduser(
        "~/nimit/graphrouter/data/units_slices.jsonl"))
    a = ap.parse_args()

    n = pos = skipped = skipped_pos = 0
    with open(a.out, "w") as o:
        for line in open(a.slices):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("sliceable", 1):
                skipped += 1
                skipped_pos += r["label"]
                continue
            hdr = (f"// CodeQL alert: {r['rule']}  ({r['query_cwe']})\n"
                   f"// {r['message'][:200]}\n\n")
            o.write(json.dumps(dict(
                unit_id=r["path_id"], code=hdr + r["slice"],
                gold_label=r["label"],
                commit=r["project"], project=r["project"],
                gold_cwe=r["gold_cwe"], query_cwe=r["query_cwe"],
                rule=r["rule"], pair_id=None, split="graph",
                n_hops=r["n_hops"], n_methods=r["n_methods"],
                cross_file=r["cross_file"], slice_loc=r["slice_loc"])) + "\n")
            n += 1
            pos += r["label"]
    print(f"wrote {a.out}: {n} units, {pos} positives ({pos/max(n,1):.2%})")
    print(f"  unsliceable paths held back: {skipped} "
          f"({skipped_pos} of them true positives)")
