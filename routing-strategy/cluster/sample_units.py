#!/usr/bin/env python3
"""Build the probe unit sets from PrimeVul's own splits.

Two sets, and the split between them is the whole of protocol §1 (fit -> freeze
-> score once):

  fit/    valid_paired  + a sample of unpaired valid   -- everything the router
                                                          is allowed to learn from
  test/   test_paired   + a sample of unpaired test    -- scored once, at the end

PrimeVul's splits are already temporal and group-aware, so this script never
re-splits; it only SUBSAMPLES within a split, because probing 24,788 test units
x 6 models sequentially is not affordable and is not needed to separate models.

Two properties the sampling must preserve, or the metrics stop meaning anything:

  * pairs stay together. A (vulnerable, patched) pair split across the sample
    makes pair accuracy undefined for that pair, which is the one metric that
    catches a model reacting to surface form (problem doc §1.2).
  * the unpaired sample keeps the natural base rate. PrimeVul's test split is
    2.2% vulnerable; the paired file is 50%. F1 and VD-Score read completely
    differently on the two, so both are probed and reported separately rather
    than pooled into one misleading number.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def to_unit(r: dict, split: str, pair_id: str | None) -> dict:
    cwe = r.get("cwe") or []
    return dict(
        unit_id=str(r["idx"]),
        code=r["func"],
        gold_label=int(r["target"]),
        commit=str(r.get("commit_id") or r["idx"]),
        project=r.get("project"),
        # GOLD CWE. Fit-split supervision ONLY -- never an inference-time input.
        # PrimeVul leaves `cwe` empty on benign functions, so "has a CWE" is
        # close to the label itself; feeding it to a router at inference leaks
        # the answer through the S11 graph's task nodes (docs/04 §leakage).
        gold_cwe=(cwe[0] if cwe else None),
        cve=r.get("cve"),
        pair_id=pair_id,
        split=split,
    )


def pairs_from(records: list[dict]) -> list[tuple[dict, dict]]:
    """PrimeVul's *_paired files store a pair as two CONSECUTIVE records that
    share a commit_id: the vulnerable function then its patched version.

    Do NOT pair on func_hash -- it hashes the function body, so the two members
    of a pair have different values by construction. data.py's `pair_id` does
    exactly that and is wrong for this file; it is fixed here at the source.
    """
    out, i = [], 0
    while i + 1 < len(records):
        a, b = records[i], records[i + 1]
        if a.get("commit_id") == b.get("commit_id") and a["target"] != b["target"]:
            out.append((a, b))
            i += 2
        else:
            i += 1              # unpaired straggler, dropped with a count
    return out


def build(data: Path, split: str, n_pairs: int, n_unpaired: int,
          seed: int) -> list[dict]:
    rng = random.Random(seed)
    units: list[dict] = []

    prs = pairs_from(read(data / f"primevul_{split}_paired.jsonl"))
    take = prs if n_pairs <= 0 else rng.sample(prs, min(n_pairs, len(prs)))
    for a, b in take:
        pid = f"{split}-pair-{a['idx']}"
        units += [to_unit(a, split, pid), to_unit(b, split, pid)]
    print(f"[{split}] {len(prs)} pairs available, {len(take)} taken "
          f"-> {len(take) * 2} units")

    if n_unpaired > 0:
        seen = {u["unit_id"] for u in units}
        pool = [r for r in read(data / f"primevul_{split}.jsonl")
                if str(r["idx"]) not in seen]
        vuln = [r for r in pool if r["target"] == 1]
        benign = [r for r in pool if r["target"] == 0]
        rate = len(vuln) / max(len(pool), 1)
        n_v = max(1, round(n_unpaired * rate))
        smp = (rng.sample(vuln, min(n_v, len(vuln)))
               + rng.sample(benign, min(n_unpaired - n_v, len(benign))))
        rng.shuffle(smp)
        units += [to_unit(r, split, None) for r in smp]
        print(f"[{split}] unpaired: {len(smp)} units at the natural base rate "
              f"{rate:.3%} ({n_v} vulnerable)")
    return units


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path.home() / "data/primevul")
    ap.add_argument("--out", type=Path, default=Path.home() / "data/units")
    ap.add_argument("--split", required=True, choices=["train", "valid", "test"])
    ap.add_argument("--pairs", type=int, default=0, help="0 = all pairs")
    ap.add_argument("--unpaired", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    units = build(a.data, a.split, a.pairs, a.unpaired, a.seed)

    # A function can take part in more than one pair (one patch, several
    # vulnerable revisions), so the paired file can list it twice. Left in, it
    # becomes a duplicate (unit_id, model) row and table.load() cannot pivot.
    seen_ids, deduped = set(), []
    for u in units:
        if u["unit_id"] in seen_ids:
            continue
        seen_ids.add(u["unit_id"])
        deduped.append(u)
    if len(deduped) != len(units):
        print(f"deduped {len(units) - len(deduped)} repeated unit_ids "
              f"(a function appearing in several pairs)")
    units = deduped

    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"{a.name or a.split}.jsonl"
    with open(dst, "w") as fh:
        for u in units:
            fh.write(json.dumps(u) + "\n")
    pos = sum(u["gold_label"] for u in units)
    print(f"wrote {dst}: {len(units)} units, {pos} vulnerable "
          f"({pos / max(len(units),1):.1%})")
