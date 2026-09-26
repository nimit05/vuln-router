"""E14 — does the LLMDFA per-case routing rule transfer to IRIS?

LLMDFA's `ROUTER_V2_PERCASE.md` found a rule that beats the best fixed model by
+0.060 F1 on held-out source/sink forms: route branchy cases (n_if >= 12) to the
small model, simple ones to the large model. Its section 8 leaves the obvious
question open -- "whether the same shape recurs in IRIS and RepoAudit. One
system is a curiosity; three would be a finding."

This tests IRIS. No GPU: `data/gr/slices.jsonl` already carries n_if per path
from b3_slice.py, and `data/gr/probe5/` has all five models' verdicts on the
same path ids.

Three tables, in the order that matters:
  1. precision excess over the band base rate, by n_if quartile, with a
     permutation z -- is there a structural crossover at all?
  2. the trend across quartiles -- does any model's edge rise with branchiness,
     which is what the LLMDFA rule monetises?
  3. the per-project breakdown of any cell that looked good -- because pooled
     precision across projects with base rates from 0.04 to 0.46 is Simpson's
     paradox waiting to happen (E13).

Run: python nullcheck/llmdfa_transfer.py
"""
from __future__ import annotations

import collections
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLICES = ROOT / "data/gr/slices.jsonl"
PROBE = ROOT / "data/gr/probe5"

# deepseek-6.7b is excluded throughout: HANDOFF.md records kappa = -0.029
# between two runs of it at temperature 0, i.e. irreproducible.
MODELS = ["qwen-1.5b", "phi-3.8b", "qwen-7b", "granite-8b"]
DRAWS = 2000
SEED = 0


def load():
    slices = {}
    for line in SLICES.open():
        r = json.loads(line)
        slices[r["path_id"]] = r
    preds = {}
    for m in MODELS:
        rows = {}
        for line in (PROBE / f"{m}__units_paths.jsonl").open():
            r = json.loads(line)
            # in_tokens == 0 marks a failed call written as data on purpose;
            # counting it as a verdict is the defect HANDOFF.md warns about.
            if r.get("in_tokens", 0) > 0:
                rows[r["unit_id"]] = r
        preds[m] = rows
    ids = sorted(set(slices) & set.intersection(*[set(p) for p in preds.values()]))
    return slices, preds, ids


def perm_z(labels: list[int], kept_n: int, hits: int, rng: random.Random) -> float:
    """z of the observed hit count against keeping the same NUMBER at random.

    The matched-count null, not "no filter": per-project precision rises at
    small keep rates by chance alone (E3), so the baseline has to hold the
    count fixed.
    """
    if kept_n in (0, len(labels)):
        return float("nan")
    sims = [sum(rng.sample(labels, kept_n)) for _ in range(DRAWS)]
    sd = statistics.pstdev(sims)
    return (hits - statistics.mean(sims)) / sd if sd > 0 else float("nan")


def quartile_bands(slices, ids):
    nifs = sorted(slices[i]["n_if"] for i in ids)
    cuts = [nifs[int(f * len(nifs))] for f in (0.25, 0.50, 0.75)]
    return [(0, cuts[0]), (cuts[0] + 1, cuts[1]), (cuts[1] + 1, cuts[2]),
            (cuts[2] + 1, 10 ** 9)], cuts


def main() -> None:
    rng = random.Random(SEED)
    slices, preds, ids = load()
    base = sum(slices[i]["label"] for i in ids) / len(ids)
    print(f"{len(ids)} paths, base rate {base:.3f}, models {', '.join(MODELS)}")

    bands, cuts = quartile_bands(slices, ids)
    print(f"n_if quartile cuts: {cuts}\n")

    print("TABLE 1 - precision excess over band base rate, by branchiness")
    print(f"{'band':>8}{'n':>6}{'pos%':>7}  | " + " ".join(f"{m:>20}" for m in MODELS))
    print(f"{'':>8}{'':>6}{'':>7}  | " + " ".join(f"{'keep%  excess      z':>20}" for _ in MODELS))
    trend = {m: [] for m in MODELS}
    for lo, hi in bands:
        band = [i for i in ids if lo <= slices[i]["n_if"] <= hi]
        if not band:
            continue
        labels = [slices[i]["label"] for i in band]
        b_base = sum(labels) / len(band)
        name = f"{lo}-{hi if hi < 10 ** 9 else '+'}"
        row = f"{name:>8}{len(band):>6}{100 * b_base:>6.1f}  | "
        for m in MODELS:
            kept = [i for i in band if preds[m][i]["pred_label"] == 1]
            hits = sum(slices[i]["label"] for i in kept)
            prec = hits / len(kept) if kept else float("nan")
            trend[m].append(prec - b_base)
            row += f"{100 * len(kept) / len(band):>6.1f}{prec - b_base:>+8.3f}{perm_z(labels, len(kept), hits, rng):>6.1f} "
        print(row)

    print("\nTABLE 2 - does the edge rise with branchiness, as the LLMDFA rule needs?")
    for m in MODELS:
        t = trend[m]
        print(f"  {m:<12} " + " ".join(f"{x:+.3f}" for x in t) + f"    Q4-Q1 {t[-1] - t[0]:+.3f}")

    print("\nTABLE 3 - per-project breakdown on the simplest quartile")
    band = [i for i in ids if slices[i]["n_if"] <= cuts[0]]
    projects = collections.defaultdict(list)
    for i in band:
        projects[slices[i]["project"].split("_CVE")[0][:28]].append(i)
    for m in MODELS:
        wins = considered = 0
        excesses = []
        for rows in projects.values():
            kept = [i for i in rows if preds[m][i]["pred_label"] == 1]
            if len(kept) < 5:
                continue
            b_base = sum(slices[i]["label"] for i in rows) / len(rows)
            prec = sum(slices[i]["label"] for i in kept) / len(kept)
            excesses.append(prec - b_base)
            considered += 1
            wins += prec > b_base
        mean = sum(excesses) / len(excesses) if excesses else float("nan")
        print(f"  {m:<12} beats its own project base rate in {wins}/{considered}, "
              f"mean per-project excess {mean:+.3f}")

    print("\n  composition of what the pooled number rewards:")
    print(f"  {'project':<30}{'n':>5}{'pos':>5}{'base':>7}{'kept':>6}{'hits':>6}{'prec':>7}")
    for p, rows in sorted(projects.items(), key=lambda kv: -len(kv[1])):
        kept = [i for i in rows if preds["qwen-1.5b"][i]["pred_label"] == 1]
        if not kept and len(rows) < 20:
            continue
        pos = sum(slices[i]["label"] for i in rows)
        hits = sum(slices[i]["label"] for i in kept)
        prec = hits / len(kept) if kept else float("nan")
        print(f"  {p:<30}{len(rows):>5}{pos:>5}{pos / len(rows):>7.2f}"
              f"{len(kept):>6}{hits:>6}{prec:>7.3f}")


if __name__ == "__main__":
    main()
