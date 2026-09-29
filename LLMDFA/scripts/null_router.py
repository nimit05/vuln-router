"""Null battery for the LLMDFA per-case router, applied to the held-out probes.

`ROUTER_V2_PERCASE.md` reports +0.060 F1 over the best fixed model on 111
held-out-form cases. That number predates the reporting rules the sibling
routing-strategy project adopted in September, where two apparent wins and one
oracle "headroom" all evaporated once the right null was attached:

  * routing-strategy E2/E8 -- an oracle must be compared against a SHUFFLED
    oracle that preserves each model's marginal accuracy and destroys per-case
    alignment. On IRIS and on RouterBench the shuffled oracle scored HIGHER than
    the real one, meaning the headroom was diversity in guessing, not
    complementary skill.
  * routing-strategy E13 -- a filter must be compared against keeping the same
    NUMBER of items at random, not against no filter, and a pooled gain must be
    broken down per subgroup, because pooled precision over subgroups with
    different base rates is Simpson's paradox waiting to happen.

So this asks four questions of the +0.060:

  1. Does the RULE do work, or only the mixing proportion? Route the same NUMBER
     of cases to the small model, chosen at random, 2,000 times.
  2. How wide is the interval on 111 cases? Bootstrap over cases.
  3. Is the gain one form carrying the other two? Per-form breakdown.
  4. Is the oracle headroom real? Shuffled-oracle null.

    export LLMDFA_BENCH=/path/to/LLMDFA/benchmark
    python3 null_router.py --suffix off815 --lo Nemo --hi Phi --thr 12
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from route_feats import extract, index  # noqa: E402
from score_probe import BASE, agg, f1_of, load_probe  # noqa: E402

DRAWS = 2000
SEED = 0


def f1_from(stats, cases, pick):
    return agg(stats, cases, pick)["f1"]


def random_assignment_null(stats, cases, feat, thr, lo, hi, rng):
    """Same NUMBER of cases to `hi`, chosen at random. The matched-count null.

    If the rule carries no information, sending 45 arbitrary cases to the small
    model scores the same as sending the 45 branchiest.
    """
    n_hi = sum(1 for f in cases if feat[f] >= thr)
    out = []
    for _ in range(DRAWS):
        chosen = set(rng.sample(cases, n_hi))
        out.append(f1_from(stats, cases, lambda f: hi if f in chosen else lo))
    return out, n_hi


def bootstrap_gap(stats, cases, feat, thr, lo, hi, rng):
    """Resample cases with replacement; recompute the router's margin.

    Two margins, because "over the best fixed model" is ambiguous and the
    difference matters: `hindsight` compares against whichever fixed model wins
    on this held-out data (not a choice anyone could have made in advance),
    `deployable` compares against the model the fitted data would have told you
    to pick.
    """
    hindsight, deployable = [], []
    models = sorted(stats)
    for _ in range(DRAWS):
        resample = [rng.choice(cases) for _ in cases]
        router = f1_from(stats, resample, lambda f: hi if feat[f] >= thr else lo)
        fixed = {m: f1_from(stats, resample, lambda f, m=m: m) for m in models}
        hindsight.append(router - max(fixed.values()))
        deployable.append(router - fixed[hi])
    return hindsight, deployable


def shuffled_oracle_null(stats, cases, rng):
    """Oracle over models whose per-case outcomes have been permuted.

    Each model keeps its own total true and false positives; only the alignment
    between models is destroyed. If the shuffled oracle matches the real one,
    the headroom is models erring on different cases by chance rather than any
    of them being right where the others are wrong.
    """
    models = sorted(stats)
    gt = {f: stats[models[0]][f][2] for f in cases}
    out = []
    for _ in range(DRAWS):
        shuffled = {}
        for m in models:
            outcomes = [(stats[m][f][0], stats[m][f][1], stats[m][f][3]) for f in cases]
            rng.shuffle(outcomes)
            shuffled[m] = {f: o for f, o in zip(cases, outcomes)}
        tp = fp = total_gt = 0
        for f in cases:
            g = gt[f]
            best = min(models, key=lambda m: (g - min(shuffled[m][f][0], g)) + shuffled[m][f][1])
            a, b, _ = shuffled[best][f]
            tp += min(a, g)
            fp += b
            total_gt += g
        out.append(f1_of(tp, fp, total_gt))
    return out


def pct(value, dist):
    return 100.0 * sum(1 for d in dist if d >= value) / len(dist)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--logdir", default=os.path.join(BASE, "logs_probe"))
    ap.add_argument("--feature", default="n_if")
    ap.add_argument("--thr", type=int, default=12)
    ap.add_argument("--lo", default="Nemo")
    ap.add_argument("--hi", default="Phi")
    a = ap.parse_args()
    rng = random.Random(SEED)

    stats = load_probe(a.logdir, a.suffix)
    if not stats:
        sys.exit(f"no logs matching '{a.suffix}' in {a.logdir}")
    idx = index()
    cases = sorted(f for f in set.intersection(*[set(v) for v in stats.values()]) if f in idx)
    feat = {f: extract(idx[f])[a.feature] for f in cases}
    models = sorted(stats)

    router = f1_from(stats, cases, lambda f: a.hi if feat[f] >= a.thr else a.lo)
    fixed = {m: f1_from(stats, cases, lambda f, m=m: m) for m in models}
    best_fixed = max(fixed, key=fixed.get)

    print(f"probe '{a.suffix}': {len(cases)} cases, models {models}")
    print(f"rule: {a.feature} >= {a.thr} ? {a.hi} : {a.lo}\n")
    print(f"  router                      {router:.3f}")
    for m in sorted(fixed, key=fixed.get, reverse=True):
        print(f"  always {m:<20} {fixed[m]:.3f}")
    print(f"\n  margin over best fixed on this data ({best_fixed}): {router - fixed[best_fixed]:+.3f}")
    print(f"  margin over {a.hi} (what the fitted data would have picked): {router - fixed[a.hi]:+.3f}")

    print("\n[1] MATCHED-COUNT RANDOM ROUTING -- does the rule beat an arbitrary split?")
    null, n_hi = random_assignment_null(stats, cases, feat, a.thr, a.lo, a.hi, rng)
    mu, sd = statistics.mean(null), statistics.pstdev(null)
    z = (router - mu) / sd if sd else float("nan")
    print(f"  {n_hi} of {len(cases)} cases go to {a.hi}")
    print(f"  random assignment of the same count: {mu:.3f} +- {sd:.3f}")
    print(f"  router {router:.3f}  ->  z = {z:+.1f}, beaten by {pct(router, null):.1f}% of random splits")
    reversed_rule = f1_from(stats, cases, lambda f: a.lo if feat[f] >= a.thr else a.hi)
    print(f"  reversed rule ({a.lo} on branchy): {reversed_rule:.3f}  "
          f"({'sign is real' if reversed_rule < mu else 'SIGN NOT REAL -- reversal also beats random'})")

    print("\n[2] BOOTSTRAP over cases (2,000 resamples)")
    hind, dep = bootstrap_gap(stats, cases, feat, a.thr, a.lo, a.hi, rng)
    for name, dist, point in (("vs best fixed (hindsight)", hind, router - fixed[best_fixed]),
                              ("vs " + a.hi + " (deployable)", dep, router - fixed[a.hi])):
        lo_ci, hi_ci = sorted(dist)[int(0.025 * DRAWS)], sorted(dist)[int(0.975 * DRAWS)]
        sign = "excludes 0" if lo_ci > 0 else "INCLUDES 0"
        print(f"  {name:<28} {point:+.3f}  95% CI [{lo_ci:+.3f}, {hi_ci:+.3f}]  {sign}")

    print("\n[3] PER-FORM -- is one form carrying the result?")
    forms = collections.defaultdict(list)
    for f in cases:
        forms[f.split("__", 1)[1].rpartition("_")[0] if "__" in f else "?"].append(f)
    print(f"  {'form':<30}{'n':>4}{'router':>9}{'best fixed':>12}{'margin':>9}")
    wins = 0
    for name, fs in sorted(forms.items()):
        r = f1_from(stats, fs, lambda f: a.hi if feat[f] >= a.thr else a.lo)
        bf = max(f1_from(stats, fs, lambda f, m=m: m) for m in models)
        wins += r > bf
        print(f"  {name:<30}{len(fs):>4}{r:>9.3f}{bf:>12.3f}{r - bf:>+9.3f}")
    print(f"  beats the best fixed model in {wins}/{len(forms)} forms")

    print("\n[4] SHUFFLED-ORACLE NULL -- is the headroom real complementarity?")
    oracle = agg(stats, cases, lambda f: min(
        models, key=lambda m: ((stats[m][f][2] - stats[m][f][0]) + stats[m][f][1], stats[m][f][3])))["f1"]
    null_oracle = shuffled_oracle_null(stats, cases, rng)
    mu_o, sd_o = statistics.mean(null_oracle), statistics.pstdev(null_oracle)
    print(f"  real {len(models)}-model oracle      {oracle:.3f}")
    print(f"  shuffled oracle             {mu_o:.3f} +- {sd_o:.3f}")
    print(f"  excess over shuffle         {oracle - mu_o:+.3f}  "
          f"({'REAL complementarity' if oracle > mu_o + 2 * sd_o else 'headroom is not real -- do not chase it'})")


if __name__ == "__main__":
    main()
