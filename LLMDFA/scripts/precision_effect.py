"""Does the router raise PRECISION, and at what recall?

The project's research question, asked directly. F1 can rise for the wrong
reason -- trading recall away buys precision, and a precision number quoted
without its recall is not a claim about anything. So every row here carries
both, and the router is compared against:

  * each fixed model,
  * the fixed model with the CLOSEST recall (precision at matched recall is the
    only honest way to say a filter is better),
  * a matched-count random split (same number of cases to the small model,
    chosen at random) -- the null that killed the sibling project's precision
    results in its E13.

It then repeats the question across the three bug types of the fitted corpus,
which is where the condition for routing to work at all becomes visible.

    export LLMDFA_BENCH=/path/to/LLMDFA/benchmark
    python3 precision_effect.py
"""
from __future__ import annotations

import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from route_feats import extract, index  # noqa: E402
from score_probe import BASE, load_probe  # noqa: E402

PROBES = ("off815", "off1481", "fam2")
LO, HI = "Nemo", "Phi"
THR = 12
DRAWS = 2000
SEED = 0


def counts(stats, cases, pick):
    tp = fp = gt = 0
    for f in cases:
        a, b, g, _ = stats[pick(f)][f]
        tp += a
        fp += b
        gt += g
    return tp, fp, gt


def prf(tp, fp, gt):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / gt if gt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> None:
    rng = random.Random(SEED)
    idx = index()
    for suffix in PROBES:
        stats = load_probe(os.path.join(BASE, "logs_probe"), suffix)
        cases = sorted(f for f in set.intersection(*[set(v) for v in stats.values()])
                       if f in idx)
        feat = {f: extract(idx[f])["n_if"] for f in cases}
        models = sorted(stats)
        rule = lambda f: HI if feat[f] >= THR else LO

        print(f"=== {suffix}  ({len(cases)} cases, {models}) ===")
        print(f"{'strategy':<28}{'prec':>8}{'recall':>8}{'F1':>8}{'TP':>6}{'FP':>6}")
        rows = {}
        for m in models:
            p, r, f = prf(*counts(stats, cases, lambda f, m=m: m))
            tp, fp, _ = counts(stats, cases, lambda f, m=m: m)
            rows[m] = (p, r, f)
            print(f"{'always ' + m:<28}{p:>7.1%}{r:>8.1%}{f:>8.3f}{tp:>6}{fp:>6}")
        rp, rr, rf = prf(*counts(stats, cases, rule))
        rtp, rfp, _ = counts(stats, cases, rule)
        print(f"{'ROUTER':<28}{rp:>7.1%}{rr:>8.1%}{rf:>8.3f}{rtp:>6}{rfp:>6}")

        # precision at matched recall: the fixed model whose recall is closest
        nearest = min(models, key=lambda m: abs(rows[m][1] - rr))
        np_, nr, _ = rows[nearest]
        print(f"\n  closest fixed model by recall: {nearest} "
              f"(recall {nr:.1%} vs router {rr:.1%})")
        print(f"  precision {np_:.1%} -> {rp:.1%}  = {rp - np_:+.1%} at "
              f"{'equal' if abs(nr - rr) < 1e-9 else 'near-equal'} recall")
        best_p = max(models, key=lambda m: rows[m][0])
        print(f"  best fixed precision of any model: {best_p} {rows[best_p][0]:.1%} "
              f"(recall {rows[best_p][1]:.1%})  -> router {rp - rows[best_p][0]:+.1%}")

        # matched-count random null on precision
        k = sum(1 for f in cases if feat[f] >= THR)
        null_p, null_r = [], []
        for _ in range(DRAWS):
            chosen = set(rng.sample(cases, k))
            p, r, _ = prf(*counts(stats, cases, lambda f: HI if f in chosen else LO))
            null_p.append(p)
            null_r.append(r)
        mu, sd = statistics.mean(null_p), statistics.pstdev(null_p)
        print(f"  matched-count random split: precision {mu:.1%} +- {sd:.1%}, "
              f"recall {statistics.mean(null_r):.1%}")
        print(f"  router precision {rp:.1%}  ->  z {(rp - mu) / sd:+.1f}, "
              f"beaten by {100.0 * sum(1 for x in null_p if x >= rp) / DRAWS:.1f}% of random splits")

        # bootstrap the precision gain over the nearest-recall fixed model
        gains, recall_deltas = [], []
        for _ in range(DRAWS):
            resample = [rng.choice(cases) for _ in cases]
            p_router, r_router, _ = prf(*counts(stats, resample, rule))
            p_fixed, r_fixed, _ = prf(*counts(stats, resample, lambda f: nearest))
            gains.append(p_router - p_fixed)
            recall_deltas.append(r_router - r_fixed)
        gains.sort()
        lo_ci, hi_ci = gains[int(0.025 * DRAWS)], gains[int(0.975 * DRAWS)]
        print(f"  bootstrap precision gain vs {nearest}: {rp - np_:+.1%} "
              f"95% CI [{lo_ci:+.1%}, {hi_ci:+.1%}] "
              f"{'excludes 0' if lo_ci > 0 else 'INCLUDES 0'}")
        print(f"  recall change over the same resamples: "
              f"{statistics.mean(recall_deltas):+.1%}\n")


def by_bug_type() -> None:
    """The same question on XSS, OSCI and DBZ.

    IN-SAMPLE: these bug types were part of the corpus the rule was selected on,
    so this is not held-out evidence. It is here because it shows the condition
    under which routing can help at all -- whether the models disperse on the
    metric you care about. Where they do not, no router has anything to sell.
    """
    import route_core as rc

    rng = random.Random(SEED)
    lo, hi = "Mistral-Nemo-Instruct-2407", "Phi-4-mini-instruct"
    print("=" * 66)
    print("BY BUG TYPE (fitted corpus -- IN-SAMPLE, shown for the condition only)")
    print("=" * 66)
    for bug in ("xss", "osci", "dbz"):
        cases = sorted(rc.ALL[bug])
        feat = {f: rc.FEAT[(bug, f)]["n_if"] for f in cases}
        rule = lambda f: hi if feat[f] >= THR else lo

        def c(pick):
            tp = fp = gt = 0
            for f in cases:
                a, b, g, _ = rc.STATS[bug][f][pick(f)]
                tp += a
                fp += b
                gt += g
            return tp, fp, gt

        rows = {m: prf(*c(lambda f, m=m: m)) for m in rc.MODELS}
        rp, rr, rf = prf(*c(rule))
        spread = max(p for p, _, _ in rows.values()) - min(p for p, _, _ in rows.values())
        print(f"\n{bug}  ({len(cases)} cases)   "
              f"precision spread across models: {spread:.1%}")
        for m in sorted(rows, key=lambda m: -rows[m][2]):
            p, r, f = rows[m]
            print(f"  always {rc.SHORT[m]:<10} prec {p:>6.1%}  recall {r:>6.1%}  F1 {f:.3f}")
        print(f"  ROUTER            prec {rp:>6.1%}  recall {rr:>6.1%}  F1 {rf:.3f}")
        near = min(rc.MODELS, key=lambda m: abs(rows[m][1] - rr))
        k = sum(1 for f in cases if feat[f] >= THR)
        null = []
        for _ in range(1000):
            chosen = set(rng.sample(cases, k))
            null.append(prf(*c(lambda f: hi if f in chosen else lo))[0])
        mu, sd = statistics.mean(null), statistics.pstdev(null)
        print(f"  vs nearest recall ({rc.SHORT[near]}): precision "
              f"{rp - rows[near][0]:+.1%} at {rr - rows[near][1]:+.1%} recall")
        print(f"  matched-count random precision {mu:.1%} +- {sd:.1%}  ->  "
              f"router z {(rp - mu) / sd:+.1f}")
    print("\nThe router only sells something where the models disperse. On XSS and")
    print("OSCI every model sits within a point of every other, and the router is")
    print("at the random-split null. On DBZ they span 37.6-51.0% and it is not.")


if __name__ == "__main__":
    main()
    by_bug_type()
