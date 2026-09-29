"""Absolute vs relative branch-count threshold on the held-out probes.

`ROUTER_V2_PERCASE.md` §8 calls threshold transfer the live weakness: the rule's
DIRECTION transfers to every held-out condition, but the cut point `n_if >= 12`
is "measurably off-centre" on the second form family. If a relative formulation
-- route the branchiest q of whatever corpus you are given -- is stable where the
absolute one drifts, that is the deployable rule and the stronger claim.

q is taken from the FITTED corpus only (the `float_*` DBZ cases in logs/ and
logs_router/), never from the probe being scored, so the relative rule is as
blind to the held-out data as the absolute one is.

    export LLMDFA_BENCH=/path/to/LLMDFA/benchmark
    python3 relative_threshold.py
"""
from __future__ import annotations

import collections
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_core as rc  # noqa: E402  (loads the fitted corpus on import)
from route_feats import extract, index  # noqa: E402
from score_probe import BASE, agg, load_probe  # noqa: E402

PROBES = ("off815", "off1481", "fam2")
LO, HI = "Nemo", "Phi"          # lo = below the cut (large), hi = above it (small)
ABS_THR = 12
DRAWS = 2000
SEED = 0


def f1(stats, cases, pick):
    return agg(stats, cases, pick)["f1"]


def quantile_threshold(feat, cases, q):
    """Smallest cut whose selected set is closest to the branchiest `q` of cases.

    Cases tied on n_if are routed together. Splitting a tie by filename would
    let an arbitrary naming convention decide which model sees a program, which
    is not a rule anyone could defend or reproduce on a new corpus.
    """
    values = sorted({feat[f] for f in cases}, reverse=True)
    target = q * len(cases)
    best, best_gap = values[0], None
    for t in values:
        n = sum(1 for f in cases if feat[f] >= t)
        gap = abs(n - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = t, gap
    return best


def matched_null(stats, cases, k, rng):
    """Same NUMBER routed to the small model, chosen at random."""
    out = []
    for _ in range(DRAWS):
        chosen = set(rng.sample(cases, k))
        out.append(f1(stats, cases, lambda f: HI if f in chosen else LO))
    return statistics.mean(out), statistics.pstdev(out)


def main() -> None:
    rng = random.Random(SEED)
    idx = index()

    fitted = sorted(rc.ALL["dbz"])
    fitted_nif = [rc.FEAT[("dbz", f)]["n_if"] for f in fitted]
    q_star = sum(1 for x in fitted_nif if x >= ABS_THR) / len(fitted_nif)
    print(f"fitted corpus: {len(fitted)} dbz cases, "
          f"median n_if {statistics.median(fitted_nif):.0f}, "
          f"fraction with n_if >= {ABS_THR} = {q_star:.3f}")
    print(f"relative rule under test: route the branchiest {q_star:.1%} of the corpus "
          f"to {HI}\n")

    probes = {}
    for suffix in PROBES:
        stats = load_probe(os.path.join(BASE, "logs_probe"), suffix)
        cases = sorted(f for f in set.intersection(*[set(v) for v in stats.values()])
                       if f in idx)
        feat = {f: extract(idx[f])["n_if"] for f in cases}
        probes[suffix] = (stats, cases, feat)

    # Two of the probes are the same case set under different engines; say so
    # rather than counting them as independent evidence.
    sig = {s: tuple(sorted(c)) for s, (_, c, _) in probes.items()}
    dupes = collections.defaultdict(list)
    for s, c in sig.items():
        dupes[c].append(s)
    for group in dupes.values():
        if len(group) > 1:
            print(f"note: {' and '.join(group)} are the SAME {len(sig[group[0]])} cases "
                  f"under different engines -- one case set, not two\n")

    print(f"{'probe':<9}{'rule':<12}{'thr':>5}{'->small':>9}{'F1':>8}"
          f"{'vs Phi':>9}{'random':>9}{'z':>7}")
    summary = {}
    for suffix, (stats, cases, feat) in probes.items():
        n = len(cases)
        phi = f1(stats, cases, lambda f: HI)
        rows = []

        k_abs = sum(1 for f in cases if feat[f] >= ABS_THR)
        rows.append(("absolute", ABS_THR, k_abs,
                     f1(stats, cases, lambda f: HI if feat[f] >= ABS_THR else LO)))

        thr_rel = quantile_threshold(feat, cases, q_star)
        k_rel = sum(1 for f in cases if feat[f] >= thr_rel)
        rows.append(("relative", thr_rel, k_rel,
                     f1(stats, cases, lambda f: HI if feat[f] >= thr_rel else LO)))

        for name, thr, k, score in rows:
            mu, sd = matched_null(stats, cases, k, rng)
            z = (score - mu) / sd if sd else float("nan")
            print(f"{suffix:<9}{name:<12}{thr:>5}{k:>9}{score:>8.3f}"
                  f"{score - phi:>+9.3f}{mu:>9.3f}{z:>+7.1f}")
            summary.setdefault(suffix, {})[name] = score
        print()

    print("threshold sweep on each probe -- FITTED ON THE PROBE, not deployable;")
    print("shown only to locate the optimum and measure how off-centre the cut is:")
    header = f"  {'thr':>4}" + "".join(f"{s:>11}" for s in PROBES)
    print(header)
    for t in (4, 6, 8, 10, 12, 14, 16, 18, 20):
        line = f"  {t:>4}"
        for suffix in PROBES:
            stats, cases, feat = probes[suffix]
            line += f"{f1(stats, cases, lambda f, t=t: HI if feat[f] >= t else LO):>11.3f}"
        print(line)

    print("\npaired bootstrap of (relative - absolute), 2,000 resamples over cases:")
    for suffix in PROBES:
        stats, cases, feat = probes[suffix]
        thr_rel = quantile_threshold(feat, cases, q_star)
        diffs = []
        for _ in range(DRAWS):
            resample = [rng.choice(cases) for _ in cases]
            rel = f1(stats, resample, lambda f: HI if feat[f] >= thr_rel else LO)
            ab = f1(stats, resample, lambda f: HI if feat[f] >= ABS_THR else LO)
            diffs.append(rel - ab)
        diffs.sort()
        lo_ci, hi_ci = diffs[int(0.025 * DRAWS)], diffs[int(0.975 * DRAWS)]
        point = summary[suffix]["relative"] - summary[suffix]["absolute"]
        share = 100.0 * sum(1 for d in diffs if d > 0) / len(diffs)
        print(f"  {suffix:<9} {point:+.3f}  95% CI [{lo_ci:+.3f}, {hi_ci:+.3f}]  "
              f"relative wins in {share:.0f}% of resamples")


if __name__ == "__main__":
    main()
