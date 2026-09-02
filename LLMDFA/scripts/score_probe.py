"""Score a held-out-form probe: fixed models, threshold router, oracle.

Unlike route_core.py (which joins the 4-model x 3-bug fitted corpus), this
takes an explicit set of .out logs that all cover the SAME case set and
evaluates the deployed rule on them. Used for the offset-815 / offset-1481
held-out-form probes.

    python3 score_probe.py --suffix off815 --lo Phi --hi Nemo --thr 12
"""
import os, sys, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from route_logsig import parse
from route_feats import index, extract

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reproduction")
SHORT = {"Phi-4-mini-instruct": "Phi", "Qwen2.5-Coder-7B-Instruct": "Qwen",
         "granite-3.1-8b-instruct": "Granite", "Mistral-Nemo-Instruct-2407": "Nemo",
         "Qwen2.5-Coder-1.5B-Instruct": "Qwen1.5B"}


def f1_of(tp, fp, gt):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / gt if gt else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def load_probe(logdir, suffix):
    """model_short -> {file: (tp_capped, fp, gt, sec)}"""
    out = {}
    for fn in sorted(os.listdir(logdir)):
        if not fn.endswith(".out") or suffix not in fn:
            continue
        r = parse(os.path.join(logdir, fn))
        if not r:
            continue
        m = SHORT.get(r["model"], r["model"])
        d = out.setdefault(m, {})
        for row in r["rows"]:
            tp = min(row["analysis_result"]["TPs"], row["ground_truth"]["TPs"])
            d[row["file"]] = (tp, row["analysis_result"]["FPs"],
                              row["ground_truth"]["TPs"], row["single time cost"])
    return out


def agg(stats, common, pick):
    """pick(file) -> model short name."""
    tp = fp = gt = 0
    cost = 0.0
    for f in common:
        a, b, g, t = stats[pick(f)][f]
        tp += a; fp += b; gt += g; cost += t
    return dict(f1=f1_of(tp, fp, gt), tp=tp, fp=fp, gt=gt, sec=cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", required=True, help="substring identifying the probe logs")
    ap.add_argument("--logdir", default=os.path.join(BASE, "logs_probe"))
    ap.add_argument("--feature", default="n_if")
    ap.add_argument("--thr", type=int, default=12)
    ap.add_argument("--lo", default="Nemo", help="model used when feature < thr")
    ap.add_argument("--hi", default="Phi", help="model used when feature >= thr")
    ap.add_argument("--sweep", action="store_true", help="threshold sweep (oracle, not deployable)")
    a = ap.parse_args()

    stats = load_probe(a.logdir, a.suffix)
    if not stats:
        sys.exit(f"no logs matching '{a.suffix}' in {a.logdir}")
    common = sorted(set.intersection(*[set(v) for v in stats.values()]))
    idx = index()
    feat = {f: extract(idx[f])[a.feature] for f in common if f in idx}
    missing = [f for f in common if f not in idx]
    common = [f for f in common if f in idx]

    print(f"probe='{a.suffix}'  models={sorted(stats)}  "
          f"per-model n={[len(v) for _, v in sorted(stats.items())]}  "
          f"matched={len(common)}" + (f"  UNINDEXED={len(missing)}" if missing else ""))
    forms = collections.Counter(f.split('__', 1)[1].rpartition('_')[0] for f in common if '__' in f)
    print("forms: " + ", ".join(f"{k} ({v})" for k, v in sorted(forms.items())))
    print()

    print(f"{'strategy':<44}{'F1':>7}{'prec':>8}{'rec':>8}{'TP':>6}{'FP':>6}{'GPU-s':>9}")
    rows = []
    for m in sorted(stats):
        rows.append((f"always {m}", agg(stats, common, lambda f, m=m: m)))
    if a.lo in stats and a.hi in stats:
        rule = lambda f: a.hi if feat[f] >= a.thr else a.lo
        rows.append((f"ROUTER {a.feature}>={a.thr} ? {a.hi} : {a.lo}",
                     agg(stats, common, rule)))
    best = lambda f: min(stats, key=lambda m: ((stats[m][f][2] - stats[m][f][0])
                                               + stats[m][f][1], stats[m][f][3]))
    rows.append((f"Oracle ({len(stats)}-model bound)", agg(stats, common, best)))
    for name, r in rows:
        p = 100 * r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else 0
        rc = 100 * r["tp"] / r["gt"] if r["gt"] else 0
        print(f"{name:<44}{r['f1']:>7.3f}{p:>7.1f}%{rc:>7.1f}%"
              f"{r['tp']:>6}{r['fp']:>6}{r['sec']:>9.0f}")

    # false alarms per case by branch band -- the mechanism check
    print(f"\nFP/case and TP/case by {a.feature} band:")
    bands = [("0-7", 0, 7), ("8-11", 8, 11), (f">={a.thr}", a.thr, 10**9)]
    if a.thr != 12:
        bands = [("0-7", 0, 7), ("8-11", 8, 11), (">=12", 12, 10**9)]
    ms = sorted(stats)
    print(f"{'band':<8}{'n':>5}" + "".join(f"{m+' FP':>13}" for m in ms)
          + "".join(f"{m+' TP':>13}" for m in ms))
    for label, lo, hi in bands:
        fs = [f for f in common if lo <= feat[f] <= hi]
        if not fs:
            continue
        line = f"{label:<8}{len(fs):>5}"
        for m in ms:
            line += f"{sum(stats[m][f][1] for f in fs)/len(fs):>13.2f}"
        for m in ms:
            line += f"{sum(stats[m][f][0] for f in fs)/len(fs):>13.2f}"
        print(line)

    if a.sweep and a.lo in stats and a.hi in stats:
        print(f"\nthreshold sweep ({a.hi} above / {a.lo} below) -- FITTED ON THIS DATA, not deployable:")
        for t in (4, 6, 8, 10, 12, 14, 16, 18):
            r = agg(stats, common, lambda f, t=t: a.hi if feat[f] >= t else a.lo)
            n_hi = sum(1 for f in common if feat[f] >= t)
            print(f"  {a.feature}>={t:<3} F1={r['f1']:.3f}  ->{a.hi}: {n_hi}/{len(common)}")


if __name__ == "__main__":
    main()
