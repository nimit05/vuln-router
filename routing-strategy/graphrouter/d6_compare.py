#!/usr/bin/env python3
"""D6 analysis -- does hiding project identity change the model's skill?

For each model: within-project AUC and fixed-budget recall@10 on the original
code, a second run on the original (run-to-run noise), renamed (a) and strict (b).
Differences are PAIRED: the same paths, resampled together (bootstrap over paths
within each project), so the interval reflects only the change of text.

Reading: if renamed/strict AUC stays near the original and the drop is no bigger
than the rerun's wobble, the model reads the code; a fall toward 0.5 means it was
recognising known CVEs.
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pv(r):
    if r.get("p_vuln") is not None:
        return float(r["p_vuln"])
    return 0.5 if r.get("pred_label") is None else float(r["pred_label"])


def load(path):
    out = {}
    if not os.path.exists(path):
        return out
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            if not r.get("error"):
                out[r["unit_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d6", default=os.path.expanduser("~/nimit/graphrouter/d6_out"))
    ap.add_argument("--d1", default=os.path.join(ROOT, "data/gr/d1"))
    ap.add_argument("--slices", default=os.path.join(ROOT, "data/gr/slices.jsonl"))
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    meta = {json.loads(l)["path_id"]: json.loads(l) for l in open(a.slices) if l.strip()}

    runs = {
        "qwen3-32b": {"original": f"{a.d1}/qwen3-32b-nothink__units_paths.jsonl",
                      "original, rerun": f"{a.d6}/qwen3-32b-nothink__orig.jsonl",
                      "renamed (a)": f"{a.d6}/qwen3-32b-nothink__anonA.jsonl",
                      "strict (b)": f"{a.d6}/qwen3-32b-nothink__anonB.jsonl"},
        "qwen3-8b": {"original": f"{a.d1}/qwen3-8b-nothink__units_paths.jsonl",
                     "renamed (a)": f"{a.d6}/qwen3-8b-nothink__anonA.jsonl",
                     "strict (b)": f"{a.d6}/qwen3-8b-nothink__anonB.jsonl"},
    }
    for model, variants in runs.items():
        R = {k: load(v) for k, v in variants.items()}
        R = {k: v for k, v in R.items() if v}
        ids = sorted(set.intersection(*[set(v) for v in R.values()]))
        y = np.array([int(meta[i]["label"]) for i in ids])
        proj = np.array([meta[i]["project"] for i in ids])
        bug = [p for p in sorted(set(proj)) if 0 < y[proj == p].sum() < (proj == p).sum()]
        pos = {p: np.where((proj == p) & (y == 1))[0] for p in bug}
        neg = {p: np.where((proj == p) & (y == 0))[0] for p in bug}
        S = {k: np.array([pv(v[i]) for i in ids]) for k, v in R.items()}
        K = {k: np.array([v[i].get("pred_label") != 0 for i in ids]) for k, v in R.items()}

        def auc(s, pi=None, ni=None):
            num = den = 0.0
            for p in bug:
                sp = s[pos[p] if pi is None else pi[p]]; sn = s[neg[p] if ni is None else ni[p]]
                d = sp[:, None] - sn[None, :]
                num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
            return num / den

        def recall10(s):
            rec = []
            for p in bug:
                ix = np.where(proj == p)[0]
                top = ix[np.argsort(-(s[ix] + rng.random(len(ix)) * 1e-9))[:10]]
                rec.append(y[top].sum() / y[ix].sum())
            return float(np.mean(rec))

        boots = []
        for _ in range(a.boot):
            pi = {p: rng.choice(pos[p], len(pos[p])) for p in bug}
            ni = {p: rng.choice(neg[p], len(neg[p])) for p in bug}
            boots.append({k: auc(s, pi, ni) for k, s in S.items()})
        print(f"\n== {model}: {len(ids)} paths, {int(y.sum())} real bugs, {len(bug)} projects")
        print(f"   {'version':18} {'AUC':>6} {'95% CI':>15} {'change vs original':>20} {'95% CI':>17} "
              f"{'recall@10':>9} {'same verdict as orig':>20}")
        for k, s in S.items():
            b = np.array([x[k] for x in boots]); lo, hi = np.percentile(b, [2.5, 97.5])
            if k == "original":
                ch = ""
            else:
                d = np.array([x[k] - x["original"] for x in boots]); dlo, dhi = np.percentile(d, [2.5, 97.5])
                ch = f"{auc(s) - auc(S['original']):+.3f}  [{dlo:+.3f},{dhi:+.3f}]"
            agree = f"{100 * np.mean(K[k] == K['original']):.1f}%"
            print(f"   {k:18} {auc(s):6.3f} [{lo:.3f},{hi:.3f}] {ch:>38} {recall10(s):9.3f} {agree:>20}")


if __name__ == "__main__":
    main()
