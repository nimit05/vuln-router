#!/usr/bin/env python3
"""Q5 -- the positive control for E12, with the significance test E12 lacked.

E12 measured 15 configurations on paired PrimeVul and found every one BELOW a
guesser that answers "vulnerable" at the configuration's own rate. All 15 are
8-14B open models, so the claim it supports is "models up to 14B fail", not
"the task defeats LLMs". This adds a 32B reasoner on the SAME 300 pairs and
reports every row against the same null.

Metrics, all per pair (one buggy version, one fixed version of the same
function):

  pair-both-right  fraction of pairs where the buggy one is called vulnerable
                   AND the fixed one is not. It rewards answering DIFFERENTLY
                   within a pair, so a guesser saying "vulnerable" with
                   probability r, independently per function, scores r*(1-r) --
                   0.25 at r=0.5. That is the baseline, and it moves with the
                   model's own yes-rate.
  excess           observed minus that baseline. Negative = worse than guessing
                   at its own rate. This is the number E12 reports.
  bootstrap CI     resample PAIRS with replacement, recompute the excess with
                   the baseline recomputed at the resampled yes-rate. NEW here:
                   E12 gave a point estimate, and a positive control needs an
                   interval before "it crossed zero" means anything.
  agreement        how often the model gives the SAME verdict to both versions.
                   Independent guessing agrees r^2+(1-r)^2 of the time; more
                   than that means the model is reacting to the function's
                   surface form rather than to the defect.
"""
import argparse, json, os
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "..", "data", "think")
TABLE = os.path.join(HERE, "..", "data", "table_dedup.jsonl")


def paired(df, col="q"):
    """(n_pairs, both-right, agreement) on pairs that have both versions."""
    v = df.pivot_table(index="pair_id", columns="gold_label", values=col).dropna()
    if 0 not in v.columns or 1 not in v.columns:
        return 0, float("nan"), float("nan")
    return len(v), float(((v[1] == 1) & (v[0] == 0)).mean()), float((v[1] == v[0]).mean())


def row(tag, d, n_perm=300, n_boot=2000):
    d = d.copy()
    # a row with no parsed verdict is scored 0 ("not vulnerable"), which is what
    # probe.py does and what E12 did -- kept for comparability, counted here so
    # a configuration that mostly failed to answer cannot pass as a decisive one.
    n_unparsed = int(pd.to_numeric(d["pred"], errors="coerce").isna().sum())
    d["q"] = pd.to_numeric(d["pred"], errors="coerce").fillna(0).astype(int)
    d["gold_label"] = pd.to_numeric(d["gold_label"], errors="coerce").astype(int)
    d = d.dropna(subset=["pair_id"])

    # the yes-rate must be measured on the units that ENTER the pair metric.
    # table_dedup has 5,820 units but only 913 paired ones, and the unpaired
    # functions pull r away from the rate the baseline is supposed to describe.
    complete = d.pivot_table(index="pair_id", columns="gold_label", values="q").dropna().index
    dp = d[d.pair_id.isin(complete)]
    r = float(dp.q.mean())
    n_pairs, obs, ag = paired(d)
    analytic, ag_ind = r * (1 - r), r**2 + (1 - r) ** 2

    perm = []
    for _ in range(n_perm):
        dn = d.copy()
        dn["q"] = rng.permutation(dn.q.values)
        perm.append(paired(dn)[1])
    perm_mu, perm_sd = float(np.mean(perm)), float(np.std(perm))

    # bootstrap the excess over pairs, baseline recomputed inside each resample
    wide = d.pivot_table(index="pair_id", columns="gold_label", values="q").dropna()
    bug, fix = wide[1].values, wide[0].values
    idx = np.arange(len(bug))
    boot = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        b, f = bug[s], fix[s]
        rb = float(np.concatenate([b, f]).mean())
        boot.append(float(((b == 1) & (f == 0)).mean()) - rb * (1 - rb))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    tok = pd.to_numeric(dp.get("out_tokens"), errors="coerce").mean() if "out_tokens" in dp else float("nan")
    return dict(config=tag, n_pairs=n_pairs, unparsed=n_unparsed, yes_rate=r, pair_both=obs,
                analytic=analytic, perm=perm_mu, perm_sd=perm_sd,
                excess=obs - analytic, lo=lo, hi=hi,
                agree=ag, agree_ind=ag_ind, out_tokens=tok)


def load(path):
    return pd.DataFrame([json.loads(l) for l in open(path)]).drop_duplicates("unit_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR, help="directory of probe jsonl files")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    want = [
        ("qwen32_nothink.jsonl", "Qwen3-32B no-think"),
        ("qwen32_think.jsonl",   "Qwen3-32B think"),
        ("qwen3_nothink.jsonl",  "Qwen3-8B no-think"),
        ("qwen3_think.jsonl",    "Qwen3-8B think"),
        ("budget_4096.jsonl",    "Qwen3-8B budget 4096"),
        ("phi4_direct.jsonl",    "phi-4 direct"),
        ("phi4_cot.jsonl",       "phi-4 CoT"),
        ("sc5_1024.jsonl",       "self-consistency k=5"),
    ]
    rows = []
    for fn, tag in want:
        p = os.path.join(a.dir, fn)
        if os.path.exists(p):
            rows.append(row(tag, load(p), n_boot=a.boot))
        else:
            print(f"[skip] {fn} not in {a.dir}")

    if os.path.exists(TABLE):
        t = pd.read_json(TABLE, lines=True)
        for m in ["granite-8b", "phi-3.8b", "qwen-7b", "qwen-1.5b", "deepseek-6.7b"]:
            x = t[t.model == m].rename(columns={"pred_label": "pred"})
            rows.append(row(m + " (all 913 pairs)", x[["pair_id", "gold_label", "pred"]], n_boot=a.boot))

    df = pd.DataFrame(rows)
    print()
    hdr = (f'{"configuration":24s} {"pairs":>5s} {"yes":>5s} {"pair-both":>9s} '
           f'{"r(1-r)":>7s} {"excess":>7s} {"95% CI":>18s} {"agree":>12s} {"out tok":>8s} {"n/a":>4s}')
    print(hdr); print("-" * len(hdr))
    for _, x in df.iterrows():
        print(f'{x.config:24s} {x.n_pairs:5d} {x.yes_rate:5.2f} {x.pair_both:9.3f} '
              f'{x.analytic:7.3f} {x.excess:+7.3f} [{x.lo:+.3f}, {x.hi:+.3f}] '
              f'{x.agree:5.2f} vs {x.agree_ind:4.2f} '
              f'{"        " if pd.isna(x.out_tokens) else f"{x.out_tokens:8.0f}"} {x.unparsed:4d}')
    print()
    print("excess > 0 with a CI excluding zero = the first configuration in this")
    print("project to beat a coin flip at its own yes-rate.")
    out = os.path.join(a.dir, "q5_pair_null.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
