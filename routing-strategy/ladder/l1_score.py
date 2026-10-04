#!/usr/bin/env python3
"""L1 -- score the OWASP model ladder (small -> medium -> large -> extra large).

Reads one jsonl per model from --dir (rows from d1_probe_reasoner.py) plus
timing.jsonl, and the 70/30 split. Prints, per model and per split:

* acc       share of units answered correctly (an unparsed verdict counts as wrong)
* auc       pooled AUC of p_vuln: pick one real and one fake alert, how often the
            real one gets the higher score (0.5 = coin flip)
* auc_cat   the same, but only pairs from the same OWASP category, averaged
* tpr-fpr   OWASP's own score: share of real alerts flagged minus share of fake
            alerts flagged (0 = saying "vulnerable" to everything)
* gpu_s     GPU-seconds per unit = run wall time / units (1 GPU, 32 concurrent)

Then the routing ceiling on the train split: the oracle (pick a right model per
unit if any exists) and the same oracle after shuffling each model's verdicts
inside each category. If the shuffled oracle is as high as the real one, the
headroom is luck, not complementary skill.
"""
import argparse, collections, glob, json, os, random


def auc(pairs):
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def score(rows, units):
    ok = [r for r in rows if r["pred_label"] is not None]
    acc = sum(r["pred_label"] == r["gold_label"] for r in ok) / len(rows)
    pairs = [((r["p_vuln"] if r["p_vuln"] is not None else 0.5), r["gold_label"]) for r in rows]
    bycat = collections.defaultdict(list)
    for r, p in zip(rows, pairs):
        bycat[units[r["unit_id"]]["project"]].append(p)
    cat = [a for a in (auc(v) for v in bycat.values()) if a is not None]
    pos = [r for r in rows if r["gold_label"] == 1]
    neg = [r for r in rows if r["gold_label"] == 0]
    tpr = sum(r["pred_label"] == 1 for r in pos) / len(pos)
    fpr = sum(r["pred_label"] == 1 for r in neg) / len(neg)
    return dict(n=len(rows), unparsed=len(rows) - len(ok), acc=acc, auc=auc(pairs),
                auc_cat=sum(cat) / len(cat), tpr_fpr=tpr - fpr, say_yes=(tpr * len(pos) + fpr * len(neg)) / len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/owasp_ladder/results")
    ap.add_argument("--units", default="data/owasp_ladder/units_owasp.jsonl")
    ap.add_argument("--split", default="data/owasp_ladder/split.json")
    a = ap.parse_args()

    units = {u["unit_id"]: u for u in map(json.loads, open(a.units))}
    split = json.load(open(a.split))["split"]
    timing = {}
    tp = os.path.join(a.dir, "timing.jsonl")
    if os.path.exists(tp):
        for l in open(tp):
            t = json.loads(l)
            c = t["wall_s"] / t["units"]          # fastest run per model, see l3_route.py
            timing[(t["model"], "owasp")] = min(c, timing.get((t["model"], "owasp"), c))
    def load(sfx):
        out = {}
        for f in sorted(glob.glob(os.path.join(a.dir, f"*__{sfx}.jsonl"))):
            rows = {}
            for l in open(f):
                r = json.loads(l)
                rows[r["unit_id"]] = r                  # last row per unit wins
            out[os.path.basename(f).split("__")[0]] = rows
        return out
    res, anon = load("owasp"), load("owasp_anon")

    print(f"{'model':22s} {'split':5s} {'n':>5s} {'unparsed':>8s} {'acc':>6s} {'auc':>6s} "
          f"{'auc_cat':>7s} {'tpr-fpr':>7s} {'say_yes':>7s} {'gpu_s':>6s}")
    for name, rows in res.items():
        for sp in ("train", "test", "all"):
            rs = [r for u, r in rows.items() if sp == "all" or split.get(u) == sp]
            s = score(rs, units)
            g = timing.get((name, "owasp"))
            print(f"{name:22s} {sp:5s} {s['n']:5d} {s['unparsed']:8d} {s['acc']:6.3f} "
                  f"{s['auc']:6.3f} {s['auc_cat']:7.3f} {s['tpr_fpr']:7.3f} {s['say_yes']:7.3f} "
                  f"{'' if g is None else f'{g:6.3f}'}")

    # memorisation control: same model, benchmark names hidden (all units)
    for name, rows in anon.items():
        if name in res:
            o, h = score(list(res[name].values()), units), score(list(rows.values()), units)
            print(f"anon {name:17s} auc {o['auc']:.3f} -> {h['auc']:.3f}  auc_cat {o['auc_cat']:.3f} -> "
                  f"{h['auc_cat']:.3f}  tpr-fpr {o['tpr_fpr']:.3f} -> {h['tpr_fpr']:.3f}")

    # routing ceiling on the train split: real oracle vs category-shuffled oracle
    names = list(res)
    ids = [u for u in units if split.get(u) == "train" and all(u in res[m] for m in names)]
    if len(names) < 2 or not ids:
        return
    right = {m: {u: res[m][u]["pred_label"] == res[m][u]["gold_label"] for u in ids} for m in names}
    oracle = sum(any(right[m][u] for m in names) for u in ids) / len(ids)
    best = max(names, key=lambda m: sum(right[m].values()))
    rng, nulls = random.Random(0), []
    bycat = collections.defaultdict(list)
    for u in ids:
        bycat[units[u]["project"]].append(u)
    for _ in range(200):
        sh = {}
        for m in names:
            sh[m] = {}
            for cat, us in bycat.items():
                v = [res[m][u]["pred_label"] for u in us]
                rng.shuffle(v)
                for u, p in zip(us, v):
                    sh[m][u] = p == units[u]["gold_label"]
        nulls.append(sum(any(sh[m][u] for m in names) for u in ids) / len(ids))
    nulls.sort()
    print(f"\ntrain n={len(ids)}  best single={best} acc {sum(right[best].values())/len(ids):.3f}  "
          f"oracle {oracle:.3f}  shuffled oracle {sum(nulls)/len(nulls):.3f} "
          f"[{nulls[5]:.3f}, {nulls[194]:.3f}]")


if __name__ == "__main__":
    main()
