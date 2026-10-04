#!/usr/bin/env python3
"""L8 -- score the zero-shot Jev stand-in (l7) on the professor's ladder
(Qwen3-1.7B / 8B / 32B / gpt-oss-120b), on IRIS paths and on the OWASP test split.

Each unit goes to the tier the stand-in gives the highest probability (what
jev-router does). Its cost = the chosen model's GPU-s + the judge call (Qwen3-8B,
measured). Compared with:
  single models      each ladder model on everything
  best-on-OWASP      the ladder model with the best OWASP-train accuracy
  cascade (E20)      Qwen3-8B on everything; in each project/category the 30% of units
                     with the highest 8B P(vuln) also go to Qwen3-32B, whose score
                     then replaces the 8B's (the IRIS result of E20)
  random same mix    the stand-in's model mix, units shuffled (2,000 draws)
Metrics: accuracy, TPR - FPR, AUC within project, recall@10 (per project, share of
its real bugs among the 10 highest-scored units, averaged over projects with a real
bug), GPU-s per unit.
"""
import argparse, collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l6_eval_cross import auc_within, cost_table, load_runs   # noqa: E402

TIERS = ["qwen3-1.7b-nothink", "qwen3-8b-nothink", "qwen3-32b-nothink", "gpt-oss-120b-low"]


def recall_at(score, y, grp, k=10):
    vals = []
    for g in set(grp):
        m = np.where(grp == g)[0]
        if y[m].sum() == 0:
            continue
        top = m[np.argsort(-score[m], kind="stable")[:k]]
        vals.append(y[top].sum() / y[m].sum())
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owasp", default="results/2026-09-owasp-ladder")
    ap.add_argument("--iris", default="results/2026-09-iris-cross")
    ap.add_argument("--iris-units", default="data/gr/units_paths.jsonl")
    ap.add_argument("--jev", default="results/2026-09-jev-standin")
    a = ap.parse_args()

    ow_lab = {r["unit_id"]: r for r in map(json.loads, open(f"{a.owasp}/labels.jsonl"))}
    split = json.load(open(f"{a.owasp}/split.json"))["split"]
    ir_units = {u["unit_id"]: u for u in map(json.loads, open(a.iris_units))}
    jt = {(t["variant"], t["data"]): t["wall_s"] / t["units"] for t in map(json.loads, open(f"{a.jev}/timing.jsonl"))}

    # best ladder model on OWASP train (accuracy)
    ow = load_runs(f"{a.owasp}/probes", "owasp")
    tr = [u for u in ow_lab if split[u] == "train"]
    best = max(TIERS, key=lambda m: np.mean([ow[m][u]["pred_label"] == ow_lab[u]["gold_label"] for u in tr]))

    for data in ("iris", "owasp"):
        if data == "iris":
            runs, cost = load_runs(f"{a.iris}/probes", "iris"), cost_table(f"{a.iris}/probes/timing.jsonl")
            ids = sorted(ir_units)
            lab = {u: (int(ir_units[u]["gold_label"]), ir_units[u]["project"]) for u in ids}
            title = f"IRIS, {len(ids)} paths (15% real), groups = projects"
        else:
            runs, cost = ow, cost_table(f"{a.owasp}/probes/timing.jsonl")
            ids = sorted(u for u in ow_lab if split[u] == "test")
            lab = {u: (int(ow_lab[u]["gold_label"]), ow_lab[u]["project"]) for u in ids}
            title = f"OWASP test split, {len(ids)} units (65% real), groups = categories"
        n = len(ids)
        Y = np.array([lab[u][0] for u in ids]); G = np.array([lab[u][1] for u in ids])
        PR = np.array([[runs[m][u]["pred_label"] or 0 for m in TIERS] for u in ids])
        PV = np.array([[0.5 if runs[m][u]["p_vuln"] is None else runs[m][u]["p_vuln"] for m in TIERS] for u in ids])
        cv = np.array([cost[m] for m in TIERS])

        def stats(verdict, score, g):
            tf = verdict[Y == 1].mean() - verdict[Y == 0].mean()
            return (verdict == Y).mean(), tf, auc_within(score, Y, G), recall_at(score, Y, G), float(np.mean(g))

        rows = []
        for j, m in enumerate(TIERS):
            rows.append((f"single {m}" + ("  <- best on OWASP train" if m == best else ""),
                         stats(PR[:, j], PV[:, j], cv[j]), None))
        # E20 cascade: 8B everywhere, top 30% per group by 8B score -> 32B
        esc = np.zeros(n, bool)
        for g in set(G):
            m_ = np.where(G == g)[0]
            k = int(round(0.3 * len(m_)))
            esc[m_[np.argsort(-PV[m_, 1], kind="stable")[:k]]] = True
        vc = np.where(esc, PR[:, 2], PR[:, 1]); sc = np.where(esc, PV[:, 2], PV[:, 1])
        rows.append((f"cascade 8B -> 32B, top 30% per group ({esc.mean():.0%} up)",
                     stats(vc, sc, cv[1] + esc * cv[2]), None))
        for v in ("generic", "informed"):
            J = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{a.jev}/jev_{v}__{data}.jsonl")}
            ch = np.array([TIERS.index(max(J[u]["probs"], key=J[u]["probs"].get)) for u in ids])
            judge = jt[(v, data)]
            s = stats(PR[np.arange(n), ch], PV[np.arange(n), ch], cv[ch] + judge)
            rng = np.random.default_rng(0)
            sims = []
            for _ in range(2000):
                sh = rng.permutation(ch)
                sims.append(((PR[np.arange(n), sh] == Y).mean(), auc_within(PV[np.arange(n), sh], Y, G)))
            sims = np.array(sims)
            z = ((s[0] - sims[:, 0].mean()) / (sims[:, 0].std() + 1e-12),
                 (s[2] - sims[:, 1].mean()) / (sims[:, 1].std() + 1e-12))
            mix = collections.Counter(TIERS[i].split("-")[0] + "-" + TIERS[i].split("-")[1] for i in ch)
            rows.append((f"Jev stand-in, {v} descriptions", s, (z, mix, judge)))

        print(title)
        print(f"  {'router':52s} {'acc':>6s} {'tpr-fpr':>7s} {'auc':>6s} {'rec@10':>6s} {'gpu_s':>6s}")
        for name, (acc, tf, auc, r10, g), extra in rows:
            print(f"  {name:52s} {acc:6.3f} {tf:7.3f} {auc:6.3f} {r10:6.3f} {g:6.3f}")
            if extra:
                (za, zu), mix, judge = extra
                print(f"  {'':52s} vs random same mix: z(acc) {za:+.1f}, z(auc) {zu:+.1f}; judge {judge:.3f} GPU-s; "
                      "mix " + ", ".join(f"{k} {v}" for k, v in mix.most_common()))
        print()


if __name__ == "__main__":
    main()
