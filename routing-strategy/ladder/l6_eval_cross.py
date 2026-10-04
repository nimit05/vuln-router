#!/usr/bin/env python3
"""L6 -- cross-benchmark test: routers fitted on OWASP, scored on IRIS paths.

Nothing here sees an IRIS answer before scoring. Every router decides from OWASP:
  best-on-OWASP  the single model with the best OWASP reward (what you would deploy)
  cwe            per CWE, the model that was best on the OWASP category with that CWE
                 (IRIS path's query CWE -> OWASP category); best-on-OWASP otherwise
  cascade        cheap model, escalate if |P(vuln) - 0.5| < w; pair and w from OWASP
  learned        l4 --apply: argmax P(right) - lam * gpu_s, fitted on all OWASP units
  direct         l4 --target label --apply: the same encoder answering "vulnerable?"
                 itself, no LLM (control: is the router just a classifier?)
Reference rows, not deployable: best-on-IRIS (hindsight) and random routing with the
learned router's model mix.

IRIS is 85% false alarms, so plain accuracy rewards saying "no". Reported per router:
acc, TPR - FPR (share of real bugs flagged minus share of false alarms flagged), AUC
within project of the routed P(vuln) (the ranking IRIS triage uses), GPU-s per path.
Decisions use OWASP GPU-s (known before deployment); cost is scored with IRIS GPU-s.
"""
import argparse, collections, glob, json, os
import numpy as np

LAMS = [0.0, 0.25, 1.0]


def cost_table(path):
    cost = {}
    for l in open(path):
        t = json.loads(l)
        c = t["wall_s"] / t["units"]
        cost[t["model"]] = min(c, cost.get(t["model"], c))
    return cost


def load_runs(d, sfx):
    return {os.path.basename(f).split("__")[0]: {json.loads(l)["unit_id"]: json.loads(l) for l in open(f)}
            for f in sorted(glob.glob(os.path.join(d, f"*__{sfx}.jsonl")))}


def auc_within(score, y, grp):
    """mean over projects of P(real path scored above a false alarm), ties = 1/2"""
    vals = []
    for g in set(grp):
        m = grp == g
        pos, neg = score[m & (y == 1)], score[m & (y == 0)]
        if len(pos) and len(neg):
            vals.append(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                        / (len(pos) * len(neg)))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owasp", default="results/2026-09-owasp-ladder")
    ap.add_argument("--iris", default="results/2026-09-iris-cross")
    ap.add_argument("--iris-units", default="data/gr/units_paths.jsonl", help="labels, project, CWE")
    ap.add_argument("--pred", nargs="*", default=[], help="l4 --apply outputs")
    a = ap.parse_args()

    ow = load_runs(f"{a.owasp}/probes", "owasp")
    ir = load_runs(f"{a.iris}/probes", "iris")
    M = sorted(set(ow) & set(ir))
    ow_lab = {r["unit_id"]: r for r in map(json.loads, open(f"{a.owasp}/labels.jsonl"))}
    ir_units = {u["unit_id"]: u for u in map(json.loads, open(a.iris_units))}
    c_ow, c_ir = cost_table(f"{a.owasp}/probes/timing.jsonl"), cost_table(f"{a.iris}/probes/timing.jsonl")
    cvo = np.array([c_ow[m] for m in M]); cvi = np.array([c_ir[m] for m in M])

    oids = sorted(ow_lab)
    Yo = np.array([int(ow_lab[u]["gold_label"]) for u in oids])
    Co = np.array([[ow[m][u]["pred_label"] == Yo[i] for m in M] for i, u in enumerate(oids)], float)
    Po = np.array([[0.5 if ow[m][u]["p_vuln"] is None else ow[m][u]["p_vuln"] for m in M] for u in oids])
    ocat = np.array([ow_lab[u]["project"] for u in oids])
    ocwe = {ow_lab[u]["project"]: int(str(ow_lab[u]["gold_cwe"]).split("-")[-1]) for u in oids}

    iids = sorted(u for u in ir_units if all(u in ir[m] for m in M))
    Yi = np.array([int(ir_units[u]["gold_label"]) for u in iids])
    Ci = np.array([[ir[m][u]["pred_label"] == Yi[i] for m in M] for i, u in enumerate(iids)], float)
    Pi = np.array([[0.5 if ir[m][u]["p_vuln"] is None else ir[m][u]["p_vuln"] for m in M] for u in iids])
    PRi = np.array([[ir[m][u]["pred_label"] or 0 for m in M] for u in iids])
    proj = np.array([ir_units[u]["project"] for u in iids])
    icwe = [int(str(ir_units[u].get("query_cwe") or ir_units[u].get("gold_cwe") or "0").split("-")[-1] or 0)
            for u in iids]
    n = len(iids)
    print(f"{len(M)} models on both benchmarks; IRIS paths {n}, real {Yi.sum()} ({Yi.mean():.0%}); "
          f"'always no' accuracy {1 - Yi.mean():.3f}\n")

    print(f"{'model on IRIS':22s} {'acc':>6s} {'tpr-fpr':>7s} {'auc_proj':>8s} {'gpu_s':>6s}")
    for j, m in enumerate(M):
        print(f"{m:22s} {Ci[:, j].mean():6.3f} {PRi[Yi == 1, j].mean() - PRi[Yi == 0, j].mean():7.3f} "
              f"{auc_within(Pi[:, j], Yi, proj):8.3f} {cvi[j]:6.3f}")
    print()

    preds = [json.load(open(p)) for p in a.pred]
    for p in preds:
        assert "apply" in p, f"{p['encoder']}: not an --apply output"

    def row(name, ch=None, verdict=None, score=None, g=None, mix=None):
        if ch is not None:
            verdict, score, g = PRi[np.arange(n), ch], Pi[np.arange(n), ch], cvi[ch]
        corr = (verdict == Yi).astype(float)
        tf = verdict[Yi == 1].mean() - verdict[Yi == 0].mean()
        return dict(name=name, corr=corr, acc=corr.mean(), tf=tf, auc=auc_within(score, Yi, proj),
                    g=np.broadcast_to(g, (n,)).astype(float), ch=ch)

    for lam in LAMS:
        rows = []
        b = (Co - lam * cvo).mean(0).argmax()
        rows.append(row(f"best-on-OWASP ({M[b]})", ch=np.full(n, b)))
        bi = (Ci - lam * cvi).mean(0).argmax()
        rows.append(row(f"[hindsight] best-on-IRIS ({M[bi]})", ch=np.full(n, bi)))
        cm = {}
        for c in set(ocat):
            cm[ocwe[c]] = (Co[ocat == c] - lam * cvo).mean(0).argmax()
        rows.append(row("cwe (OWASP category by CWE)", ch=np.array([cm.get(c, b) for c in icwe])))
        best_c = None
        for j in range(len(M)):
            for e in range(len(M)):
                if cvo[e] <= cvo[j]:
                    continue
                for w in np.linspace(0, 0.5, 26):
                    esc = np.abs(Po[:, j] - 0.5) < w
                    v = (Co[np.arange(len(oids)), np.where(esc, e, j)] - lam * (cvo[j] + esc * cvo[e])).mean()
                    if best_c is None or v > best_c[0]:
                        best_c = (v, j, e, w)
        if best_c is not None:
            _, j, e, w = best_c
            esc = np.abs(Pi[:, j] - 0.5) < w
            ch = np.where(esc, e, j)
            r = row(f"cascade {M[j]}->{M[e]} ({esc.mean():.0%} up)", ch=ch)
            r["g"] = cvi[j] + esc * cvi[e]
            rows.append(r)
        for p in preds:
            P = np.array([p["apply"][u] for u in iids])
            if p.get("target") == "label":
                if lam == 0:
                    rows.append(row(f"direct {p['encoder']} (no LLM)", verdict=(P[:, 0] > 0.5).astype(int),
                                    score=P[:, 0], g=0.0))
                continue
            assert p["models"] == M, "router trained on a different model list"
            ch = (P - lam * cvo).argmax(1)
            rows.append(row(f"learned {p['encoder']}", ch=ch))
            # random routing with the same model mix
            rng = np.random.default_rng(0)
            sims = [(Ci[np.arange(n), s] - lam * cvi[s]).mean() for s in (rng.permutation(ch) for _ in range(1000))]
            rows[-1]["null"] = (np.mean(sims), np.std(sims))

        base = rows[0]
        rb = base["corr"] - lam * base["g"]
        rng = np.random.default_rng(2)
        BS = [rng.integers(0, n, n) for _ in range(2000)]
        print(f"lam = {lam}")
        print(f"  {'router':44s} {'acc':>6s} {'tpr-fpr':>7s} {'auc_proj':>8s} {'gpu_s':>6s} {'reward':>7s} "
              f"{'z(mix)':>6s} {'vs best-on-OWASP, 95% CI':>26s}")
        for r in rows:
            rw = r["corr"] - lam * r["g"]
            z = f"{(rw.mean() - r['null'][0]) / (r['null'][1] + 1e-12):6.1f}" if "null" in r else ""
            d = np.array([(rw[i] - rb[i]).mean() for i in BS])
            ci = "" if r is base else f"{d.mean():+.3f} [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]"
            print(f"  {r['name']:44s} {r['acc']:6.3f} {r['tf']:7.3f} {r['auc']:8.3f} {r['g'].mean():6.3f} "
                  f"{rw.mean():7.3f} {z:>6s} {ci:>26s}")
            if r["ch"] is not None and r["name"].startswith("learned"):
                mix = collections.Counter(M[i] for i in r["ch"]).most_common(4)
                print(f"  {'':44s} mix: " + ", ".join(f"{k} {v}" for k, v in mix))
        print()


if __name__ == "__main__":
    main()
