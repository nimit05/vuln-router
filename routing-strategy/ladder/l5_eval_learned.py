#!/usr/bin/env python3
"""L5 -- score the learned (Jev-style) router from l4 against the same baselines as l3.

For each lam, on the 588 test units: best single model, category router, the base-rate
trick (best model or "always yes/no" per category), kNN on MiniLM, and the learned
router (argmax P(right) - lam * gpu_s). Each row: accuracy, OWASP's TPR - FPR, GPU-s,
reward, random routing with the same model mix (z), and paired 95% intervals of the
learned router minus best / category / kNN.

Also: are the probabilities any good? Per model, on test,
  AUC   does P(right) rank the units the model gets right above those it gets wrong
  ECE   average gap between predicted and actual hit rate over 10 probability bins
        (0 = "0.9" really means right 9 times in 10)
And leave-one-category-out, pooled over held-out units, as in l3.
"""
import argparse, collections, glob, json, os
import numpy as np

LAMS = [0.0, 0.25, 1.0, 4.0]


def auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


def ece(p, y, bins=10):
    e, n = 0.0, len(p)
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins) if lo < 0.9 else (p >= lo)
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return e


def main():
    ap = argparse.ArgumentParser()
    R = "results/2026-09-owasp-ladder"
    ap.add_argument("--probes", default=f"{R}/probes")
    ap.add_argument("--labels", default=f"{R}/labels.jsonl")
    ap.add_argument("--split", default=f"{R}/split.json")
    ap.add_argument("--emb", default=f"{R}/emb_minilm")
    ap.add_argument("--pred", nargs="+", required=True, help="l4 prediction files")
    a = ap.parse_args()

    labels = {r["unit_id"]: r for r in map(json.loads, open(a.labels))}
    split = json.load(open(a.split))["split"]
    cost = {}
    for l in open(os.path.join(a.probes, "timing.jsonl")):
        t = json.loads(l)
        c = t["wall_s"] / t["units"]
        cost[t["model"]] = min(c, cost.get(t["model"], c))
    res = {os.path.basename(f).split("__")[0]: {json.loads(l)["unit_id"]: json.loads(l) for l in open(f)}
           for f in sorted(glob.glob(os.path.join(a.probes, "*__owasp.jsonl")))}
    allp = [json.load(open(p)) for p in a.pred]
    direct = [p for p in allp if p.get("target") == "label"]
    preds = [p for p in allp if p.get("target", "models") == "models"]
    M = preds[0]["models"]
    assert all(p["models"] == M for p in preds) and set(M) == set(res)
    ids = sorted(labels)
    Y = np.array([int(labels[u]["gold_label"]) for u in ids])
    C = np.array([[res[m][u]["pred_label"] == Y[i] for m in M] for i, u in enumerate(ids)], float)
    PR = np.array([[res[m][u]["pred_label"] or 0 for m in M] for u in ids])
    cvec = np.array([cost[m] for m in M])
    cat = np.array([labels[u]["project"] for u in ids])
    itr = np.array([i for i, u in enumerate(ids) if split[u] == "train"])
    ite = np.array([i for i, u in enumerate(ids) if split[u] == "test"])
    E = np.load(a.emb + ".npy")
    eids = json.load(open(a.emb + ".ids.json"))
    E = E[[eids.index(u) for u in ids]]
    # constant answers for the base-rate trick: correct iff the label matches
    CY, CN = (Y == 1).astype(float), (Y == 0).astype(float)
    rng = np.random.default_rng(2)
    BS = [rng.integers(0, len(ite), len(ite)) for _ in range(2000)]

    def knn(train, query, k, lam):
        R_ = C[train] - lam * cvec
        nn = np.argsort(-(E[query] @ E[train].T), 1)[:, :k]
        return R_[nn].mean(1).argmax(1)

    def knn_cv(train, lam):
        f5 = np.array_split(np.random.default_rng(1).permutation(train), 5)
        best = None
        for k in (5, 10, 20, 50, 100):
            v = np.mean([(lambda q, ch: (C[q, ch] - lam * cvec[ch]).mean())(
                f5[f], knn(np.concatenate([f5[g] for g in range(5) if g != f]), f5[f], k, lam))
                for f in range(5)])
            if best is None or v > best[0]:
                best = (v, k)
        return best[1]

    print("calibration of P(model right) on test (per model: AUC / ECE)")
    for p in preds:
        P = np.array([p["main"][ids[i]] for i in ite])
        a_ = [auc(P[:, j], C[ite, j]) for j in range(len(M))]
        e_ = [ece(P[:, j], C[ite, j]) for j in range(len(M))]
        print(f"  {p['encoder']:9s} mean AUC {np.nanmean(a_):.3f}  mean ECE {np.mean(e_):.3f}   "
              + "  ".join(f"{m.split('-')[0]}{'-' + m.split('-')[1] if m.startswith(('qwen', 'gpt')) else ''}:"
                          f"{x:.2f}/{y:.2f}" for m, x, y in zip(M, a_, e_)))
    print()

    for lam in LAMS:
        rows = {}
        Rtr = C[itr] - lam * cvec
        b = Rtr.mean(0).argmax()
        rows["best:" + M[b]] = (C[ite, b], np.full(len(ite), cvec[b]), np.full(len(ite), b), PR[ite, b])
        cm = {c: (C[[i for i in itr if cat[i] == c]] - lam * cvec).mean(0).argmax() for c in set(cat)}
        ch = np.array([cm[cat[i]] for i in ite])
        rows["cat"] = (C[ite, ch], cvec[ch], ch, PR[ite, ch])
        # base-rate trick: best model, or a free constant answer, per category
        opt = {}
        for c in set(cat):
            tr_c = [i for i in itr if cat[i] == c]
            v = [(C[tr_c, b] - lam * cvec[b]).mean(), CY[tr_c].mean(), CN[tr_c].mean()]
            opt[c] = int(np.argmax(v))
        corr = np.array([[C[i, b], CY[i], CN[i]][opt[cat[i]]] for i in ite])
        gs = np.array([[cvec[b], 0.0, 0.0][opt[cat[i]]] for i in ite])
        prd = np.array([[PR[i, b], 1, 0][opt[cat[i]]] for i in ite])
        rows["base-rate trick"] = (corr, gs, None, prd)
        k = knn_cv(itr, lam)
        ch = knn(itr, ite, k, lam)
        rows[f"knn(k={k})"] = (C[ite, ch], cvec[ch], ch, PR[ite, ch])
        for p in preds:
            P = np.array([p["main"][ids[i]] for i in ite])
            ch = (P - lam * cvec).argmax(1)
            rows["learned " + p["encoder"]] = (C[ite, ch], cvec[ch], ch, PR[ite, ch])

        def rew(r):
            return r[0] - lam * r[1]
        print(f"lam = {lam}")
        print(f"  {'router':24s} {'acc':>6s} {'tpr-fpr':>7s} {'gpu_s':>6s} {'reward':>7s} {'z(mix)':>6s}"
              f"  {'- best':>16s} {'- cat':>16s} {'- knn':>16s}")
        ref = {n: rew(r) for n, r in rows.items()}
        kn = [n for n in rows if n.startswith("knn")][0]
        bn = [n for n in rows if n.startswith("best")][0]
        for n, r in rows.items():
            acc, g = r[0].mean(), r[1].mean()
            y = Y[ite]
            tf = r[3][y == 1].mean() - r[3][y == 0].mean()
            z = ""
            if r[2] is not None:
                sims = []
                sh_rng = np.random.default_rng(0)
                for _ in range(1000):
                    s = sh_rng.permutation(r[2])
                    sims.append((C[ite, s] - lam * cvec[s]).mean())
                z = f"{(ref[n].mean() - np.mean(sims)) / (np.std(sims) + 1e-12):6.1f}"
            ci = []
            for other in (bn, "cat", kn):
                if not n.startswith("learned"):
                    ci.append(""); continue
                d = np.array([(ref[n][i] - ref[other][i]).mean() for i in BS])
                ci.append(f"[{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]")
            print(f"  {n:24s} {acc:6.3f} {tf:7.3f} {g:6.3f} {ref[n].mean():7.3f} {z:>6s}"
                  f"  {ci[0]:>16s} {ci[1]:>16s} {ci[2]:>16s}")
        print()

    for p in direct:
        print(f"direct classifier {p['encoder']} (no LLM; answer = P(vulnerable) > 0.5), GPU-s ~0")
        for name, key, rows in (("test", "main", ite), ("leave-one-category-out, all units", "loco",
                                                         np.arange(len(ids)))):
            if not p[key]:
                continue
            pv = np.array([p[key][ids[i]][0] for i in rows])
            pr, y = (pv > 0.5).astype(int), Y[rows]
            print(f"  {name:36s} acc {(pr == y).mean():.3f}  tpr-fpr {pr[y == 1].mean() - pr[y == 0].mean():.3f}"
                  f"  AUC {auc(pv, y):.3f}")
        cats30 = [c for c in set(cat) if (cat == c).sum() >= 30]
        if p["loco"]:
            rows = np.array([i for i in range(len(ids)) if cat[i] in cats30])
            pv = np.array([p["loco"][ids[i]][0] for i in rows])
            print(f"  {'leave-one-category-out, cats >= 30':36s} acc {((pv > 0.5) == Y[rows]).mean():.3f}")
        print()

    print("leave-one-category-out (categories with >= 30 units; pooled reward over held-out units)")
    cats = sorted(c for c in set(cat) if (cat == c).sum() >= 30)
    print(f"  {'lam':>5s} {'best single':>12s} {'knn(k=20)':>10s} " + " ".join(f"{'learned ' + p['encoder']:>18s}" for p in preds))
    ci_rows = []
    for lam in LAMS:
        per = collections.defaultdict(list)          # per held-out unit reward, by router
        for c in cats:
            ho = np.where(cat == c)[0]
            ot = np.array([i for i in range(len(ids)) if cat[i] != c and cat[i] in cats])
            b_ = (C[ot] - lam * cvec).mean(0).argmax()
            per["best"] += list(C[ho, b_] - lam * cvec[b_])
            ch = knn(ot, ho, 20, lam)
            per["knn"] += list(C[ho, ch] - lam * cvec[ch])
            for p in preds:
                if not p["loco"]:
                    continue
                P = np.array([p["loco"][ids[i]] for i in ho])
                ch = (P - lam * cvec).argmax(1)
                per[p["encoder"]] += list(C[ho, ch] - lam * cvec[ch])
        print(f"  {lam:5.2f} {np.mean(per['best']):12.3f} {np.mean(per['knn']):10.3f} "
              + " ".join(f"{np.mean(per[p['encoder']]) if per[p['encoder']] else float('nan'):18.3f}"
                         for p in preds))
        b0 = np.array(per["best"])
        rng_l = np.random.default_rng(3)
        bs = [rng_l.integers(0, len(b0), len(b0)) for _ in range(2000)]
        for p in preds:
            if per[p["encoder"]]:
                d = np.array(per[p["encoder"]]) - b0
                v = np.array([d[i].mean() for i in bs])
                ci_rows.append(f"  lam {lam:4.2f} {p['encoder']:10s} - best single {d.mean():+.3f} "
                               f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]")
    print("\n".join(ci_rows))


if __name__ == "__main__":
    main()
