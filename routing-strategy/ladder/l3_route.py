#!/usr/bin/env python3
"""L3 -- per-unit routers on the OWASP ladder, fitted on train, scored on test.

Each unit is sent to ONE model. The router is fitted to maximise, per unit,

    reward(m, u) = correct(m, u) - lam * gpu_s(m)

correct = 1 if model m's verdict on unit u matches the answer key, else 0;
gpu_s(m) = GPU-seconds per unit for model m (timing.jsonl); lam = how much
accuracy one GPU-second is worth (lam = 0: accuracy only; lam = 1: paying
0.1 GPU-s must buy at least 0.1 accuracy).

Routers:
  best     the one model with the highest mean train reward
  cat      per OWASP category, the model with the best mean train reward
           (knows only the task type -- the "router is a task classifier" null)
  knn      the k nearest train units by MiniLM embedding vote on the model; k by
           5-fold CV on train
  cascade  a cheap model answers; if its P(vulnerable) is within w of 0.5 the
           unit goes to a dearer model. Pair and w are picked on train.

Every router is compared with random routing that uses the SAME number of units
per model (2,000 shuffles): if the router is not above that, its choice of WHICH
units go where carries no information, only its mix does.
"""
import argparse, collections, glob, json, os, random
import numpy as np

LAMS = [0.0, 0.25, 1.0, 4.0]


def load(a):
    units = {u["unit_id"]: u for u in map(json.loads, open(a.units))}
    split = json.load(open(a.split))["split"]
    cost = {}
    for l in open(os.path.join(a.dir, "timing.jsonl")):
        t = json.loads(l)
        # fastest run per model (original or names-hidden: same prompts, same size);
        # one run can be inflated by stuck requests (gpt-oss-20b: ~20 hung ~435 s)
        c = t["wall_s"] / t["units"]
        cost[t["model"]] = min(c, cost.get(t["model"], c))
    res = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*__owasp.jsonl"))):
        name = os.path.basename(f).split("__")[0]
        rows = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f)}
        if name in cost and len(rows) == len(units):
            res[name] = rows
    return units, split, cost, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/owasp_ladder/results")
    ap.add_argument("--units", default="data/owasp_ladder/units_owasp.jsonl")
    ap.add_argument("--split", default="data/owasp_ladder/split.json")
    ap.add_argument("--emb", default="data/owasp_ladder/emb_minilm")
    ap.add_argument("--models", default="", help="comma list; default all complete runs")
    ap.add_argument("--constants", action="store_true",
                    help="add free 'always-yes' / 'always-no' answers to the pool: if routers "
                         "gain as much with these, the gain is the category base rate, not model skill")
    a = ap.parse_args()
    units, split, cost, res = load(a)
    if a.constants:
        for name, v in (("always-yes", 1), ("always-no", 0)):
            res[name] = {u: dict(pred_label=v, p_vuln=float(v), gold_label=int(units[u]["gold_label"]))
                         for u in units}
            cost[name] = 0.0
        if a.models:
            a.models += ",always-yes,always-no"
    M = [m for m in (a.models.split(",") if a.models else sorted(res, key=lambda m: cost[m]))]
    ids = sorted(units)
    tr = [u for u in ids if split[u] == "train"]
    te = [u for u in ids if split[u] == "test"]
    ix = {u: i for i, u in enumerate(ids)}
    C = np.array([[res[m][u]["pred_label"] == units[u]["gold_label"] for m in M] for u in ids], float)
    P = np.array([[0.5 if res[m][u]["p_vuln"] is None else res[m][u]["p_vuln"] for m in M] for u in ids])
    Y = np.array([int(units[u]["gold_label"]) for u in ids])
    PR = np.array([[res[m][u]["pred_label"] or 0 for m in M] for u in ids])
    cvec = np.array([cost[m] for m in M])
    cat = [units[u]["project"] for u in ids]
    itr, ite = np.array([ix[u] for u in tr]), np.array([ix[u] for u in te])
    E = np.load(a.emb + ".npy")
    eids = json.load(open(a.emb + ".ids.json"))
    E = E[[eids.index(u) for u in ids]]

    print("pool (cheapest first):")
    for j, m in enumerate(M):
        print(f"  {m:22s} gpu_s {cvec[j]:.3f}  test acc {C[ite, j].mean():.3f}")
    print(f"test oracle acc {C[ite].max(1).mean():.3f}   n_test {len(ite)}\n")

    def evaluate(choice, rows):
        """choice[i] = model index for unit rows[i] -> acc, tpr-fpr, gpu_s"""
        c = C[rows, choice]
        pr, y = PR[rows, choice], Y[rows]
        tf = pr[y == 1].mean() - pr[y == 0].mean()
        return c.mean(), tf, cvec[choice].mean()

    def null(choice, rows, lam, draws=2000, seed=0):
        rng = np.random.default_rng(seed)
        real = (C[rows, choice] - lam * cvec[choice]).mean()
        vals = []
        for _ in range(draws):
            sh = rng.permutation(choice)
            vals.append((C[rows, sh] - lam * cvec[sh]).mean())
        vals = np.array(vals)
        return real, vals.mean(), (real - vals.mean()) / (vals.std() + 1e-12)

    def knn_choice(train_rows, query_rows, k, lam, exclude_self=False):
        R = C[train_rows] - lam * cvec
        S = E[query_rows] @ E[train_rows].T
        if exclude_self:
            S[S > 0.9999] = -1
        nn = np.argsort(-S, 1)[:, :k]
        return R[nn].mean(1).argmax(1)

    for lam in LAMS:
        Rtr = C[itr] - lam * cvec
        out = {}
        # best single
        b = Rtr.mean(0).argmax()
        out["best:" + M[b]] = np.full(len(ite), b)
        # category router
        cm = {}
        for c_ in set(cat):
            rows = [i for i in itr if cat[i] == c_]
            cm[c_] = (C[rows] - lam * cvec).mean(0).argmax() if rows else b
        out["cat"] = np.array([cm[cat[i]] for i in ite])
        # kNN, k by 5-fold CV on train
        rng = np.random.default_rng(1)
        folds = np.array_split(rng.permutation(itr), 5)
        best_k, best_v = None, -9
        for k in (5, 10, 20, 50, 100):
            v = []
            for f in range(5):
                q = folds[f]
                t_ = np.concatenate([folds[g] for g in range(5) if g != f])
                ch = knn_choice(t_, q, k, lam)
                v.append((C[q, ch] - lam * cvec[ch]).mean())
            if np.mean(v) > best_v:
                best_k, best_v = k, np.mean(v)
        out[f"knn(k={best_k})"] = knn_choice(itr, ite, best_k, lam)
        # cascade: cheap j answers, unsure (|p-0.5|<w) -> dear e
        best_c, best_cv = None, -9
        for j in range(len(M)):
            for e in range(len(M)):
                if cvec[e] <= cvec[j]:
                    continue
                for w in np.linspace(0, 0.5, 26):
                    esc = np.abs(P[itr, j] - 0.5) < w
                    ch = np.where(esc, e, j)
                    v = (C[itr, ch] - lam * (cvec[j] + esc * cvec[e])).mean()
                    if v > best_cv:
                        best_cv, best_c = v, (j, e, w)
        j, e, w = best_c
        esc = np.abs(P[ite, j] - 0.5) < w
        casc = np.where(esc, e, j)

        bsel = out["best:" + M[b]]
        rng_b = np.random.default_rng(2)
        BS = [rng_b.integers(0, len(ite), len(ite)) for _ in range(2000)]

        def ci(ch, extra_cost=None):
            """95% CI of (router reward - best-single reward) over test resamples"""
            r = C[ite, ch] - lam * (cvec[ch] if extra_cost is None else extra_cost)
            r0 = C[ite, bsel] - lam * cvec[bsel]
            d = np.array([(r[i] - r0[i]).mean() for i in BS])
            return f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"

        print(f"lam = {lam}")
        print(f"  {'router':34s} {'acc':>6s} {'tpr-fpr':>7s} {'gpu_s':>6s} {'reward':>7s} {'random-mix':>10s} {'z':>6s} {'vs best, 95% CI':>17s}  mix")
        for name, ch in out.items():
            acc, tf, g = evaluate(ch, ite)
            real, nm, z = null(ch, ite, lam)
            mix = collections.Counter(M[i] for i in ch)
            print(f"  {name:34s} {acc:6.3f} {tf:7.3f} {g:6.3f} {real:7.3f} {nm:10.3f} {z:6.1f} {ci(ch):>17s}  "
                  + ", ".join(f"{k}:{v}" for k, v in mix.most_common()))
        # does kNN know more than the category? paired CI of knn - cat
        kn = [k_ for k_ in out if k_.startswith("knn")][0]
        rk = C[ite, out[kn]] - lam * cvec[out[kn]]
        rc = C[ite, out["cat"]] - lam * cvec[out["cat"]]
        d = np.array([(rk[i] - rc[i]).mean() for i in BS])
        print(f"  {kn} - cat: {d.mean():+.3f}  95% CI [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
        # cascade pays the cheap model on every unit
        acc, tf, _ = evaluate(casc, ite)
        g = cvec[j] + esc.mean() * cvec[e]
        rng = np.random.default_rng(0)
        rv = []
        for _ in range(2000):
            r_esc = np.zeros(len(ite), bool)
            r_esc[rng.choice(len(ite), esc.sum(), replace=False)] = True
            rv.append(C[ite, np.where(r_esc, e, j)].mean())
        rv = np.array(rv)
        name = f"cascade {M[j]}->{M[e]} w={w:.2f}"
        print(f"  {name:34s} {acc:6.3f} {tf:7.3f} {g:6.3f} {acc - lam * g:7.3f} "
              f"{rv.mean() - lam * g:10.3f} {(acc - rv.mean()) / (rv.std() + 1e-12):6.1f} "
              f"{ci(casc, cvec[j] + esc * cvec[e]):>17s}  "
              f"escalated {esc.mean():.0%}")
        print()

    # leave-one-category-out: fit on the other categories, score on the held-out one
    print("leave-one-category-out (train = all units of the other categories)")
    print(f"  {'lam':>5s} {'best single':>12s} {'knn(k=20)':>10s} {'cascade':>8s} {'oracle':>7s}   reward, pooled over held-out units")
    cats = sorted(c_ for c_ in set(cat) if sum(x == c_ for x in cat) >= 30)
    for lam in LAMS:
        tot = collections.Counter(); n = 0
        for c_ in cats:
            ho = np.array([i for i in range(len(ids)) if cat[i] == c_])
            ot = np.array([i for i in range(len(ids)) if cat[i] != c_ and cat[i] in cats])
            b_ = (C[ot] - lam * cvec).mean(0).argmax()
            tot["best"] += (C[ho, b_] - lam * cvec[b_]).sum()
            ch = knn_choice(ot, ho, 20, lam)
            tot["knn"] += (C[ho, ch] - lam * cvec[ch]).sum()
            bc, bv = None, -9
            for j in range(len(M)):
                for e in range(len(M)):
                    if cvec[e] <= cvec[j]:
                        continue
                    for w in np.linspace(0, 0.5, 26):
                        esc = np.abs(P[ot, j] - 0.5) < w
                        v = (C[ot, np.where(esc, e, j)] - lam * (cvec[j] + esc * cvec[e])).mean()
                        if v > bv:
                            bv, bc = v, (j, e, w)
            j, e, w = bc
            esc = np.abs(P[ho, j] - 0.5) < w
            tot["cascade"] += (C[ho, np.where(esc, e, j)] - lam * (cvec[j] + esc * cvec[e])).sum()
            tot["oracle"] += (C[ho] - lam * cvec).max(1).sum()
            n += len(ho)
        print(f"  {lam:5.2f} {tot['best']/n:12.3f} {tot['knn']/n:10.3f} {tot['cascade']/n:8.3f} {tot['oracle']/n:7.3f}")


if __name__ == "__main__":
    main()
