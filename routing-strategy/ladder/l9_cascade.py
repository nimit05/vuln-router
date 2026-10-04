#!/usr/bin/env python3
"""L9 -- the proposed router, checked without hindsight: a confidence cascade whose
model pair and escalation share are chosen from data the evaluation never sees.

Cascade(cheap, strong, f): the cheap model scores every unit; in each project (IRIS)
or category (OWASP) the f share of units it finds most suspicious also go to the
strong model, whose verdict and score replace the cheap one's. Cost per unit =
cheap GPU-s + f * strong GPU-s.

  a) lopo    IRIS, pair Qwen3-8B -> Qwen3-32B (E20). f is chosen on the other projects
             (smallest f whose recall@10 is within --tol of sending everything to the
             32B) and applied to the held-out project. Same-repo projects (antisamy
             2016/2017) are held out together. Compared with random escalation of the
             same number of units per project.
  b) calib   IRIS. A new deployment labels k units; from them: keep models with AUC >=
             0.65 (skill gate), cheap = cheapest kept, strong = highest AUC kept. The
             labelled units come from one random half of the repos, the cascade is scored
             on the other half; repeated over many draws. Compared on the same half with
             the E20 pair, all-32B, and the pair chosen from OWASP data instead.
  c) owasp   the same calibration on OWASP (k units from the train split, scored on the
             test split), so the method is shown on both benchmarks.
If cheap GPU-s + f * strong GPU-s >= strong GPU-s, the strong model is used alone.
Metrics: recall@10 (per group, share of its real bugs among its 10 highest-scored
units, averaged over groups with a real bug), AUC within group, TPR - FPR, accuracy.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l6_eval_cross import auc_within, cost_table, load_runs   # noqa: E402
from l8_eval_jev import recall_at                              # noqa: E402

FGRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def load(probes, sfx, labels):
    runs, cost = load_runs(probes, sfx), cost_table(os.path.join(probes, "timing.jsonl"))
    M = sorted(runs)
    ids = sorted(labels)
    Y = np.array([labels[u][0] for u in ids])
    G = np.array([labels[u][1] for u in ids])
    P = np.array([[0.5 if runs[m][u]["p_vuln"] is None else runs[m][u]["p_vuln"] for m in M] for u in ids])
    V = np.array([[runs[m][u]["pred_label"] or 0 for m in M] for u in ids])
    return M, ids, Y, G, P, V, np.array([cost[m] for m in M])


def escalate(score, G, f, rows):
    """boolean mask over rows: top-f share per group by score"""
    esc = np.zeros(len(rows), bool)
    for g in set(G[rows]):
        m = np.where(G[rows] == g)[0]
        k = int(round(f * len(m)))
        if k:
            esc[m[np.argsort(-score[rows][m], kind="stable")[:k]]] = True
    return esc


def run_cascade(c, s, f, rows, P, V, G):
    esc = escalate(P[:, c], G, f, rows)
    sc = np.where(esc, P[rows, s], P[rows, c])
    vd = np.where(esc, V[rows, s], V[rows, c])
    return sc, vd, esc


def metrics(sc, vd, Y, G, cost):
    tf = vd[Y == 1].mean() - vd[Y == 0].mean() if (Y == 1).any() and (Y == 0).any() else float("nan")
    return dict(r10=recall_at(sc, Y, G), auc=auc_within(sc, Y, G), tf=tf, acc=(vd == Y).mean(), gpu=cost)


def pooled_auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return 0.5
    return ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))


def sample_auc(s, y, g):
    """AUC from pairs inside the same group only (pooling across projects with different
    base rates rewards a model for ranking projects, not paths -- Simpson, E13/E14);
    0.5 when the sample has no within-group pair"""
    w = n = 0.0
    for grp in set(g):
        m = g == grp
        pos, neg = s[m & (y == 1)], s[m & (y == 0)]
        w += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        n += len(pos) * len(neg)
    return w / n if n else 0.5


def pick_pair(rows, P, Y, cvec, G, gate=0.65):
    a = np.array([sample_auc(P[rows, j], Y[rows], G[rows]) for j in range(P.shape[1])])
    keep = np.where(a >= gate)[0]
    if not len(keep):
        j = int(a.argmax())
        return j, j, a
    return int(keep[np.argmin(cvec[keep])]), int(keep[np.argmax(a[keep])]), a


def short(m):
    return m.rsplit("-", 1)[0]


def fmt(name, m):
    return (f"  {name:46s} {m['r10']:6.3f} {m['auc']:6.3f} {m['tf']:7.3f} {m['acc']:6.3f} {m['gpu']:6.3f}")


HDR = f"  {'':46s} {'rec@10':>6s} {'auc':>6s} {'tpr-fpr':>7s} {'acc':>6s} {'gpu_s':>6s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owasp", default="results/2026-09-owasp-ladder")
    ap.add_argument("--iris", default="results/2026-09-iris-cross")
    ap.add_argument("--iris-units", default="data/gr/units_paths.jsonl")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--ks", default="25,50,100,200,400")
    a = ap.parse_args()
    ks = [int(k) for k in a.ks.split(",")]

    iu = {u["unit_id"]: u for u in map(json.loads, open(a.iris_units))}
    lab_i = {u: (int(iu[u]["gold_label"]), iu[u]["project"]) for u in iu}
    Mi, idi, Yi, Gi, Pi, Vi, ci = load(f"{a.iris}/probes", "iris", lab_i)
    repo = np.array([g.split("_CVE")[0] for g in Gi])
    ol = {r["unit_id"]: r for r in map(json.loads, open(f"{a.owasp}/labels.jsonl"))}
    lab_o = {u: (int(ol[u]["gold_label"]), ol[u]["project"]) for u in ol}
    Mo, ido, Yo, Go, Po, Vo, co = load(f"{a.owasp}/probes", "owasp", lab_o)
    assert Mi == Mo
    M = Mi
    j8, j32 = M.index("qwen3-8b-nothink"), M.index("qwen3-32b-nothink")
    alli = np.arange(len(idi))

    # ---------------------------------------------------------------- a) LOPO f on IRIS
    print("a) IRIS, Qwen3-8B -> Qwen3-32B; f chosen on the other projects (leave one repo out)")
    print("   rule: smallest f whose recall@10 on the other projects is within "
          f"{a.tol} of sending everything to the 32B\n")
    curve = []
    for f in FGRID:
        sc, vd, esc = run_cascade(j8, j32, f, alli, Pi, Vi, Gi)
        curve.append((f, metrics(sc, vd, Yi, Gi, ci[j8] + esc.mean() * ci[j32])))
    print("   hindsight curve, all IRIS (for reference only):")
    print("   " + "  ".join(f"f={f:.2f}:{m['r10']:.3f}" for f, m in curve[::2]))
    esc_all = np.zeros(len(idi), bool); chosen = {}
    for r in sorted(set(repo)):
        tr = np.where(repo != r)[0]; ho = np.where(repo == r)[0]
        if Yi[tr].sum() == 0:
            continue
        full = recall_at(Pi[tr, j32], Yi[tr], Gi[tr])
        f_pick = 1.0
        for f in FGRID:
            sc, _, _ = run_cascade(j8, j32, f, tr, Pi, Vi, Gi)
            if recall_at(sc, Yi[tr], Gi[tr]) >= full - a.tol:
                f_pick = f
                break
        chosen[r] = f_pick
        esc_all[ho] = escalate(Pi[:, j8], Gi, f_pick, ho)
    sc = np.where(esc_all, Pi[:, j32], Pi[:, j8]); vd = np.where(esc_all, Vi[:, j32], Vi[:, j8])
    lopo = metrics(sc, vd, Yi, Gi, ci[j8] + esc_all.mean() * ci[j32])
    rng = np.random.default_rng(0); rnd = []
    for _ in range(500):
        e = np.zeros(len(idi), bool)
        for g in set(Gi):
            m = np.where(Gi == g)[0]
            e[rng.choice(m, esc_all[m].sum(), replace=False)] = True
        rnd.append(recall_at(np.where(e, Pi[:, j32], Pi[:, j8]), Yi, Gi))
    rnd = np.array(rnd)
    print(f"\n   f chosen per held-out repo: " + ", ".join(f"{r.split('__')[-1][:14]} {f:.2f}" for r, f in chosen.items()))
    print(HDR)
    print(fmt("Qwen3-8B on everything", metrics(Pi[:, j8], Vi[:, j8], Yi, Gi, ci[j8])))
    print(fmt("Qwen3-32B on everything", metrics(Pi[:, j32], Vi[:, j32], Yi, Gi, ci[j32])))
    print(fmt(f"cascade, f from held-out rule ({esc_all.mean():.0%} up)", lopo))
    print(f"  {'random escalation, same count per project':46s} {rnd.mean():6.3f}   "
          f"(cascade z = {(lopo['r10'] - rnd.mean()) / (rnd.std() + 1e-12):+.1f})")
    f_iris = float(np.median(list(chosen.values())))

    # ---------------------------------------------------------------- b) calibration on IRIS
    print(f"\nb) IRIS, pair picked from k labelled units of one half of the repos, scored on the other half;"
          f" f = {f_iris:.2f} (median of a); {a.draws} draws per k")
    o_cheap, o_strong, _ = pick_pair(np.arange(len(ido)), Po, Yo, co, Go)
    print(f"   pair picked from all OWASP labels instead: {M[o_cheap]} -> {M[o_strong]}")
    repos = np.array(sorted(set(repo)))
    print(f"  {'k':>4s} {'picks 8B->32B':>13s} {'cascade':>8s} {'E20 pair':>8s} {'OWASP pair':>10s} "
          f"{'all-32B':>8s} {'gpu_s':>6s}   recall@10 on the unseen half (mean over draws)")
    rng = np.random.default_rng(1)
    for k in ks:
        hits, r_c, r_e, r_o, r_32, g_c, picks = 0, [], [], [], [], [], {}
        for _ in range(a.draws):
            side = rng.permutation(repos)
            cal_r, ev_r = set(side[: len(side) // 2]), set(side[len(side) // 2:])
            cal = np.where(np.isin(repo, list(cal_r)))[0]; ev = np.where(np.isin(repo, list(ev_r)))[0]
            if Yi[ev].sum() == 0 or len(cal) < k:
                continue
            samp = rng.choice(cal, k, replace=False)
            c, s, _ = pick_pair(samp, Pi, Yi, ci, Gi)
            picks[(M[c], M[s])] = picks.get((M[c], M[s]), 0) + 1
            hits += (c, s) == (j8, j32)
            for (cc, ss), acc in (((c, s), r_c), ((j8, j32), r_e), ((o_cheap, o_strong), r_o)):
                if cc != ss and ci[cc] + f_iris * ci[ss] >= ci[ss]:
                    cc = ss                      # cascade dearer than the strong model alone
                scv, _, esc = run_cascade(cc, ss, f_iris if cc != ss else 0.0, ev, Pi, Vi, Gi)
                acc.append(recall_at(scv, Yi[ev], Gi[ev]))
                if acc is r_c:
                    g_c.append(ci[cc] + (esc.mean() * ci[ss] if cc != ss else 0))
            r_32.append(recall_at(Pi[ev, j32], Yi[ev], Gi[ev]))
        n = len(r_c)
        top = sorted(picks.items(), key=lambda x: -x[1])[:2]
        print(f"  {k:4d} {hits / n:13.0%} {np.mean(r_c):8.3f} {np.mean(r_e):8.3f} {np.mean(r_o):10.3f} "
              f"{np.mean(r_32):8.3f} {np.mean(g_c):6.3f}   most picked: "
              + "; ".join(f"{short(p[0])}->{short(p[1])} {v / n:.0%}"
                          for p, v in top))

    # ---------------------------------------------------------------- c) the same on OWASP
    split = json.load(open(f"{a.owasp}/split.json"))["split"]
    tr = np.array([i for i, u in enumerate(ido) if split[u] == "train"])
    te = np.array([i for i, u in enumerate(ido) if split[u] == "test"])
    print(f"\nc) OWASP: pair picked from k labelled train units, f chosen on train (smallest f within {a.tol}"
          " AUC of all-strong), scored on the 588 test units")
    print(HDR)
    for j in sorted(range(len(M)), key=lambda j: co[j]):
        if M[j] in ("qwen3-8b-nothink", "qwen3-32b-nothink", "gpt-oss-20b-low", "gpt-oss-120b-low"):
            print(fmt(f"{M[j]} on everything", metrics(Po[te, j], Vo[te, j], Yo[te], Go[te], co[j])))
    for k in ks + [len(tr)]:
        res, picks = [], {}
        for d in range(a.draws if k < len(tr) else 1):
            samp = rng.choice(tr, k, replace=False) if k < len(tr) else tr
            c, s, _ = pick_pair(samp, Po, Yo, co, Go)
            picks[(M[c], M[s])] = picks.get((M[c], M[s]), 0) + 1
            f_pick = 0.0
            if c != s:
                full = auc_within(Po[tr, s], Yo[tr], Go[tr]); f_pick = 1.0
                for f in FGRID:
                    scv, _, _ = run_cascade(c, s, f, tr, Po, Vo, Go)
                    if auc_within(scv, Yo[tr], Go[tr]) >= full - a.tol:
                        f_pick = f
                        break
            if c != s and co[c] + f_pick * co[s] >= co[s]:
                c, f_pick = s, 0.0               # cascade dearer than the strong model alone
            scv, vdv, esc = run_cascade(c, s, f_pick, te, Po, Vo, Go)
            res.append(metrics(scv, vdv, Yo[te], Go[te], co[c] + (esc.mean() * co[s] if c != s else 0)))
        mean = {key: float(np.mean([r[key] for r in res])) for key in res[0]}
        top = sorted(picks.items(), key=lambda x: -x[1])[0]
        name = (f"cascade, k={k if k < len(tr) else 'all train'} "
                f"({short(top[0][0])}->{short(top[0][1])} {top[1] / len(res):.0%})")
        print(fmt(name, mean))


if __name__ == "__main__":
    main()
