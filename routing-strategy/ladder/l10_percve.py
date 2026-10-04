#!/usr/bin/env python3
"""L10 -- IRIS scored per vulnerability, not per path.

IRIS labels a path "real" when it passes through a method the CVE fix changed, so one
CVE yields many real paths (ff4j: 181 of 1,413) and path-level metrics count the same
vulnerability many times. Here the unit is the CVE (one project = one CVE):
  IFA        false alarms ranked above the first real path: what an analyst reads before
             the first hit (LineVul's "initial false alarm")
  share      IFA / false paths in the project, so big and small projects weigh the same
  found@k    the CVE has a real path among the k highest-scored paths
  random     exact expectations for a random order (F false, R real paths):
             E[IFA] = F / (R + 1),  P(found@k) = 1 - C(F, k) / C(F + R, k)
Only CVEs with both real and false paths count. Uncertainty: bootstrap over CVEs.
Ties in a model's score are broken at random and averaged over 20 orders.

Rankers: the 12 single models (E22-E26 runs) and the Qwen3-8B -> Qwen3-32B cascade with
  fixed f = 0.30   top 30% per project by the 8B go to the 32B (E20; f picked in hindsight)
  LOPO f           f chosen on the other repos with the E26 rule
  stop rule        send the 8B's next `batch` suspects to the 32B; stop once a batch adds
                   nothing to the 32B's top 10 (no labels, no f)
Fixed and LOPO f rank as E20/E26 did (the 32B's score replaces the 8B's); the stop rule
ranks the paths the 32B saw first. Second table: the E20 runs with identifiers renamed.
"""
import argparse, json, os, sys
from math import comb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l6_eval_cross import cost_table, load_runs   # noqa: E402
from l8_eval_jev import recall_at                 # noqa: E402

KS = (1, 5, 10)
FGRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def scores(path, ids):
    d = {}
    for l in open(path):
        r = json.loads(l)
        d[r["unit_id"]] = 0.5 if r["p_vuln"] is None else r["p_vuln"]
    return np.array([d[u] for u in ids])


class Cves:
    def __init__(self, Y, G):
        self.Y, self.G = Y, G
        self.rows = []
        for g in sorted(set(G), key=lambda g: -(G == g).sum()):
            m = np.where(G == g)[0]
            if 0 < Y[m].sum() < len(m):
                self.rows.append((g, m))
        self.F = np.array([(Y[m] == 0).sum() for _, m in self.rows])
        self.R = np.array([(Y[m] == 1).sum() for _, m in self.rows])

    def ifa(self, s, orders=20):
        """IFA per CVE, ties broken at random, averaged over `orders` draws"""
        rng = np.random.default_rng(0)
        out = np.zeros(len(self.rows))
        for _ in range(orders):
            jit = rng.random(len(s)) * 1e-9
            for i, (_, m) in enumerate(self.rows):
                o = m[np.argsort(-(s[m] + jit[m]), kind="stable")]
                out[i] += int(np.argmax(self.Y[o] == 1))
        return out / orders

    def random(self):
        ifa = self.F / (self.R + 1)
        found = {k: np.array([1 - comb(f, k) / comb(f + r, k) if f + r >= k else 1.0   # fewer paths than k: all read
                              for f, r in zip(self.F, self.R)]) for k in KS}
        return ifa, found


def summary(ifa, cv, found=None, boot=4000, seed=0):
    found = found if found is not None else {k: (ifa < k).astype(float) for k in KS}
    share = ifa / cv.F
    rng = np.random.default_rng(seed)
    bs = np.array([share[rng.integers(0, len(share), len(share))].mean() for _ in range(boot)])
    return dict(found={k: found[k].sum() for k in KS}, med=float(np.median(ifa)), share=share.mean(),
                lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), ifa=ifa, sh=share)


def escalate_share(s, G, f):
    esc = np.zeros(len(s), bool)
    for g in set(G):
        m = np.where(G == g)[0]
        k = int(round(f * len(m)))
        if k:
            esc[m[np.argsort(-s[m], kind="stable")[:k]]] = True
    return esc


def stop_rule(s8, s32, G, batch=10, patience=1, k=10):
    esc = np.zeros(len(s8), bool)
    for g in set(G):
        m = np.where(G == g)[0]
        order = m[np.argsort(-s8[m], kind="stable")]
        quiet, top = 0, set()
        for i in range(0, len(order), batch):
            esc[order[i:i + batch]] = True
            e = m[esc[m]]
            new = set(e[np.argsort(-s32[e], kind="stable")[:k]])
            quiet = quiet + 1 if (i > 0 and new == top) else 0
            top = new
            if quiet >= patience:
                break
    return esc


def lopo_f(s8, s32, Y, G, repo, tol=0.01):
    """E26 rule: per held-out repo, smallest f whose recall@10 on the other repos is within
    tol of sending everything to the 32B; returns the escalation mask"""
    esc = np.zeros(len(s8), bool)
    for r in sorted(set(repo)):
        tr, ho = np.where(repo != r)[0], np.where(repo == r)[0]
        full = recall_at(s32[tr], Y[tr], G[tr])
        pick = 1.0
        for f in FGRID:
            e = escalate_share(s8[tr], G[tr], f)
            if recall_at(np.where(e, s32[tr], s8[tr]), Y[tr], G[tr]) >= full - tol:
                pick = f
                break
        esc[ho] = escalate_share(s8[ho], G[ho], pick)
    return esc


def row(name, cost, res, r10=None):
    f = res["found"]
    r10s = f"{r10:6.3f}" if r10 is not None else f"{'':6s}"
    return (f"  {name:40s} {cost:6.3f}  {f[1]:4.1f} {f[5]:4.1f} {f[10]:5.1f}  {res['med']:6.1f}  "
            f"{res['share']:5.3f} [{res['lo']:.3f},{res['hi']:.3f}]  {r10s}")


def header(n):
    return (f"  {'ranker':40s} {'gpu_s':>6s}  {'found@1/5/10 (of ' + str(n) + ')':>16s}  {'medIFA':>6s}  "
            f"{'share read before hit':>21s}  {'r@10*':>6s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iris", default="results/2026-09-iris-cross")
    ap.add_argument("--e20", default="results/2026-09-skilled-cascade")
    ap.add_argument("--units", default="data/gr/units_paths.jsonl")
    a = ap.parse_args()

    units = {u["unit_id"]: u for u in map(json.loads, open(a.units))}
    ids = sorted(units)
    Y = np.array([int(units[u]["gold_label"]) for u in ids])
    G = np.array([units[u]["project"] for u in ids])
    repo = np.array([g.split("_CVE")[0] for g in G])
    cv = Cves(Y, G)
    n = len(cv.rows)
    runs = load_runs(f"{a.iris}/probes", "iris")
    cost = cost_table(f"{a.iris}/probes/timing.jsonl")
    P = {m: np.array([0.5 if runs[m][u]["p_vuln"] is None else runs[m][u]["p_vuln"] for u in ids]) for m in runs}

    print(f"IRIS-16 per vulnerability: {n} CVEs with both real and false paths "
          f"({int(cv.R.sum())} real, {int(cv.F.sum())} false paths)")
    print("  share = false alarms read before the first real path / false paths in the project;"
          " mean over CVEs, 95% bootstrap CI over CVEs")
    print("  r@10* = the old path-level recall@10, for comparison\n")
    print(header(n))
    rifa, rfound = cv.random()
    print(row("random order (exact expectation)", 0.0, summary(rifa, cv, rfound)))
    for m in sorted(P, key=lambda m: cost[m]):
        print(row(m, cost[m], summary(cv.ifa(P[m]), cv), recall_at(P[m], Y, G)))

    s8, s32 = P["qwen3-8b-nothink"], P["qwen3-32b-nothink"]
    c8, c32 = cost["qwen3-8b-nothink"], cost["qwen3-32b-nothink"]
    print()
    casc = {}
    esc = escalate_share(s8, G, 0.30)
    casc["8B->32B, fixed f = 0.30 (hindsight)"] = (np.where(esc, s32, s8), esc)
    esc = lopo_f(s8, s32, Y, G, repo)
    casc["8B->32B, LOPO f (E26 rule)"] = (np.where(esc, s32, s8), esc)
    for b, p in ((10, 1), (10, 2), (5, 1), (20, 1)):
        esc = stop_rule(s8, s32, G, b, p)
        casc[f"8B->32B, stop rule batch {b}, patience {p}"] = (np.where(esc, 2 + s32, s8), esc)
    for name, (sc, esc) in casc.items():
        print(row(name + f" ({esc.mean():.0%} up)", c8 + esc.mean() * c32, summary(cv.ifa(sc), cv),
                  recall_at(sc, Y, G)))

    # per-CVE detail
    main_ = {"random": rifa, "8B": cv.ifa(s8), "32B": cv.ifa(s32),
             "stop": cv.ifa(casc["8B->32B, stop rule batch 10, patience 1"][0]),
             "LOPO": cv.ifa(casc["8B->32B, LOPO f (E26 rule)"][0])}
    print("\nper CVE: false alarms before the first real path")
    print(f"  {'CVE':44s} {'paths':>5s} {'real':>4s} {'r@10 cap':>8s}  " + " ".join(f"{k:>6s}" for k in main_))
    for i, (g, m) in enumerate(cv.rows):
        print(f"  {g.split('__')[-1][:44]:44s} {len(m):5d} {int(cv.R[i]):4d} {min(1, 10 / cv.R[i]):8.2f}  "
              + " ".join(f"{v[i]:6.1f}" for v in main_.values()))
    for x, y in (("stop", "32B"), ("stop", "8B"), ("32B", "8B"), ("8B", "random"), ("32B", "random")):
        d = main_[x] - main_[y]
        print(f"  {x} vs {y}: fewer false alarms first on {int((d < 0).sum())} CVEs, "
              f"same on {int((d == 0).sum())}, more on {int((d > 0).sum())}")

    # renamed identifiers (E20 runs)
    mem = f"{a.e20}/memorisation"
    variants = {"original": (f"{a.e20}/probes/qwen3-8b-nothink__units_paths.jsonl", f"{mem}/qwen3-32b-nothink__orig.jsonl"),
                "renamed": (f"{mem}/qwen3-8b-nothink__anonA.jsonl", f"{mem}/qwen3-32b-nothink__anonA.jsonl"),
                "strict renamed": (f"{mem}/qwen3-8b-nothink__anonB.jsonl", f"{mem}/qwen3-32b-nothink__anonB.jsonl")}
    print("\nE20 runs (other prompt), identifiers kept / renamed / strict renamed (also comments, strings)")
    print(header(n))
    for v, (p8, p32) in variants.items():
        a8, a32 = scores(p8, ids), scores(p32, ids)
        esc_f = escalate_share(a8, G, 0.30)
        esc_s = stop_rule(a8, a32, G, 10, 1)
        for name, sc, c in ((f"{v}: 8B", a8, c8), (f"{v}: 32B", a32, c32),
                            (f"{v}: fixed f = 0.30", np.where(esc_f, a32, a8), c8 + esc_f.mean() * c32),
                            (f"{v}: stop rule ({esc_s.mean():.0%} up)", np.where(esc_s, 2 + a32, a8),
                             c8 + esc_s.mean() * c32)):
            print(row(name, c, summary(cv.ifa(sc), cv), recall_at(sc, Y, G)))


if __name__ == "__main__":
    main()
