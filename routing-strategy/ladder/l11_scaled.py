#!/usr/bin/env python3
"""L11 -- scaled IRIS: which of IRIS's two LLM stages should get the big model?

  spec stage    an LLM labels library APIs as taint sources/sinks; CodeQL then finds dataflow
                paths. Decides whether a real path exists at all. Runs: s_q8 (Qwen3-8B),
                s_dsc (deepseek-coder-7b-v1.5, the paper's small model), s_q32 (Qwen3-32B)
  filter stage  an LLM scores every path; the analyst reads each project's list from the top.
                Qwen3-8B, Qwen3-32B, and the 8B -> 32B stop-rule cascade (l10)

Unit = CVE (one project = one CVE). A CVE counts as found within k when one of its real paths is
among the k highest-scored paths of its project; a CVE whose spec run produced no real path is
missed whatever the filter does. Cost = GPU-seconds:
  spec    seconds the vLLM server was busy during the run (10-s engine log lines with running or
          waiting requests), plus the tokens IRIS sent and received (IRIS_USAGE_LOG)
  filter  wall seconds per path of the probe run (32 concurrent requests, one A100), times paths
Data: results/2026-10-scaled-iris/ (copied back from gpu7:/tmp/nimit/scaled).
"""
import argparse, datetime, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l10_percve import Cves, summary, stop_rule, escalate_share, KS   # noqa: E402

RUNS = {"s_q8": "Qwen3-8B", "s_q32": "Qwen3-32B"}   # deepseek-7b (s_dsc) crashed in window 1, not run
FULL = {"s_q8": "s_q8_full", "s_q32": "s_q32_full"}  # path sets incl. projects recovered by the CodeQL retry
SERVER = {"s_q8": "qwen3-8b_w1", "s_dsc": "dsc7b_w1", "s_q32": "qwen3-32b_w1"}   # window-1 server logs
LINE = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*Running: (\d+) reqs, Waiting: (\d+) reqs")


def busy_seconds(log, t0, t1, year):
    """10 s per engine log line inside [t0, t1] that shows running or waiting requests"""
    n = 0
    for l in open(log, errors="replace"):
        m = LINE.search(l)
        if not m:
            continue
        ts = datetime.datetime.strptime(f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=datetime.timezone.utc).timestamp()
        if t0 <= ts <= t1 + 10 and (int(m.group(2)) or int(m.group(3))):
            n += 1
    return 10.0 * n


def scores(path, ids):
    if not os.path.exists(path):
        return None
    d = {}
    for l in open(path):
        r = json.loads(l)
        d[r["unit_id"]] = 0.5 if r["p_vuln"] is None else r["p_vuln"]
    return np.array([d.get(u, 0.5) for u in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/2026-10-scaled-iris")
    ap.add_argument("--year", type=int, default=2026)
    a = ap.parse_args()
    D = a.dir
    projects = [l.strip() for l in open(f"{D}/projects.txt") if l.strip()]
    timing = {}
    if os.path.exists(f"{D}/probe/timing.jsonl"):
        for l in open(f"{D}/probe/timing.jsonl"):
            t = json.loads(l)
            timing[(t["model"], t["units_file"])] = t["wall_s"] / max(t["units"], 1)

    print("1) spec stage, per model")
    print(f"  {'model':20s} {'projects ok':>11s} {'CVE w/ real path':>16s} {'paths':>6s} {'real':>5s} "
          f"{'tokens in/out (M)':>17s} {'GPU busy s':>10s} {'GPU-s/project':>13s}")
    spec_cost, data = {}, {}
    for run, name in RUNS.items():
        st = f"{D}/status/{run}.csv"
        if not os.path.exists(st):
            continue
        ok = sum(1 for l in open(st) if l.split(",")[1] == "0")
        tin = tout = 0
        for f in glob.glob(f"{D}/usage/{run}/*.jsonl"):
            for l in open(f):
                u = json.loads(l)
                tin += u["prompt_tokens"]; tout += u["completion_tokens"]
        t0 = float(open(f"{D}/status/{run}.t0").read()); t1 = float(open(f"{D}/status/{run}.t1").read())
        busy = busy_seconds(f"{D}/logs/vllm_{SERVER[run]}.log", t0, t1, a.year)
        spec_cost[run] = busy
        cq = {}
        for f in (f"{D}/status/{run}_cq.csv", f"{D}/status/{run}_cq_retry.csv"):
            if os.path.exists(f):
                for l in open(f):
                    p, rc = l.split(",")[:2]
                    cq[p] = rc
        failed_cq = sorted(p for p, rc in cq.items() if rc != "0")
        U = f"{D}/data/{FULL[run]}/units_paths.jsonl"
        if not os.path.exists(U):
            print(f"  {name:20s} {ok:11d}   (no units yet)")
            continue
        units = [json.loads(l) for l in open(U)]
        ids = [u["unit_id"] for u in units]
        Y = np.array([int(u["gold_label"]) for u in units]); G = np.array([u["project"] for u in units])
        det = sorted({g for g, y in zip(G, Y) if y == 1})
        data[run] = (ids, Y, G, det)
        print(f"  {name:20s} {ok:11d} {len(det):16d} {len(ids):6d} {int(Y.sum()):5d} "
              f"{tin / 1e6:8.2f}/{tout / 1e6:<8.2f} {busy:10.0f} {busy / max(len(projects), 1):13.1f}")
        print(f"  {'':20s} CodeQL could not finish (64 GB, 3 h): {len(failed_cq)} projects "
              f"{[p.split('__')[-1][:28] for p in failed_cq]}")

    N = len(projects)
    print(f"\n2) whole pipeline: CVEs found when the analyst reads k paths per project (of {N} CVEs run)")
    print(f"  {'spec':18s} {'filter':28s} {'found@1':>7s} {'@5':>5s} {'@10':>5s} {'medIFA':>6s} "
          f"{'GPU-s spec':>10s} {'filter':>7s} {'total':>7s}")
    rows = []
    for run, (ids, Y, G, det) in data.items():
        P8 = scores(f"{D}/probe/qwen3-8b-nothink__{FULL[run]}.jsonl", ids)
        P32 = scores(f"{D}/probe/qwen3-32b-nothink__{FULL[run]}.jsonl", ids)
        c8 = timing.get(("qwen3-8b-nothink", FULL[run])); c32 = timing.get(("qwen3-32b-nothink", FULL[run]))
        cv = Cves(Y, G)
        n = len(ids)
        rifa, rfound = cv.random()
        allreal = {g for g, _ in cv.rows}
        only_real = [g for g in det if g not in allreal]          # every path real: found at once
        rankers = [("no filter (random order)", rifa, rfound, 0.0)]
        if P8 is not None and c8:
            rankers.append(("Qwen3-8B", cv.ifa(P8), None, c8 * n))
        if P32 is not None and c32:
            rankers.append(("Qwen3-32B", cv.ifa(P32), None, c32 * n))
        if P8 is not None and P32 is not None and c8 and c32:
            esc = stop_rule(P8, P32, G, 10, 1)
            rankers.append((f"8B->32B stop rule ({esc.mean():.0%} up)", cv.ifa(np.where(esc, 2 + P32, P8)), None,
                            c8 * n + c32 * esc.sum()))
            esc = escalate_share(P8, G, 0.30)
            rankers.append(("8B->32B fixed f = 0.30", cv.ifa(np.where(esc, P32, P8)), None, c8 * n + c32 * esc.sum()))
        for name, ifa, found, fcost in rankers:
            fk = {k: (found[k] if found is not None else (ifa < k).astype(float)).sum() + len(only_real) for k in KS}
            med = float(np.median(ifa)) if len(ifa) else float("nan")
            tot = spec_cost.get(run, 0) + fcost
            rows.append((run, name, fk, med, ifa, cv))
            print(f"  {RUNS[run]:18s} {name:28s} {fk[1]:7.1f} {fk[5]:5.1f} {fk[10]:5.1f} {med:6.1f} "
                  f"{spec_cost.get(run, 0):10.0f} {fcost:7.0f} {tot:7.0f}")

    print("\n3) per spec run: the filter rankers compared CVE by CVE (only CVEs with real and false paths)")
    for run in data:
        rr = {name: (ifa, cv) for r, name, _, _, ifa, cv in rows if r == run}
        if "Qwen3-32B" not in rr:
            continue
        cv = rr["Qwen3-32B"][1]
        print(f"  {RUNS[run]}: {len(cv.rows)} CVEs")
        for name, (ifa, _) in rr.items():
            s = summary(ifa, cv)
            print(f"    {name:30s} share read before hit {s['share']:.3f} [{s['lo']:.3f},{s['hi']:.3f}]")
        names = [k for k in rr if k.startswith("8B->32B stop")]
        for x, y in ((names[0] if names else None, "Qwen3-32B"), ("Qwen3-8B", "Qwen3-32B"),
                     ("Qwen3-8B", "no filter (random order)")):
            if x is None or x not in rr:
                continue
            d = rr[x][0] - rr[y][0]
            print(f"    {x} vs {y}: fewer false alarms first on {int((d < 0).sum())}, same {int((d == 0).sum())}, "
                  f"more {int((d > 0).sum())}")


def found_at(y, g, s, projects, k=10, orders=20):
    """per CVE (all projects run): 1 if a real path is among the k highest-scored, ties at random"""
    rng = np.random.default_rng(0)
    out = []
    for p in projects:
        m = np.where(g == p)[0]
        if not len(m) or y[m].sum() == 0:
            out.append(0.0)
            continue
        hit = sum(float(y[m[np.argsort(-(s[m] + rng.random(len(m)) * 1e-9), kind="stable")[:k]]].sum() > 0)
                  for _ in range(orders))
        out.append(hit / orders)
    return np.array(out)


def paired(D, projects, best="spec Qwen3-32B + filter Qwen3-8B"):
    """section 4: every pipeline against the best, CVE by CVE over all projects, bootstrap over CVEs"""
    P = {}
    for run in RUNS:
        U = f"{D}/data/{FULL[run]}/units_paths.jsonl"
        if not os.path.exists(U):
            continue
        units = [json.loads(l) for l in open(U)]
        ids = [u["unit_id"] for u in units]
        y = np.array([int(u["gold_label"]) for u in units]); g = np.array([u["project"] for u in units])
        s8 = scores(f"{D}/probe/qwen3-8b-nothink__{FULL[run]}.jsonl", ids)
        s32 = scores(f"{D}/probe/qwen3-32b-nothink__{FULL[run]}.jsonl", ids)
        if s8 is None or s32 is None:
            continue
        P[f"spec {RUNS[run]} + filter Qwen3-8B"] = found_at(y, g, s8, projects)
        P[f"spec {RUNS[run]} + filter Qwen3-32B"] = found_at(y, g, s32, projects)
        esc = stop_rule(s8, s32, g, 10, 1)
        P[f"spec {RUNS[run]} + stop rule"] = found_at(y, g, np.where(esc, 2 + s32, s8), projects)
    if best not in P:
        return
    print(f"\n4) CVE by CVE (all {len(projects)} run): found within 10 paths, '{best}' minus each other pipeline")
    rng = np.random.default_rng(1)
    for k, v in P.items():
        if k == best:
            continue
        d = P[best] - v
        bs = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(4000)]
        print(f"  minus {k:36s} {d.sum():+5.1f} CVEs  95% CI [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]"
              f"  better on {int((d > 0.5).sum())}, worse on {int((d < -0.5).sum())}")
    for run in RUNS:
        sec = 0.0
        for f in (f"{D}/status/{run}_cq.csv", f"{D}/status/{run}_cq_retry.csv"):
            if os.path.exists(f):
                sec += sum(float(l.split(",")[2]) for l in open(f))
        print(f"  CodeQL time for the {RUNS[run]} spec run (seconds summed over projects, incl. retries): {sec / 3600:.1f} h")


if __name__ == "__main__":
    main()
    _a = argparse.ArgumentParser(); _a.add_argument("--dir", default="results/2026-10-scaled-iris"); _a.add_argument("--year")
    _d = _a.parse_args().dir
    paired(_d, [l.strip() for l in open(f"{_d}/projects.txt") if l.strip()])
