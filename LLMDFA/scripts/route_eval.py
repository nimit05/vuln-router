import sys, os, random, collections, statistics, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_core
from route_core import *

def per_bug_f1(model, files_by_bug):
    out = {}
    for bug, fs in files_by_bug.items():
        tp = fp = gt = c = 0
        for f in fs:
            a, b, g, t = STATS[bug][f][model]
            tp += a; fp += b; gt += g; c += t
        out[bug] = (f1_of(tp, fp, gt), c)
    return out

def pareto_pool(SEL):
    """Drop models dominated on the FIT set: worse-or-equal F1 on every bug AND
    more expensive. Uses select data only - no test-set information."""
    prof = {m: per_bug_f1(m, SEL) for m in MODELS}
    keep = []
    for m in MODELS:
        dominated = any(
            all(prof[o][b][0] >= prof[m][b][0] for b in BUGS) and
            sum(prof[o][b][1] for b in BUGS) <= sum(prof[m][b][1] for b in BUGS) and o != m
            for o in MODELS)
        if not dominated:
            keep.append(m)
    return keep

class PoolIndex(Index):
    def __init__(self, files_by_bug, pool):
        self.pool = pool
        super().__init__(files_by_bug)
    def search(self):
        out = []
        for field in FIELDS:
            for thr in self.thresholds(field):
                for lo, hi in itertools.permutations(self.pool, 2):
                    mac, cost = self.score_rule(field, thr, lo, hi)
                    out.append((mac, -cost, field, thr, lo, hi))
        out.sort(reverse=True)
        return out

def stability_pick(SEL, pool, n_boot=40, topk=5, seed=7):
    rng = random.Random(seed)
    sv = {b: sorted({VAR[(b, f)] for f in SEL[b]}) for b in BUGS}
    fam, thrs = collections.Counter(), collections.defaultdict(list)
    for _ in range(n_boot):
        sub = {}
        for b in BUGS:
            keep = set(rng.sample(sv[b], max(2, int(0.7 * len(sv[b])))))
            sub[b] = {f for f in SEL[b] if VAR[(b, f)] in keep}
        for mac, nc, fl, th, lo, hi in PoolIndex(sub, pool).search()[:topk]:
            fam[(fl, lo, hi)] += 1; thrs[(fl, lo, hi)].append(th)
    return fam, thrs

print("=== procedure v2: Pareto filter (fit-set only) + stability selection ===\n")
print(f"{'split':<12}{'pool kept':<26}{'selected rule':<30}{'held':>7}{'bestfix':>10}{'dF1':>7}{'cost%':>7}")
rows = []
for label, sd in [("PRE-REG", None)] + [(f"seed {s}", s) for s in range(101, 111)]:
    SEL, HELD = make_split(12, sd)
    pool = pareto_pool(SEL)
    fam, thrs = stability_pick(SEL, pool, n_boot=30)
    bf, hits = fam.most_common(1)[0]
    thr = int(statistics.median(thrs[bf]))
    fl, lo, hi = bf
    m2, c2 = Index(HELD).score_rule(fl, thr, lo, hi)
    fxh = {m: score_fixed(m, HELD) for m in MODELS}
    bfh = max(fxh.values()); bfn = [SHORT[m] for m in MODELS if fxh[m] == bfh][0]
    rows.append((label, m2 - bfh[0], c2 / bfh[1]))
    print(f"{label:<12}{'+'.join(SHORT[m] for m in pool):<26}"
          f"{fl+'>='+str(thr)+' ? '+SHORT[hi]+' : '+SHORT[lo]:<30}{m2:>7.3f}"
          f"{bfh[0]:>7.3f}({bfn[:3]}){m2-bfh[0]:>7.3f}{100*(c2/bfh[1]-1):>6.0f}%")

d = [r[1] for r in rows]
print(f"\nbeat best fixed model on {sum(1 for x in d if x>0)}/{len(d)} splits; "
      f"mean {statistics.mean(d):+.3f}  worst {min(d):+.3f}  best {max(d):+.3f}")
print(f"cost vs best fixed: mean {100*(statistics.mean([r[2] for r in rows])-1):+.0f}%")

# what the pre-registered split now yields, in full
SEL, HELD = make_split()
pool = pareto_pool(SEL)
print(f"\n=== PRE-REGISTERED SPLIT, full table (pool = {', '.join(SHORT[m] for m in pool)}) ===")
fam, thrs = stability_pick(SEL, pool, n_boot=30)
bf, _ = fam.most_common(1)[0]; thr = int(statistics.median(thrs[bf])); fl, lo, hi = bf
print(f"{'strategy':<42}{'avgF1':>8}{'XSS':>8}{'OSCI':>8}{'DBZ':>8}{'GPU-sec':>10}")
print("-"*84)
def show(name, mac, cost, per):
    print(f"{name:<42}{mac:>8.3f}" + "".join(f"{per[b]:>8.3f}" for b in BUGS) + f"{cost:>10.0f}")
for m in MODELS:
    pb = per_bug_f1(m, HELD)
    show("always "+SHORT[m], sum(pb[b][0] for b in BUGS)/3, sum(pb[b][1] for b in BUGS),
         {b: pb[b][0] for b in BUGS})
idxH = Index(HELD)
per = {}
tot_c = 0
for b in BUGS:
    m1, c1 = Index({b: HELD[b]}).score_rule(fl, thr, lo, hi)
    per[b] = m1; tot_c += c1
mac, cost = idxH.score_rule(fl, thr, lo, hi)
print("-"*84)
show(f"ROUTER {fl}>={thr} ? {SHORT[hi]} : {SHORT[lo]}", mac, cost, per)
om, oc = score_oracle(HELD)
pero = {}
for b in BUGS:
    pero[b] = score_oracle({b: HELD[b]})[0]
print("-"*84); show("ORACLE per-case", om, oc, pero)
