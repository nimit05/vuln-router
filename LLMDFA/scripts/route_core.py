"""Fast rule search: per (bug, feature) prefix sums make any (threshold, lo, hi)
rule an O(1) evaluation instead of a rescan."""
import sys, os, itertools, collections, random, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from route_logsig import load, case_err
from score_split import split_variants
from route_feats import index, extract

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reproduction")
BUGS = ("xss", "osci", "dbz")
SHORT = {"Phi-4-mini-instruct": "Phi", "Qwen2.5-Coder-7B-Instruct": "Qwen",
         "granite-3.1-8b-instruct": "Granite", "Mistral-Nemo-Instruct-2407": "Nemo"}
_merged = load([f"{BASE}/logs", f"{BASE}/logs_router"])
MODELS = sorted({m for (m, b) in _merged}, key=lambda m: SHORT[m])
_IDX = index()

# stats[bug][file][model] = (tp_capped, fp, gt, sec)
STATS, FEAT, ALL, VAR = {}, {}, {}, {}
for bug in BUGS:
    common = set.intersection(*[set(_merged[(m, bug)]) for m in MODELS])
    S, keep = {}, set()
    for f in common:
        if f not in _IDX:
            continue
        ft = extract(_IDX[f])
        ft["n_summaries"] = _merged[(MODELS[0], bug)][f]["n_summaries"]
        ft["n_callees"] = _merged[(MODELS[0], bug)][f]["n_callees"]
        FEAT[(bug, f)] = ft
        S[f] = {}
        for m in MODELS:
            r = _merged[(m, bug)][f]
            tp = min(r["analysis_result"]["TPs"], r["ground_truth"]["TPs"])
            S[f][m] = (tp, r["analysis_result"]["FPs"], r["ground_truth"]["TPs"],
                       r["single time cost"])
        VAR[(bug, f)] = _merged[(MODELS[0], bug)][f]["variant"]
        keep.add(f)
    STATS[bug], ALL[bug] = S, keep
FIELDS = sorted(next(iter(FEAT.values())).keys())


def f1_of(tp, fp, gt):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / gt if gt else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def score_fixed(model, files_by_bug):
    macro, cost = [], 0.0
    for bug, fs in files_by_bug.items():
        tp = fp = gt = 0
        for f in fs:
            a, b, g, t = STATS[bug][f][model]
            tp += a; fp += b; gt += g; cost += t
        macro.append(f1_of(tp, fp, gt))
    return sum(macro) / len(macro), cost


def score_oracle(files_by_bug):
    macro, cost = [], 0.0
    for bug, fs in files_by_bug.items():
        tp = fp = gt = 0
        for f in fs:
            best = min(MODELS, key=lambda m: (
                (STATS[bug][f][m][2] - STATS[bug][f][m][0]) + STATS[bug][f][m][1],
                STATS[bug][f][m][3]))
            a, b, g, t = STATS[bug][f][best]
            tp += a; fp += b; gt += g; cost += t
        macro.append(f1_of(tp, fp, gt))
    return sum(macro) / len(macro), cost


class Index:
    """Prefix sums per (bug, field) over cases sorted by feature value."""
    def __init__(self, files_by_bug):
        self.fb = {b: sorted(fs) for b, fs in files_by_bug.items()}
        self.pre = {}
        for bug, fs in self.fb.items():
            for field in FIELDS:
                order = sorted(fs, key=lambda f: FEAT[(bug, f)][field])
                vals = [FEAT[(bug, f)][field] for f in order]
                acc = {m: [(0, 0, 0, 0.0)] for m in MODELS}
                for f in order:
                    for m in MODELS:
                        p = acc[m][-1]; c = STATS[bug][f][m]
                        acc[m].append((p[0]+c[0], p[1]+c[1], p[2]+c[2], p[3]+c[3]))
                self.pre[(bug, field)] = (vals, acc)

    def thresholds(self, field, cap=12):
        vs = sorted({v for bug in self.fb for v in self.pre[(bug, field)][0]})
        if len(vs) <= cap:
            return vs[1:]
        step = len(vs) / cap
        return sorted({vs[int(i*step)] for i in range(1, cap)})

    def score_rule(self, field, thr, lo, hi):
        macro, cost = [], 0.0
        for bug in self.fb:
            vals, acc = self.pre[(bug, field)]
            n = len(vals)
            k = 0
            while k < n and vals[k] < thr:
                k += 1
            L, H = acc[lo], acc[hi]
            tp = L[k][0] + H[n][0] - H[k][0]
            fp = L[k][1] + H[n][1] - H[k][1]
            gt = L[k][2] + H[n][2] - H[k][2]
            cost += L[k][3] + H[n][3] - H[k][3]
            macro.append(f1_of(tp, fp, gt))
        return sum(macro) / len(macro), cost

    def search(self):
        out = []
        for field in FIELDS:
            for thr in self.thresholds(field):
                for lo, hi in itertools.permutations(MODELS, 2):
                    mac, cost = self.score_rule(field, thr, lo, hi)
                    out.append((mac, -cost, field, thr, lo, hi))
        out.sort(reverse=True)
        return out


def make_split(holdout=12, seed=None):
    SEL, HELD = {}, {}
    for bug in BUGS:
        variants = {VAR[(bug, f)] for f in ALL[bug]}
        if seed is None:
            sel, held = split_variants(variants, holdout)
        else:
            ordered = sorted(variants); rng = random.Random(seed)
            h = set(rng.sample(ordered, min(holdout, len(ordered))))
            sel, held = sorted(set(ordered) - h), sorted(h)
        SEL[bug] = {f for f in ALL[bug] if VAR[(bug, f)] in sel}
        HELD[bug] = {f for f in ALL[bug] if VAR[(bug, f)] in held}
    return SEL, HELD
