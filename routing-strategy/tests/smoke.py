"""End-to-end smoke test on a SYNTHETIC probe table.

Verifies the offline machinery works before any GPU time is spent. The synthetic
generator deliberately builds a pool with COMPLEMENTARY errors (the cheap model
is better on complex units, the big model on simple ones) so that G1 must pass
and the oracle must beat every fixed model. If this test stops showing that, the
bug is in the machinery, not in the data.
"""
import sys, json, os, tempfile
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vulnrouter.table import load
from vulnrouter.strategies import Fixed, Random, Oracle, PredictiveDefer, BinaryThreshold, DualAscent
from vulnrouter import evaluate as ev

rng = np.random.default_rng(7)
N = 800
complexity = rng.gamma(2.0, 3.0, N)            # the routable signal
gold = (rng.random(N) < 0.35).astype(int)

MODELS = {                                     # (base skill, cost, complexity penalty)
    "small-1.5b": (0.60, 1.0, +0.004),         # degrades slowly with complexity
    "mid-7b":     (0.72, 3.5, -0.020),         # degrades fast
    "big-32b":    (0.80, 9.0, -0.028),         # best on simple, worst on complex
}

rows = []
for u in range(N):
    for m, (skill, base_cost, slope) in MODELS.items():
        acc = np.clip(skill + slope * complexity[u], 0.05, 0.98)
        correct = rng.random() < acc
        pred = gold[u] if correct else 1 - gold[u]
        # cost is ENDOGENOUS: a wrong answer means retries, so it costs more
        secs = base_cost * (1 + 0.06 * complexity[u]) * (1.6 if not correct else 1.0)
        rows.append(dict(unit_id=f"u{u}", model=m, commit=f"c{u // 4}",
                         pred_label=int(pred), gold_label=int(gold[u]),
                         seconds=float(secs)))

with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
    path = fh.name

t = load(path, c_fn=10.0, c_fp=1.0)
print(f"table: {t.shape[0]} units x {t.shape[1]} models {t.models}\n")

print("--- references (all fixed models + oracle) ---")
print(ev.references(t).to_string(index=False,
      columns=["strategy", "f1", "loss", "cost"], float_format="%.3f"), "\n")

print("--- G1  pool complementarity ---")
print(ev.gate_g1_pool_complementarity(t).to_string(index=False, float_format="%.4f"), "\n")

# features MUST be a unit_id-indexed frame -- see evaluate.align()
X = pd.DataFrame({"complexity": complexity, "complexity2": complexity ** 2},
                 index=[f"u{u}" for u in range(N)])
print("--- G2  cost endogeneity (R^2 vs constant c(m)) ---")
print(ev.gate_g2_cost_endogeneity(t, X).to_string(index=False, float_format="%.3f"), "\n")

print("--- G3  cascade fire rates ---")
cx = X.reindex(t.units).complexity.to_numpy()
fires = {"disagree@complexity>6": cx > 6, "always": np.ones(len(t.units), bool)}
print(ev.gate_g3_cascade_fire_rate(fires, t).to_string(index=False, float_format="%.3f"), "\n")

# S5/S6 -- binary threshold routed on the observable feature, swept by quantile
bt = BinaryThreshold(score=cx, weak="small-1.5b", strong="big-32b")
curve = ev.frontier(t, bt, np.linspace(0.0, 1.0, 21))
lo, hi = curve.cost.min(), curve.cost.max()
print("--- S5/S6 binary-threshold frontier (head) ---")
print(curve.head(4).to_string(index=False,
      columns=["lam", "f1", "cost"], float_format="%.3f"))
print(f"AIQ = {ev.aiq(curve, lo, hi):.4f}\n")

# S8 -- dual ascent against a workload budget, wrapping the oracle's argmin form
budget = 0.6 * ev.score(t, Fixed('big-32b').route(t, 0))["cost"]
da = DualAscent(inner=Oracle(), budget=budget)
choice = da.route(t)
s = ev.score(t, choice)
print("--- S8 dual ascent ---")
print(f"budget={budget:.0f}  realised={s['cost']:.0f} "
      f"({100*s['cost']/budget:.0f}% of budget)  f1={s['f1']:.3f}  final_lam={da.final_lam:.4f}\n")

best_fixed = max(t.models, key=lambda m: ev.score(t, Fixed(m).route(t, 0))["f1"])
pt, lo_ci, hi_ci = ev.bootstrap_margin(t, Oracle().route(t, 0.0),
                                       Fixed(best_fixed).route(t, 0), n_boot=500)
print(f"--- bootstrap: oracle - always-{best_fixed} ---")
print(f"F1 margin {pt:+.4f}  90% CI [{lo_ci:+.4f}, {hi_ci:+.4f}]")
os.unlink(path)
