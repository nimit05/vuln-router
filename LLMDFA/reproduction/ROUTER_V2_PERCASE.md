# Router Findings v2 — Per-Case Routing on LLMDFA

Companion to [`ROUTER_FINDINGS.md`](ROUTER_FINDINGS.md), which tested a
**per-bug-type** router and found it degenerate. Its §5.4 argued the failure was
one of *granularity*, not of the routing idea, and predicted that a per-case
router on pre-call features could capture the headroom. This document tests that
prediction.

Analysis date: 2026-08-28. **No new GPU hours were spent** on anything below:
every result replays the 4-model × 3-bug corpus already measured (~40 GPU-hours,
jobs 1853–2310). Two small GPU jobs were queued for the two questions replay
cannot answer (§6).

---

## TL;DR

| Claim | Verdict |
|---|---|
| **The rule transfers to source/sink forms never seen by any model** (§6b) | **Yes — +0.060 F1 over the best fixed model at 6% lower cost**, on 111 held-out-form cases |
| **The rule also survives a change of inference engine** (§6d) | **Yes — +0.043 F1** under HF transformers on the same cases, where the fixed-model ranking flips |
| **The rule survives a second unseen family** (§6e) | **Yes — +0.042 F1** version-matched (job 2417), +0.015 under HF transformers. **4 of 4 held-out conditions positive**, but the margin varies 3x and the threshold `12` is measurably off-centre on family 2 |
| **Fixed model choice does NOT transfer**: rankings invert between form families | Phi > Nemo on `float_*` (0.601 / 0.568); **Nemo > Phi on `int_*`** (0.639 / 0.575) |
| Per-case routing on **static pre-call code features** beats the best fixed model | **Yes — on 10 of 11 splits**, mean **+0.019 F1** at **+1% cost** |
| …including the **pre-registered** split | **No.** −0.019 F1 at +20% cost. Diagnosed in §4, and the diagnosis is not "bad luck" |
| Headroom at per-case granularity | **+0.092 F1** (oracle 0.895 vs best fixed 0.803) — 2.8× the +0.033 available per-bug-type |
| Verifier-triggered cascade (escalate when the Z3 script fails) | **Rejected.** Best variant costs 2.1× always-Phi to gain 0.028 F1 |
| Mechanism: why the small model wins on branchy programs | **Confirmed, but it is not a size law** (§6f). Phi-4-mini emits 1.42 FP/case at `n_if>=12`; Qwen-7B emits 2.27 — the same as the 12B |
| **The pre-registered split's own rule `n_if>=12 ? Qwen : Nemo` on held-out forms** (§6f) | **No — −0.008 F1** vs best fixed. It fails on the pre-registered holdout *and* on unseen forms; §4's diagnosis is confirmed, not excused |
| **Which model pairs can carry the rule at all** (§6f) | **2 of 12** ordered pairs beat the best fixed model — both with Phi-4-mini on the branchy side. Reversing the direction costs −0.106 to −0.126, so the *sign* is real and the *substitutability* is not |

The selected rule is a single comparison, computable from source in
microseconds, with no LLM in the decision:

```
branch conditions >= 12  ->  Phi-4-mini 3.8B   (small)
otherwise                ->  Mistral-Nemo 12B  (large)
```

In effect: **large model on the cheap easy bug types, small model on the
expensive branchy ones.** That is the opposite of the usual instinct, and it is
why the router improves quality and cost at the same time instead of trading
one against the other.

---

## 1. New instrumentation — stage signals

`score_split.py` pairs each result dict with its `.java` path. LLMDFA also
prints, between those two lines, a trace of what the pipeline did:

```
Applying path checker with solver...          -> solver_checks
Constructing solving program...               -> z3_construct
Refining solving program...                   -> z3_refine
Solver-aided path checker succeeded/failed... -> z3_ok / z3_fail
Applying path reachability check with LLM...  -> llm_fallback
start to summarize... / Processing callees... -> n_summaries / n_callees
```

[`route_logsig.py`](../scripts/route_logsig.py) extracts these per case. Two
consequences:

1. `z3_fail` is a **candidate escalation trigger that costs nothing to compute** —
   the Z3 interpreter, not the model, decides whether the script ran.
2. `n_summaries` and `n_callees` are **identical across all four models** on
   100% of XSS and OSCI cases (74% on DBZ), confirming they are parse-derived
   static properties, not model behaviour — so they are legitimate *pre-call*
   features.

[`route_feats.py`](../scripts/route_feats.py) adds 11 static features read
straight from the Java source (`loc`, `n_if`, `n_cond`, `cond_calls`,
`cond_math`, `cond_nondet`, `has_static`, …).

**Join fix.** Juliet splits variants such as `_22` across `_22a.java` and
`_22b.java`. Naive stem matching dropped 563 of 1,396 cases — and those are the
cross-file variants, i.e. the *hard* ones. The extractor now concatenates all
parts of a variant. Coverage after the fix: **1,396 / 1,396**.

---

## 2. Strategies tested

All fitted on the 25 select variants, reported on the 12 held-out variants of
the pre-registered split (`SPLIT_SEED = 20260826`).

| strategy | avgF1 | XSS | OSCI | DBZ | GPU-sec |
|---|---|---|---|---|---|
| always Granite | 0.730 | 0.965 | 0.772 | 0.452 | 14,420 |
| always Nemo | 0.798 | 0.984 | 0.905 | 0.506 | 10,811 |
| always Qwen | 0.784 | 0.981 | 0.902 | 0.467 | 7,167 |
| **always Phi (best fixed)** | **0.803** | 0.897 | 0.894 | 0.617 | **6,985** |
| per-bug-type router (fitted) | 0.785 | 0.984 | 0.905 | 0.467 | 7,841 |
| best verifier cascade `Nemo→Phi [z3_fail≥2]` | 0.831 | 0.984 | 0.905 | 0.604 | 14,719 |
| **ORACLE per-case** | **0.895** | 0.993 | 0.962 | **0.730** | **6,584** |

Two facts in that table drive everything else.

**The oracle is better *and* cheaper than every fixed model** (0.895 at 6,584s
vs Phi 0.803 at 6,985s). Per-case, the right model is usually also the fast one:
the cheapest model is already error-optimal on **81.4%** of XSS, **76.6%** of
OSCI and **50.3%** of DBZ cases. The oracle's picks are correspondingly
small-model-heavy — Phi 41–63%, Qwen 29–31%, Nemo 6–18%.

**Per-bug-type routing reproduces its published failure** independently
(0.785 vs always-Phi 0.803), on a re-derived case-level join. §5.4 of v1 stands.

### 2.1 Verifier-triggered cascade — tested and rejected

The premise was that `z3_fail` is a free, mechanically-grounded escalation
signal. It is free, but it is only *informative* where it is rare:

| bug | model | fires on | error lift when it fires |
|---|---|---|---|
| XSS | Nemo | 2.7% | **6.8×** |
| XSS | Phi | 14.3% | 2.5× |
| OSCI | all | 6–10% | **0.1–0.3× (anti-predictive)** |
| DBZ | all | **84–93%** | 0.9–1.7× |

On DBZ — the only bug type with headroom — the trigger fires on nearly every
case, so it selects almost everything and discriminates almost nothing. And a
cascade always pays the base model first, so the best variant found
(`Nemo→Phi [z3_fail≥2]`, 0.831) costs **2.1× always-Phi**. It buys quality; it
does not save money.

**Recorded because the idea is intuitive and wrong.** A verifier signal is only
a router feature when the verifier usually succeeds.

### 2.2 Per-case static features — works

Depth-1 rules `(feature, threshold, model_low, model_high)`, searched
exhaustively (1,152 rules, 1.2s via prefix sums in
[`route_core.py`](../scripts/route_core.py)).

Selecting the single best-scoring rule on the fit set **fails** — the same
error one level up from v1 §5.1. `cond_calls≥3 ? Qwen : Granite` tops the fit
set and scores 0.781 held-out, below always-Phi.

Selecting by **stability** instead — 30–40 bootstrap resamples of the select
variants, take the rule *family* that recurs in the top-5 — converges on a
single family in 10 of 11 splits:

```
n_if / n_cond  >=  12-15   ?   Phi-4-mini   :   Mistral-Nemo
```

| split | selected rule | held avgF1 | best fixed | ΔF1 | Δcost |
|---|---|---|---|---|---|
| **PRE-REGISTERED** | `n_if≥12 ? Qwen : Nemo` | 0.783 | 0.803 (Phi) | **−0.019** | +20% |
| seed 101 | `n_if≥12 ? Phi : Nemo` | 0.828 | 0.811 | +0.017 | −17% |
| seed 102 | `n_cond≥13 ? Phi : Nemo` | 0.815 | 0.803 | +0.012 | −9% |
| seed 103 | `n_if≥14 ? Phi : Nemo` | 0.846 | 0.820 | +0.026 | +46% |
| seed 104 | `n_if≥15 ? Phi : Nemo` | 0.833 | 0.802 | +0.031 | −16% |
| seed 105 | `n_cond≥13 ? Phi : Nemo` | 0.813 | 0.804 | +0.009 | −10% |
| seed 106 | `n_cond≥13 ? Phi : Nemo` | 0.888 | 0.849 | +0.039 | +52% |
| seed 107 | `n_cond≥12 ? Phi : Nemo` | 0.818 | 0.806 | +0.012 | −13% |
| seed 108 | `n_cond≥12 ? Phi : Nemo` | 0.846 | 0.820 | +0.027 | −18% |
| seed 109 | `n_cond≥13 ? Phi : Nemo` | 0.841 | 0.813 | +0.029 | −17% |
| seed 110 | `n_cond≥14 ? Phi : Nemo` | 0.860 | 0.838 | +0.022 | −9% |

**10/11 splits, mean +0.019 F1, mean cost +1%** — and 8 of the 10 wins were
also *cheaper* (−9% to −18%). `best fixed` is chosen with hindsight on the
held-out set, so the comparison is deliberately unfavourable to the router.

**Not a bug-type proxy.** `n_if ≥ 13` routes 39.5% of DBZ cases to Phi and
60.5% to Nemo, i.e. it discriminates *within* the bug type — precisely the
granularity per-bug-type routing could not reach.

---

## 3. Why the rule looks backwards

DBZ programs are branchy (median 9 `if`s, up to 26); XSS/OSCI are not (median
5, max 14). DBZ is also 4–10× more expensive per case. So the rule resolves to:
**large model on XSS/OSCI (4–11 s/case), small model on branchy DBZ (40–73
s/case)**. Cost savings come from the expensive bug type; quality gains come
from both sides.

---

## 4. The pre-registered split fails, and the reason is not luck

Per-bug F1, **same models, same pipeline**, either side of the pre-registered
split boundary:

| model | DBZ select | DBZ held | drop | OSCI select | OSCI held |
|---|---|---|---|---|---|
| Granite | 0.579 | 0.452 | −0.127 | 0.782 | 0.772 |
| Nemo | 0.602 | 0.506 | −0.096 | 0.892 | 0.905 |
| Qwen | 0.611 | 0.467 | −0.144 | 0.865 | 0.902 |
| **Phi** | 0.597 | **0.617** | **+0.020** | 0.782 | **0.894** |

**Phi is unremarkable in the fitting data and the best model in the held-out
data.** Its OSCI F1 rises 0.112 across the boundary; on DBZ it is the only model
that does not degrade. A Pareto filter computed on select data therefore *drops
Phi from the candidate pool*, and no router fitted on select can choose a model
whose advantage is invisible there.

This follows from something v1 §3.3 already disclosed: the pre-registered DBZ
held-out set contains `_09`, `_12` and `_22` — all three variants the paper's
Appendix A.4.2 documents as breaking path validation. Phi's edge *is* robustness
on hard variants, so the split concentrates the entire signal on the wrong side
of the boundary.

**The conclusion is about protocol, not about the router.** A single
12-variant / 91-DBZ-case holdout is too high-variance to evaluate a router on.
Reporting should be over repeated splits with the spread shown, with the
pre-registered split named as one draw. The pre-registered result is reported
first above precisely so this is not read as seed-shopping.

---

## 5. Mechanism — confirmed

Average **false alarms per case**, by branch count, all matched cases:

| bug | `n_if` band | cases | Granite | Nemo | Phi | Qwen |
|---|---|---|---|---|---|---|
| DBZ | 0–7 | 122 | 0.14 | 0.50 | **0.24** | 0.61 |
| DBZ | 8–11 | 47 | 0.43 | 0.79 | **0.64** | 0.96 |
| DBZ | **≥12** | 117 | 3.26 | 2.34 | **1.24** | 2.23 |
| XSS | any | 666 | ≤0.04 | ≤0.04 | ≤0.02 | ≤0.04 |

On branchy DBZ programs the 3.8B emits **roughly half** the false alarms of the
12B and the 8B. This is the documented DBZ failure mode (precision collapses,
recall does not) resolved by model size — and it inverts the usual assumption.

Z3 refinement rounds per case on DBZ rise from ~5–7 in the low band to
**30.2 (Granite), 22.6 (Phi), 15.5 (Nemo), 11.4 (Qwen)** at `n_if ≥ 12`.
Note Phi refines *more* than Nemo yet emits *fewer* false alarms: it retries
more and concedes less. Refinement count alone is therefore not the mechanism —
what matters is what the model does when repair fails.

---

## 6. Power analysis — why the missing 1,551 DBZ cases were **not** run

286 of 1,851 DBZ cases have all four models. The obvious next step was to
complete them (~20 GPU-hours). Bootstrap (with replacement) over the router's
margin against the best fixed model says not to:

| DBZ n | status | margin | 90% CI | width |
|---|---|---|---|---|
| 91 | have | +0.021 | [+0.009, +0.032] | 0.024 |
| 286 | have | +0.021 | **[+0.014, +0.029]** | 0.015 |
| 600 | would need runs | +0.021 | [+0.017, +0.026] | 0.009 |
| 1,851 | would need runs | +0.022 | [+0.019, +0.024] | 0.005 |

The interval already excludes zero at n=286. Twenty GPU-hours would narrow the
width from 0.015 to 0.005 and change no conclusion. **Not run.**

What the bootstrap *cannot* answer is coverage: all 286 cases come from the
sorted prefix, which is **9 of 50 source/sink forms, every one of them
`float_*`**. That is a generalisation gap, not a variance gap, and only new
cases can close it.

### Completed (jobs 2372 / 2373, 2026-08-30)

Both ran on `gpu-A100` (A100-PCIE-40GB) under vLLM 0.27.1 — **version- and
hardware-matched to the whole existing corpus**. Sanity counters clean:
111/111 and 150/150 cases, `empty=0 failed=0` throughout.

---

## 6b. The decisive test — held-out FORMS

Job 2372 ran Phi-4-mini and Mistral-Nemo over DBZ offset 815, count 111:
`int_Environment_divide`, `int_Environment_modulo`, `int_File_divide` — three
source/sink forms **never measured for any model**, from the `int_*` family,
where every previously measured case was `float_*`.

The rule under test (`n_if >= 12 ? Phi : Nemo`) was fixed **before** these ran,
selected by stability selection over `float_*` forms only.

| strategy | F1 | TP | FP | GPU-sec |
|---|---|---|---|---|
| always Phi-4-mini (3.8B) | 0.575 | 86 | 102 | 3,619 |
| always Mistral-Nemo (12B) | 0.639 | 106 | 115 | 4,408 |
| **ROUTER `n_if>=12 ? Phi : Nemo`** | **0.699** | 101 | **77** | **4,147** |
| Oracle (2-model upper bound) | 0.751 | 101 | 57 | 3,686 |

**+0.060 F1 over the best fixed model, at 6% lower cost**, capturing 54% of the
available oracle headroom — on forms the rule had never seen.

### Why this is the result that matters

The two models **swap places** between form families:

| | `float_*` (fitted) | `int_*` (held out) |
|---|---|---|
| Phi-4-mini | **0.601** | 0.575 |
| Mistral-Nemo | 0.568 | **0.639** |

So *"just pick the best model"* does not survive a change of program family —
whichever you pick from the fitted data is the wrong one on the new data. The
router survives because it conditions on a **property of the code** rather than
betting on a model. It takes Nemo's recall where recall is cheap and Phi's
precision where precision is expensive: TP 101 (near Nemo's 106) with FP 77
(below *both* Phi's 102 and Nemo's 115).

### The mechanism replicates on unseen forms

Average false alarms per case, 111 held-out-form cases:

| `n_if` band | cases | Phi FP | Nemo FP | Phi TP | Nemo TP |
|---|---|---|---|---|---|
| 0–7 | 42 | 0.57 | **0.05** | 0.76 | 0.90 |
| 8–11 | 24 | 0.58 | **0.46** | 0.58 | 0.96 |
| **≥12** | 45 | **1.42** | 2.27 | 0.89 | 1.00 |

The crossover predicted from the `float_*` data appears again: the 12B is far
cleaner on simple programs (0.05 vs 0.57) and markedly worse on branchy ones
(2.27 vs 1.42). That crossover is what the rule exploits, and it is not an
artifact of the forms it was fitted on.

---

## 6c. Cost frontier below 3.8B — job 2373

Qwen2.5-Coder-**1.5B**, DBZ, 140 cases matched against every other model:

| model | params | F1 | prec | recall | s/case |
|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | 1.5B | 0.450 | 37.68% | 55.71% | **33.5** |
| Qwen2.5-Coder-7B | 7B | 0.555 | 40.67% | 87.14% | 38.5 |
| **Phi-4-mini** | 3.8B | **0.589** | 48.83% | 74.29% | 45.2 |
| Mistral-Nemo | 12B | 0.556 | 40.06% | 90.71% | 72.5 |
| Granite-3.1 | 8B | 0.526 | 37.65% | 87.14% | 97.2 |

The frontier *does* extend lower — 1.5B is the cheapest point measured — but it
buys 26% less time for **0.139 less F1** than Phi. It is a Pareto endpoint, not
a replacement. Note again that seconds are not monotonic in parameters: the 12B
is cheaper than the 8B, and the 1.5B is only 26% cheaper than a model 2.5x its
size.

---

## 6d. These results are NOT portable across inference engines

While jobs 2372/2373 were queued, the same probe was set up on a second box
(A100-SXM4-80GB). Its driver (535 / CUDA 12.2) cannot load the cu13 wheels vLLM
0.27.1 requires, and PyPI on that box was throttled to ~26 KB/s, so the models
were served through HF transformers 4.57.1 instead, with the LLMDFA client code
byte-identical.

A control block — the same 37 `float_Environment_divide` cases already measured
on csecluster, same weights, same bf16, same greedy decoding — gives:

| model | engine | TP | FP | F1 | cases differing |
|---|---|---|---|---|---|
| Mistral-Nemo | HF transformers | 29 | 30 | **0.617** | |
| | vLLM 0.27.1 | 30 | 49 | **0.526** | **17/35** |
| Phi-4-mini | HF transformers | 24 | 23 | **0.585** | |
| | vLLM 0.27.1 | 28 | 22 | **0.659** | **23/35** |

**Roughly half of all cases return a different verdict, and F1 moves by up to
0.09 — in opposite directions for the two models**, so it cannot be calibrated
away. Cumulative input tokens are comparable (Nemo 40.6K vs 43.0K; Phi 45.2K vs
47.4K), ruling out prompt truncation.

The explanation consistent with the evidence: attention-kernel numerics differ
between engines, at temperature 0 a near-tie flips a token, and LLMDFA chains
~40 LLM calls per case — spec extraction, CoT summarisation, Z3 synthesis,
repair — so one flipped token propagates into a different Z3 script and a
different verdict.

### But the ROUTER survives the engine change — only the fixed choices don't

The full 111-case held-out-form probe was also completed under HF transformers,
giving the same cases under two engines:

| condition | Phi-4-mini | Mistral-Nemo | ROUTER | router vs best fixed |
|---|---|---|---|---|
| `float_*` forms, vLLM 0.27.1 | **0.601** | 0.568 | — | — |
| `int_*` forms, vLLM 0.27.1 | 0.575 | **0.639** | **0.699** | **+0.060** |
| `int_*` forms, HF transformers | **0.677** | 0.628 | **0.719** | **+0.043** |

**The better fixed model flips twice** — across program family *and* across
inference engine. Any static "use model X" decision is wrong in two of the three
rows. The router is ahead in both rows where it can be evaluated.

The mechanism is what carries it. False alarms per case, same 111 cases:

| `n_if` band | cases | vLLM Phi | vLLM Nemo | HF Phi | HF Nemo |
|---|---|---|---|---|---|
| 0–7 | 42 | 0.57 | **0.05** | 0.24 | **0.14** |
| 8–11 | 24 | 0.58 | **0.46** | **0.50** | 0.75 |
| **≥12** | 45 | **1.42** | 2.27 | **0.96** | 2.04 |

At `n_if >= 12` the small model emits fewer false alarms **under both engines**
(1.42 vs 2.27, and 0.96 vs 2.04). The absolute rates move with the engine; the
ordering that the rule depends on does not.

---

## 6e. A SECOND held-out family — the rule holds under both engines

DBZ offset 1481 (`int_getParameter_Servlet_divide`, `int_getParameter_Servlet_modulo`,
`int_getQueryString_Servlet_divide`), 111 cases. Run twice: **job 2417 under vLLM
0.27.1** (version-matched, authoritative) and on gpu0 under HF transformers. The two
engines are kept in separate tables throughout.

### Version-matched (job 2417, vLLM 0.27.1)

| strategy | F1 | TP | FP | GPU-sec |
|---|---|---|---|---|
| always Phi-4-mini (3.8B) | 0.594 | 93 | 109 | 6,998 |
| always Mistral-Nemo (12B) | **0.649** | 111 | 120 | 8,531 |
| **ROUTER `n_if>=12 ? Phi : Nemo`** | **0.691** | 103 | **84** | **7,491** |

**+0.042 F1 over the best fixed model at 12% lower cost.** Crossover intact:
at `n_if >= 12`, Phi 1.27 FP/case vs Nemo 2.17; at `n_if` 0–7, Nemo 0.28 vs Phi 0.78.

### Four held-out conditions, all positive

| family | engine | best fixed | router | Δ |
|---|---|---|---|---|
| `int_*` family 1 | vLLM 0.27.1 | 0.639 | **0.699** | **+0.060** |
| `int_*` family 1 | HF transformers | 0.677 | **0.719** | **+0.042** |
| `int_*` family 2 | vLLM 0.27.1 | 0.649 | **0.691** | **+0.042** |
| `int_*` family 2 | HF transformers | 0.698 | **0.714** | **+0.015** |

The rule is ahead in **4 of 4** held-out conditions spanning two program families and
two inference engines. The margin varies 3x (+0.015 to +0.060), so the *size* of the
win is not stable even though its *sign* is.

### The HF-transformers run of the same family

Reported separately below and never merged into the vLLM tables.

| strategy | F1 | TP | FP | GPU-sec |
|---|---|---|---|---|
| always Phi-4-mini (3.8B) | 0.640 | 79 | 57 | 22,077 |
| always Mistral-Nemo (12B) | **0.698** | 110 | 94 | 17,213 |
| **ROUTER `n_if>=12 ? Phi : Nemo`** | **0.714** | 96 | 62 | 19,976 |
| Oracle (2-model bound) | 0.791 | 102 | 45 | 16,802 |

**+0.015 over the best fixed model** — the rule wins a third consecutive
held-out condition, but the margin has collapsed from +0.060 / +0.043 to +0.015.

The crossover itself survives:

| `n_if` band | cases | Phi FP | Nemo FP |
|---|---|---|---|
| 0–7 | 67 | 0.30 | 0.27 |
| 8–11 | 4 | **0.75** | 2.50 |
| ≥12 | 40 | **0.85** | 1.65 |

But a threshold sweep on this family shows the deployed cut point is **not**
optimal here:

| threshold | F1 |
|---|---|
| `n_if>=6` | **0.743** |
| `n_if>=8` | 0.723 |
| `n_if>=12` (deployed) | 0.714 |
| `n_if>=18` | 0.698 |

That sweep is fitted on the data it is scored on, so `>=6` is an *oracle*
threshold, not a deployable one — it cannot be claimed as an improvement. What
it does establish is that **12 is off-centre for this family**, costing ~0.03 F1
against the best available cut.

**Honest reading.** The *direction* — route branchy programs to the small model —
transfers across three held-out conditions. The *exact cut point* does not: this
family's branch distribution is far more bimodal (67 cases at 0–7, only 4 at
8–11, 40 at ≥12) than the fitted family's (42/24/45). A deployable router
probably needs either a per-family calibration step or a formulation less
brittle than a hard threshold — a relative one (e.g. per-corpus branch-count
quantile) rather than an absolute count. That is the next experiment, not a
claim this document can make.

Cost note: Phi consumed **more** GPU-seconds than Nemo here (22,077 vs 17,213),
a third instance of seconds not being monotonic in parameter count.

---

## 6f. Four models on held-out forms — the pre-registered rule FAILS, and the mechanism is not model size

Job 2418 (vLLM 0.27.1, gpu-A100-02, `cases=111 empty=0 failed=0` for both models)
added **Qwen2.5-Coder-7B** and **Granite-3.1-8B** to the offset-815 held-out-form
probe, so all four models now cover the identical 111 `int_*` cases of §6b. This
closes the gap §8 recorded: the pre-registered split's *own* selected rule
(`n_if>=12 ? Qwen : Nemo`) could finally be evaluated on forms it had never seen.

| strategy | F1 | prec | recall | TP | FP | GPU-sec |
|---|---|---|---|---|---|---|
| always Granite-3.1 (8B) | 0.549 | 41.5% | 81.1% | 90 | 127 | 6,935 |
| always Phi-4-mini (3.8B) | 0.575 | 45.7% | 77.5% | 86 | 102 | **3,619** |
| always Qwen2.5-Coder (7B) | 0.620 | 47.0% | 91.0% | 101 | 114 | 3,964 |
| **always Mistral-Nemo (12B)** — best fixed | **0.639** | 48.0% | 95.5% | 106 | 115 | 4,408 |
| PRE-REGISTERED router `n_if>=12 ? Qwen : Nemo` | **0.630** | 47.5% | 93.7% | 104 | 115 | 4,061 |
| **DEPLOYED router `n_if>=12 ? Phi : Nemo`** | **0.699** | 56.7% | 91.0% | 101 | **77** | 4,147 |
| Oracle (4-model bound) | 0.805 | 72.1% | 91.0% | 101 | **39** | 3,760 |

**The pre-registered rule loses on held-out forms too: 0.630 vs 0.639 best fixed
(−0.008).** It failed on the pre-registered holdout (§2.2, −0.019) and it fails
again here. Two independent failures of the same rule, on different data, is not
a variance story — that rule is simply wrong, and §4's diagnosis of *why* the
pre-registered split selected it now has its confirmation.

### Why it fails: Qwen has none of the property the rule depends on

False alarms per case, same 111 held-out-form cases:

| `n_if` band | cases | Phi (3.8B) | Qwen (7B) | Granite (8B) | Nemo (12B) |
|---|---|---|---|---|---|
| 0–7 | 42 | 0.57 | **0.02** | 0.60 | 0.05 |
| 8–11 | 24 | 0.58 | **0.46** | 0.67 | **0.46** |
| **≥12** | 45 | **1.42** | 2.27 | 1.91 | 2.27 |

At `n_if >= 12` **Qwen emits exactly as many false alarms as Nemo (2.27)**. The
crossover the router monetises does not exist for the Qwen/Nemo pair, so routing
between them can only shuffle cases between two equally-noisy models — which is
precisely what the numbers show (FP 115 for the router, 115 for always-Nemo).

The same ordering was already visible in the fitted `float_*` corpus (§5:
Phi 1.24, Qwen 2.23, Nemo 2.34, Granite 3.26 at `n_if >= 12`) and is reproduced
here on unseen forms.

### The direction is real; "smaller is cleaner on branchy code" is not

All 12 ordered model pairs, deployed threshold:

| rule | F1 | TP | FP | GPU-sec | vs best fixed |
|---|---|---|---|---|---|
| `n_if>=12 ? Phi : Nemo` | **0.699** | 101 | **77** | 4,147 | **+0.060** |
| `n_if>=12 ? Phi : Qwen` | **0.688** | 98 | 76 | 4,051 | **+0.049** |
| `n_if>=12 ? Qwen : Nemo` (pre-registered) | 0.630 | 104 | 115 | 4,061 | −0.008 |
| `n_if>=12 ? Nemo : Qwen` | 0.628 | 103 | 114 | 4,311 | −0.011 |
| `n_if>=12 ? Granite : Nemo` | 0.627 | 96 | 99 | 5,063 | −0.011 |
| `n_if>=12 ? Granite : Qwen` | 0.616 | 93 | 98 | 4,966 | −0.023 |
| `n_if>=12 ? Phi : Granite` | 0.611 | 95 | 105 | 6,019 | −0.028 |
| `n_if>=12 ? Nemo : Granite` | 0.565 | 100 | 143 | 6,280 | −0.074 |
| `n_if>=12 ? Qwen : Granite` | 0.557 | 98 | 143 | 5,933 | −0.082 |
| `n_if>=12 ? Nemo : Phi` (reversed) | 0.532 | 91 | 140 | 3,880 | −0.106 |
| `n_if>=12 ? Qwen : Phi` (reversed) | 0.524 | 89 | 140 | 3,533 | −0.115 |
| `n_if>=12 ? Granite : Phi` (reversed) | 0.513 | 81 | 124 | 4,535 | −0.126 |

Two readings, and the second is the uncomfortable one.

**The routing direction is confirmed.** Every rule that sends branchy programs to
the *large* model is catastrophic (−0.106 to −0.126) — worse than any fixed
model. The sign of the effect is not an artifact.

**But only Phi carries it.** Exactly 2 of 12 pairs beat the best fixed model, and
both have Phi-4-mini on the high-branch side. Ordered by parameter count, false
alarms at `n_if >= 12` go 3.8B → **1.42**, 7B → 2.27, 8B → 1.91, 12B → 2.27:
not monotone, and Qwen (7B) is no better than the model three times its size.

So §5's phrase "resolved by model size" is **overstated, and this probe is what
corrects it**. What the router actually exploits is a property of *one model* —
Phi-4-mini concedes less and retries more when Z3 repair fails (§5) — that
happens to be attached to the cheapest model in the pool. That is a genuine and
replicated effect (fitted `float_*`, held-out family 1, held-out family 2), and
it is what makes the router simultaneously better and cheaper. It is **not** a
law about parameter counts, and a practitioner who substituted a different small
model would get the pre-registered rule's result, not the deployed rule's.

### The 4-model oracle

Adding two models raises the per-case upper bound from 0.751 (Phi+Nemo, §6b) to
**0.805**, at *lower* cost than three of the four fixed models. Pairwise bounds:

| pair | oracle F1 |
|---|---|
| Nemo + Phi | **0.751** |
| Granite + Nemo | 0.736 |
| Phi + Qwen | 0.731 |
| Granite + Qwen | 0.714 |
| Nemo + Qwen | 0.695 |
| Granite + Phi | 0.667 |

**Nemo+Phi is the best pair**, i.e. the deployed router's two models are also the
pair with the most complementary errors — chosen, note, without access to this
data. The deployed rule captures 54% of its pair's headroom and 36% of the
4-model headroom, so a better rule over the same four models is still worth
looking for. Depth-1 over `n_if` is not the ceiling.

---

**Consequence for anyone reproducing this work: the inference engine is part of
the experimental configuration and must be reported.** Every number in this
document and in [`ROUTER_FINDINGS.md`](ROUTER_FINDINGS.md)/[`FINDINGS.md`](FINDINGS.md)
is vLLM 0.27.1. The gpu0 transformers runs are therefore **not** merged into any
table above; they are reported only as this control and as the engine-robustness
check immediately above.

## 7. Reproducing

No GPU or cluster access needed. Requires the Juliet sources (not vendored):

```bash
export LLMDFA_BENCH=/path/to/LLMDFA/benchmark
python3 LLMDFA/scripts/route_eval.py        # Pareto filter + stability selection + 11 splits
```

| Script | Purpose |
|---|---|
| [`route_logsig.py`](../scripts/route_logsig.py) | log parser with per-case stage signals |
| [`route_feats.py`](../scripts/route_feats.py) | static code features (handles multi-file variants) |
| [`route_core.py`](../scripts/route_core.py) | prefix-sum rule search, 1,152 rules in 1.2 s |
| [`route_eval.py`](../scripts/route_eval.py) | strategy bake-off, stability selection, robustness |
| [`score_probe.py`](../scripts/score_probe.py) | held-out-form probe scorer: fixed / router / oracle over one matched case set |

## 8. Open

* **Threshold transfer is the live weakness** (§6e). Direction transfers; the cut
  point does not. Test a relative formulation (per-corpus branch-count quantile)
  against the absolute `n_if>=12` across all three held-out conditions — if the
  relative form is stable where the absolute one drifts, that is the deployable
  rule and the stronger claim.
* ~~Qwen2.5-Coder-7B was not run at offset 815~~ — **answered by job 2418 (§6f):
  the pre-registered rule loses on held-out forms too (−0.008), because Qwen has
  none of the low-false-alarm behaviour on branchy code that the rule monetises.**
* **The effect is model-specific, not size-specific** (§6f), which is the bigger
  open question this document leaves. Is Phi-4-mini's precision-under-branching
  a property of that model's post-training, or of the 3–4B scale generally? Only
  more small models (Llama-3.2-3B, Gemma-3-4B, Qwen3-4B) at offset 815 can
  separate those, and the answer decides whether the rule is a recipe or an
  anecdote about one checkpoint.
* The 4-model oracle is **0.805** vs the deployed rule's 0.699 (§6f), so 64% of
  the per-case headroom over four models is still unclaimed by a depth-1 rule.
* The rule is depth-1 by choice — with 195 select DBZ cases, deeper trees overfit
  faster than they help. Revisit only if held-out DBZ coverage grows.
* Whether the same shape (**cheap model on the expensive complex work, large
  model on the cheap simple work**) recurs in IRIS and RepoAudit. One system is a
  curiosity; three would be a finding.
