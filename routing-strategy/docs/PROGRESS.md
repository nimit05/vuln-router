# Progress log

Running record of what was tried, how, and what came out. Newest session first.
Every experiment gets: **method**, **result**, **verdict**. A verdict is one of
*dead end* / *real effect* / *open*. Numbers here are reproducible from the
scripts named in each entry.

**Status line (2026-09-18, end of session):** on paired PrimeVul, every
configuration measured — 5 models, 2 reasoning-mode families, 6 thinking budgets,
self-consistency — scores **below a coin flip at its own yes-rate** (E12). Per-unit
routing has null-level headroom across models, modes and budgets (E2, E7, E10).
An earlier claim in E6 that reasoning multiplies skill 5-8x is **withdrawn**; it
lacked the yes-rate baseline. The project's viable output is a measurement paper
for a security/SE venue, and its missing piece is a positive control (Q5).
Precision-first filtering over the same pool also fails (E13): consensus voting is
worse than random, and the ranking gains vanish under leave-one-project-out
selection and per-project breakdown. The remaining untested lever is verification,
not model choice.

**Amended (same day, later):** the LLMDFA rule does **not** transfer to IRIS
(E14 — the apparent structural signal is ESAPI again), but it **does** survive
this project's full null battery where it was measured (E15): it beats a
matched-count random split on 3 of 3 held-out probes at z +5.5 / +4.4 / +2.3,
with a deployable margin of +0.074 to +0.124 whose CIs exclude zero. That is the
first routing result here that passes, and it is a result about *verification-
backed, per-case* routing rather than per-unit model confidence — which is the
lever E13 named. Its oracle headroom is still not real (shuffled oracle 0.940 vs
0.805), so the rule is near the ceiling of per-case model choice, not far below
it.

**Amended (2026-09-23, later):** Q6 is **done** too (E19), and it is the one the
reframed paper leads with. Under a fixed review budget — an analyst reviews k
alerts per project, so being stingy buys nothing — **no LLM filter configuration
beats a matched-budget coin flip on detection** at k = 5, 10 or 20, and several
are significantly worse. The single positive recall row (qwen-7b ranking, z +2.3)
is an artifact of small projects: restricted to the 6 projects where the budget
actually binds it is **+0.001, 2 of 6**, and on the two largest projects the model
finds **zero** true positives in its top 10 where random finds some. That moves
the filter-stage claim off the metric E3 discredited and onto one that cannot be
gamed. A third reporting rule follows: **report any per-project mean alongside
the subset where the budget binds.**

**Amended (2026-09-23):** Q5 is **done** and scale is not the answer (E18). A
32B reasoner on the same 299 pairs lands at excess **−0.018 [−0.060, +0.025]**
against the 8B's −0.029; the paired difference is **+0.011 [−0.048, +0.070]**,
i.e. nothing. Reasoning is the only lever that moves the number (−0.211 → −0.018
inside the 32B). Two corrections to E12 fall out of the same analysis, neither
changing a verdict: its yes-rate for the five pool models was measured over all
5,820 PrimeVul units when only 1,814 enter the pair metric, and it reported point
estimates with no intervals — with a bootstrap, the **reasoning** configurations
sit *on* the line rather than below it. So the headline is now "no configuration
is above a same-rate coin flip, at 8B or 32B, with or without reasoning", and the
remaining gap is a frontier API reasoner, not a bigger local one.

**Amended (2026-09-21):** the verifier-gated cascade is now closed at full
coverage -- 12 ordered model pairs x 3 bug types, 1 non-degenerate win worth
+0.005 F1 on the bug type with no dispersion -- and the mechanism is identified:
`z3_fail` fires on 31.8% of Qwen's dbz cases and 92.8% of Nemo's, so it measures
whether a model can emit valid Z3, not whether a case is hard. The pre-flight
test of E17 is corrected in three places: the shuffled oracle must not gate (it
rejects DBZ), gate 1 is a screen rather than a gate (marginally equal models can
still be conditionally routable), and the routing direction must be fixed before
the run (on IRIS the reversal moves z from -5.8 to +7.8). Full record and the
nine tool defects found by cross-dataset testing:
[`09-findings-2026-09.md`](09-findings-2026-09.md). No GPU hours.

---

## Session 2026-09-17/18

### Context

Goal was to try routing strategies and find a direction for "cost-aware LLM
router for software vulnerability". Two measurement sets already existed:

| Set | Units | Models | Notes |
|---|---|---|---|
| PrimeVul (`data/table_dedup.jsonl`) | 5,820 C/C++ functions, 913 buggy/fixed pairs | 5 | 16% positive |
| IRIS / CWE-Bench-Java (`data/gr/probe5/`) | 2,257 dataflow paths, 14 projects | 5 | 15% positive |

Pool: qwen-1.5b, phi-3.8b, qwen-7b, granite-8b, deepseek-6.7b.

---

### E1 — Does any model in the pool have skill at all?

**Method.** Per model, turn the verdict's token log-probability into a score
`p_vuln`, then measure AUC (chance that a random real bug scores above a random
non-bug; 0.5 = coin flip) and pair-rank (how often the buggy version of a pair
scores above its own fix). Script: scratchpad `common.py` + inline.

**Result.**

| Model | PrimeVul AUC | PrimeVul pair-rank | IRIS AUC | says "vulnerable" |
|---|---|---|---|---|
| qwen-1.5b | 0.539 | 0.495 | 0.463 | 9% / 25% |
| phi-3.8b | 0.482 | 0.530 | 0.523 | 55% / 68% |
| qwen-7b | 0.591 | 0.477 | 0.507 | 2% / 20% |
| granite-8b | 0.555 | 0.531 | 0.408 | 81% / 97% |
| deepseek-6.7b | 0.497 | 0.054 | 0.524 | 0% / 68% |

**Verdict: dead end for the pool.** Every model is at or near chance. The models
differ mainly in how often they say "vulnerable", not in what they know. Routing
between them can only redistribute bias.

---

### E2 — Is the oracle headroom real? (the null-oracle test)

The oracle is a cheating router that reads the answer and picks whichever model
got it right. Every routing paper reports it as the ceiling worth chasing.

**Method.** Shuffle each model's verdicts *within each project*. That destroys
any per-unit skill but preserves each model's positive rate. Recompute the
oracle. If the shuffled oracle matches the real one, the headroom is luck from
models guessing differently. 50-200 draws. Script: scratchpad `null_oracle.py`,
`exp2.py`.

**Result.**

| Oracle | Real | Shuffled (skill-free) |
|---|---|---|
| IRIS, 5 models (F1) | 0.859 | **0.944 ± 0.007** |
| PrimeVul, 5 models (F1) | 0.951 | **0.952 ± 0.006** |
| qwen-7b→granite pair (IRIS AvgF1) | 0.508 | 0.506 ± 0.007 |
| qwen-1.5b→granite pair | 0.463 | 0.458 ± 0.005 |
| phi-3.8b→granite pair | 0.292 | 0.297 ± 0.003 |

**Verdict: dead end, and it explains the earlier negative results.** The
shuffled oracle is as good or better than the real one in every configuration.
The 0.529 per-path oracle in `RESULTS_TABLE.md` and the 0.405 pair oracle in
`RESULTS_HYBRIDLLM.md` are not reachable ceilings; they are artifacts of
diversity in guessing. `06-objective-misalignment.md` diagnosed the objective;
the deeper cause is that the pool has no complementary skill to exploit.

---

### E3 — Is the IRIS metric itself gameable?

**Method.** Keep `k` paths per project chosen uniformly at random — no model in
the loop — and score with IRIS's own per-project AvgF1. 200 draws. Script:
scratchpad `exp2.py`.

**Result.**

| Policy | AvgF1 | #Detected |
|---|---|---|
| random 3 paths/project | 0.209 ± 0.054 | 4.3/16 |
| random 5 paths/project | 0.226 ± 0.046 | 5.7/16 |
| random 10 paths/project | **0.236 ± 0.028** | 7.6/16 |
| always-granite-8b (best single) | 0.207 | 10/16 |
| router + floor, best point (earlier work) | 0.244 | 10/16 |
| floor-only, k=10 (earlier work) | 0.237 | 9/16 |

**Verdict: dead end as a headline metric.** Per-project AvgF1 rewards keeping
few paths, so a coin flip scores 0.236 against the router's 0.244. The
`RESULTS_FLOOR.md` and `RESULTS_CONTROLS.md` gains are mostly the metric. A
random-k row belongs in every future IRIS table.

---

### E4 — Benchmark shortcuts on PrimeVul

**Method.** Compare a features-only predictor against the LLMs: function length
alone, and stacked (5 model scores + 5 verdicts + length), trained on the valid
split and scored on test. Then check length inside pairs. Script: scratchpad
`exp3.py`.

**Result.**
* Length alone: **AUC 0.734**.
* All 5 models stacked + length: AUC 0.713.
* Best single model: AUC 0.591.
* Positive rate climbs with length: 5.5% → 8.8% → 17.7% → 32.2% across quartiles.
* Inside pairs, the **fixed version is longer in 83.8%** of cases (median
  difference 73 characters) — fixes add checks.

**Verdict: dead end unless controlled.** A trivial feature beats the entire
pool. Any pair-accuracy number on PrimeVul needs a length-only control, or it
may be measuring "shorter = buggy".

---

### E5 — Does per-unit cost vary enough to be worth predicting?

Problem doc §1.4 claims `c(m,x)` is materially non-constant (gate G2).

**Method.** Per model, spread of wall-clock seconds across units, and its
correlation with input/output tokens. Script: scratchpad `exp3.py`.

**Result.** p90/p10 spread is 1.3× (qwen-1.5b) to 2.3× (granite-8b,
deepseek-6.7b). Cost tracks input tokens for the small models (r = 0.82-0.87)
and output tokens for the larger ones (r = 0.92-0.95).

**Verdict: gate G2 fails in direct-answer mode.** Treating cost as a per-model
constant is fine here. It only becomes false once the model reasons (see E6).

---

### E6 — Reasoning instead of model choice (GPU run, gpu0)

If no model has skill, the question becomes whether *more computation per unit*
creates skill. Tested with one model in two modes, so weights and data are held
fixed and only the reasoning changes.

**Method.** gpu0 (A100-80GB), vLLM 0.27.1, 299 PrimeVul pairs (599 functions).
Qwen3-8B with its thinking mode off vs on; phi-4 (14B) answering directly vs
step-by-step. Scripts: `think_probe.py`, `run_think.sh`/`chain_all.sh` (on gpu0
under `~/nimit/vr_explore/`), `analyze_think.py`. 30 min wall clock, GPU
released after.

**Result.** Metric is pair-both-right — both versions of a pair labeled
correctly — which cannot be gamed by always answering "vulnerable".

| Configuration | pair-both-right | F1 | AUC | says "vulnerable" | output tokens |
|---|---|---|---|---|---|
| Qwen3-8B no-think | 0.047 | 0.594 | 0.506 | 0.72 | 18 |
| Qwen3-8B think | **0.220** | 0.525 | 0.536 | 0.49 | 2,417 (134×) |
| phi-4 direct | 0.027 | 0.625 | 0.501 | 0.82 | 64 (capped) |
| phi-4 step-by-step | **0.217** | 0.581 | 0.553 | 0.57 | 484 (7.6×) |

**Verdict: WITHDRAWN — see E12.** This entry originally read "real effect, the
first one measured: reasoning raises real skill 5-8×". That claim was wrong. It
compared reasoning against the non-reasoning mode and against nothing else;
`pair-both-right` rewards answering *differently* within a pair, so it has a
yes-rate-dependent random baseline of `r(1-r)` that was never computed. When it
is (E12), every configuration here sits **below** its own random baseline.

What survives from this entry: reasoning cuts the yes-bias from 0.72 to 0.49 and
reduces within-pair stickiness, F1 *falls* while `pair-both-right` rises (so F1
was partly rewarding bias), and cost now varies 3.9× across units (p90/p10 of
thinking tokens), which is where gate G2 becomes true.

---

### E7 — Can the think / don't-think decision be routed?

**Method.** Label each unit with "did reasoning fix it". Predict from static and
cheap-mode features (log code length, cheap `p_vuln`, its distance from 0.5,
input tokens, cheap verdict), 5-fold grouped by pair so a pair never straddles
the split. Also the mode-level oracle against its shuffled null, and
quality when reasoning is spent on a random fraction of pairs.

**Result.**

| Question | Answer |
|---|---|
| "Does thinking fix this unit" learnable? | AUC 0.543 ± 0.053 (Qwen3), 0.547 ± 0.028 (phi-4); base rate 0.22 |
| Cheap-mode confidence as escalation signal | AUC 0.536 / 0.557 |
| Mode-level oracle vs shuffled null (F1) | 0.750 vs **0.777** (Qwen3); 0.776 vs 0.773 (phi-4) |
| Per-pair oracle vs always-think | 0.250 vs 0.220 |
| Think on 25 / 50 / 75% of pairs at random | 0.088 / 0.134 / 0.179 |

**Verdict: dead end.** The decision is unlearnable, the mode oracle is again at
shuffle level, and a perfect router would gain +0.03 over simply always
thinking. Spending interpolates linearly, so there is no routing bargain to
find. Reasoning should be spent uniformly.

---

### E8 — Does the inflated oracle hold on the routing literature's own benchmark?

If E2 is a property of routing evaluation rather than of vulnerability data, it
should reproduce on **RouterBench** (Hu et al., 2024) — 36,497 prompts × 11
models with per-model correctness and dollar cost, the dataset whose oracle and
AIQ are the field's standard reporting objects.

**Method.** Same null test: shuffle each model's per-prompt score within each
benchmark, which preserves every model's accuracy on every benchmark and
destroys per-prompt skill. Then (a) the oracle, (b) the cost-quality frontier
`argmax_m (score - lambda*cost)` swept over lambda, and (c) AIQ, the normalised
area under that frontier. Scripts: `nullcheck/rb_null.py`, `rb_frontier.py`.

**Result.**

| | value |
|---|---|
| best single model (gpt-4-1106-preview) | 0.7814 |
| oracle, best model per prompt | 0.9121 |
| **shuffle-null oracle** | **0.9440 ± 0.0002** |
| real complementarity (oracle − null) | **−0.032** |
| AIQ real | 0.9099 |
| **AIQ null** | **0.9424 ± 0.0002** |

Holds in **85 of 88 benchmarks** (the 3 exceptions are ties at 1.000 or n=15).
The null frontier dominates the real frontier at every cost level.

Robustness (`nullcheck/rb_robust.py`), both RouterBench splits, 200 bootstrap
draws over prompts:

| split | n | best single | oracle | null | gap (95% CI) | mean pairwise error correlation |
|---|---|---|---|---|---|---|
| 0-shot | 36,497 | 0.7814 | 0.9121 | 0.9440 | **−0.032** [−0.034, −0.030] | +0.238 |
| 5-shot | 36,483 | 0.8048 | 0.9184 | 0.9449 | **−0.027** [−0.028, −0.025] | +0.293 |

The error correlation is the mechanism: correctness across the 11 models is
positively correlated (+0.24 to +0.29), so hard prompts are hard for everyone
and the oracle has less to exploit than independence would give.

**Verdict: real effect, and it generalises.** The oracle sits *below* what
independent errors with identical per-model accuracies would produce, i.e. the
11 models fail on the *same* prompts — they are less complementary than chance.
So oracle headroom and AIQ are functions of marginal accuracy and price
dispersion, not evidence of complementary skill a router could learn. This is
not specific to security data.

---

### E9 — What do routers actually learn? (mechanism)

**Method.** Label = "does the cheap model (mixtral-8x7b) already get this prompt
right". Four routers on RouterBench, 5-fold: **A** prompt text (TF-IDF char
3-5-grams + logistic regression, what the papers fit); **B** benchmark identity
one-hot, no text at all; **C** router A scored *within* each benchmark, so task
identity is held constant; **D** the benchmark's base rate, no learning
whatsoever. Plus A under leave-one-benchmark-out. Scripts:
`nullcheck/rb_predict.py`, `rb_mechanism.py`.

**Result.**

| Router | AUC |
|---|---|
| A prompt text, random split | 0.705 ± 0.005 |
| B benchmark identity only (one-hot) | 0.691 ± 0.007 |
| **D task base rate, zero learning** | **0.694** |
| C prompt text, within benchmark | 0.575 ± 0.119 |
| A prompt text, leave-one-benchmark-out | 0.607 ± 0.075 |
| benchmark identity recoverable from prompt text | 88.3% over 86 classes |

**Verdict: real effect — routers are task classifiers.** Knowing nothing but
which benchmark a prompt came from scores 0.694 against the text router's 0.705,
so ~99% of the apparent skill is task identification. Per-prompt skill with the
task held fixed is 0.575, and it degrades to 0.607 on an unseen task family.

This closes the loop with E1/E2/E7: vulnerability detection is a **single-domain**
task, so there is no task-family signal to identify, and routing collapses to
chance exactly as this mechanism predicts. The earlier within-project AUC 0.95 vs
leave-one-project-out 0.49 gap in `07-hybrid-llm.md` is the same phenomenon with
"project" in place of "benchmark".

---

### E8b — Literature check: the diagnostic is already published

Run before writing E8 up as a contribution.

**Found, and it scoops E8 directly.** **RouteGuard** (arXiv 2608.07583, Aug 2026)
defines *excess over independence* as
`E = A* - (1 - prod_j(1 - p_j))` — the same closed form validated above — proves
`E <= 0` follows generically from shared pretraining, and measures it on
RouterBench's 36,497 prompts × 11 models plus an incident-triage pool, reporting
the same redundancy. **"Opportunity Is Not Realizability"** (arXiv 2608.08265)
certifies with selection-valid intervals that deployable routers recover only
7.5-14.4% of a 9.7-30.7 point oracle gap. **"How Much of the Routing Gap Is
Real?"** (arXiv 2607.03436) splits the router-to-oracle gap into reproducible
specialist advantage and single-draw label noise.

Adjacent, on the positive lever: **"Learning When to Think"** (arXiv 2608.20256)
learns NoThink/Short/Long modes inside GRPO and cuts mean response length 4,796
-> 2,811 tokens at held-out MATH500 accuracy — i.e. per-instance budget routing
**works on maths**, which is the direct contrast to E7 failing here.
**Triage** (arXiv 2604.07494) routes software-engineering tasks to model tiers by
code-health signals, but not vulnerability triage.

**Verdict: E8/E9 are a reproduction, not a contribution.** Our numbers match
theirs independently, which validates the measurement code, and they are the
right citations for the diagnostic. The contribution has to move to the parts
the literature has not covered — see below.

---

### E10 — The thinking-budget frontier (GPU run, gpu0)

**Method.** Qwen3-8B, 299 PrimeVul pairs (597 functions), s1-style budget
forcing: cap the reasoning at B tokens, and if the trace overruns, re-prompt with
the partial reasoning plus `</think>\n\nFinal answer` so the model must commit.
Cost is charged as the sum of both stages, so a truncated unit pays for what it
used. Budget 0 is Qwen3's own no-think form. Raw `/v1/completions` with
hand-written chat markers, because the chat endpoint gives no control over where
thinking is cut. Scripts: `nullcheck/probe2.py`, `chain2.sh`, `analyze_budget.py`.
GPU 10:20-12:08, released cleanly.

**Result.**

| budget | out tokens | pair-both | accuracy | AUC | yes-rate | forced to commit |
|---|---|---|---|---|---|---|
| 0 | 18 | 0.047 | 0.504 | 0.506 | 0.72 | — |
| 256 | 273 | 0.091 | 0.504 | 0.514 | 0.73 | 100% |
| 512 | 528 | 0.124 | 0.514 | 0.534 | 0.73 | 99% |
| 1024 | 1,022 | 0.134 | 0.513 | 0.525 | 0.74 | 92% |
| 2048 | 1,775 | 0.134 | 0.513 | 0.512 | 0.68 | 56% |
| 4096 | 2,319 | 0.208 | 0.524 | 0.535 | 0.53 | 9% |

Per-unit budget routing: oracle cheapest-correct budget reaches accuracy 0.831
at 806 tokens, but the shuffle null reaches **0.950** — headroom +0.307 real
against +0.425 null, so budget choice is no more routable than model choice (E7).

**Verdict: no usable frontier.** `pair-both-right` rises monotonically with
budget, but accuracy moves 0.504 -> 0.524 and AUC stays in [0.51, 0.54]. E12
shows the `pair-both-right` rise is the yes-rate falling, not skill appearing.
Also note thresholding the cheap mode's own confidence does *not* reproduce the
rise (`nullcheck/calib_vs_reason.py`: budget 0 raw 0.047 vs thresholded 0.044),
so the change is not merely a decision threshold — it is the model becoming less
sticky, which E12 quantifies.

---

### E11 — Self-consistency: is depth or breadth the better spend?

**Method.** 5 independent reasoning samples at budget 1024 (temperature 0.6,
different seeds), majority vote, on the same 299 pairs. Also tests whether vote
disagreement is the confidence signal that token log-probs failed to provide.
Script: `nullcheck/probe2.py --mode sc:5@1024`, `analyze_budget.py`.

**Result.**

| Configuration | out tokens | pair-both | accuracy |
|---|---|---|---|
| 5 samples @1024, majority vote | 5,103 | 0.117 | 0.524 |
| 1 sample @1024 | 1,022 | 0.134 | 0.513 |
| 1 sample @4096 | 2,319 | 0.208 | 0.524 |

Vote fraction as a bug score: AUC 0.539. Vote agreement predicting its own
correctness: AUC 0.522. Unanimous units (55% of them) are right 54.0% of the
time against 52.4% overall.

**Verdict: dead end, and breadth is strictly worse than depth.** Five samples
cost 2.2× one long trace and score below it. Vote agreement is not a usable
confidence signal, so the one remaining hope for a cost-aware escalation
trigger is gone.

---

### E12 — The control that overturns E6: same-yes-rate random guessing

`pair-both-right` counts pairs where the buggy version is called vulnerable and
the fixed one is not. It therefore rewards *disagreeing within a pair*, and a
guesser that answers "vulnerable" independently with probability `r` scores
`r*(1-r)` — 0.25 at `r = 0.5`. E6 never computed that baseline.

**Method.** For every configuration: the analytic baseline `r(1-r)` at its own
observed yes-rate, plus a permutation null (shuffle verdicts across units, which
preserves the rate and destroys within-pair structure, 300 draws). Also
within-pair agreement — how often the model gives the *same* verdict to both
versions — against the `r^2 + (1-r)^2` expected under independence. Script:
`nullcheck/pair_null.py`.

**Result.**

| Configuration | yes-rate | pair-both | same-rate null | excess | within-pair agreement vs independent |
|---|---|---|---|---|---|
| Qwen3 no-think | 0.72 | 0.047 | 0.202 | **−0.155** | 0.91 vs 0.60 |
| Qwen3 budget 256 | 0.73 | 0.091 | 0.196 | −0.106 | 0.83 vs 0.60 |
| Qwen3 budget 512 | 0.73 | 0.124 | 0.200 | −0.075 | 0.78 vs 0.60 |
| Qwen3 budget 1024 | 0.74 | 0.134 | 0.197 | −0.063 | 0.76 vs 0.62 |
| Qwen3 budget 2048 | 0.68 | 0.134 | 0.218 | −0.084 | 0.76 vs 0.56 |
| Qwen3 budget 4096 | 0.53 | 0.208 | 0.251 | −0.043 | 0.63 vs 0.50 |
| Qwen3 think, unbounded | 0.49 | 0.221 | 0.252 | **−0.032** | 0.62 vs 0.50 |
| phi-4 direct | 0.82 | 0.027 | 0.144 | −0.117 | 0.95 vs 0.71 |
| phi-4 CoT | 0.57 | 0.217 | 0.243 | −0.026 | 0.67 vs 0.51 |
| self-consistency k=5 | 0.76 | 0.117 | 0.180 | −0.063 | 0.81 vs 0.64 |
| granite-8b | 0.81 | 0.029 | 0.155 | −0.127 | 0.96 vs 0.69 |
| phi-3.8b | 0.55 | 0.065 | 0.249 | **−0.184** | 0.89 vs 0.51 |
| qwen-7b | 0.02 | 0.008 | 0.024 | −0.017 | 0.98 vs 0.95 |
| qwen-1.5b | 0.09 | 0.003 | 0.078 | −0.075 | 0.98 vs 0.84 |
| deepseek-6.7b | 0.00 | 0.000 | 0.001 | −0.001 | 1.00 vs 1.00 |

**Verdict: real effect, and it is the project's central finding.** Fifteen
configurations — five models, two families of reasoning mode, six budgets,
self-consistency — and **every one is below a coin flip at its own yes-rate**.
Reasoning monotonically shrinks the deficit (−0.155 -> −0.032) and never crosses
zero.

The mechanism is the agreement column: models answer the buggy and fixed versions
*identically* far more often than independence allows (0.91 vs 0.60 with no
reasoning; 0.62 vs 0.50 with the longest). They respond to the function's
surface form, not to the defect; the median patch that separates the two versions
is 73 characters (E4) and it does not move the verdict. Reasoning reduces this
stickiness — which is what the `pair-both-right` rise in E6/E10 actually
measured — without producing discrimination.

This supersedes E6's withdrawn claim, and it explains E1 (no AUC signal), E7 and
E10's null-level oracles: there is no per-unit competence anywhere in this pool
to route, at any reasoning budget.

**Corrected 2026-09-23 (E18), two places, no verdict changed.**

1. *The yes-rate for the five pool-model rows was measured on the wrong set.*
   `table_dedup.jsonl` holds 5,820 units but only 1,814 sit in complete pairs —
   the rest carry no `pair_id` and never enter `pair-both-right`, so the baseline
   described a population the metric does not score. On the paired units the rows
   become: granite-8b **−0.077** (was −0.127), qwen-7b **−0.032** (was −0.017),
   qwen-1.5b **−0.106** (was −0.075), phi-3.8b −0.185 (unchanged), deepseek-6.7b
   0.000. All still negative.
2. *The table reports point estimates only.* With a 2,000-resample bootstrap over
   pairs (baseline recomputed at each resample's own yes-rate), the
   non-reasoning rows stay clearly below zero — Qwen3 no-think −0.155
   [−0.183, −0.124] — while every **reasoning** row straddles it: Qwen3 think
   −0.029 [−0.076, +0.016], phi-4 CoT −0.028 [−0.071, +0.020], budget 4096
   −0.041 [−0.088, +0.006], and the E18 32B think −0.018 [−0.060, +0.025].

So "every configuration is below a coin flip" holds for every point estimate, but
the defensible statement is **"no configuration is above one, and the reasoning
ones sit on the line"**. That is what E18 was run to test, and it is why the
distinction matters.

---

### E12b — Literature check on E12

**Mostly covered.** PrimeVul's own paper ("Vulnerability Detection with Code
Language Models: How Far Are We?", arXiv 2403.18624) already reports that GPT-4
does not beat random guessing on the paired split, and already breaks pairs into
four outcomes — pairwise-correct, pairwise-vulnerable, pairwise-benign,
pairwise-reversed — which is the within-pair agreement measure in another form.
**"Words Speak Louder Than Code"** (arXiv 2606.30587) goes further on exactly our
mechanism: PrimeVul pairs, within-pair agreement, surface/linguistic framing
effects, **and reasoning budgets / thinking modes**, on much stronger models
(Claude 4.6, Gemini 2.5, GPT, Llama 4, DeepSeek V3, Qwen Coder 3).

**What is left of E12 as a contribution:** the explicit `r(1-r)` null with a
permutation test (they report the raw pair categories, not a same-rate baseline),
and the budget sweep showing the deficit closing monotonically without crossing
zero. That is a tightening of a known result, not a new one.

**Not found in the literature, after three searches:** any critique of IRIS's
per-project AvgF1 with a random-k control (E3), or any application of routing
diagnostics to static-analysis alert triage. Nearest neighbours are JavaVulBench
(arXiv 2607.02825, leakage-aware Java benchmark) and QLCoder (arXiv 2511.08462,
query synthesis), neither of which touches the metric. **E3 is the one clearly
unoccupied result this session produced.**

---

### E13 — Precision-first: can any filter over this pool raise precision?

Asked because the field's live concern is precision, not cost. On IRIS the
deployment question is: of the alerts kept, what fraction are real (base rate
15.2%)? The honest null is **not** "no filter" but "keep the same NUMBER of
alerts per project at random", since per-project precision rises by chance alone
at small keep-rates (E3). Scripts: `nullcheck/precision_first.py`,
`precision_honest.py`, `precision_twostage.py`.

**Result, thresholding and voting** (matched-count random baseline alongside):

| filter | kept | precision | matched random | excess |
|---|---|---|---|---|
| no filter | 100% | 0.152 | — | — |
| best single verdict (phi-3.8b) | 67.6% | 0.161 | 0.152 | +0.009 |
| **unanimous, all 5 say vuln** | 5.6% | **0.110** | 0.151 | **−0.041** |
| >=4 of 5 say vuln | 19.5% | 0.120 | 0.152 | −0.031 |
| majority, >=3 of 5 | 61.1% | 0.160 | 0.150 | +0.010 |
| top 10%/project by qwen-7b score | 10.1% | 0.220 | 0.156 | +0.064 (z=3.1) |
| top 25%/project by qwen-7b score | 25.1% | 0.187 | 0.154 | +0.033 (z=2.8) |

**Consensus voting is worse than random** — the first thing most people try. Only
*ranking within a project by the score* looked promising, so it was tested for the
three ways it could be an artifact:

| check | result |
|---|---|
| leave-one-project-out selection of the ranker | precision **0.119**, below the 0.152 base rate |
| per-project, qwen-7b top 10% | beats its own project base rate in **2 of 10** |
| bootstrap over the 14 projects | +0.040, 95% CI **[−0.084, +0.124]** |

**Two-stage** (rank top 25% by one model, confirm with a second, pair chosen
leave-one-project-out) pools to precision **0.522** on 46 kept against a 0.152
base rate — and it is **Simpson's paradox**: 19 of its 24 true positives come from
ESAPI, a project whose own base rate is 0.50. Per project it beats its own base
rate in 2 of 9, mean per-project excess over matched random is **−0.020**, recall
is 0.070, and the bootstrap CI is [−0.098, +0.292].

**Verdict: dead end.** Precision cannot be bought by re-weighting verdicts from
models that do not discriminate — filtering, voting, ranking and two-stage
confirmation all reshuffle the same 15% base rate. Two traps are worth carrying
into any future precision table: a *matched-count* random baseline (not "no
filter"), and a *per-project* breakdown (pooled precision on projects with very
different base rates is Simpson's paradox waiting to happen).

**Where precision could come from instead.** Not from the model pool but from
*verification*, and the evidence is in this repo's own LLMDFA reproduction:
precision there tracked the Z3 path-feasibility stage, not the model — XSS 97.5%
and OSCI 84.0% held, while DBZ fell to 45.4% against the paper's 73.8%, localised
to Z3 falling back to LLM judgement 17.7% of the time versus the paper's 0.61%.
That suggests routing alerts to *verification effort* (symbolic feasibility, then
test/PoV generation, LLM reasoning last) with precision as the objective and cost
as the constraint — a decision made on a checkable signal rather than on model
confidence, which is exactly why E7's routing had nothing to learn from. **Not
started; needs the user's go-ahead to use LLMDFA data inside this project.**

---

### E14 — Does the LLMDFA per-case routing rule transfer to IRIS?

Asked because LLMDFA is the one place in this project where routing *works*:
`LLMDFA/reproduction/ROUTER_V2_PERCASE.md` reports **+0.060 F1 over the best
fixed model at 6% lower cost** on 111 held-out source/sink forms, from a rule
with no LLM in the decision — `n_if >= 12 -> small model (Phi-4-mini), else
large (Mistral-Nemo)`. Its §8 leaves the transfer question open: "whether the
same shape recurs in IRIS and RepoAudit. One system is a curiosity; three would
be a finding."

**Method.** No GPU. `data/gr/slices.jsonl` already carries `n_if` per path from
`b3_slice.py`; `data/gr/probe5/` has four reproducible models' verdicts on the
same path ids (deepseek-6.7b excluded, kappa −0.029 between runs). Split the
2,257 paths into `n_if` quartiles and ask, per band, whether a model's kept set
beats the band base rate — against a **matched-count** permutation null (same
number kept, drawn at random, 2,000 draws), then break the good cells down **per
project**. Script: `nullcheck/llmdfa_transfer.py`.

**Result.** Precision excess over the band base rate, by branchiness:

| n_if band | n | pos% | qwen-1.5b | phi-3.8b | qwen-7b | granite-8b |
|---|---|---|---|---|---|---|
| 0–3 | 624 | 7.4 | **+0.250** (z 8.3) | −0.016 | **+0.145** (z 5.6) | −0.004 |
| 4–7 | 511 | 17.6 | −0.088 | +0.034 | −0.055 | −0.026 |
| 8–12 | 575 | 15.5 | +0.034 | −0.009 | −0.047 | −0.013 |
| 13+ | 547 | 21.8 | +0.006 | +0.034 | −0.069 | −0.012 |

The LLMDFA shape requires a model whose edge **rises** with branchiness. No
model does that: phi-3.8b oscillates (−0.016, +0.034, −0.009, +0.034) rather
than trending, and the two models with large z-scores have them on the
*simplest* quartile and decay from there — the opposite direction. granite-8b
keeps 95–98% of everything, so it has no room for a precision excess at all.

The two significant cells are the **same artifact E13 documented**. Of the 68
simple-band paths qwen-1.5b keeps, 25 are ESAPI (base rate 0.46, 22 hits) and 36
are ff4j (base rate 0.04, **0** hits). Pooled precision 0.324 against a 0.074
band base rate is entirely "it kept ESAPI paths, and half of ESAPI is positive".
Per project it beats its own base rate in 1 of 2; phi-3.8b in 1 of 9,
granite-8b in 1 of 9.

**Verdict: dead end — the rule does not transfer.** This answers LLMDFA §8 for
IRIS: the shape is one system's, not a law. The reason is the E9 mechanism with
"project" in place of "benchmark" — IRIS paths inside a project share a CWE, a
codebase and a base rate, so a structural feature is largely a project
identifier, and conditioning on it reproduces the project's base rate rather
than any per-path competence. LLMDFA differs in the way that matters: its cases
are independent synthesised programs with a **verifier** (Z3) downstream, so
`n_if` is a property of the case rather than a proxy for which repository it
came from.

**What this does not touch.** The LLMDFA result itself. Nothing here tests it —
it is a different pipeline, unit and metric, and its held-out gain stands as
reported. What it does say is that the way to build on it is to go *deeper into
LLMDFA*, not to port it here.

---

### E15 — The LLMDFA router through this project's null battery

E14 showed the LLMDFA rule does not transfer here. The complementary question is
whether it survives *where it was measured*, since `ROUTER_V2_PERCASE.md`
predates the reporting rules adopted this session and claims both a +0.060
held-out gain and +0.092 of unclaimed oracle headroom.

**Method.** No GPU; replays the existing probe logs. Four nulls on each of the
three held-out probes (off815 4-model, off1481, fam2): matched-count random
routing (same number of cases to the small model, chosen at random, 2,000
draws), bootstrap CI over cases, per-form breakdown, and the shuffled-oracle
null from E2/E8. Script: `LLMDFA/scripts/null_router.py`; results written up as
§9 of `LLMDFA/reproduction/ROUTER_V2_PERCASE.md`.

**Result.**

| probe | router | best fixed | vs best fixed (95% CI) | vs deployable fixed choice | matched-count random | forms won |
|---|---|---|---|---|---|---|
| off815 | 0.699 | 0.639 | +0.060 [+0.009, +0.100] | **+0.124** [+0.081, +0.173] | 0.615 ± 0.015, z **+5.5** | 3/3 |
| off1481 | 0.691 | 0.649 | +0.042 [−0.001, +0.088] | **+0.097** [+0.053, +0.142] | 0.631 ± 0.014, z **+4.4** | 3/3 |
| fam2 | 0.714 | 0.698 | +0.015 [−0.021, +0.050] | **+0.074** [+0.023, +0.130] | 0.681 ± 0.015, z **+2.3** | 2/3 |

Oracle against its shuffled null: 0.805 vs **0.940**, 0.754 vs 0.825, 0.791 vs
0.856 — negative excess in all three, the same sign as E2 and E8.

**Verdict: real effect — the first routing result in this project that passes.**
Two halves, and they separate cleanly:

* **The rule is real.** It beats an arbitrary split of the same size in 3 of 3
  held-out conditions and the reversed rule loses every time, so the sign
  carries information. The honest margin is the **deployable** one (+0.074 to
  +0.124, all CIs excluding zero) rather than the +0.060 over a fixed model that
  could only be picked with hindsight — and against the best-on-held-out-data
  model, only off815 is significant at n=111.
* **The oracle is not.** The shuffled oracle beats the real one everywhere, so
  the "64% of headroom still unclaimed" in that document's §8 is not claimable
  by any router. Recorded there; it removes a GPU-shaped open question.

**Why this one works when E2/E7/E10/E13/E14 did not.** LLMDFA cases are
independent synthesised programs with a **verifier** (Z3) downstream, so `n_if`
is a property of the case rather than a proxy for which repository it came from,
and the routing decision is made on a checkable code property rather than on
model confidence. That is the distinction E13 pointed at and E14 confirmed from
the other side. It is also consistent with E9: routing needs heterogeneity the
decision can see, and single-domain per-unit confidence is not it.

---

### E16 — The LLMDFA threshold: absolute vs relative cut

The live weakness §8 of `ROUTER_V2_PERCASE.md` names: the rule's direction
transfers to every held-out condition but the cut point `n_if >= 12` does not.

**Method.** No GPU. Replace the fixed cut with "route the branchiest *q* of
whatever corpus you are given", *q* taken from the fitted `float_*` corpus only
(0.409), so the relative rule is as blind to the held-out data as the absolute
one. Ties on `n_if` route together. Matched-count random null and a paired
bootstrap on each probe. Script: `LLMDFA/scripts/relative_threshold.py`;
write-up as §10 of that document.

**Result.** The fitted corpus and family 2 have different shapes — median `n_if`
9 vs 3 — while off815 matches the fitted distribution almost exactly (fraction
≥12 of 0.405 vs 0.409).

| probe | absolute (12) | relative (q=0.409) | difference | 95% CI | matched-count z |
|---|---|---|---|---|---|
| off815 | 0.699 | 0.699 | +0.000 | — (identical 45 cases) | +5.5 → +5.3 |
| off1481 | 0.691 | 0.701 (cut 9) | +0.010 | [−0.011, +0.037] | +4.3 → **+5.0** |
| fam2 | 0.714 | 0.723 (cut 9) | +0.009 | [−0.002, +0.024] | +2.4 → **+3.0** |

**Verdict: real but small — adopt the relative form for transferability, not for
the F1.** The gain is under +0.010 and both intervals include zero, and off1481
and fam2 are the same 111 cases under two engines, so it is one piece of
evidence rather than two. What the table does establish is that the absolute
rule's clean run on off815 is a **distributional coincidence** — that family
happens to share the fitted branchiness — and that the relative form is never
worse while lifting the weakest matched-count z from +2.4 to +3.0. It recovers
part of the drift, not all: the probe-fitted optimum on family 2 is near `n_if
>= 6`, the relative rule reaches 9 without seeing the data, the absolute one
stays at 12.

---

### E17 — Does a router raise PRECISION, or only F1? And when?

The project's research question, asked directly. F1 can rise for the wrong
reason: trading recall away buys precision, so a precision number quoted without
its recall is not a claim about anything.

**Method.** No GPU. Every strategy reported with precision AND recall, compared
against (a) each fixed model, (b) the fixed model with the **closest recall**,
(c) a matched-count random split. Bootstrap on the precision gain, with the
recall change reported over the same resamples. Then the same question across
the three bug types of the fitted corpus. Script:
`LLMDFA/scripts/precision_effect.py`.

**Result — held-out probes.**

| probe | router prec / recall | nearest-recall fixed model | precision gain | 95% CI | recall change | vs random split |
|---|---|---|---|---|---|---|
| off815 | **56.7% / 91.0%** | Qwen 47.0% / **91.0%** | **+9.8 pts** | [+4.5, +14.5] | **−0.1%** | z **+6.0**, beaten by 0.0% |
| off1481 | 55.1% / 92.8% | Nemo 48.1% / 100% | +7.0 pts | [+2.1, +12.5] | −7.2% | z +5.1, beaten by 0.0% |
| fam2 | 60.8% / 86.5% | Nemo 53.9% / 99.1% | +6.8 pts | [+2.7, +11.3] | −12.6% | z +3.5, beaten by 0.1% |

**off815 is the clean one: +9.8 points of precision at exactly equal recall**
(91.0% both), CI excluding zero, and −0.1% mean recall change across 2,000
resamples. That is not precision bought with recall. On family 2 the gain is
real but partly paid for in recall (−7 to −13 points), though the router still
dominates Phi outright there (60.8%/86.5% against 58.1%/71.2%).

**Result — by bug type (in-sample, and the more useful half).**

| bug type | precision spread across the 4 models | router vs matched-count random |
|---|---|---|
| XSS (666 cases) | **0.7 pts** (97.5–98.1%) | z **−0.4** — nothing |
| OSCI (444 cases) | **2.2 pts** (84.0–86.1%) | z **−0.7** — nothing |
| DBZ (286 cases) | **13.3 pts** (37.6–51.0%) | z **+6.7**, +12.9 pts over the nearest-recall model |

**Verdict: real effect, with a stated condition.** A router raises precision
only where the models **disperse** on the metric being routed for. On XSS and
OSCI every model sits within a couple of points of every other, there is nothing
to sell, and the router lands exactly on the random-split null. On DBZ they span
13 points and it does not.

This gives the project a pre-flight test that costs nothing and would have saved
most of E1–E14:

1. **Do the candidate models disperse on the target metric?** If the spread is
   ~1 point (XSS, OSCI), stop — no router can help.
2. **Is the dispersion predictable from a property of the case available before
   any LLM call?** Test against the shuffled-oracle null. If the real oracle is
   at or below the shuffle (IRIS, PrimeVul, RouterBench), stop.
3. Only if both hold is there a router worth building — and then report
   precision **at matched recall**, against a **matched-count random split**.

Under that test, IRIS and PrimeVul fail at step 2 and the XSS/OSCI corpora fail
at step 1. DBZ passes both, and is the one place in either project where routing
pays.

---

### E18 — Q5, the positive control: does scale rescue it? (GPU run, gpu0)

The control E12 needed. Every one of its 15 configurations is an 8-14B open
model, so the claim it supports is "models up to 14B fail", while the sentence a
reader takes away is "LLMs cannot do this". This closes that gap.

**Method.** Qwen3-32B on gpu0 (A100-SXM4-80GB, vLLM 0.27.1, bf16), no-think and
think, over the **same 599 units / 299 pairs** E6 and E10 used — same box, same
engine, same prompts, and `think_probe.py` byte-identical to the file that
produced E6 (md5 verified on both sides), so the only variable against the
measured Qwen3-8B rows is scale. Weights verified against HF's published SHA256
before launch. Scored with the E12 metric plus a bootstrap CI over pairs, and a
**paired** bootstrap against Qwen3-8B on the same pairs. Scripts:
`nullcheck/q5_chain.sh` (gpu0 driver), `nullcheck/q5_pair_null.py` (analysis).
GPU 12:41-13:47 IST, 66 min, released cleanly.

**Result.**

| configuration | yes-rate | pair-both | `r(1-r)` | excess | 95% CI | agreement vs independent | out tokens |
|---|---|---|---|---|---|---|---|
| Qwen3-32B no-think | 0.55 | 0.037 | 0.248 | **−0.211** | [−0.232, −0.188] | 0.91 vs 0.50 | 20 |
| Qwen3-32B think | 0.70 | 0.191 | 0.209 | **−0.018** | [−0.060, +0.025] | 0.70 vs 0.58 | 1,641 |
| Qwen3-8B think (E6) | 0.49 | 0.221 | 0.250 | −0.029 | [−0.076, +0.016] | 0.62 vs 0.50 | 2,418 |

Paired bootstrap on the same 299 pairs, 5,000 resamples:

| comparison | value |
|---|---|
| 32B think − 8B think, excess difference | **+0.011, 95% CI [−0.048, +0.070]** |
| resamples where 32B is ahead | 63.6% |
| P(excess > 0), 32B think | 20.7% |
| P(excess > 0), 8B think | 11.8% |

**Verdict: real effect — scale does not rescue it, and reasoning is the only
lever that moves the number.** Four times the parameters buys +0.011 with an
interval six times wider than the effect. Within the 32B, reasoning moves the
deficit −0.211 → −0.018, a swing ~17x larger than anything scale produced. The
stickiness mechanism persists: the 32B gives the same verdict to both halves of
a pair 70% of the time against 58% under independence.

The no-think row is the **worst in the whole table** (−0.211), and the reason is
instructive: at 8B the model was yes-biased (r = 0.72, baseline 0.202), while the
32B is nearly balanced (r = 0.55, baseline 0.248). Better calibration raises the
bar a same-rate guesser sets, and the 32B is no better at clearing it. Scale
changed the bias, not the blindness.

**What this does NOT establish.** One family, one jump. A frontier API reasoner
is still untested, so the claim is "no gain from scale up to 32B", not "scale can
never help". Think mode samples at temperature 0.6, so this is one draw — as was
the E6 row it is compared against.

---

### E19 — Q6: the IRIS filter stage under a metric that cannot be gamed

E3 left every claim about the IRIS filter resting on a metric E3 itself broke:
per-project AvgF1 is matched by keeping 10 random paths per project. So "the LLM
filter contributes nothing" was, as stated, a claim about the metric as much as
about the filter. This re-asks it under a fixed review budget.

**Method.** No GPU; replays `data/gr/probe5`. An analyst reviews **k alerts per
project** (k = 5, 10, 20), so spending is equal by construction and being stingy
buys nothing — a filter now has to be good at *ordering*. Two readings of each
model: **filter** (the deployed semantics — kept alerts reviewed first, random
order inside the kept set) and **rank** (the most generous — order every path by
the model's `p_vuln`, review the top k). Scored as `detected` (projects where the
analyst sees at least one real vulnerability — IRIS's own #Detected, made
budget-aware) and mean per-project recall, against a **matched-budget random**
null of 2,000 draws. Script: `nullcheck/q6_budget_recall.py`.

**Result — nothing beats random on detection, at any budget.**

| configuration | det@10 (of 10) | z | recall@10 | z |
|---|---|---|---|---|
| random, matched budget | 7.5 | — | 0.311 | — |
| qwen-1.5b filter | 4.4 | **−2.6** | 0.258 | −0.8 |
| phi-3.8b filter | 7.9 | +0.3 | 0.253 | −0.9 |
| qwen-7b filter | 6.4 | −1.0 | 0.286 | −0.4 |
| granite-8b filter | 7.5 | +0.0 | 0.305 | −0.1 |
| deepseek-6.7b filter | 8.0 | +0.4 | 0.315 | +0.1 |
| **qwen-7b rank** | 8.0 | +0.4 | **0.461** | **+2.3** |
| phi-3.8b rank | 4.0 | −2.9 | 0.131 | −2.8 |
| granite-8b rank | 6.0 | −1.3 | 0.315 | +0.1 |
| ORACLE (TPs first) | 10.0 | — | 0.743 | — |

The best filter row is +0.4 sd; several are significantly **worse** than random.
Ranking by model score is often actively harmful to detection (phi-3.8b −2.9,
qwen-1.5b rank −2.1 at k=10, granite-8b rank −3.0 at k=20) because the scores are
a project-level signal: a model ranks one repository's paths highly and starves
the others, which is E9's mechanism with "project" for "benchmark".

**The one positive row dies under this project's own controls.** `qwen-7b rank`
carries +2.2 / +2.3 / +1.6 on recall across the three budgets. Per project, k=10:

| project | paths | TPs | rank | random | excess |
|---|---|---|---|---|---|
| ff4j | 1413 | 181 | **0.000** | 0.007 | −0.007 |
| commons-text | 230 | 36 | 0.056 | 0.043 | +0.012 |
| ESAPI | 198 | 99 | 0.020 | 0.051 | −0.030 |
| plexus-utils | 152 | 8 | **0.000** | 0.068 | −0.068 |
| zt-zip | 71 | 7 | 0.286 | 0.143 | +0.143 |
| antisamy | 33 | 4 | 0.250 | 0.294 | −0.044 |
| plexus-archiver | 30 | 3 | 1.000 | 0.345 | +0.655 |
| cron-utils | 26 | 1 | 1.000 | 0.383 | +0.617 |
| workflow-cps | 13 | 4 | 1.000 | 0.766 | +0.234 |

Pooled it is +0.151, 95% CI [+0.008, +0.326], 5 of 10 projects — a coin flip.
Restricted to the **6 projects where the budget actually binds** (> 3k paths):
**+0.001, 2 of 6**. Every win is on a project small enough that k = 10 reviews a
third or more of it, where recall approaches 1 for any policy. On the two largest
projects the model finds **zero** true positives in its top 10 while random finds
some. Leave-one-project-out selection picks qwen-7b every time and beats matched
random in **5 of 10** held-out projects.

**Verdict: real effect — the result survives the metric change, and it is now
metric-proof.** The LLM filter stage carries no usable ordering signal at a fixed
review budget: no configuration beats a matched-budget coin flip on detection,
and the single apparent recall win is an artifact of small projects. This is the
mirror image of E13's Simpson's paradox — there pooled precision was inflated by
one large project, here mean per-project recall is inflated by several tiny ones.
**A third reporting rule follows: report any per-project mean alongside the
subset where the budget binds, or a metric can be satisfied by projects too small
to pose the problem.**

This is the row the reframed paper needs. The earlier statement — always-granite
0.207 against no-filter 0.212 and random-k 0.236 — was true but stood on a metric
E3 discredited. The claim now holds under a metric that cannot be gamed by
keeping few alerts, measured on IRIS's own alert stream.

---

### E20 — GraphRouter v2 and the first skilled model on IRIS (GPU run, gpu0, 2026-09-26)

Full write-up: [`10-graphrouter-v2.md`](10-graphrouter-v2.md).

* **Objective fix works as designed.** Security-utility edge target (rho = 10)
  instead of `argmax(effect)`: 0.198 → 0.218 on the old pool, but the same-mix
  random router scores 0.222. With a skilled model present, the router picks it.
* **Positive control found.** Qwen3-32B no-think: within-project AUC 0.868
  [0.845, 0.892], 0.810 at sink-method level, recall@10 0.678 vs random 0.304 and
  0.463 vs 0.098 where the budget binds. It passes the E19 control. Qwen3-8B no-think
  0.687 at 0.23 GPU-s. Think mode is worse (0.696 vs 0.835) at ~55x cost. This
  overturns E18 for IRIS: on paired PrimeVul the 32B failed, on IRIS it works.
* **Per-path routing still adds nothing.** Every routed configuration ties a
  same-mix random router on the fixed-budget metric. The skilled models fail on the
  same paths (both wrong 15.0% vs 10.6% independent; pair oracle 0.850 < shuffled 0.904).
  The one AvgF1 "win" (z +6.4) is one path in antisamy-2017, and antisamy-2016/2017
  are the same repo, so leave-one-project-out leaks between them.
* **What works is a cascade, not a learned router.** Qwen3-8B on every path, its
  top 30% per project to the 32B: all-32B quality (recall@10 0.677 vs 0.678) at 50%
  of the cost, z +4.5 over random escalation. On IRIS's table at k = 10: AvgF1 0.374,
  10/16, FDR 65.71, next to the paper's IRIS + GPT-4 (0.366, 10/16, 65.69). GraphRouter
  v3 (given the 8B's score) re-learns this rule and never beats it.
* **No second skilled family among local models:** Mistral-Nemo-12B 0.462,
  gpt-oss-20b 0.613, phi-4 0.610. All err on the same paths as the 32B.
* **Memorisation control (2026-09-27, D6).** Project names hidden (names used in <5
  of 79 repos renamed): 32B 0.868 → 0.795; strict (only everyday Java kept, no
  comments/strings) 0.719. Rerun noise ±0.002. Text4Shell does not drop at all
  (0.948 → 0.970), so the loss looks like helpful names, not recalled CVEs. The 8B
  falls to chance (0.544 / 0.511): its skill is names. Main claim stands: IRIS's AvgF1
  ranks deepseek-6.7b (AUC 0.489) above the 32B (0.795 with its project hidden).

### E21 — Findings consolidated, raw outputs tracked (2026-09-28)

* Plain-language summary of E20, the LLMDFA router, and the proposed
  "test the signal, then route" strategy: [`11-findings-summary.md`](11-findings-summary.md).
* Raw model outputs and router results now tracked in
  [`../results/2026-09-skilled-cascade/`](../results/2026-09-skilled-cascade/); they
  reproduce the E20 numbers with `d4_skilled_combos.py` and `d6_compare.py`.
* Recorded caveats: the 30% escalation share was chosen after seeing the sweep (every
  share beats random escalation; fix it leave-one-project-out before reporting), and
  the cascade *matches* the paper's GPT-4 AvgF1 (0.374 vs 0.366), it does not beat it.
* Scaled run paused: 57/96 new DBs built, spec inference and the OWASP probes not run.

### E22 — OWASP model ladder and per-unit routers (GPU run, gpu7, 2026-09-29)

Full write-up: [`12-owasp-ladder.md`](12-owasp-ladder.md). Raw outputs:
[`../results/2026-09-owasp-ladder/`](../results/2026-09-owasp-ladder/).

* 12 models on 1,956 OWASP Benchmark units (CodeQL alerts, 65% real), 70/30 split.
  Ladder: Qwen3-1.7B AUC 0.447 (chance) → Qwen3-8B 0.717 → Qwen3-32B 0.841 →
  gpt-oss-120b 0.907 (acc 0.902, 0.156 GPU-s). gpt-oss-20b nearly matches it (AUC 0.912,
  acc 0.881) at 0.061 GPU-s.
* Memorisation control: names hidden, AUC unchanged (gpt-oss-20b 0.912 → 0.914, 32B
  0.841 → 0.835, 8B 0.717 → 0.705).
* Routers beat the best single model on test: all 12 models, λ = 1, kNN 0.932 acc at
  0.058 GPU-s vs gpt-oss-120b 0.905 at 0.156 (reward gain +0.041 to +0.091); ladder only,
  category router 0.929 vs 0.905. All beat random same-mix routing (z 5.6-10.8).
* The gain is mostly category-level: at λ = 0, 0.905 → 0.922 from answering "always yes"
  in high-base-rate categories (no model), → 0.934 from a better model per category;
  per-unit kNN adds ~0 over the category router at λ = 0 and +0.02-0.03 at λ ≥ 1.
* Does not transfer: leave-one-category-out, every router loses to the best single
  model (λ = 0: 0.901 vs kNN 0.847). Confidence cascades never beat the best model.
* Run defects fixed: gpt-oss-20b straggler run (cost from clean rerun); DeepSeek
  tokenizer (`PreTrainedTokenizerFast`), 53% unreadable verdicts before the fix.

### E23 — Our own Jev-style router, trained on the saved outcomes (gpu0, 2026-09-30)

Write-up: [`12-owasp-ladder.md`](12-owasp-ladder.md) section 7. Code `ladder/l4`, `l5`,
`jevlike_route.py`; outputs `../results/2026-09-owasp-ladder/learned/`.

* Code in → calibrated P(model right) for all 12 models → pick argmax P − λ·GPU-s.
  UniXcoder fine-tune: per-model AUC 0.93, calibration error 0.034.
* Test: 0.971–0.978 accuracy (3 seeds) vs best single 0.905, at lower cost; λ = 1:
  0.96–0.97 at 0.025–0.033 GPU-s.
* **Control:** the same UniXcoder trained on the answer key alone scores 0.997 on test.
  In-distribution, the router's gain is a small model that has learned OWASP.
* **New category (LOCO, 3 seeds):** router 0.919 / 0.944 / 0.923 > best LLM 0.901 (gains
  +0.019 to +0.043, every CI above 0) > direct classifier 0.875. First router here that
  transfers to an unseen category. Seeds 1-2 ran on gpu7 after gpu0 was withdrawn.

### E24 — Train on OWASP, route on IRIS: does not transfer (gpu7, 2026-09-30)

Write-up: [`12-owasp-ladder.md`](12-owasp-ladder.md) section 8. Outputs
`../results/2026-09-iris-cross/` (12 models × 2,257 IRIS paths, same settings as OWASP;
routers in `learned/`; `eval.txt`).

* Model ranking flips: gpt-oss-120b AUC 0.907 on OWASP → 0.605 on IRIS; Qwen3-32B is the
  only strong IRIS model (AUC 0.855, TPR − FPR 0.311).
* Every router fitted on OWASP loses on IRIS to the OWASP-chosen single model (gpt-oss-120b,
  acc 0.767): emb-lr 0.605 (−0.162, CI −0.178 to −0.146), fine-tuned 0.37-0.59 (−0.18 to
  −0.40), and mostly below random same-mix routing (z −6.1 to +0.7).
* Cause: base-rate shift (65% real → 15%). The router learned "Granite (says yes 94%) is
  right" on OWASP and sends ~60% of IRIS paths to Granite, which is right on 15% there.
  The direct classifier swings 0.15-0.83 across seeds for the same reason.
* Run notes: all 12 runs clean (no errors, no hung requests; DeepSeek 10 unparsed). Run
  paused 75 min when another user's job took gpu7 in a gap between models.

### E25 — Free zero-shot Jev stand-in: acts as "always Qwen3-8B" (gpu7, 2026-09-30)

Write-up: [`12-owasp-ladder.md`](12-owasp-ladder.md) section 9. Outputs
`../results/2026-09-jev-standin/`.

* Qwen3-8B plays Jev: code + one description per ladder tier → probability per tier.
  Generic and OWASP-informed descriptions; 4 runs, 8,426 decisions, 5 GPU-min.
* Picks the 8B for 74-98% of IRIS paths, the 120B never. IRIS: acc 0.670 / 0.695, AUC
  0.767 / 0.775, recall@10 0.553 / 0.600 — ≈ the 8B alone (0.695, 0.784, 0.603), no better
  than random same-mix, below the 8B→32B cascade (0.755, 0.830, 0.691 at 0.059 GPU-s).
* OWASP test: ≈ 8B again (0.670 / 0.701) vs gpt-oss-120b 0.905.
* Reading: zero-shot "how hard does this look" does not identify the right model; on IRIS
  the confidence cascade remains the only routing that helps.

### E26 — Cascade router without hindsight (CPU, 2026-10-01)

Write-up: [`13-cascade-router.md`](13-cascade-router.md). Output
`../results/2026-10-cascade/cascade.txt`, script `ladder/l9_cascade.py`.

* a) f chosen leave-one-repo-out (8B → 32B, IRIS): f = 0.30 in 12/13 folds; recall@10
  0.648 at 0.059 GPU-s vs 8B 0.603 / 32B 0.690 (0.115), random escalation 0.570 (z +2.0).
  **Corrects E20:** "matches the 32B" (0.677 vs 0.678) relied on a hindsight f.
* b) Pair chosen from k target labels (within-project AUC gate): picks 8B → 32B in 4% /
  15% / 13% / 25% / 45% of draws at k = 25 / 50 / 100 / 200 / 400; recall@10 0.45 → 0.66 vs
  known pair 0.67-0.73; the pair chosen on OWASP (1.5B → 120B) is worst (0.44-0.50).
  Pooled-across-projects AUC gate was wrong (Simpson) and was replaced.
* c) OWASP: calibrated cascade 0.872-0.880 acc at 0.12-0.15 GPU-s, no better than
  gpt-oss-120b (0.905, 0.156); gpt-oss-20b alone 0.869 at 0.061 is the sensible choice.

### E27 — IRIS re-scored per vulnerability (CPU, 2026-10-02)

Write-up: [`14-per-cve-and-stages.md`](14-per-cve-and-stages.md) sections 1-2. Output
`../results/2026-10-percve/percve.txt`, script `ladder/l10_percve.py`.

* IRIS-16 = 9 CVEs with both real and false paths (one CVE gives up to 181 real paths).
  Random order finds 6.6 of 9 in the top 10; Qwen3-8B 8, Qwen3-32B 9, cascades 9.
* Only LLM vs random separates (8 of 9 CVEs); 8B vs 32B vs cascades all overlap.
* Per CVE, gpt-oss-20b ≈ Qwen3-32B (share 0.059 vs 0.052) though per path it looked far worse.
* New label-free **stop rule** cascade: 20% sent up, 0.048 GPU-s, 32B-level per CVE.
* **Corrects E20/doc 13:** renaming identifiers does not hurt per CVE (renamed 8B finds
  9 of 9); the AUC/recall@10 drop was a path-level effect of ff4j/ESAPI.

### E28 — Scaled IRIS: which stage gets the big model (gpu7, started 2026-10-02)

Write-up: [`14-per-cve-and-stages.md`](14-per-cve-and-stages.md) sections 3 and 5. Output
`../results/2026-10-scaled-iris/scaled.txt`. 80 projects; spec stage with Qwen3-8B and
Qwen3-32B (deepseek-coder-7b crashed past its 4k context, not run); filter stage with Qwen3-8B,
Qwen3-32B and the stop rule on every spec run. Scripts `ladder/run_scaled_iris.sh`,
`patch_iris_scaled.py`, `iris_gpt_scaled.py`, `run_window2.sh`, `filter_dedup.py`, analysis
`ladder/l11_scaled.py`. GPU: window 1 2026-10-02 17:49-20:27 IST, window 2 2026-10-03 11:08-14:59 IST.

* Spec: 8B detects 52 of 80 CVEs (81,596 paths), 32B 47 (55,720 paths). 8B spec costs 2,970
  GPU-s vs 5,740, but 12.6 h vs 7.6 h of CodeQL and 4 projects CodeQL never finishes.
* **Best pipeline: 32B spec + 8B filter, 28 of 80 CVEs found within 10 paths per project, 7,189
  GPU-s.** Reverse allocation (8B spec + 32B filter): 20 CVEs, 10,412 GPU-s; paired +8.0
  [+1.0, +15.0]. All-32B 23 (12,347 GPU-s); all-8B 24 (4,602 GPU-s, −3.7 [−10.7, +3.0]).
* The 32B filter has within-project AUC 0.70-0.75 but finds the first real path later than
  random order (share 0.18-0.19 vs 0.14-0.15): confident false alarms at the top. The IRIS-16
  filter-stage conclusions (E20, E26, E27) do not carry over at this scale.

---

### Where that leaves the project

Two independent walls, both measured:

* **No routable competence.** Per-unit routing has null-level headroom across
  models (E2), across reasoning modes (E7) and across reasoning budgets (E10).
  The diagnostic that shows why is already published (E8b), so it is a citation,
  not a contribution.
* **No competence at all.** Fifteen configurations are below a same-yes-rate coin
  flip on paired PrimeVul (E12), with the deficit explained by within-pair
  stickiness. Reasoning shrinks the deficit and never crosses zero.

So "cost-aware routing over this pool on these benchmarks" has no positive result
available, and no amount of GPU time will produce one. What *is* available is a
measurement paper the security ML community needs, made of four things this
session established and the literature does not cover:

1. **Paired evaluation with the right null.** `pair-both-right` has a yes-rate
   baseline `r(1-r)` that nobody reports; against it, every model and every
   reasoning budget is negative (E12). The companion diagnostic — within-pair
   agreement against `r^2+(1-r)^2` — localises the failure to surface-form
   reactivity rather than miscalibration, and `calib_vs_reason.py` rules out a
   decision threshold as the explanation.
2. **Two broken benchmark metrics, with the controls that expose them.** IRIS's
   per-project AvgF1 is matched by keeping 10 random paths per project (E3:
   0.236 random vs 0.244 for the published router). PrimeVul has a length
   shortcut: length alone (AUC 0.734) beats all five LLMs stacked (0.713), and the
   fixed version is the longer one in 83.8% of pairs (E4). F1 rewards yes-bias,
   so it *falls* as within-pair discrimination rises (E6/E12).
3. **Why routing works elsewhere and not here, mechanistically.** Routers are
   task classifiers: on RouterBench a base-rate-only "router" with no text and no
   learning scores AUC 0.694 against a fitted text router's 0.705, and per-prompt
   skill within a benchmark is 0.575 (E9). Vulnerability detection is
   single-domain, so there is no task identity to exploit and routing collapses to
   chance — the same shape as the within-project 0.95 vs across-project 0.49 gap
   in `07-hybrid-llm.md`.
4. **A negative result with a matched positive comparison.** Per-instance budget
   routing cuts 41% of tokens at equal accuracy on MATH ("Learning When to
   Think", arXiv 2608.20256) and is unlearnable here (E7: AUC 0.543; E10: oracle
   below null). Difference: maths has a verifiable answer and heterogeneous
   difficulty; paired vulnerability data has neither.

**Venue implication.** This is an empirical software-engineering / security paper
(ICSE, FSE, ISSTA, ACSAC), not ICLR/NeurIPS. The project's original constraint —
anchor every decision to ICLR/NeurIPS/ICML — pushed toward a generic-ML framing
where RouteGuard and "Opportunity Is Not Realizability" already occupy the space.
In security triage evaluation the seat is empty.

Queued experiments, highest value first:

| # | Experiment | Cost | Tests |
|---|---|---|---|
| ~~Q5~~ | **DONE — E18.** Qwen3-32B, both modes, same 299 pairs. Scale adds +0.011 [−0.048, +0.070]; the 32B reasoner does not cross zero. | gpu0, 66 min | Answered for local scale up to 32B. What remains is the **frontier API** half of the original Q5: one run, no weights, and it closes the last objection to E12/E18. |
| ~~Q6~~ | **DONE — E19.** Recall/detection at a fixed review budget, matched-budget random null, per-project breakdown. No filter beats random on detection; the one positive recall row is small-project artifact. | CPU, done | Answered: the filter-stage result is now metric-proof and is the reframed paper's headline table. |
| Q7 | Within-pair agreement vs patch size / CWE / function length | CPU | is stickiness worst on small patches, which would sharpen the mechanism claim |
| Q8 | Length-matched PrimeVul subset (pairs whose two versions differ by < N chars) | CPU | removes the E4 shortcut from the evaluation entirely |

Dropped: Q1-Q2 are done (E10, E11). Q3 (bigger local reasoner) is superseded by
Q5, which asks the same question with the right control attached. Q4 is done and
scooped (E8, E8b).

### Reporting rules adopted from this session

Every future table carries:

* a **shuffled-oracle** row next to any oracle (E2, E8);
* a **matched-count random** row for any filter — same number of alerts kept per
  project, chosen at random — not "no filter" (E3, E13);
* a **length-only** row for PrimeVul (E4);
* **pair-both-right** alongside F1, with its **`r(1-r)` same-rate baseline** at the
  configuration's own yes-rate (E12); F1 alone rewards bias;
* a **per-project breakdown** whenever precision is pooled across projects with
  different base rates, because pooled precision is Simpson's paradox waiting to
  happen (E13);
* **leave-one-project-out selection** of any model, ranker or threshold that was
  chosen by looking at results — hindsight selection produced two apparent wins
  this session that both vanished (E13).

### Infrastructure notes

* **csecluster** home is over its disk quota (`mkdir` fails, no downloads).
  Largest item visible is Mistral-Nemo at 46 GB from the LLMDFA work. Nothing
  was deleted.
* **gpu0** runs are fine and fast: vLLM 0.27.1 in `~/nimit/vllmenv`, weights in
  `~/nimit/hf`, work under `~/nimit/vr_explore/`. vLLM needs `ninja` on PATH
  (it is in that venv's `bin`) or the engine dies at startup with
  `FileNotFoundError: ninja`. Always stop the server on exit; a run of this size
  took 30 min and released the GPU cleanly.
* Everything in this log reproduces from `nullcheck/` in this repo (22 files):

  | script | entries |
  |---|---|
  | `common.py` | loaders, `p_vuln` from token log-probs — imported by the rest |
  | `null_oracle.py` | E2 |
  | `exp2.py` | E2 pairs, E3 |
  | `exp3.py` | E4, E5 |
  | `think_probe.py`, `run_think.sh`, `chain_all.sh` | E6 GPU run |
  | `analyze_think.py` | E6, E7 |
  | `rb_null.py`, `rb_frontier.py`, `rb_robust.py` | E8 |
  | `rb_predict.py`, `rb_mechanism.py` | E9 |
  | `null_closed_form.py` | closed-form null vs permutation, all 6 datasets |
  | `probe2.py`, `chain2.sh` | E10, E11 GPU run (budget forcing, self-consistency) |
  | `analyze_budget.py` | E10, E11 |
  | `calib_vs_reason.py` | E10 (thresholding does not reproduce the budget rise) |
  | `pair_null.py` | **E12 — the key script** |
  | `q5_chain.sh`, `q5_pair_null.py` | **E18 — Q5 positive control; adds bootstrap CIs and the paired 8B-vs-32B test** |
  | `q6_budget_recall.py` | **E19 — Q6; IRIS filter stage at a fixed review budget, with the binding-budget control** |
  | `precision_first.py`, `precision_honest.py`, `precision_twostage.py` | E13 |
  | `llmdfa_transfer.py` | **E14 — LLMDFA rule tested on IRIS, does not transfer** |
  | `../../LLMDFA/scripts/null_router.py` | **E15 — null battery on the LLMDFA router; the rule passes, its oracle does not** |
  | `../../LLMDFA/scripts/relative_threshold.py` | **E16 — relative vs absolute branch-count cut** |
  | `../../LLMDFA/scripts/precision_effect.py` | **E17 — precision at matched recall; the dispersion condition** |

  Needs numpy / pandas / scikit-learn — the project `.venv` has no scikit-learn,
  so a separate venv was used. Probe outputs are in `data/think/` (13
  files: 4 from E6, 6 budgets from E10, 1 self-consistency from E11, 2 from E18).
  Qwen3-32B weights live on gpu0 at `~/nimit/models/Qwen3-32B` as a plain
  directory (SHA256-verified against HF), not an HF cache entry.
  RouterBench pickles (E8/E9, 271 MB) are **not** stored — re-download from
  `huggingface.co/datasets/withmartian/routerbench` (`routerbench_0shot.pkl`,
  `routerbench_5shot.pkl`) into the working directory.
