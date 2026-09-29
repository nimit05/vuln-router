# GraphRouter v2: fixed objective, a model with real skill, and what routing still cannot do

Session 2026-09-26. All GPU work and router training on gpu0 (A100-SXM4-80GB, vLLM 0.27.1, the same
engine and prompt as the B6 pool, so every row is comparable).

## Short version

| Question | Answer |
|---|---|
| Does fixing the objective stop GraphRouter losing to the best single model? | **Yes.** 0.198 → 0.218 on the old pool; with a skilled model in the pool it picks that model |
| Is there a model that can actually triage IRIS paths? | **Yes: Qwen3-32B no-think** (within-project AUC 0.868), and Qwen3-8B (0.687) at 18% of its cost |
| Does thinking help? | **No.** Think mode 0.696 vs no-think 0.835 on the same 358 paths, at ~55x the cost |
| Does a learned per-path router beat a random router with the same mix? | **No**: GraphRouter v2 (pre-call) and v3 (sees the 8B's score) never beat the simple rule below |
| What does work? | **A two-stage cascade**: 8B on every path, then its top 30% per project to the 32B. **All-32B quality at 50% of the cost**; beats random escalation at z +4.5 |
| Best IRIS-table row? | **Cascade, top 10 per project: AvgF1 0.374, 10/16 detected, FDR 65.71**, vs the paper's IRIS + GPT-4 at 0.366 / 10/16 / 65.69 |
| Is there a second skilled model from another family? | **No.** Mistral-Nemo-12B 0.462, gpt-oss-20b 0.613, phi-4 0.610: all fail the gate and err on the same paths as the 32B |

## 1. What changed in GraphRouter (v2)

Script: `graphrouter/d2_graphrouter_v2.py`. Upstream network, masking and
optimiser are unchanged (`third_party/GraphRouter/model/graph_nn.py`). Three
changes:

1. **Edge target.** Upstream: `label = eye(n)[argmax(effect)]`, i.e. "which model
   won this path". v2: each model's **security utility** on the path, where a
   missed bug costs `rho = 10` times a false alarm:
   - true bug: `utility = p` (p = the model's P(vulnerable))
   - false alarm: `utility = 1 - p/rho`

   Example: granite keeps a real bug at p = 0.95 and gets 0.95. qwen-7b drops
   it (p = 0.05) and gets 0.05. On a false alarm, granite keeping it at p = 0.95
   costs only 0.095, so it scores 0.905.
2. **Decision.** `argmax(predicted utility - lambda * mean cost)`, sweeping lambda.
3. **Controls on every routed row.** v1 (upstream label) through the same
   trainer, and a **same-mix random router**: the same number of paths per
   project to each model, assigned at random, 500 draws.

Leave-one-project-out throughout.

## 2. The old pool: the fix works, but routing has nothing to route

4 models (qwen-1.5b, phi-3.8b, qwen-7b, granite-8b), `data/gr/d2_pool4.json`.

| Config | AvgF1 | Same-mix random |
|---|---|---|
| always-granite-8b (best single) | 0.207 | — |
| v1, upstream label | 0.198 | 0.200 |
| **v2, lambda = 0** | **0.218** | **0.222 (z −7.0)** |
| v2, lambda ≥ 0.05 | 0.105 (all to qwen-1.5b) | — |

v2 now sends 403 paths to granite (v1: 30). But random assignment at the same
mix does as well, because every model's mean utility is 0.888–0.892: nothing to
choose between.

## 3. A model with skill (D1)

Scripts: `graphrouter/d1_probe_reasoner.py`, `d1_chain_gpu0.sh`, `d1_gate.py`.
The gate was fixed before any result: within-project AUC ≥ 0.65 and 95% CI above
0.5. Within-project AUC = over (true bug, false alarm) pairs from the **same**
project, how often the model scores the bug higher. 0.5 is a coin flip.

| Model | Within-project AUC, all 2,257 paths | 95% CI | Batch-1 GPU-s/path |
|---|---|---|---|
| qwen-1.5b | 0.378 | [0.348, 0.418] | 0.11 |
| granite-8b | 0.475 | [0.429, 0.519] | 0.86 |
| phi-3.8b | 0.540 | [0.502, 0.578] | 0.19 |
| qwen-7b | 0.586 | [0.540, 0.623] | 0.25 |
| **Qwen3-8B no-think** | **0.687** | [0.648, 0.719] | **0.23** |
| **Qwen3-32B no-think** | **0.868** | [0.845, 0.892] | 1.31 |
| Qwen3-32B think (358-path sample) | 0.696 | [0.638, 0.753] | ~65 |

Checks on the 32B result:

- **Not one bug counted many times.** Averaged per sink method (110 vulnerable,
  390 clean), AUC is still 0.810 (qwen-7b 0.622).
- **Broad, not one project.** 7 of 9 projects with bugs score above 0.85,
  including ff4j (1,413 paths, 181 TPs, 0.877). Weak: ESAPI 0.603, workflow-cps 0.556.
- **Passes the fixed-budget control that killed qwen-7b (E19).** Analyst reviews
  the top 10 alerts per project, ranked by the model's score:

  | | det@10 | recall@10 | recall@10, 6 projects where the budget binds |
  |---|---|---|---|
  | random order | 7.5/10 | 0.304 | 0.098 |
  | qwen-7b rank (E19) | 8/10 | 0.461 | 0.102 |
  | **Qwen3-32B rank** | **10/10** | **0.678** (z +6.6) | **0.463** |
  | oracle (bugs first) | 10/10 | 0.743 | — |

- **As a keep/drop filter, AvgF1 hides it.** always-32B scores AvgF1 0.213
  (no filter: 0.212) because it drops every bug in 2 projects. Matched-count
  random keeps score 0.136 (z +2.3).

Caveat: every CVE here predates Qwen3's training cutoff, so memorisation of
known-vulnerable code cannot be ruled out. The sink-method and per-project
spread make it less likely, not impossible.

## 4. GraphRouter with skilled models: model selection, not routing

`data/gr/d2_pool5_q32.json`, `d2_pair_8b_32b.json`, `d2_pool6.json`.

**Pool + 32B.** v1 and v2 (lambda = 0) both send all 2,257 paths to the 32B:
0.213, identical to always-32B. Any cost weight sends paths to chance-level
models and loses (0.166 at lambda = 0.01).

**{Qwen3-8B, Qwen3-32B}**, fixed-budget metric (AvgF1 is swung by single paths,
see below):

| Config | Paths to 32B | GPU-s | recall@10 | binding | Same-mix random recall@10 |
|---|---|---|---|---|---|
| always-8B | 0 | 535 | 0.616 | 0.359 | — |
| always-32B | 2,257 | 2,951 | 0.678 | 0.463 | — |
| v2, lambda = 0 | 1,077 | 1,728 | 0.681 | 0.469 | 0.680 ± 0.001 (z +1.3) |
| v2, lambda = 0.005 | 366 | 943 | 0.430 | 0.217 | 0.558 ± 0.053 (z −2.4) |
| v2, lambda ≥ 0.01 | ≤ 102 | ≤ 642 | 0.616 | 0.359 | 0.616 (z ≤ 0) |
| v1, upstream label | 140 | 680 | 0.616 | 0.359 | 0.616 (z −0.2) |

v2 at lambda = 0 matches always-32B at 59% of the cost, but a random router
with the same 48% share does the same. Nothing is learned per path.

**One apparent win, and why it is not one.** On AvgF1, v2 at lambda = 0.03
scores 0.207 against a same-mix null of 0.167 (z +6.4). It sends only 12 paths
to the 32B; one of them is a true bug in antisamy-2017, and under per-project
AvgF1 that single path flips the project from missed to detected. Worse,
antisamy-2016 and antisamy-2017 are the **same repository at two versions**, so
leave-one-project-out does not hold one of them out from the other. On the
fixed-budget metric this row equals always-8B.

## 5. Why routing still cannot beat the best model

- **The skilled models fail on the same paths.** Qwen3-8B and 32B are both wrong
  on 15.0% of paths against 10.6% if independent. The pair oracle is *below*
  shuffle level (0.850 real vs 0.904 shuffled). With no complementary errors,
  the best per-path choice is almost always "the 32B".
- **The router decides before calling any model**, from a MiniLM text embedding
  of the slice. Nothing in that embedding says which paths the 8B gets right.
- **Only 14 held-out folds, two of them the same codebase.**

## 6. What works: a cascade on the cheap model's own score (D4)

Script: `graphrouter/d4_skilled_combos.py`. The difference from GraphRouter:
the routing decision is made **after** the cheap call, on the 8B's P(vulnerable),
which has skill (AUC 0.687). GraphRouter decides **before** any call, from a text
embedding that has none.

Policy: run Qwen3-8B on every path; in each project send the top fraction f of
paths (by the 8B's score) to Qwen3-32B; review escalated paths first, ranked by
the 32B, then the rest by the 8B. Null: escalate the same number per project at
random. Cost counts both models.

| Policy | GPU-s | det@10 | recall@10 | binding | Random escalation, same count |
|---|---|---|---|---|---|
| 8B only | 535 | 9/10 | 0.616 | 0.359 | — |
| cascade f = 0.1 | 849 | 9/10 | 0.646 | 0.410 | 0.552 ± 0.015 (z +6.1) |
| cascade f = 0.2 | 1,162 | 9/10 | 0.660 | 0.434 | 0.484 ± 0.050 (z +3.5) |
| **cascade f = 0.3** | **1,473** | **10/10** | **0.677** | **0.461** | 0.398 ± 0.062 (z +4.5) |
| cascade f = 0.5 | 2,065 | 10/10 | 0.678 | 0.464 | 0.458 ± 0.067 (z +3.3) |
| 32B only | 2,951 | 10/10 | 0.678 | 0.463 | — |

Example: ff4j has 1,413 paths. The 8B scores all of them (~335 GPU-s), and its
top 424 go to the 32B (~555 GPU-s instead of ~1,850 for all 1,413). The analyst's
top 10 then come from the 32B's ranking of those 424.

Averaging the two models' scores does **not** help (AUC 0.811 < 0.868).

**On IRIS's own table** (AvgF1 via their `score_subset.metrics`), keeping the top
k paths per project. k = 10 was fixed in advance (E19's budget); other k are shown
so no cherry-pick hides:

| Policy | k | AvgF1 | det | FDR% | GPU-s |
|---|---|---|---|---|---|
| random k per project (E3 null) | 10 | 0.207 ± 0.024 | — | — | 0 |
| **cascade 8B → 32B, f = 0.3** | **10** | **0.374** | **10/16** | **65.71** | **1,473** |
| cascade 8B → 32B, f = 0.3 | 3 / 5 | 0.400 / 0.395 | 8/16 | ~60 | 1,473 |
| 32B only | 10 | 0.380 | 10/16 | 65.00 | 2,951 |
| 32B only | 1 / 3 / 5 | 0.375 / 0.369 / 0.360 | 6–8/16 | 57–64 | 2,951 |
| paper: IRIS + GPT-4 | — | 0.366 | 10/16 | 65.69 | API |

Not like-for-like with the paper row: IRIS's filter is keep/drop with no budget,
and these rows keep a fixed k. The fair control for a top-k row is random-k, which
it beats by +0.17.

## 7. GraphRouter v3: learning the escalation (D2 `--probe-feature`, D5)

v3 gives GraphRouter the 8B's score as a query feature (post-call), so it can
learn *which* paths to escalate. Scored by `graphrouter/d5_eval_escalation.py`
against the rule above at the **same number of escalations in every project**:

| Router | Escalated | recall@10 | Rule, same count | Random, same count |
|---|---|---|---|---|
| v2 (pre-call, no 8B feature), lambda = 0.002 | 679 | 0.564 | **0.680** | 0.643 (z −3.3) |
| v3, upstream label | 527 | 0.631 | **0.632** | 0.540 (z +3.4) |
| v3, security label, lambda = 0 | 2,051 | 0.675 | 0.674 | 0.635 (z +1.3) |
| v3, security label, lambda = 0.002–0.02 | 94–1,581 | 0.35–0.54 | **0.61–0.63** | 0.48–0.59 (z −4.5 to −0.6) |

At best the GNN re-learns the one-line rule; it never beats it. Use the rule.

## 8. The search for a second skilled model (D3)

A routing gain beyond the cascade needs a model that is right where the 32B is
wrong. Every candidate on gpu0, same prompt and engine:

| Model | Mode | Within-project AUC | Batch-1 GPU-s/path | Both wrong with 32B (independent) |
|---|---|---|---|---|
| Qwen3-8B | no-think | **0.687** | 0.23 | 0.150 (0.106) |
| Mistral-Nemo-12B | direct | 0.462 | 1.39 | 0.239 (0.195) |
| gpt-oss-20b | low reasoning | 0.613 | — | 0.133 (0.077) |
| phi-4 (14B) | direct, explains first | 0.610 | 6.57 | 0.272 (0.208) |

Every pair errs together more often than chance. Only the Qwen3 family clears
the gate. Probing notes: Nemo rejects Qwen's `enable_thinking` switch (use
`--plain`); phi-4 ignores "JSON only" and needs `--mode verbose`; gpt-oss runs on
the A100 under vLLM 0.27.1 with `--mode low`.

## 9. What to use

- **Deploy:** the 8B → 32B cascade at f = 0.3 as IRIS's filter/ranker. It has the
  32B's quality at half its cost. Report the IRIS-table row at k = 10 (0.374) next
  to the random-k null and the 32B-only row.
- **Report GraphRouter honestly:** the objective fix (v2) turns it into correct
  model selection, and v3 recovers the cascade rule. Neither beats the rule, so
  the routing contribution is the cascade, not the GNN.
- **What could still move the number:** a cheaper first stage with the 8B's skill
  (e.g. Qwen3-4B), or a second skilled family. Local models tried so far all fail.
  Memorisation caveat from §3 still applies: all CVEs predate Qwen3's cutoff.

## 10. Memorisation control: hide the project, keep the logic (D6, 2026-09-27)

Every CVE here predates Qwen3's training data, so the 32B could be recognising
known-vulnerable code. `graphrouter/d6_anonymise.py` rewrites every path:

- **Renamed (a):** any name used in fewer than 5 of the 79 CWE-Bench-Java
  repositories is project vocabulary and becomes `Cls1` / `name1` / `CONST1`
  (consistently within a path); file paths become `F1.java`. `HashTrie`, `FF4j`,
  `FeatureStore`, `ESAPI` hidden; `File`, `exec`, `getParameter` kept. 21% of names renamed.
- **Strict (b):** keep only names used in 40+ repositories, drop comments, blank
  string literals. Even `StringSubstitutor` (Text4Shell) and `ScriptEngine` hidden.
  37% renamed.

Same models, prompt, mode and engine (`d6_chain_gpu0.sh`, 18 GPU-min); paired
bootstrap over paths (`d6_compare.py`):

| | Original | Rerun of original | Renamed (a) | Strict (b) |
|---|---|---|---|---|
| Qwen3-32B within-project AUC | 0.868 | 0.870 (99.6% same verdicts) | **0.795** (−0.073 [−0.099, −0.043]) | **0.719** (−0.149 [−0.186, −0.105]) |
| Qwen3-8B within-project AUC | 0.687 | — | 0.544 (−0.142) | 0.511 (−0.175) |

- **The 32B reads the code.** Rerun noise is ±0.002, and with the project hidden it
  keeps 0.795 (strict 0.719), above the 0.65 gate in both.
- **The drop looks like lost descriptive names, not forgotten CVEs.** Per project,
  the most famous CVE (commons-text, Text4Shell) does not drop at all (0.948 →
  0.970 → 0.951). Most of the pooled drop is ff4j, an obscure CVE with 181 of the
  344 bugs (0.877 → 0.810 → 0.710).
- **The 8B's skill is names.** Hidden, it falls to chance. The cascade still works on
  real code (names exist there), but the 8B cannot be described as reading the code.

**Report:** 0.795 as the 32B's code-reading skill, 0.868 with real names, 0.719 as a
strict lower bound. The main claim is unchanged: IRIS's AvgF1 ranks chance-level
models (e.g. deepseek-6.7b, AUC 0.489, AvgF1 0.223) above a model that keeps
0.795 with its project hidden (AvgF1 0.213).

## Reproduce

```
# gpu0, vLLM 0.27.1 (probes), hybridllm_nimit venv + torch_geometric (router)
bash graphrouter/d1_chain_gpu0.sh       # Qwen3-32B no-think (all) + think (gate) + calibration
bash graphrouter/d3_chain_gpu0.sh "qwen3-8b|<path>|0.45||--dtype bfloat16|nothink" \
    "nemo-12b|<path>|0.55|--plain|--dtype bfloat16|nothink" \
    "gpt-oss-20b|<path>|0.5|--plain||low" "phi-4|<path>|0.5|--plain|--dtype bfloat16|verbose"
python graphrouter/d1_gate.py --extra name=path ...
python graphrouter/d4_skilled_combos.py                      # ensemble, cascade, IRIS top-k
python graphrouter/d2_graphrouter_v2.py --extra qwen3-8b-nothink=... qwen3-32b-nothink=... \
    --cost-per-token data/gr/d1/cost_per_token.json [--drop ...] [--probe-feature qwen3-8b-nothink]
python graphrouter/d5_eval_escalation.py data/gr/d2_v3_cascade.npz
python graphrouter/d6_anonymise.py && bash graphrouter/d6_chain_gpu0.sh && python graphrouter/d6_compare.py
```
