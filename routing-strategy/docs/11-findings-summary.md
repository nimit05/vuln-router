# Findings summary, 2026-09-26 → 09-28

Plain-language summary of what the GraphRouter v2 / skilled-model work established,
what it means for a cost-aware vulnerability router, and what is still open. Full
details and tables: [`10-graphrouter-v2.md`](10-graphrouter-v2.md); the LLMDFA router:
[`../../LLMDFA/reproduction/ROUTER_V2_PERCASE.md`](../../LLMDFA/reproduction/ROUTER_V2_PERCASE.md).
Raw model outputs and router results: [`../results/2026-09-skilled-cascade/`](../results/2026-09-skilled-cascade/).

Data for every IRIS number below: 16 CWE-Bench-Java projects, 2,257 CodeQL paths,
344 real vulnerabilities, local models on vLLM 0.27.1 (A100-80GB).

## 1. Why GraphRouter and HybridLLM did not beat a single model

- The small-model pool was at chance on IRIS: within-project AUC 0.38–0.61.
  Routing between models that guess gives a guess.
- GraphRouter's training label (`argmax` model per path) does not match how it is
  scored (per-project recall).

**AUC here** = take one real bug and one false alarm from the same project; how often
does the model give the real bug the higher score? 0.5 is a coin flip, 1.0 is perfect.

## 2. A model with real skill exists: Qwen3-32B

| Model | Within-project AUC | GPU-s per path |
|---|---|---|
| Qwen3-32B, no thinking | **0.868** | 1.31 |
| Qwen3-8B, no thinking | 0.687 | 0.23 |
| gpt-oss-20b / phi-4 / Mistral-Nemo-12B | 0.613 / 0.610 / 0.462 | — |
| Qwen3-32B, thinking | 0.696 | ~65 |

Thinking mode is worse and ~55x slower.

## 3. The cost-aware result: an 8B → 32B cascade

**How it works.**
1. Qwen3-8B reads every path and gives it a confidence score: its probability of
   answering "vulnerable" (P("true") from the answer token's log-probabilities,
   renormalised over true/false). The score comes from the same call, at no extra cost.
2. In each project, the 30% of paths with the highest 8B score go to Qwen3-32B.
3. Escalated paths are ranked by the 32B's score, the rest by the 8B's.

The score is used only to **rank** paths, never read as a calibrated probability.

**Result.** Recall@10 (analyst reviews each project's top 10; share of its real bugs
found) is **0.677 vs 0.678** for the 32B on everything, at **1,473 vs 2,951 GPU-s**.

**Which paths go up is not random; how many go up was tuned.**

| Share sent to 32B | GPU-s | recall@10 | Random paths, same count |
|---|---|---|---|
| 0% (8B only) | 535 | 0.616 | — |
| 10% | 849 | 0.646 | 0.552 |
| 20% | 1,162 | 0.660 | 0.484 |
| **30%** | **1,473** | **0.677** | 0.398 (z +4.5) |
| 50% | 2,065 | 0.678 | 0.458 |
| 100% (32B only) | 2,951 | 0.678 | — |

Caveat: 30% was picked **after** seeing these results, as the smallest share that
reached the 32B's quality. Every share beats random escalation, and the curve is
smooth, so the policy is sound; the exact share still needs to be fixed in advance
(leave-one-project-out) and confirmed on new projects and OWASP.

**Comparison with the IRIS paper.** Keeping each project's top 10, the cascade scores
AvgF1 0.374, 10/16 detected, FDR 65.71; the paper's IRIS + GPT-4 row on the same 16
projects is 0.366, 10/16, 65.69. That is a **match, not a win**, and:
- it depends on the cascade's ranking (random top-10: 0.207; the 32B's plain yes/no: 0.213);
- GPT-4 was not re-run here, so the numbers are from the paper, not path-by-path;
- AvgF1 itself is unreliable (section 5).

## 4. Learned routers add nothing on top

- GraphRouter v2 (reward a caught bug, penalise a false alarm) picks the skilled model
  but ties a random router with the same model mix.
- GraphRouter v3 (given the 8B's score) re-learns the cascade rule and never beats it.
- Reason: the 8B and 32B are wrong on the same paths (both wrong 15.0% vs 10.6% if
  independent), so there are no complementary errors to route around.

## 5. IRIS's official score cannot see skill

**How AvgF1 works, per project:** recall = 1 if at least one real bug path was kept,
else 0 (all-or-nothing); precision = real bugs kept / paths kept; F1 combines them;
AvgF1 averages over projects.

**Worked example.** Two projects, 10 alarms each, 1 real bug each.

| Model | Project A | Project B | AvgF1 |
|---|---|---|---|
| Keeps all 10 | 0.18 | 0.18 | **0.18** |
| Keeps 5 at random (lucky in A) | 0.33 | 0 | **0.17** |
| Ranks the bug 2nd of 10, keeps only its top 1 | 0 | 0 | **0** |

The model that reads the code best scores lowest.

**Measured on IRIS:** deepseek-6.7b (AUC 0.489, a coin flip) has the best AvgF1
(0.223); Qwen3-32B (AUC 0.868) scores 0.213; no filter at all scores 0.212.
Kendall τ between skill and AvgF1 across 10 models: 0.20 (τ = +1 same order, 0 unrelated).

Use instead: within-project AUC and fixed-budget recall@10, always next to a random
baseline with the same count.

## 6. The 32B reads code; the 8B reads names

Project-specific identifiers renamed (names used in fewer than 5 of 79 repos):

| | Original | Renamed | Strict (also comments and strings stripped) |
|---|---|---|---|
| Qwen3-32B | 0.868 | **0.795** | 0.719 |
| Qwen3-8B | 0.687 | 0.544 | 0.511 |

Rerun noise ±0.002. Text4Shell, the most famous CVE, does not drop (0.948 → 0.970 →
0.951), so the 32B's drop is lost helpful names, not forgotten CVEs. The 8B's ranking
skill comes from names, which the cheap tier of a router must be checked for.

## 7. Routing on LLMDFA works with a different signal

| Benchmark | Router | Result |
|---|---|---|
| IRIS | GraphRouter v1–v3, HybridLLM | no gain over one model or a random router |
| IRIS | cascade on the 8B's confidence | 32B quality at half the cost |
| LLMDFA | per-case rule: ≥ 12 `if` conditions → Phi-4-mini, else Mistral-Nemo | **+0.060 F1 at 6% lower cost** on held-out forms; beats random at z +5.5 / +4.4 / +2.3 |
| LLMDFA | small model → Z3 → large model on rejects | 1 of 12 pairs "wins", by +0.005 (noise) |
| IRIS | the LLMDFA rule, transferred | does not transfer |

GraphRouter was never run on LLMDFA: the rule already sits near the oracle there.

## 8. What this means for the cost-aware router

No single signal works everywhere. On IRIS the cheap model's confidence works; on
LLMDFA a static code feature works; each fails on the other benchmark. The proposed
strategy is a procedure, not one rule:

1. **Skill gate.** Keep only models with within-group AUC ≥ 0.65.
2. **Candidate signals, cheapest first.** Static code features (free), cheap-model
   confidence (one cheap call), a verifier (Z3, CodeQL).
3. **Null test on a development split.** Keep a signal only if it beats random
   routing with the same count per model.
4. **Route and report** recall at a fixed review budget against cost, as a curve.

The claim to test: chosen on a dev split, the procedure picks the working signal on
a benchmark it has not seen (OWASP). Known prior work on cascades (e.g. FrugalGPT)
means the contribution is the security-specific signals, the gates, and the metric,
not the cascade itself.

## 9. Open: the scaled run (paused 2026-09-27)

- Code databases built for 57 of 96 more IRIS projects (on gpu0).
- IRIS spec inference on them not yet run. The vLLM-served deepseek-7b needs the
  tokenizer fix (`tokenizer_class: PreTrainedTokenizerFast`); transformers 5.16.1
  otherwise loads a slow tokenizer that drops spaces and returns empty labels.
- OWASP Benchmark Java: 1,956 CodeQL alerts matched to the answer key (1,263 real),
  built by `graphrouter/o1_owasp_units.py`; models not yet run.
- Analysis script ready: `graphrouter/s1_scaled_analysis.py`. About 6 GPU hours remain.
