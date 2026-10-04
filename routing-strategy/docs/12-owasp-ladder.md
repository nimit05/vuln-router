# 12 — OWASP model ladder and per-unit routing (2026-09-29)

Task from the professor: run a small → medium → large → extra-large model ladder plus a
pool of other models over one dataset, save every model's answer on every data point,
and build a router that picks the model for each new data point. Raw outputs and
reproduction commands: [`../results/2026-09-owasp-ladder/`](../results/2026-09-owasp-ladder/).

**Dataset.** OWASP Benchmark (Java), as triage units: each unit is one CodeQL alert on
one benchmark test, and the question is "is this alert a real vulnerability?". 1,956
units, 1,263 real (65%), 10 categories (sqli 479, pathtraver 428, xss 336, ...).
Chosen over IRIS because every test is its own small program with its own answer, so
per-unit routing is not confounded by project identity (the IRIS failure, E13/E14).
Split 70/30 by unit, stratified by category and label: 1,368 train, 588 test.

**Setup.** gpu7, vLLM 0.27.1, one A100-80GB, same prompt as all earlier probes.
Cost = GPU-seconds per unit at 32 concurrent requests.

## 1. The models

| Role | Model | Accuracy | AUC | TPR − FPR | Says "yes" | GPU-s / unit |
|---|---|---|---|---|---|---|
| **small** | Qwen3-1.7B | 0.606 | 0.447 | −0.024 | 90% | 0.013 |
| **medium** | Qwen3-8B | 0.701 | 0.717 | 0.326 | 68% | 0.033 |
| **large** | Qwen3-32B | 0.762 | 0.841 | 0.414 | 75% | 0.140 |
| **extra large** | gpt-oss-120b (low reasoning) | **0.902** | 0.907 | **0.789** | 64% | 0.156 |
| pool | gpt-oss-20b (low reasoning) | 0.881 | **0.912** | 0.762 | 61% | 0.061 |
| pool | phi-4 (14B) | 0.778 | 0.732 | 0.408 | 81% | 0.305 |
| pool | Granite-3.1-8B | 0.692 | 0.727 | 0.137 | 94% | 0.062 |
| pool | Mistral-Nemo-12B | 0.668 | 0.648 | 0.117 | 89% | 0.073 |
| pool | Phi-4-mini (3.8B) | 0.651 | 0.590 | 0.060 | 92% | 0.026 |
| pool | Qwen2.5-Coder-7B | 0.632 | 0.594 | 0.111 | 78% | 0.043 |
| pool | DeepSeek-Coder-6.7B | 0.585 | 0.527 | 0.094 | 65% | 0.153 |
| pool | Qwen2.5-Coder-1.5B | 0.468 | 0.645 | 0.109 | 22% | 0.012 |

* **AUC**: pick one real and one fake alert; how often does the model give the real one
  the higher score? 0.5 = coin flip.
* **TPR − FPR**: OWASP's own score. Share of real alerts flagged minus share of fake
  alerts flagged. Saying "vulnerable" to everything scores 0, even though that gets 65%
  accuracy here.
* The ladder works as a ladder: the small model is at chance, each step up is better,
  and the extra-large model is best. gpt-oss-20b is nearly as good at 40% of the cost:
  it is a mixture-of-experts model that uses ~3.6B parameters per token.
* On IRIS the same gpt-oss-20b had AUC 0.613 (E20). OWASP is much easier.

## 2. Are the models reading the code?

OWASP has been public since 2015. Every name that identifies it was rewritten
(`BenchmarkTest00001` → `RequestHandler`, `org.owasp.benchmark` → `com.example.app`)
and three models were re-run:

| Model | AUC, names kept → hidden |
|---|---|
| gpt-oss-20b | 0.912 → 0.914 |
| Qwen3-32B | 0.841 → 0.835 |
| Qwen3-8B | 0.717 → 0.705 |

No real drop, so the skill is code reading, not recognising the benchmark (on IRIS the
8B fell to chance under the same test, E20).

## 3. Routing: does a per-unit router beat the best single model?

Each test unit is sent to one model. The router is fitted on train to maximise

    reward = correct − λ × GPU-seconds

`correct` is 1 or 0; λ says how much accuracy one GPU-second is worth. λ = 0 means
accuracy only. λ = 1 means paying 0.1 GPU-s must buy at least 0.1 accuracy.
Example at λ = 1: gpt-oss-120b right on a unit scores 1 − 0.156 = 0.844; gpt-oss-20b
right on it scores 1 − 0.061 = 0.939, so the router prefers the 20b when both are right.

Routers:
* **best**: one model for everything, the best on train.
* **cat**: per OWASP category, the best model on train. Knows only the bug type.
* **kNN**: find the 20 most similar train units (MiniLM text embedding); send the unit
  to the model that did best on them.
* **cascade**: a cheap model answers; if unsure, the unit goes to a dearer model.
* **base-rate trick**: best single model, except in categories where "always yes" or
  "always no" did better on train. No second model.

Test split, 588 units. "Gain" = reward minus best single, with its 95% interval over
resampled test sets.

| Pool | λ | Router | Accuracy | GPU-s / unit | Gain vs best, 95% CI |
|---|---|---|---|---|---|
| ladder (4) | 0 | best = gpt-oss-120b | 0.905 | 0.156 | — |
| ladder (4) | 0 | cat | **0.929** | 0.135 | +0.003 to +0.044 |
| ladder (4) | 0 | kNN | 0.912 | 0.132 | −0.014 to +0.027 |
| ladder (4) | 1 | cat | 0.923 | 0.127 | +0.025 to +0.072 |
| ladder (4) | 1 | cascade 8B → 120b | 0.869 | 0.120 | −0.019 to +0.019 |
| all 12 | 0 | best = gpt-oss-120b | 0.905 | 0.156 | — |
| all 12 | 0 | base-rate trick (120b + always-yes/no) | 0.922 | 0.135 | +0.000 to +0.034 |
| all 12 | 0 | cat | **0.934** | 0.134 | +0.010 to +0.049 |
| all 12 | 0 | kNN | 0.932 | 0.096 | +0.005 to +0.049 |
| all 12 | 1 | best = gpt-oss-20b | 0.869 | 0.061 | — |
| all 12 | 1 | base-rate trick (20b + always-yes/no) | 0.889 | 0.047 | +0.010 to +0.059 |
| all 12 | 1 | cat | 0.913 | 0.055 | +0.028 to +0.074 |
| all 12 | 1 | kNN | **0.932** | 0.058 | +0.041 to +0.091 |

Every router in the table beats random routing that uses the same model mix (z 5.6
to 10.8), so *which* units go to which model matters, not only the mix.

**Headline.** With all 12 models, the kNN router gets **0.932 accuracy at 0.058
GPU-s per unit**, against gpt-oss-120b's 0.905 at 0.156: better answers at 37% of the
cost. With the 4-model ladder alone, the best router is the category router: 0.929 at
0.135.

## 4. What the gain is made of

Best model per category (accuracy, all units; full table in
`routers/per_category.txt`):

| Category | Real | Qwen3-8B | Qwen3-32B | gpt-oss-120b | Granite-8B | Best |
|---|---|---|---|---|---|---|
| sqli | 57% | 0.643 | 0.693 | **0.956** | 0.566 | 120b |
| pathtraver | 57% | 0.654 | 0.799 | **0.953** | 0.603 | 120b |
| hash | 61% | 0.786 | 0.655 | 0.731 | **0.952** | Granite (real skill) |
| crypto | 83% | 0.764 | 0.815 | 0.777 | **0.828** | Granite (says yes 94%) |
| trustbound | 78% | 0.402 | 0.654 | 0.449 | 0.776 | "always yes" (0.78) |

Three sources, measured at λ = 0 with all 12 models (0.905 → 0.934, +0.029):

1. **Base rate, about +0.017.** In `trustbound` and `crypto` most alerts are real, and
   gpt-oss-120b says "safe" too often. Answering "yes" there beats every model. The
   base-rate trick alone reaches 0.922. It needs no second model; it is a property of
   this benchmark's category mix.
2. **A better model for a category, about +0.012.** Example: `hash`, where Granite-8B
   is right on 95% and gpt-oss-120b on 73%.
3. **Per-unit choice within a category, small.** kNN minus the category router: −0.002
   at λ = 0 (not different), +0.015 to +0.020 at λ = 1 (interval touches zero), +0.032
   at λ = 4. It pays mainly when cost matters and cheap models must be used where they
   are safe.

When cost matters (λ = 1) routing does add real value: kNN's reward is 0.874 against
0.842 for the base-rate trick on the cheap gpt-oss-20b.

## 5. What does not work

* **A new kind of bug.** Leave-one-category-out (train on 9 categories, test on the
  10th): every router loses to the best single model. λ = 0, all 12 models: best single
  0.901, kNN 0.847. The router learns "which model is good at *this* category"; that
  knowledge does not exist for a category it has never seen.
* **Confidence cascades.** Sending the cheap model's unsure units up never beats the
  best single model here (ladder, λ = 1: 0.869 vs 0.905 at similar cost). Unlike IRIS
  (E20), the best model on OWASP is also cheap enough to run on everything.
* **Oracle headroom is not informative with 12 models.** Some model is right on 99.8%
  of test units, and the shuffled oracle is 99.5%: with that many models, someone is
  right by chance. The random-same-mix null is the test that matters.

## 6. Caveats

* **Cost is noisy.** Repeat runs of the same model differ by ~25% in wall time.
  gpt-oss-20b's first run had ~20 requests hang for ~435 s; its cost is taken from the
  clean re-run (same prompts). Costs are at 32 concurrent requests, not batch-1.
* **One benchmark, one split.** OWASP tests are generated from templates, so a test
  unit usually has a near-twin in train. That is the setting the router is good at
  (new units from a known benchmark); section 5 shows it does not transfer beyond it.
* **65% of units are real.** Accuracy rewards saying "yes"; TPR − FPR is reported
  beside it for that reason.
* **DeepSeek-Coder** needed a tokenizer fix (README); its score is from the fixed run.

## 7. Our own Jev-style router (2026-09-30)

Jev is paid and zero-shot, so we built the same interface ourselves and trained it on
the saved outcomes: code in, one calibrated P(model right) per model out, no text
generated; pick = argmax P(right) − λ × GPU-s. Code: `ladder/l4_learned_router.py`
(fit), `ladder/l5_eval_learned.py` (score), `ladder/jevlike_route.py` (use it on new
code). Outputs: `results/2026-09-owasp-ladder/learned/`.

Two versions, both fitted on the 1,368 train units only:
* **emb-lr**: MiniLM embedding → one logistic regression per model (CPU, 20 s).
* **finetune**: UniXcoder (125M code encoder) fine-tuned with 12 outputs (GPU, ~80 s),
  3 seeds.

**Are the probabilities any good?** On test, per model: fine-tuned AUC 0.93 (it ranks
the units a model gets right above those it gets wrong), calibration error 0.034
("0.9" means right ~9 times in 10). emb-lr: AUC 0.82, error 0.052.

**Test split (588 units):**

| λ | Router | Accuracy | GPU-s / unit | Gain vs best, 95% CI |
|---|---|---|---|---|
| 0 | best single (gpt-oss-120b) | 0.905 | 0.156 | — |
| 0 | kNN / category router | 0.935 / 0.934 | 0.125 / 0.137 | |
| 0 | emb-lr | 0.942 | 0.120 | +0.017 to +0.060 |
| 0 | finetune, 3 seeds | 0.971–0.978 | 0.079–0.126 | +0.043 to +0.095 |
| 1 | best single (gpt-oss-20b) | 0.869 | 0.061 | — |
| 1 | finetune, 3 seeds | 0.959–0.969 | 0.025–0.033 | +0.091 to +0.167 |

**Control: the router may just be solving the task.** Some pool models say "yes" to
almost everything (Granite 94%), one says "no" to most (Qwen2.5-Coder-1.5B 78%). A
router that knows the answer can send vulnerable units to a yes-sayer and safe ones to
a no-sayer. So the same UniXcoder was fine-tuned on the answer key directly, no LLM:
**test accuracy 0.997**. On units from categories it was trained on, OWASP is solved by
a 125M model trained on OWASP; the router's in-distribution win is that, not routing.

**New bug category** (leave-one-category-out, categories with ≥ 30 units, λ = 0):

| | Accuracy |
|---|---|
| **finetune router, 3 seeds** | **0.919 / 0.944 / 0.923** (vs best: +0.019 to +0.043, every CI above 0) |
| best single LLM | 0.901 |
| direct classifier, no LLM | 0.875 |
| kNN / emb-lr router | 0.865 / 0.852 |

Here the router beats both the best LLM and the classifier trained on the same data:
where its own knowledge runs out, it relies on the LLMs. At λ = 1 it is +0.042 to +0.067
over the best single choice (all three seeds, every CI above 0). With heavy cost weight
(λ = 4) the free classifier wins (0.875 vs ~0.80 reward). Seed 0 ran on gpu0, seeds 1-2
on gpu7 (same vLLM env, torch 2.13); seed-to-seed spread 0.919-0.944 is the noise level.

## 8. Train on OWASP, route on IRIS (2026-09-30)

The real "new data" test. The same 12 models, same prompt and settings, were run on
IRIS's 2,257 CodeQL paths (gpu7). Every router was fitted on all 1,956 OWASP units and
saw no IRIS answer. Outputs: [`../results/2026-09-iris-cross/`](../results/2026-09-iris-cross/),
scored by `ladder/l6_eval_cross.py` (`eval.txt`). IRIS is 15% real (OWASP 65%), so
"always no" already scores 0.848 accuracy; TPR − FPR and within-project AUC matter more.

**The models on IRIS** (selected): Qwen3-32B is the only strong one (AUC 0.855,
TPR − FPR 0.311). gpt-oss-120b, best on OWASP (0.907), falls to AUC 0.605 on IRIS and
says "vulnerable" to only 14% of paths. The model ranking does not carry over.

**Routers, λ = 0** (accuracy; 95% CI of the difference to the best OWASP-chosen model):

| Router (fitted on OWASP) | Accuracy on IRIS | TPR − FPR | vs best-on-OWASP |
|---|---|---|---|
| best-on-OWASP: gpt-oss-120b | 0.767 | 0.062 | — |
| [hindsight] best-on-IRIS: gpt-oss-20b | 0.772 | 0.137 | +0.005 [−0.011, +0.020] |
| CWE router (OWASP category with the same CWE) | 0.767 | 0.062 | 0 |
| emb-lr router | 0.605 | −0.089 | −0.162 [−0.178, −0.146] |
| finetune router, 3 seeds | 0.366 / 0.374 / 0.586 | −0.08 to +0.12 | −0.18 to −0.40 |
| direct classifier, 3 seeds (no LLM) | 0.833 / 0.440 / 0.150 | −0.02 to −0.12 | unstable |

**It fails, badly.** Every learned router is worse than simply using the model that was
best on OWASP, and mostly worse than random routing with the same model mix (z −6.1 to
+0.7). Same picture at λ = 0.25 and 1.

**Why.** On OWASP most alerts are real, so the router learned "send it to Granite":
Granite says "vulnerable" to 94% of OWASP units and is right most of the time. On IRIS
85% are false alarms; Granite still says "vulnerable" to 97% and is right on 15%. The
fine-tuned router sends 1,273–1,413 of 2,257 IRIS paths to Granite. What it learned on
OWASP was the benchmark's base rate and which models say "yes", not which model reads
code well. The direct classifier shows the same thing: it swings from 0.15 to 0.83
between seeds depending on whether it leans to "yes" or "no".

**What this means.** A router trained on one benchmark learns that benchmark's answer
mix. It needs outcomes from the target data (or at least its base rate) before it can
be trusted. The in-benchmark and new-category wins in section 7 do not transfer to a
new benchmark.

## 9. A free zero-shot Jev stand-in (2026-09-30)

Jev is paid, so its job was given to a model we host: Qwen3-8B (no thinking) gets the
code plus one line per ladder tier (1.7B / 8B / 32B / 120B) and answers "which model
should check this alert? pick the cheapest capable one"; the probability of each option
letter is read from its first answer token, as Jev returns one probability per tier.
Nothing is learned from outcomes. Two description sets: **generic** (size and "how
complex a case it handles", what a Jev user writes) and **informed** (plus each model's
OWASP accuracy). Code `ladder/l7_jev_standin.py`, scoring `ladder/l8_eval_jev.py`,
outputs [`../results/2026-09-jev-standin/`](../results/2026-09-jev-standin/). Judge cost
0.018 GPU-s per IRIS path, included in its cost.

**It sends almost everything to Qwen3-8B** (generic: 1,672 of 2,257 IRIS paths to the
8B, 583 to the 1.7B, 2 to the 32B, 0 to the 120B; informed: 2,214 to the 8B).

| IRIS, 2,257 paths | Accuracy | TPR − FPR | AUC | recall@10 | GPU-s |
|---|---|---|---|---|---|
| Qwen3-8B alone | 0.695 | 0.164 | 0.784 | 0.603 | 0.025 |
| Qwen3-32B alone | 0.659 | 0.311 | 0.855 | 0.690 | 0.115 |
| gpt-oss-120b alone (best on OWASP) | 0.767 | 0.062 | 0.605 | 0.381 | 0.113 |
| **cascade 8B → 32B** (top 30% per project) | **0.755** | 0.258 | 0.830 | **0.691** | 0.059 |
| Jev stand-in, generic | 0.670 | 0.129 | 0.767 | 0.553 | 0.038 |
| Jev stand-in, informed | 0.695 | 0.168 | 0.775 | 0.600 | 0.044 |

It is no better than random routing with the same mix (z of accuracy −3.6 and +0.2),
slightly below the 8B alone, and well below the cascade, which reaches the 32B's
recall@10 at half its cost. On the OWASP test split it is again ≈ the 8B (0.670 /
0.701) against gpt-oss-120b's 0.905.

**Why.** A zero-shot judge sees how complicated the code looks, and most flagged code
looks "moderate", so it picks the middle tier. Which model is right on a path is not
visible that way (E2/E7/E10), and the strong model differs by benchmark (gpt-oss-120b on
OWASP, Qwen3-32B on IRIS), which a description cannot know. Caveat: this is a stand-in,
not Jev itself; Jev may judge "difficulty" better, but difficulty is not what decides
the right model here.

## 10. Next

* **Base-rate correction**: re-weight the router's P(right) for the target base rate
  (15% real on IRIS vs 65% on OWASP) and re-score; cheap, no GPU.
* **Train on a mix** (OWASP + part of IRIS) and test on held-out IRIS projects.
* **Jev** itself (paid): pending the professor. Same baselines as above.
