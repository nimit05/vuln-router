# 13 — The cascade router, checked without hindsight (2026-10-01)

Every routing family was compared in [`12-owasp-ladder.md`](12-owasp-ladder.md): learned
routers win inside a benchmark but learn its base rate and fail on a new one; a zero-shot
Jev-style router behaves like "always the medium model". On IRIS only the confidence
cascade helped (E20). This checks the cascade the way a new deployment would meet it:
no setting chosen after seeing the evaluation data. Script `ladder/l9_cascade.py`,
output [`../results/2026-10-cascade/cascade.txt`](../results/2026-10-cascade/cascade.txt).
CPU only, on the saved answers of the 12 models.

**The router.** A cheap model scores every alert. In each project, the share *f* of
alerts it finds most suspicious also go to a strong model, whose answer replaces the
cheap one. Cost per alert = cheap GPU-s + f × strong GPU-s. If that is not cheaper than
the strong model alone, use the strong model alone.

**recall@10**: an analyst reads each project's 10 highest-scored alerts; the share of
the project's real bugs among them, averaged over projects with a real bug.

## a) Choosing f without hindsight (IRIS, Qwen3-8B → Qwen3-32B)

f is chosen on the other projects (smallest f whose recall@10 is within 0.01 of sending
everything to the 32B) and applied to the held-out project; the two antisamy projects
(same repo) are held out together.

| IRIS, 2,257 paths | recall@10 | AUC | TPR − FPR | Accuracy | GPU-s |
|---|---|---|---|---|---|
| Qwen3-8B on everything | 0.603 | 0.784 | 0.164 | 0.695 | 0.025 |
| Qwen3-32B on everything | 0.690 | 0.855 | 0.311 | 0.659 | 0.115 |
| **cascade, f from held-out rule (29% sent up)** | **0.648** | 0.818 | 0.252 | **0.754** | **0.059** |
| random escalation, same count per project | 0.570 | | | | |

* The rule picked f = 0.30 for 12 of 13 held-out repos (0.10 for zt-zip): stable.
* The cascade beats random escalation (z +2.0) and closes about half the gap between the
  8B and the 32B at half the 32B's cost; its accuracy (0.754) is above both models.
* **Correction to E20:** "matches the 32B at half the cost" (0.677 vs 0.678) used an f
  chosen after seeing the curve. Without hindsight it is 0.648 vs 0.690.

## b) Choosing the model pair from a few labelled alerts (IRIS)

A new deployment labels k alerts, keeps models with AUC ≥ 0.65 (pairs compared only
inside the same project, see below), cheap = cheapest kept, strong = highest AUC kept,
f = 0.30. The k alerts come from a random half of the repos; the cascade is scored on the
other half; 300 draws per k.

| k labelled | picks 8B → 32B | calibrated cascade | known pair (8B → 32B) | pair picked on OWASP (1.5B → 120B) | 32B alone |
|---|---|---|---|---|---|
| 25 | 4% | 0.452 | 0.694 | 0.455 | 0.694 |
| 50 | 15% | 0.516 | 0.696 | 0.465 | 0.696 |
| 100 | 13% | 0.515 | 0.670 | 0.436 | 0.669 |
| 200 | 25% | 0.588 | 0.699 | 0.464 | 0.699 |
| 400 | 45% | 0.662 | 0.731 | 0.498 | 0.731 |

(recall@10 on the unseen half, mean over draws.)

* Picking the pair from target labels beats picking it from another benchmark once
  k ≥ 50, but it needs a few hundred labels (≈ 60 real bugs at IRIS's 15%) before it
  usually finds the right pair, and even at k = 400 it trails the known pair by 0.07.
* Taking the pair from OWASP is the worst choice throughout (0.44-0.50): the model that
  is strongest on OWASP (gpt-oss-120b) is weak on IRIS.
* **Method note.** A first version scored each model's skill by pooling the k alerts
  across projects. That rewards a model for scoring ESAPI (50% real) above ff4j (13%
  real) rather than for ranking alerts, and it picked Qwen2.5-Coder-1.5B alone in a
  quarter of the draws. Only within-project pairs are used now (Simpson's paradox, as
  in E13/E14).

## c) The same method on OWASP

Pair from k labelled train units, f chosen on train, scored on the 588 test units.

| OWASP test | Accuracy | AUC | GPU-s |
|---|---|---|---|
| gpt-oss-20b on everything | 0.869 | 0.875 | 0.061 |
| gpt-oss-120b on everything | 0.905 | 0.892 | 0.156 |
| calibrated cascade, k = 100 (mostly 1.5B → 120B) | 0.875 | 0.879 | 0.117 |
| calibrated cascade, all train labels (1.5B → 120B) | 0.872 | 0.880 | 0.153 |

* No gain on OWASP: the cascade costs about as much as gpt-oss-120b and is less accurate,
  and gpt-oss-20b alone is nearly as good at 40% of the cost. Where a strong model is
  already cheap, the right router is that model alone.
* The pick rule chooses the highest-AUC model as "strong" without looking at cost
  (gpt-oss-120b 0.892 over gpt-oss-20b 0.875 at 2.5× the cost). A cost-aware pick is the
  obvious next variant; it was not tried here to avoid tuning rules on these results.

## What this gives the thesis

1. **The signal that works is the cheap model's own confidence on the alert.** Across
   all families, only the cascade survives a new benchmark (IRIS: 0.648 recall@10 at
   half the 32B's cost, above random escalation), while learned and zero-shot routers do
   not (E24, E25).
2. **The router must be calibrated on the target.** The best model changes between
   benchmarks; choosing it from another benchmark is the worst option measured. A few
   hundred labelled target alerts are needed to choose reliably.
3. **Honest limits.** The cascade recovers about half of the 8B → 32B gap without
   hindsight, not all of it; on OWASP it gives nothing because a cheap strong model
   exists.
