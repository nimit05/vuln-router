# 3. Evaluation Protocol

The protocol is written before any measurement exists, and the gates in §3 are
pre-registered: each has a stated outcome for pass and for fail.

## 1. Never select on the data you score on

The single most common way a routing result evaporates. Selecting the best model
per class, or the best rule from a search, using the same data it is then
reported on, measures an **oracle** and not a router. A deployed router must
decide before seeing outcomes.

**Rule.** Fit split -> select -> freeze -> score once on held-out. Any rule that
touches held-out data during selection is reported as an oracle and labelled as
one.

**Selection method: stability selection**, not top-1 on the fit split.
Bootstrap-resample the fit split 30-40 times, keep the rule *family* that recurs
in the top-k, not the single best-scoring rule. A top-1 rule chosen on a fit
split is fitting that split's noise; a family that survives resampling is not.
Anchor: Meinshausen & Bühlmann, *Stability Selection*, JRSS-B 2010.

## 2. Report a frontier, with three references

The reporting object is the **cost-quality curve** over `lambda`, summarised by
its area, not a single operating point.

Every plot carries: `always-cheapest`, `always-strongest`, **all** other fixed
models as points, and the **per-unit oracle** (S1). The oracle is what makes a
small delta interpretable — "+0.02 F1" means something different when the oracle
headroom is +0.03 than when it is +0.15.

Anchors: RouterBench (Agentic Markets @ ICML 2024, workshop — protocol only) for
the frontier/AIQ convention; Hybrid LLM (ICLR 2024) for the test-time-tunable
quality knob that generates the curve.

## 3. Pre-registered gates

Run these in order. Each is cheap and each can kill the project early, which is
the point.

**G1 — Does the pool have complementary errors?**
Compute the pairwise oracle matrix over all model pairs (§2 of the strategy doc).
*Pass:* at least one pair's oracle materially exceeds its better member.
*Fail:* the pool is effectively ordered; no router can win. Change the pool
before writing any router code — and report the negative, because "we checked
and there was nothing to route" is a real finding about a model family.

**G2 — Is cost actually endogenous?**
Regress `c(m,x)` on unit features per model and compare against the constant
`c(m)` baseline.
*Pass (R² materially > 0):* the standard fixed-$/token formulation is wrong for
this workload and the cost regressor is a contribution in its own right.
*Fail (R² ~ 0):* adopt the standard formulation, drop the regressor, say so.
Either way the answer is reported — this is the cheapest novel measurement in
the project.

**G3 — Do cascades escalate rarely enough to save money?**
Measure the fire rate of each escalation trigger (S2-S4) before tuning.
*Pass (low fire rate):* family B is viable.
*Fail (high fire rate):* cascades cost more than always-strong and buy only
quality. Report them as quality baselines and make family C the primary.

## 4. Splits

**Group-aware, never random.** Split by commit, then by project, then
temporally — never by function. Functions from one commit are near-duplicates;
a random function-level split puts them on both sides and every strategy scores
beautifully for the wrong reason.

Anchor: Allamanis, *The Adverse Effects of Code Duplication in Machine Learning
Models of Code*, Onward!/SPLASH 2019. PrimeVul's own splits already enforce
this, which is a further reason to prefer it (problem §1.2).

**Repeated splits, spread shown.** Report over >= 10 group splits with the
spread, and name one pre-registered draw, reported *first*. A single holdout is
too high-variance to evaluate a router on; a mean that hides a losing draw is
not honest reporting.

## 5. What must be logged at probe time

Retrofitting any of these means re-running every model. Switch them all on
before the first job.

```
unit_id, project, commit, cwe, split_group,
model, engine, engine_version, hardware, job_id,
pred_label, pred_cwe, gold_label, gold_cwe,
seconds, in_tokens, out_tokens,
logprobs_decision_tokens,      # required by S3
n_samples, sample_verdicts,    # required by S2
self_verify_score,             # required by S4
n_retries, parse_failures      # required by G2 (cost endogeneity)
```

## 6. Reproducibility constraints

**The serving engine is part of the configuration.** Two engines running
identical weights, prompts and greedy decoding do not produce identical outputs —
kernel numerics differ, and at temperature 0 a near-tie flips a token. Record
engine and version; never merge runs across engines into one table.

**One benchmark is not a generalisation claim.** PrimeVul alone supports "this
works on PrimeVul". A second benchmark is required before any claim about
vulnerability detection in general — and it should differ in language or in
granularity, not just in sample.

## 7. Statistics

Bootstrap CIs on the router's margin against the best fixed model, over units,
with the interval reported alongside every headline delta. Before spending GPU
hours to grow a set, check whether the interval already excludes zero — widening
`n` to narrow a CI that already answers the question is wasted compute.
