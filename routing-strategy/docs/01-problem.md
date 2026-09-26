# 1. Problem

## 1.1 Task

**Unit of work.** One *code unit* `x` — a C/C++ function, together with the
commit metadata that accompanies it. The LLM is asked a single question:
*is this function vulnerable, and if so under which CWE?*

**Why function-level and not repository-level.** Function-level is the
granularity at which a routing decision is (a) frequent enough to learn from —
tens of thousands of decisions rather than tens, and (b) cheap enough that the
routing overhead is negligible against the call it controls. Repository-level
routing yields too few decision points to fit or validate anything, and the
per-decision cost variance swamps the signal.

## 1.2 Benchmark

**Primary: PrimeVul** (Ding et al., *Vulnerability Detection with Code Language
Models: How Far Are We?*, ICSE 2025). ~7k vulnerable and ~229k benign C/C++
functions over 140+ CWEs.

**Why this one, and why not the obvious alternatives.** PrimeVul exists because
BigVul, Devign and DiverseVul are broken in ways that would silently invalidate
any routing result:

| Defect in prior datasets | Why it destroys a routing study |
|---|---|
| Label accuracy 24-60% | A router trained on `l(m,x)` is fitting label noise, not model competence. The whole method reduces to noise-fitting. |
| High duplicate rate across splits | Near-identical functions on both sides of a split make every strategy look good and the oracle look reachable. |
| Random splits | Test functions come from the same commits as training ones — the strongest form of the leak. |

The headline number that makes this concrete: a 7B model scores **68.26% F1 on
BigVul and 3.09% F1 on PrimeVul**. Any router evaluated on BigVul is measuring
memorisation.

**The paired subset is the reason to prefer PrimeVul specifically.**
`primevul_test_paired.jsonl` contains each vulnerable function alongside its
patched version. A model that pattern-matches "this code looks dangerous" scores
well on unpaired F1 and near-zero on pairs. For routing this is exactly the
discriminating signal we need — it separates models that *reason about the
defect* from models that *react to surface form*, and those are precisely the
models worth routing between.

**Metrics.** Report all three:

* **F1** — comparability with the literature, nothing more.
* **VD-Score** — false-negative rate subject to a false-positive rate budget
  `r` (PrimeVul's own metric; `r` small, e.g. 0.005). This is the deployment
  metric: *how many real vulnerabilities do we miss while keeping the alert
  queue tolerable?*
* **Pair-wise correct prediction** — fraction of (vulnerable, patched) pairs
  where both are labelled correctly.

**Loader is pluggable** (`data.py`). A second benchmark is required before any
generalisation claim (protocol §6); PrimeVul is the default, not a commitment.

## 1.3 Model pool

Open-weight models served locally, spanning roughly two orders of magnitude of
compute. The pool is defined in [`../configs/pool.yaml`](../configs/pool.yaml)
and is a *variable of the study*, not a fixed constant — §2 of the strategy doc
argues that pool composition is itself a routing decision.

Design requirements on the pool:

1. **Span the cost axis.** ~0.5B through ~32B, so a frontier exists to trace.
2. **Include a same-size, different-family pair.** Otherwise family effects and
   scale effects are confounded and no claim about "small models" is separable
   from "this particular checkpoint".
3. **Errors must be complementary, not merely unequal.** If model B is worse
   than A everywhere, routing between them can only lose. This is checkable
   before any router is fitted (protocol §3) and is the first gate the project
   passes or fails.

## 1.4 Cost model

**Primary axis: GPU-seconds** on fixed hardware with a fixed serving
configuration. Not tokens — token counts are not comparable across tokenizer
families, and the thing a budget is denominated in is machine time.

**Secondary axis: dollars**, derived under a published price sheet, reported
only for comparability with the routing literature.

**Cost is measured per unit, not assumed per model.** Every routing paper —
Hybrid LLM (ICLR 2024), RouteLLM (ICLR 2025), FrugalGPT (TMLR 2024) — treats
cost as a constant $/token per model, so that `c(m,x) = c(m)`. That assumption
is safe for a single chat completion and unsafe the moment the model sits inside
a loop with a retry, a verifier, or a self-consistency vote: a weaker model then
consumes *more* compute, because it fails more and retries more. Strategies S2-S4
below are exactly such loops. So `c(m, x)` is recorded per (model, unit) and
predicted, never looked up.

Whether `c(m,x)` is materially non-constant is an **empirical question this
project answers early** (protocol §3, gate G2). If it is constant, the standard
formulation is correct and we adopt it; if not, that is a contribution.

## 1.5 Objective

With `l(m,x)` the security loss and `c(m,x)` the realised cost:

```
   l(m, x)  =  c_FN * FN(m,x)  +  c_FP * FP(m,x)          c_FN >> c_FP

   min_pi  E_x [ l(pi(x), x) ]     s.t.   E_x [ c(pi(x), x) ]  <=  B

<=>  min_pi  E_x [ l(pi(x), x)  +  lambda * c(pi(x), x) ]
```

Two commitments follow.

**Asymmetric loss, not accuracy.** A missed CVE and a spurious alert are not the
same event. Generic routers optimise accuracy or a preference win-rate, both of
which assume symmetry. `c_FN/c_FP` is swept and reported as a curve, not fixed
at one value — the same discipline as risk-coverage reporting in selective
classification (Geifman & El-Yaniv, NeurIPS 2017).

**Frontier, not a point.** `lambda` sweeps from always-cheapest to
quality-optimal. A single operating point is not a result; the frontier and its
area are (protocol §2).
