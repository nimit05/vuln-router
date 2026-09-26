# 2. The Strategy Zoo

Ten routing strategies, grouped into four families. Each has one primary anchor.
All implement the same interface — given units and the measurement table, emit a
model choice per unit — so they are directly comparable on one frontier.

| Family | Strategies | Decides |
|---|---|---|
| A. References | S0 fixed, S1 oracle | nothing (bounds) |
| B. Cascades | S2 self-consistency, S3 token uncertainty, S4 self-verification | *after* seeing a cheap answer |
| C. Predictive routers | S5 difficulty, S6 win-prediction, S7 M-way defer | *before* any call |
| D. Wrappers | S8 budget, S9 risk control, S10 calibration | modify B or C |

---

## Family A — References

Not strategies. They are the three lines every plot must carry, because a
routing delta is meaningless without them.

### S0 — Fixed and random baselines
`always-cheapest`, `always-strongest`, `uniform-random`, and *every* fixed model
individually. Reported as points, not a curve.

**Why every fixed model, not just the best two.** The "best" fixed model is
chosen with hindsight on the test set; naming it is already a mild oracle. Show
all of them so the reader can see how much of the router's margin is just
"picked the right constant".

### S1 — Per-unit oracle
Chooses, per unit, the model minimising `l(m,x) + lambda*c(m,x)`. Not
achievable — it reads the answer — but it is the **upper bound on everything in
families B and C**, and its cost tells you something no accuracy number does.

**Anchor.** RouterBench (Hu et al., *Agentic Markets @ ICML 2024* — a workshop
paper, cited for protocol only) makes the oracle and the cost-quality frontier
the standard reporting objects for routing systems.

**What to read off it.** If the oracle is both *better and cheaper* than the
best fixed model, quality and cost are not in tension for this pool and the
project should be framed as a frontier, not a trade-off. If the oracle is barely
above the best fixed model, the pool lacks error complementarity and no router
can rescue it — stop and change the pool (§Pool selection).

---

## Family B — Cascades (decide after a cheap call)

Run the cheap model first; escalate on a signal derived from its output. Cheap
to implement, and the dominant family in the literature.

**Shared precondition, stated once because it decides whether this family is
viable at all:** a cascade always pays for the base model, so it saves money only
when escalation is *rare*. If the trigger fires on most units, the cascade costs
more than always-strong and buys only quality. Measure the fire rate before
tuning anything.

### S2 — Answer-consistency escalation
Sample the cheap model `k` times; escalate when the verdicts disagree.

**Anchor.** Yue, Zhao, Zhang, Du & Yao, *Large Language Model Cascades with
Mixture of Thought Representations for Cost-Efficient Reasoning*, **ICLR 2024** —
uses the weak model's answer consistency as the difficulty signal, and mixes two
prompt representations (CoT / PoT) to decorrelate the votes.

**Adaptation here.** The vulnerability analogue of mixture-of-thought is to ask
the same question two ways: *"is this function vulnerable?"* and *"what CWE, if
any, does this function contain?"*. Agreement across two framings is a stronger
signal than `k` samples of one framing, and it costs the same.

**Cost note.** `k` samples make the cheap model `k`x its nominal price. This is
the clearest instance of `c(m,x)` being endogenous (problem §1.4).

### S3 — Token-level uncertainty deferral
Escalate on a learned rule over the cheap model's token-level logits, rather
than on a scalar sequence probability.

**Anchor.** Gupta, Narasimhan, Jitkrittum, Rawat, Menon & Kumar, *Language Model
Cascades: Token-Level Uncertainty and Beyond*, **ICLR 2024** — shows sequence
probability suffers a length bias and that a learned post-hoc deferral rule over
token-level quantiles beats simple aggregation.

**Why it fits.** Our output is short and structured (a label plus a CWE id), so
the relevant uncertainty is concentrated in a handful of decision tokens. That is
a favourable case for token-level rules and an unfavourable one for
sequence-level scores, which will be dominated by the boilerplate.

**Requirement.** Logprobs must be logged at probe time. Cheap, but it has to be
switched on *before* the runs — retrofitting means re-running.

### S4 — Few-shot self-verification
The cheap model verifies its own answer against the code; escalate on low
self-verification confidence, with the noisy signal handled by a POMDP-style
meta-verifier.

**Anchor.** Aggarwal, Madaan et al., *AutoMix: Automatically Mixing Language
Models*, **NeurIPS 2024**.

**Why it is worth testing here despite the cost.** Self-verification is posed as
an entailment check: *is the claimed data flow actually present in this code?*
That is checkable from the code itself, which makes it a better-posed question
than "are you confident?" and is closer to what a security reviewer does.

**Why it may fail here.** It adds LLM calls to the decision path. If the fire
rate is high (see the shared precondition), S4 is strictly dominated.

---

## Family C — Predictive routers (decide before any call)

Route from features of the code alone. Zero LLM in the decision path, so the
routing overhead is microseconds against seconds of inference.

### S5 — Difficulty-based binary router
Predict a difficulty/quality-gap score from the unit; send hard units to the
strong model, easy units to the weak one. One threshold, tunable at test time to
slide along the frontier.

**Anchor.** Ding, Mallick, Wang, Sim, Mukherjee, Rühle, Lakshmanan & Awadallah,
*Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing*, **ICLR 2024** —
routes on predicted query difficulty before generation, with the quality level
exposed as a test-time knob. Reported up to 40% fewer strong-model calls at no
quality drop.

**Adaptation here.** Their router is a BERT over the prompt. Ours reads the
*code*: cyclomatic complexity, nesting depth, pointer/alloc/free density, loop
and branch counts, function length, presence of the CWE-typical API surface
(`memcpy`, `strcpy`, `malloc`/`free` pairing), plus a frozen code-encoder
embedding (UniXcoder). See `features.py`.

### S6 — Win-prediction router with data augmentation
Train `P(strong beats weak | x)` and threshold it. The threshold *is* the cost
knob.

**Anchor.** Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous & Stoica,
*RouteLLM: Learning to Route LLMs with Preference Data*, **ICLR 2025** — 95% of
GPT-4 quality at 26% of GPT-4 calls; routers transfer to model pairs unseen in
training. Also the source of the constraint that router cost must be negligible.

**One deliberate deviation.** RouteLLM learns from *human preference* labels
because chat quality has no ground truth. Vulnerability detection does — the
label is a fact, not an opinion. So S6 is trained on realised win/loss from the
measurement table, not on preferences. Using a weaker supervision signal than the
one available would be a defect, and this is worth stating explicitly as a
difference between the domains rather than silently swapping it.

**Keep from RouteLLM:** the data-augmentation arm and the matrix-factorisation
router, both of which are what make their routers transfer across pools.

### S7 — M-way cost-sensitive learn-to-defer
Drop the binary strong/weak framing. Predict the full loss vector over the `M`
models and route `argmin_m [ h_m(x) + lambda * c_m(x) ]`.

**Anchors.** Mozannar & Sontag, *Consistent Estimators for Learning to Defer to
an Expert*, **ICML 2020** — reduces defer-or-predict to cost-sensitive learning
and supplies a consistent surrogate loss generalising cross-entropy, which is
what makes the training objective principled rather than a heuristic.
Madras, Pitassi & Zemel, *Predict Responsibly: Improving Fairness and Accuracy
by Learning to Defer*, **NeurIPS 2018** — deferral trained against the downstream
decision cost.

**Why this is the interesting one.** S5 and S6 presume a quality ordering
between two models. With `M` models whose errors are complementary but not
ordered, no such ordering exists, and the binary framing throws away most of the
pool. S7 is the only strategy in the zoo that can express *"model C is the right
one for this input, though it is neither the cheapest nor the strongest."*

**Also fits S7 and nothing else:** the cost regressor `ĉ_m(x)` from problem
§1.4 slots directly into the argmin.

---

## Family D — Wrappers

Applied on top of any strategy from B or C.

### S8 — Workload-level budget via online dual ascent
One budget for a whole scan, not a per-unit threshold. `lambda` is a dual
variable updated as the scan proceeds: spend faster when under budget, tighten
when over.

**Anchor.** Badanidiyuru, Kleinberg & Slivkins, *Bandits with Knapsacks*,
**FOCS 2013** (journal version J. ACM 65(3), 2018) — the canonical primal-dual
treatment of sequential decisions under a global resource constraint, with regret
bounds. Recent LLM-routing instances (OmniRouter's Lagrangian dual decomposition,
PILOT's online multi-choice knapsack) are the same idea applied.

**Why it matters and is not cosmetic.** The deployment request is "audit this
repository within N GPU-hours", never "spend at most X on this one function". A
fixed threshold cannot honour a global budget under a non-stationary stream, and
code streams *are* non-stationary — feature distributions differ sharply between
projects. A dual variable that adapts to realised spend is a self-calibrating
threshold, which is also the fix for thresholds that transfer poorly across
corpora.

**Caveat to state up front.** These regret bounds are asymptotic. A short scan
may not give the dual time to converge, in which case S8 degrades to a fixed
`lambda` and should be reported as such.

### S9 — Risk-controlled routing
Wrap the router in a distribution-free guarantee: *with probability >= 1-delta,
the router's false-negative rate exceeds always-strong's by at most alpha.*

**Anchors.** Angelopoulos, Bates, Fisch, Lei & Schuster, *Conformal Risk
Control*, **ICLR 2024 (spotlight)** — extends conformal prediction from coverage
to any monotone risk, which is exactly the FNR case. Quach, Fisch, Schuster,
Yala, Sohn, Jaakkola & Barzilay, *Conformal Language Modeling*, **ICLR 2024** —
the recipe for calibrating an abstention/stopping rule around LLM outputs.

**Why this is the component that makes the work adoptable.** No cost-aware
router in the literature ships a bound on missed detections. A security team
cannot deploy a 40%-cheaper scanner on the strength of a mean F1 delta; it can
deploy one that bounds the extra vulnerabilities it will miss. Report the price
of the guarantee — how much of the saving is handed back to buy it.

### S10 — Calibration
Temperature-scale any router head before thresholding it.

**Anchor.** Guo, Pleiss, Sun & Weinberger, *On Calibration of Modern Neural
Networks*, **ICML 2017**.

**Why it is not optional.** S5, S6 and S9 all threshold a predicted probability.
Modern predictors are systematically over-confident, so an uncalibrated threshold
of 0.7 does not mean 70% and the frontier it traces is mislabelled. Report
reliability diagrams before and after; this is one line of code and it removes an
entire class of reviewer objection.

---

## Pool selection is itself a strategy decision

Not a strategy in the zoo, but it belongs here because it dominates all of them.

The pairwise oracle over every model pair (cheap to compute from the measurement
table, no extra inference) reveals which pairs have complementary errors. A pair
with a high pairwise oracle is worth routing between; a pair whose oracle barely
exceeds its better member is not, and no strategy in families B or C will make it
so.

Report this matrix **before** reporting any router. It converts "our router
works" into "our router works because this pool has complementary errors, and
here is the evidence that it does" — and it forewarns the reviewer question of
whether the result is a property of the method or of one lucky checkpoint.
