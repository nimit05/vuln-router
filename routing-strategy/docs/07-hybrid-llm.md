# Hybrid LLM: the objective is right and the router cannot reach it

Hybrid LLM (Ding et al., **ICLR 2024**) run on the IRIS candidate stream, same
2,259 dataflow paths and same 16 projects as the GraphRouter track, scored by
IRIS's own `metrics()`.

The short version: this is **the opposite failure to GraphRouter**. There, the
target itself was worthless on a skewed population and no amount of training
helped (`06-objective-misalignment.md`). Here the target is excellent —
following it perfectly nearly doubles the best single model and beats the
published GPT-4 row at a quarter of the compute — and the router cannot predict
it across projects at better than chance.

## 1. What a perfect router on this objective would win

Quality per (path, model) is the asymmetric security loss of §1.5, so
`q = 1` for a correct verdict, `0` for a missed vulnerability, `1 - 1/rho` for
a false alarm. `det_2cls` then labels each path with whichever model is cheaper
to be right with.

Computing that label from the **ground-truth** path labels and following it
gives the **label-following** row. It is an oracle: it reads the answer, it is
not achievable at inference, and it is reported for one reason only — it is the
ceiling this objective permits, so comparing a trained router against it says
whether the objective or the predictor is the binding constraint. It differs
from the pair oracle below because quality is the *expected* quality under each
model's own verdict confidence, not its hard verdict, so the two disagree on
paths where a model is right but unsure.

| Configuration | %routed large | GPU-s | #Det | AvgFDR% | AvgF1 |
|---|---|---|---|---|---|
| no filter, all 2,259 paths | — | — | 10/16 | 82.12 | 0.212 |
| always-qwen-1.5b | 0.0 | 245.5 | 3/16 | 87.18 | 0.105 |
| always-granite-8b | 100.0 | 1943.6 | 10/16 | 82.71 | 0.207 |
| **label-following (perfect router)** | **15.0** | **512.4** | **10/16** | **58.88** | **0.393** |
| pair oracle (upper bound) | — | — | 10/16 | 57.65 | 0.405 |
| paper: IRIS + GPT-4 | — | — | 10/16 | 65.69 | 0.366 |

Three things to note. The label-following row is close to the pair oracle, so
Hybrid LLM's target is nearly the best target available for this pair. It beats
the published GPT-4 row on both false-discovery rate and F1. And it does so
sending only 15% of paths to the larger model, for 26% of always-granite's
GPU-seconds.

**Why it works where GraphRouter's did not.** GraphRouter supervises with
`argmax` over five models, which on an 85%-negative population hands the label
to whichever model is right on the negative majority and discards granite-8b,
the one model that keeps true positives. Hybrid LLM supervises with a *pairwise
comparison* against an asymmetric quality, so a path where the small model
would drop a real vulnerability labels "large" no matter how the rest of the
population behaves. The base rate cannot swamp a per-path comparison the way it
swamps a per-path argmax.

## 2. Where the ceiling sits for every pair

Ten ordered pairs, small strictly cheaper in measured GPU-seconds.
Full table with the fits in `RESULTS_HYBRIDLLM.md`.

| small → large | AvgF1 small | AvgF1 large | AvgF1 label-following | AvgF1 pair-oracle |
|---|---|---|---|---|
| qwen-7b → granite-8b | 0.142 | 0.207 | **0.405** | 0.445 |
| qwen-1.5b → granite-8b | 0.105 | 0.207 | **0.393** | 0.405 |
| qwen-7b → deepseek-6.7b | 0.142 | 0.223 | 0.386 | 0.412 |
| qwen-1.5b → deepseek-6.7b | 0.105 | 0.223 | 0.338 | 0.358 |
| qwen-1.5b → phi-3.8b | 0.105 | 0.202 | 0.290 | 0.400 |
| phi-3.8b → qwen-7b | 0.202 | 0.142 | 0.283 | 0.423 |
| phi-3.8b → deepseek-6.7b | 0.202 | 0.223 | 0.282 | 0.295 |
| granite-8b → deepseek-6.7b | 0.207 | 0.223 | 0.257 | 0.260 |
| phi-3.8b → granite-8b | 0.202 | 0.207 | 0.256 | 0.256 |
| qwen-1.5b → qwen-7b | 0.105 | 0.142 | 0.153 | 0.153 |

The pattern is consistent: every pair whose large member is granite-8b or
deepseek-6.7b clears the best single model by a wide margin, and the ordering
tracks how *complementary* the two models are rather than how good either is.
`phi-3.8b → granite-8b` pairs the two strongest single models and gains almost
nothing, which is gate G3 of §1.3 restated — errors must be complementary, not
merely unequal.

## 3. The router cannot find it

Trained leave-one-project-out on the code slice, `det_2cls` at `t = 0`.
AUC is against the router's own label.

**Two backbones, one pair** (`qwen-1.5b → granite-8b`):

| split | TF-IDF + logistic regression | structural features + GBM |
|---|---|---|
| random 5-fold, **inside** projects (leaky) | 0.955 | 0.943 |
| leave-one-project-out | 0.483 | 0.491 |

**All ten pairs**, TF-IDF + logistic regression:

| small → large | AUC LOPO | AUC random (leaky) |
|---|---|---|
| qwen-7b → granite-8b | 0.471 | 0.943 |
| qwen-1.5b → granite-8b | 0.483 | 0.955 |
| qwen-7b → deepseek-6.7b | 0.541 | 0.726 |
| qwen-1.5b → deepseek-6.7b | 0.528 | 0.714 |
| qwen-1.5b → phi-3.8b | 0.380 | 0.861 |
| phi-3.8b → qwen-7b | 0.471 | 0.869 |
| phi-3.8b → deepseek-6.7b | 0.502 | 0.726 |
| granite-8b → deepseek-6.7b | 0.551 | 0.672 |
| phi-3.8b → granite-8b | 0.485 | 0.944 |
| qwen-1.5b → qwen-7b | 0.507 | 0.909 |
| **median** | **0.494** | **0.865** |

Every backbone, every pair, same story. Within a project the label is
predictable and often nearly perfectly so. Across projects **not one pair of
ten clears 0.551**, the median is 0.494, and five of the ten sit below chance.

The threshold sweep degenerates accordingly, under **both** of upstream's losses: every
operating point between always-small and always-large is worse than
always-large, and the best point is `tau = 1.00`, which is not routing.

| loss | pair | best tau | AvgF1 at best | always-large AvgF1 |
|---|---|---|---|---|
| `det_2cls`, t=0 | qwen-1.5b → granite-8b | 1.00 | 0.207 | 0.207 |
| `prob_2cls`, t=0 | qwen-7b → granite-8b | 1.00 | 0.207 | 0.207 |

The probabilistic variant is the one with the extra information — it sees each
verdict's confidence, not just its sign — and it does not help. Full sweeps in
`RESULTS_HYBRIDLLM_PAIR.md`.

**This is not a capacity problem.** A weak backbone underfits both splits, and
these do not: the same model on the same features reaches a median 0.865 when
the split is allowed to leak. A gap that large means the thing being learned is
the *repository*, not the path. Two unrelated feature families — character
n-grams over Java source, and ten structural counts — land within 0.01 of each
other on both sides of the split, which is what you see when the ceiling is
information rather than parameters.

The pairs whose leaky AUC is lowest (0.67 to 0.73) are exactly the four
involving deepseek-6.7b, the one model whose verdicts are near-uniform on this
population. Where there is less to memorise, there is less apparent skill.

**The consequence for the literature is the part that generalises.** A Hybrid
LLM number on a vulnerability benchmark with an alert-level or function-level
random split would report roughly 0.95 AUC and a large apparent win. It would
be measuring near-duplicate paths from the same repo sitting on both sides of
the split. Protocol §4 predicted this in the abstract; this is the measurement.

## 4. What is NOT claimed

* Not that Hybrid LLM fails. Its objective is the best of the routers tried on
  this task, by a wide margin, and the ceiling it defines is real.
* **Not that 0.393 is a result.** Every number in §1 above the sweep is an
  oracle computed from ground truth. The only achievable Hybrid LLM numbers in
  this document are in §3, and they are at chance.
* Not that a DeBERTa backbone would not help. It is checkpoint H5 and it is
  unrun. The two-backbone agreement above is evidence against, not proof.
  Fourteen projects is a small universe and the honest statement is that the
  linear probes find nothing transferable.
* Not that per-project aggregation is solved. Hybrid LLM supervises per path
  like everything else. The label-following row detects the same 10 projects as
  always-granite; it wins entirely on false-discovery rate.
* Not a tuned result. Upstream's label arithmetic is copied unmodified in
  `hybridllm/hl_labels.py`; only the quality definition and the sampling of it
  are ours, and both are argued in `hybridllm/CHECKPOINTS.md`.

## 5. The two adaptations, restated for a reviewer

**Quality.** No reference answer exists for "is this path a real
vulnerability", so BARTScore is replaced by `q = 1 - l/c_FN` over the asymmetric
security loss. At `rho = c_FN/c_FP = 1` this is accuracy and reproduces the
paper's symmetric setting exactly.

A consequence worth stating plainly: quality then takes three values whose
*order* is independent of rho, so `det_2cls` at `t = 0` is **invariant to the
false-negative price**. Upstream's tolerance `t` is the only channel through
which asymmetric cost reaches the objective at all. That is a limitation of the
method on cost-asymmetric tasks, and it is not visible on the paper's own
benchmarks because there quality is continuous.

**Sampled quality.** `prob_2cls` needs a distribution of quality per model.
The probe recorded one greedy verdict plus its token log-probability, so
`p_vuln` is read from that and the samples are laid down deterministically.
Cheaper than re-running the pool and available for any open-weight model. A row
with no parsed verdict is scored at `p = 0.5`, never as a confident benign —
that is defect 1 of `05-probe-defects.md` and it is what made
`data/gr/probe5` the required snapshot.
