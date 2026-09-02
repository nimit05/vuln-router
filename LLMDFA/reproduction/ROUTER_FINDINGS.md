# Router Findings — Per-Bug-Type Model Routing on LLMDFA

Run dates: **2026-08-26 / 27**. Hardware: IIT Guwahati CSE cluster (`csecluster`),
SLURM, NVIDIA **A100-PCIE-40GB**, models served with **vLLM 0.27.1** in **bfloat16**
inside `~/containers/vllm.sif`.

Companion to [`FINDINGS.md`](FINDINGS.md), which covers the single-model LLMDFA
reproduction. This document covers the follow-on question: **can we pick a cheaper
model per bug type and come out ahead?**

---

## TL;DR

We measured **four open-weight models across all three LLMDFA bug types** (2,961
benchmark programs, ~13,000 case-analyses, ~40 GPU-hours) and then tested a
per-bug-type router on a pre-registered held-out split.

**The router lost.** Fitted honestly, it degenerated to "always use Mistral-Nemo"
and was beaten by a single fixed model — always-Phi-4-mini — on **both** quality
and cost.

| Held-out evaluation (458 unseen cases) | Avg F1 | GPU-sec |
|---|---|---|
| Oracle (upper bound, not achievable) | **0.825** | 8,671 |
| **always Phi-4-mini 3.8B** | **0.792** | **7,694** |
| **ROUTER (fitted on select variants)** | 0.787 | 12,188 |
| always Mistral-Nemo 12B | 0.787 | 12,188 |
| always Qwen2.5-Coder 7B | 0.782 | 7,523 |
| always Granite-3.1 8B | 0.729 | 15,382 |

The oracle's 0.825 shows **real headroom exists** (+0.033 F1 over the best fixed
model). Per-bug-type granularity simply cannot reach it — there are only three
decisions to make, and one of them is a coin flip.

**This is a negative result, and it is the honest one.** An earlier version of this
analysis reported the router *winning* (+0.013 F1, −24% cost). That number was
produced by selecting the best model per bug type **using the same data it was then
scored on**. It is an oracle, not a router. It is recorded in the "Discarded result"
section below so the mistake is not repeated.

---

## 1. What we set out to test

LLMDFA decomposes dataflow analysis into three LLM-driven phases (paper §3):

| Phase | What the LLM does | Frequency |
|---|---|---|
| I — Source/Sink Extraction | writes a tree-sitter script | once per bug type |
| II — Dataflow Summarization | few-shot CoT: "does A reach B?" | per candidate path |
| III — Path Feasibility Validation | writes a Z3 script, repairs it ≤3× | per source-sink pair |

The paper's own Table 1 shows no single LLM dominates: gpt-4 wins DBZ and XSS,
claude-3 wins OSCI. If that holds for open models, a router that picks per bug type
should beat any fixed choice.

**Hypothesis:** picking the best model per bug type beats always using one model,
on quality, cost, or both.

---

## 2. Setup

### Models

Four families, deliberately including a **7B/8B same-size pair** so that family
effects can be separated from size effects.

| Model | Params | Family | License |
|---|---|---|---|
| `microsoft/Phi-4-mini-instruct` | 3.8B | Microsoft | MIT |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | Alibaba | Apache-2.0 |
| `ibm-granite/granite-3.1-8b-instruct` | 8B | IBM | Apache-2.0 |
| `mistralai/Mistral-Nemo-Instruct-2407` | 12B | Mistral | Apache-2.0 |

`Qwen2.5-Coder-7B` results are reused from the earlier single-model reproduction.
`deepseek-coder-6.7b` was excluded: vLLM returns undetokenized byte-level BPE
(`ĠĊ`) for that tokenizer, which scores as total failure while the model answers
correctly. `--tokenizer-mode slow` does not fix it.

**Cross-family caveat.** Different tokenizers mean raw token counts are not
comparable across families. The cost axis throughout this document is therefore
**GPU-seconds per case**, measured on identical hardware with identical serving
config. Token counts are reported for within-family reference only.

### Benchmark

Obfuscated Juliet Test Suite as shipped in the authors' repo.

| Bug type | Programs | Coverage |
|---|---|---|
| XSS | 666 | full |
| OSCI | 444 | full |
| DBZ | **300 of 1,851** | sorted prefix (see §3.2) |

---

## 3. Methodology

### 3.1 Case-level matching

`score_llmdfa.py` reports one aggregate number per (model, bug type). That is enough
to reproduce the paper but **not** enough to compare models fairly or to split data,
because it discards which program each result came from.

LLMDFA does log the filename — on the line immediately before each result dict:

```
Start to analyze the case  1 out of  666
/…/CWE80_XSS/s01/CWE80_XSS__CWE182_Servlet_File_01.java
{'input_token_cost': 2036, 'analysis_result': {...}, 'ground_truth': {...}, …}
```

[`scripts/score_split.py`](../scripts/score_split.py) pairs each result with its
`.java` path, which makes three things possible that were not before:

1. **Matched comparison** — score every model on the *identical* set of programs.
   Used to fold Qwen's existing 1,851-case DBZ run into the 300-case comparison
   without re-running it (300/300 coverage confirmed).
2. **Grouped splits** — see §3.3.
3. **Partial-run scoring** — a job cancelled mid-flight still yields a valid
   comparison on whatever subset it completed.

It reproduces the published aggregate numbers exactly (XSS 0.950, OSCI 0.877),
which is how we know the pairing is correct.

### 3.2 Why DBZ uses a sorted 300-case prefix

DBZ is 1,851 cases at ~40–100 s/case — roughly 20–50 GPU-hours *per model*. We
sampled.

Juliet is a **complete Cartesian grid**: every source/sink FORM appears with every
control-flow VARIANT.

```
CWE80_XSS__CWE182_Servlet_File_01.java
             ^form              ^variant

XSS  = 18 forms × 37 variants = 666
OSCI = 12 forms × 37 variants = 444
```

Sorted order groups a form's variants contiguously, so a **prefix of 300 cases
covers ~8 complete forms with all 37 variants present**. Since the paper states
(§4.2) that programs "only differ … in terms of sources and sinks" — i.e. the form
is interchangeable and the variant carries the difficulty — a sorted prefix is a
form-stratified sample with full coverage of the axis that matters. It is also
trivially reproducible and shards cleanly in half.

**Caveat:** this subset is *harder* than the full benchmark. Qwen scores 0.547 on
the 300 vs 0.609 on all 1,851. Absolute DBZ numbers here are therefore not
comparable to the paper's; cross-model comparisons are, since every model sees the
same 300.

### 3.3 The train/test split — and why it is by variant, not random

A random case-level split would be **leakage**. Because of the Cartesian grid,
`Servlet_File_01` in train and `Servlet_File_02` in test are near-identical
programs. Such a split scores beautifully and proves nothing.

We split by **control-flow variant**: 12 of 37 held out, chosen by a fixed seed
(`SPLIT_SEED = 20260826`), the same variants for every model. Pre-registered before
any model was scored.

This is the axis the paper's own error analysis (Appendix A.4.2) identifies as
carrying difficulty, naming three variants that break the Z3 stage in three
different ways:

| Variant | Construct | Failure mode |
|---|---|---|
| `_09` | `Math.abs(data) > 0.000001` | encodes `And(...)` instead of `Or(...)` |
| `_12` | `staticReturnsTrueOrFalse()` | nondeterministic call read as a constant |
| `_22a` | cross-file static field | global value not tracked |

**Note:** the DBZ held-out set happens to contain `_09`, `_12` and `_22` — all three
paper-documented hard variants. The DBZ held-out set is therefore adversarially
hard, and every model's F1 drops on it. This was not engineered; it fell out of the
pre-registered seed. It is disclosed because it materially affects interpretation.

Split composition:

```
XSS / OSCI  select : 01 02 06 08 09 11 12 14 15 16 21 22 41 42 45 51 52 53 54 61 66 67 71 72 81
XSS / OSCI  held   : 03 04 05 07 10 13 17 31 68 73 74 75
DBZ         select : 00 01 06 07 08 10 11 14 15 16 17 21 31 41 42 45 51 52 53 54 61 66 68 71 72 81
DBZ         held   : 02 03 04 05 09 12 13 22 67 73 74 75
```

---

## 4. Results — full benchmark

Precision / Recall / F1, and GPU-seconds per case. Identical programs within each
bug type for every model.

### XSS — 666 cases

| Model | Prec | Recall | F1 | s/case | tok in | tok out |
|---|---|---|---|---|---|---|
| Mistral-Nemo 12B | 97.55% | 95.80% | **0.967** | 11.6 | 5,549 | 564 |
| Qwen2.5-Coder 7B | 97.47% | 92.64% | 0.950 | **9.1** | 6,283 | 696 |
| Granite-3.1 8B | 97.46% | 92.34% | 0.948 | 12.6 | 6,924 | 827 |
| Phi-4-mini 3.8B | **98.14%** | 79.43% | 0.878 | 9.6 | 6,342 | 886 |
| *paper gpt-3.5* | *100%* | *92.31%* | *0.96* | — | — | — |

### OSCI — 444 cases

| Model | Prec | Recall | F1 | s/case |
|---|---|---|---|---|
| Mistral-Nemo 12B | 84.60% | 95.27% | **0.896** | 8.4 |
| Qwen2.5-Coder 7B | 83.95% | 91.89% | 0.877 | 6.5 |
| Phi-4-mini 3.8B | **86.10%** | 78.15% | 0.819 | **5.0** |
| Granite-3.1 8B | 84.47% | 72.30% | 0.779 | 7.8 |
| *paper gpt-3.5* | *100%* | *78.38%* | *0.88* | — |

### DBZ — 300 matched cases

| Model | Prec | Recall | F1 | s/case |
|---|---|---|---|---|
| Phi-4-mini 3.8B | **50.80%** | 74.33% | **0.604** | 46.0 |
| Mistral-Nemo 12B | 40.96% | **91.33%** | 0.566 | 72.8 |
| Qwen2.5-Coder 7B | 39.41% | 89.33% | 0.547 | **38.3** |
| Granite-3.1 8B | 37.32% | 88.33% | 0.525 | 97.7 |
| *paper gpt-3.5 (full 1,851)* | *73.75%* | *92.16%* | *0.82* | — |

### Four observations

**1. The rankings invert.** Mistral-Nemo wins both easy bug types and comes third
on the hard one. Phi-4-mini is last on both easy types and **first** on DBZ. The
cheapest model wins the hardest case.

**2. Cost is not monotonic in size.** Phi-4-mini (3.8B) is *slower* than Qwen (7B)
on XSS — 9.6 vs 9.1 s/case — because it emits 27% more output tokens. Granite (8B)
is the most expensive model on DBZ at 97.7 s/case, 2.5× Qwen, while scoring worst.
**"Smaller model = cheaper" is false on this workload.**

**3. Granite is dominated everywhere.** It never appears on a Pareto frontier for
any bug type. The mechanism is measured: **22.2 Z3 refinement rounds per case**
versus Phi's 12.1 — it writes broken solver scripts and burns LLM calls repairing
them. Slowness here is a *capability* signal, not an infrastructure artifact.

**4. The DBZ collapse is universal, not a Qwen weakness.** Every open model lands
at F1 0.52–0.60 against the paper's gpt-3.5 at 0.82. Precision collapses to 37–51%
while recall stays 74–91%. The Z3 path-validation stage is the bottleneck for all
of them.

---

## 5. The router test

### 5.1 Discarded result — selection on the test set

The first analysis picked the best model per bug type by reading the §4 tables, then
reported that combination's average:

| Strategy | Avg F1 | GPU-sec |
|---|---|---|
| "ROUTER" (XSS→Nemo, OSCI→Nemo, DBZ→Phi) | 0.822 | 25,255 |
| always Mistral-Nemo (best single) | 0.810 | 33,295 |

**+0.013 F1 at −24% cost — and it is not a valid result.** The models were selected
using the same data they were then scored on. This measures an *oracle*: an upper
bound that assumes you already know the answer. A deployed router must decide before
seeing outcomes. Recorded here only so the error is not repeated.

### 5.2 The honest test

**Procedure.** Fit the routing table using **only the 25 select variants**. Then
apply that fixed table to the **12 held-out variants**, which it has never seen.

**Fitted table** (from select variants only):

```
XSS  -> Mistral-Nemo
OSCI -> Mistral-Nemo
DBZ  -> Mistral-Nemo        <-- note: not Phi
```

**Held-out evaluation — 458 cases (XSS 216, OSCI 144, DBZ 98):**

| Strategy | Avg F1 | GPU-sec |
|---|---|---|
| Oracle (upper bound) | **0.825** | 8,671 |
| **always Phi-4-mini** | **0.792** | **7,694** |
| ROUTER (fitted on select) | 0.787 | 12,188 |
| always Mistral-Nemo | 0.787 | 12,188 |
| always Qwen2.5-Coder | 0.782 | 7,523 |
| always Granite-3.1 | 0.729 | 15,382 |

**The router failed on three counts:**

1. **It degenerated.** It chose Nemo for all three bug types — making no routing
   decision at all. Its row is identical to always-Nemo.
2. **A fixed model beat it on both axes.** Always-Phi-4-mini: higher F1 (0.792 vs
   0.787) at 37% lower cost (7,694 vs 12,188 GPU-sec).
3. **The decision did not generalize.** See below.

### 5.3 Why it failed — the precise mechanism

The DBZ pick is the whole story. Per-variant-set F1 on DBZ:

| Model | SELECT (n=202) | HELD-OUT (n=98) | drop |
|---|---|---|---|
| Mistral-Nemo | **0.628** | 0.473 | −0.155 |
| Phi-4-mini | 0.614 | **0.586** | **−0.028** |
| Qwen2.5-Coder | 0.604 | 0.464 | −0.140 |
| Granite-3.1 | 0.577 | 0.449 | −0.128 |

On select, Nemo beats Phi by **0.014**. On held-out, Phi beats Nemo by **0.113** —
a reversal, and an order of magnitude larger in the other direction.

A 0.014 margin over 202 cases is noise. The router bet on it and lost. **This is
exactly the failure mode that selecting on the test set conceals.**

There is a second, more interesting signal here: **Phi-4-mini degrades far less
across unseen control-flow variants** (−0.028 vs −0.128 to −0.155). The held-out set
contains all three paper-documented hard variants, and the small model is the one
that holds up. That is a robustness property, not a capability one, and it is not
visible in any aggregate score.

### 5.4 What this means

**Per-bug-type routing cannot be validated.** There are exactly three decisions.
One of them is a coin flip. This is the lookup-table problem — three categorical
inputs give three rows, no generalization, no held-out set worth the name —
demonstrated empirically rather than argued.

**But the headroom is real.** The oracle reaches 0.825 against the best fixed
model's 0.792 — a genuine **+0.033 F1** gap, at comparable cost (8,671 vs 7,694
GPU-sec). Something is there to capture; this granularity just cannot capture it.

**The implication is per-case routing.** Decide per program path using features
available *before* any model is called — path length, branch-condition count,
presence of library/nondeterministic/global constructs in the condition, sink API,
slice token length, call depth. That gives ~9,600 labelled decision points instead
of 3, and it works on code with no dataset label, which is the actual deployment
condition.

The per-variant table in §5.3 is direct evidence the signal exists at that
granularity: model rankings differ sharply *within* a bug type depending on
control-flow shape.

---

## 6. Pipeline fixes made during this campaign

All in [`scripts/patch_llmdfa.sh`](../scripts/patch_llmdfa.sh) and
[`scripts/llm_local.py`](../scripts/llm_local.py), disclosed per the existing
convention.

| # | Problem | Fix |
|---|---|---|
| 10 | **Unbounded extractor synthesis loop.** `TSAgent/synthesis/synthesize.py` loops until the parser is correct, with no cap. Phi-4-mini hit **593 refinement rounds in 33 min** and would not have terminated. | Capped at 40 iterations (paper's worst case is 30, Table 6). |
| 11 | **Seeded sampling was not reproducible.** `CASE_SEED` shuffled `all_single_files` straight from a filesystem walk, whose order is not stable across nodes. Same seed could select different cases — silently invalidating any cross-model comparison drawn on a sample. | Sort before shuffling. |
| — | **Dead server burned GPU hours.** When vLLM died mid-run, every later call failed instantly and nothing noticed. Job 2248 logged **1,154 consecutive failures and held an A100 for 11.5 hours** after completing 78 of 666 cases. | `LLMDFA_ABORT_AFTER_FAILURES=20` — exit non-zero so SLURM marks the job FAILED and the queue advances. Zero failures in every run after this landed. |

Also added:

- [`scripts/run_llmdfa_x2.sbatch`](../scripts/run_llmdfa_x2.sbatch) — runs one
  (model × bug) as two shards on two GPUs. The `mtech` QOS caps *jobs* at 2, not
  GPUs, so `--gres=gpu:1` leaves half a node unreachable. **Deliberately does not
  batch concurrent requests into one server**: that would raise throughput but
  change what "seconds per case" means, breaking comparability with sequentially
  measured runs. Two servers, each serving one sequential client, halves wall-clock
  while preserving timing semantics.
- Per-shard logs carry their own header, because both shards stream to the job's
  stdout where they **interleave** — an interleaved `%j.out` would mis-attribute
  cases across shards in the scorer.

**Cluster-policy note.** `run_llmdfa_x2.sbatch` at 2 jobs × 2 GPUs consumes the
entire A100 partition. Manual §8 states "only one job will be executed at a time",
which the SLURM `mtech` QOS (`MaxJobsPU=2`) contradicts. The manual is demonstrably
stale — its node table, GPU list, and partition table omit the A100 nodes entirely,
while §10 grants `mtech` access to them. Final Granite run was moved back to
`--gres=gpu:1` after a co-user raised it.

---

## 7. Reproducing these numbers

No GPU or cluster access required — all scoring is offline from the logs.

```bash
# per-model, per-bug-type, with the pre-registered variant split
python3 LLMDFA/scripts/score_split.py <log>...

# full per-variant breakdown (the §5.3 table)
python3 LLMDFA/scripts/score_split.py --by-variant <log>...
```

Job IDs on `csecluster` (logs in `~/cluster/<jobid>.out`, results in
`~/job_results/<jobid>/`):

| Job | Model | Bug | Outcome |
|---|---|---|---|
| 2232 | Phi-4-mini | Gate A | cancelled — 593-round synthesis loop |
| 2247 | Phi-4-mini | XSS | completed 1:51:20 |
| 2248 | Granite-3.1 | XSS | **cancelled — dead vLLM, 11.5 h wasted** |
| 2258 | Mistral-Nemo | XSS | completed 2:18:53 |
| 2268 | Phi-4-mini | OSCI | completed 0:42:24 |
| 2289 | Granite-3.1 | OSCI | completed 1:04:46 |
| 2291 | Mistral-Nemo | OSCI | completed 1:10:26 |
| 2292 | Granite-3.1 | XSS (retry) | completed 2:27:05 |
| 2293 | Phi-4-mini | XSS ×2 smoke test | completed 0:04:29 |
| 2300 | Mistral-Nemo | DBZ ×2 | completed 3:09:56 |
| 2301 | Phi-4-mini | DBZ ×2 | completed 1:57:58 |
| 2310 | Granite-3.1 | DBZ (1 GPU) | completed 8:14:12 |

Total ≈ **40 GPU-hours**, of which 11.5 were lost to the dead-server incident.

---

## 8. Not attempted

* **Qwen2.5-Coder-32B** — the escalation ceiling. ~13 h for DBZ alone at `gpu:1`,
  ~6.5 h at `gpu:2`. Needs 62 GB staged (home would reach ~165 GB against a 200 GB
  hard cap, 50 GB soft).
* **Per-case routing** — the conclusion of §5.4, and the actual next step. Requires
  per-call-site model dispatch and per-call-site token/time logging, neither of
  which exists yet: `llm_local.py` has one global model and aggregates cost per
  case, not per phase.
* **Cross-dataset generalization** — SecBench.js (138 real JavaScript CVEs; the
  paper reports the language port as <100 LoC) and TaintBench (39 real Android
  malware apps, 203 labelled paths). Everything here is one synthetic benchmark
  whose difficulty is encoded in its filenames; the robustness result in §5.3 needs
  confirmation on real code.
* **Per-phase precision/recall** (paper Table 1's Extract/Summarize/Validate rows) —
  two of three were obtained by manual examination in the paper.
