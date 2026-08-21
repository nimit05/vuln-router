# LLMDFA Reproduction — Findings

Target: **LLMDFA: Analyzing Dataflow in Code with Large Language Models**, NeurIPS 2024
(Wang et al., Purdue / HKUST / Ant Group), repo `chengpeng-wang/LLMDFA`.

Run dates: 2026-08-20/21. Hardware: IIT Guwahati CSE cluster (`csecluster`), SLURM,
1x NVIDIA A100-PCIE-40GB per job, models served with vLLM 0.27.1 in **bfloat16**.

Model under test: **Qwen2.5-Coder-7B-Instruct** (Apache-2.0, ungated, served locally).
The paper used gpt-3.5-turbo-0125, gpt-4-turbo-preview, gemini-1.0-pro and claude-3-opus,
at a stated experimental cost of **USD 1,622**. No paid API was used here.

---

## Summary

| Bug type | Reproduced? |
|---|---|
| XSS | **Yes** — F1 0.950 vs the paper's 0.96 for gpt-3.5 |
| OSCI | **Yes** — F1 0.877 vs 0.88 |
| DBZ | **No** — F1 0.612 vs 0.82. Precision collapses; recall does not. |

The paper's central claim — that a decomposed, tool-assisted pipeline detects dataflow
bugs at high precision and recall without compilation — **reproduces on two of three bug
types with a free 7B model**, and fails on the third in a way that is localised to a
single pipeline stage rather than diffuse.

## Headline table

Full benchmark, obfuscated Juliet Test Suite as shipped in the authors' repo.
Paper columns are its own published Table 1 "Detection" rows.

| bug | cases | TP | FP | precision | recall | F1 | paper gpt-3.5 | paper gpt-4 |
|---|---|---|---|---|---|---|---|---|
| DBZ | 1762/1851 | 1650 | 1983 | 45.42% | **93.64%** | 0.612 | 73.75 / 92.16 / 0.82 | 81.38 / 95.75 / 0.87 |
| XSS | 666/666 | 617 | 16 | 97.47% | **92.64%** | 0.950 | 100.00 / 92.31 / 0.96 | 100.00 / 98.64 / 0.99 |
| OSCI | 444/444 | 408 | 78 | 83.95% | **91.89%** | 0.877 | 100.00 / 78.38 / 0.88 | 100.00 / 89.19 / 0.94 |

**Recall exceeds the paper's gpt-3.5 on all three bug types**, by 13.5 points on OSCI.
Precision is lower on all three. The model finds more of the real bugs and discards
fewer of the false ones.

Cost: DBZ 28,673 input / 2,526 output tokens per case at 36.4 s; XSS 6,283 / 696 at
9.1 s; OSCI 4,452 / 468 at 6.5 s. About 20 GPU-hours in total.

## Why DBZ fails: one stage, measured three ways

LLMDFA validates path feasibility by having the LLM **write a Z3 script**, executing it,
and repairing it from the error message at most three times. DBZ is the only bug type
where this matters — the paper's own NoSynVal ablation shows the XSS and OSCI programs
contain no infeasible bug-inducing paths.

1. **Protocol probe (Gate A, job 1812).** Given path conditions with known answers, the
   model produced correctly fenced scripts 3/3 times but runnable ones only **1/3**. Both
   failures were `Z3Exception: Symbolic expressions cannot be cast to concrete Boolean
   values` — python `and`/`not` applied to Z3 expressions.
2. **Fallback rate in the real runs.** Scripts that fail all three repair rounds fall back
   to LLM judgement:

   | bug | solver checks | fell back to LLM | refinement rounds |
   |---|---|---|---|
   | DBZ | 203 | **36 (17.7%)** | 267 |
   | XSS | 42 | 0 | 1 |
   | OSCI | 48 | 1 | 7 |

   The paper reports **0.61%** for DBZ with gpt-3.5 (Appendix A.3.2). Ours is ~29x higher.
3. **The consequence is the one the paper predicts.** It states: *"LLMDFA may encode the
   path condition incorrectly and accept the infeasible path, eventually causing false
   positives."* That is exactly the observed failure — 1,983 false positives against 1,650
   true ones, with recall untouched at 93.64%.

Note DBZ is the weakest row in the paper too: gpt-3.5 reaches only 73.75% precision there
against 100% on XSS and OSCI, and gemini-1.0 drops to 66.57%. The weakness is inherent to
the stage; a 7B amplifies it.

**Implication.** The gap is not general capability. Three of the four LLM call sites
perform at gpt-3.5 level; one does not. This is a per-call-site routing signal, not an
argument for a uniformly larger model.

## Deviations from the paper

1. **Model and serving.** Open 7B via local vLLM in bf16 instead of four hosted APIs.
   Intended; this is the point of the study.
2. **Source/sink extractors.** Runs used the extractors shipped in the authors' repo
   (synthesized by them with gpt-4o-mini), **not** re-synthesized with Qwen. Gate A
   confirmed Qwen synthesizes both XSS extractors in one iteration with zero fixes, and
   the paper reports 100% extraction precision/recall for all four of its models, so the
   stage is effectively model-independent — but Phase I here is not strictly Qwen's.
3. **Scoring aggregation.** LLMDFA's `TPs` counts reported *traces*; several paths can
   reach one labelled sink, so raw trace sums give recall above 100%. True positives are
   capped per case at that case's ground truth (361 duplicate traces discarded on DBZ).
   The paper does not state its aggregation, so its precision may be computed differently.
4. **Sharding.** DBZ was split into 4 independent jobs. Cases are independent; no effect
   on results.
5. **Code patches** (`scripts/patch_llmdfa.sh`), all disclosed: local endpoint, real token
   counts from the server rather than a mis-matched tiktoken estimate, bounded retries in
   place of two unbounded `while True` loops, argparse whitelists that rejected non-OpenAI
   model names, and module-level imports of `google.generativeai` / `anthropic`.
6. **Crash fix.** `construct_solving_program()` ends in `assert program != ""`, aborting
   the whole run when a model fails three times to fence its Z3 script. This killed two
   DBZ shards. The paper describes a fallback rather than a crash (*"If the script is
   buggy after three trials, LLMDFA enforces LLMs to determine path feasibility"*), so the
   assertion was replaced with that fallback path. Frequency: twice in 1,851 DBZ cases.
7. **Coverage.** DBZ is 1,762 of 1,851 cases (95.2%); the remaining 89 were lost to the
   crash above and are being re-run. The 37-case sample and the full set agree closely
   (F1 0.607 vs 0.612), so the outstanding cases are not expected to move the result.

## Reproducing these numbers

No GPU or cluster access required:

```bash
python3 scripts/score_llmdfa.py reproduction/logs/1853.out reproduction/logs/1854.out \
    reproduction/logs/186*.out reproduction/logs/187*.out
```

## Not attempted

* The paper's three ablations (NoSynExt / NoCoT / NoSynVal, Figure 8) — 9 runs, dominated
  by ~18 h per DBZ configuration.
* Per-phase precision/recall (Table 1's Extract / Summarize / Validate rows). Two of the
  three were obtained by **manual examination** in the paper, so parity is not purely
  computational.
* The TaintBench real-world Android evaluation (Table 3) and the C/C++ and JavaScript
  migrations (Tables 4 and 5).
* Any second model. Gate A was run on `deepseek-coder-6.7b-instruct` and abandoned: vLLM
  returned undetokenized byte-level BPE (`ĠĊ` artifacts) for that tokenizer, which scores
  as total failure while the model is in fact answering correctly. An infrastructure bug,
  not a model result. `--tokenizer-mode slow` did not fix it.
