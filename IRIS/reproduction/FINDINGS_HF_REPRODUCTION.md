# IRIS Reproduction via native HuggingFace backend — Findings

Date: 2026-08-18. Hardware: 1x NVIDIA A100-SXM4-80GB.
Scope: 6-project subset of CWE-Bench-Java, chosen because the paper's
`IRIS+DeepSeekCoder-7B.csv` detects **all 6**. Spans CWE-022/079/094.

> NOTE: this file documents the *native HF transformers* reproduction line.
> It does not replace `FINDINGS.md` (the earlier Ollama/quantization line).

---

## Headline

| Configuration | Detected | vs paper |
|---|---|---|
| Paper (IRIS + DeepSeekCoder-7B) | **6 / 6** | — |
| Ours: 6.7b-instruct, as-shipped | 1 / 6 | ✗ |
| Ours: 6.7b-instruct + JSON retry | 2 / 6 | ✗ |
| **Ours: 7b-instruct-v1.5 + JSON retry** | **5 / 6** | **✓ reproduces** |
| Ours: 7b-instruct-v1.5, *pre-filter* | **6 / 6** | **✓ exact** |

**IRIS reproduces with DeepSeek-Coder-7B.** The gap was primarily a wrong
model checkpoint, not model capability.

## Per-project (v1.5 + retry)

| Project | CWE | pre-filter | post-filter (official) | paper |
|---|---|---|---|---|
| zt-zip CVE-2018-1002201 | 022 | True 71p/7TP | **True 58p/7TP** | True 79p/2TP |
| plexus-utils CVE-2022-4244 | 022 | True 152p/8TP | **True 131p/8TP** | True 111p/8TP |
| plexus-archiver CVE-2018-1002200 | 022 | True 28p/3TP | **True 17p/3TP** | True 37p/3TP |
| cron-utils CVE-2021-41269 | 094 | True 26p/1TP | **False 3p/0TP** | True 89p/80TP |
| antisamy CVE-2016-10006 | 079 | True 46p/14TP | **True 21p/4TP** | True 61p/9TP |
| antisamy CVE-2017-14735 | 079 | True 46p/7TP | **True 21p/5TP** | True 61p/4TP |

TP counts land in the same ballpark as the paper; plexus-utils matches exactly (8 TP).

---

## Root cause #1 (dominant): WRONG MODEL CHECKPOINT

Two different models, easily confused:

| | Released | Specified by |
|---|---|---|
| `deepseek-ai/deepseek-coder-6.7b-instruct` | Oct 2023 | Paper Table 6 text (`deepseek-coder-7b-instruct`, which HF *redirects* here) |
| `deepseek-ai/deepseek-coder-7b-instruct-v1.5` | Jan 2024 | **IRIS's own `src/models/deepseek.py`** |

These are architecturally different (v1.5 is built on deepseek-llm-7b).

**The authors' code is the ground truth, not the paper's table.** Overriding
the repo's `-v1.5` to match the paper's stated ID dropped detection 5/6 -> 1/6.

**Do not "fix" IRIS's model map to match the paper.** It is already correct.

v1.5 is also markedly better at the task (same project, zt-zip):

| | 6.7b | v1.5 |
|---|---|---|
| APIs labelled | 67 (18 src/19 sink) | 104 (39 src/27 sink) |
| func-param sources | 11 | 27 |
| unparseable batches | frequent | ~0 |

## Root cause #2 (real IRIS bug, worth reporting)

IRIS **silently discards** any LLM batch whose response is not parseable JSON.
No exception, no counter — only a log line the pipeline ignores.

Measured on the 6.7b run (81 batches, using IRIS's *actual* two-strategy parser):

| Stage | Prompt style | Batches | Truly discarded |
|---|---|---|---|
| `label_apis` (external APIs) | **few-shot** (3-4 examples) | 60 | 1 (1.7%) |
| `func_params` (internal APIs) | **zero-shot** | 21 | 6 (28.6%) |

A **17x** difference. The zero-shot internal-API prompt (paper §A.3) is the weak point.

**Proven causal chain on cron-utils** (the paper's own showcase example):
1. Batch 60 — the only batch containing `CronParser.parse` — returned prose, not JSON.
   The model even named the right answer in that prose: *"such as the `parse` method
   in the `CronParser` class"*.
2. IRIS discarded the whole batch -> `CronParser.parse` never labelled as a source.
3. `fix_info.csv` ground truth: the CVE patch is in `CronParser.parse(String)`.
   No source -> no taint path through the patched method -> not detected.
4. Adding retry recovered it: func-param sources 11 -> 26, `CronParser.parse`
   labelled with `tainted_input: ["expression"]`, result **0 TP -> 12 TP, DETECTED**.

### Fix implemented
`predict_with_json_retry()` in `src/neusym_vul.py` (both LLM call sites).
Up to 2 retries, appending a corrective instruction.
**Greedy decoding is deterministic — a plain retry returns the identical failure,
so the prompt must be varied.**

## Root cause #3: environment incompatibilities (ours, fixed)

1. **transformers 5.15.0 silently corrupts this tokenizer.** Loads slow
   `LlamaTokenizer`; `decode("hello world\nline2")` -> `'helloworldline2'`.
   All whitespace destroyed; model output arrives as raw BPE (`Ġ`/`Ċ`), so
   **0 labels parse** despite the model answering correctly.
   **Fix: transformers 4.56.1 + tokenizers + huggingface_hub 0.34.5** ->
   `LlamaTokenizerFast`, correct decode.
   *This failure looks exactly like "the model is bad" — it is not.*
2. **`temperature=0.0` crashes transformers >=5.** IRIS passes it to `pipeline()`.
   Fix: `do_sample=False` (greedy == what temperature 0 means) and drop
   `temperature`/`top_p` from both `pipe()` calls in `src/models/llm.py`.
3. **Ollama backend is not paper-faithful.** Paper used HF transformers.
   Abandoned; use `--llm deepseekcoder-7b` (native path).

## Observation: the posthoc filter can destroy a true positive

cron-utils under v1.5: **detected pre-filter** (26 paths, 1 TP), then Stage 8
filtering removed the correct path (3 paths, 0 TP). The taint analysis was right;
the LLM false-positive filter threw the real vulnerability away.

Consistent with the paper's own Fig. 9 (contextual analysis only helps models with
sufficient reasoning). Worth reporting: **the filter is not monotonically beneficial.**

---

## Environment actually used

| Component | Value |
|---|---|
| Model | `deepseek-ai/deepseek-coder-7b-instruct-v1.5` (local, byte-verified) |
| Backend | HF transformers **4.56.1**, fp16, `device_map="auto"`, cuda:0 |
| torch | 2.5.1+cu121 (matches IRIS `environment.yml` pin) |
| hub / tokenizers | huggingface_hub 0.34.5, tokenizers (4.56.1-matched) |
| accelerate / psutil | 1.10.1 / 7.0.0 |
| Generation | `do_sample=False` (greedy), `max_new_tokens=2048` |
| CodeQL | 2.15.5 (IRIS patched build; paper states 2.15.3) |
| JDK | Azul Zulu 8u202 (Oracle is auth-gated) |

## Reproduce

```bash
cd ~/nimit/vuln-pred-results/IRIS
export PATH=$PWD/codeql:$PATH PYTHONPATH=$PWD:$PWD/src
./.conda-iris/bin/python src/neusym_vul.py \
    --query cwe-022wLLM --run-id <FRESH_ID> --llm deepseekcoder-7b <SLUG>
```
**Always use a fresh `--run-id`** — IRIS caches labelled specs per run-id and will
silently reuse stale ones.

Run logs: `/tmp/iris_hf2.log` (6.7b baseline), `/tmp/iris_retry.log` (6.7b+retry),
`/tmp/iris_v15.log` (v1.5+retry). Outputs under `output/<slug>/{dsc7b_hf2,dsc7b_retry,dsc7b_v15}/`.

## Caveats

- **6 projects, not 120.** Chosen as ones the paper detects, so this measures
  "can we match the paper where it succeeds" — NOT overall detection rate,
  and NOT false-discovery rate. Not comparable to the paper's headline 52/120.
- **Single run per configuration**; no variance measurement. Detection is
  all-or-nothing per project, so run-to-run variance may be material.
- The JSON-retry fix is a **deviation from the paper** and must be disclosed.
- transformers 4.56.1 chosen for compatibility; the paper's exact version is unstated.

## Corrections to earlier claims made during this work

- An intermediate figure of "31% of batches discarded" was **wrong** — it replicated
  only IRIS's first parse strategy. True rate with both strategies: **8.6%** overall
  (28.6% on the zero-shot stage).
- An intermediate conclusion that the gap was largely "model capability" was **wrong**;
  it was dominated by the checkpoint choice.
