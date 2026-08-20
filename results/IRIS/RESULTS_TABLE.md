# IRIS Reproduction — Results Tables

**Target:** IRIS (ICLR 2025, arXiv:2405.17238), `IRIS + DeepSeekCoder-7B` row
**Date:** 2026-08-18
**Hardware:** 1x NVIDIA A100-SXM4-80GB
**Benchmark:** 6-project subset of CWE-Bench-Java (CWE-022 / 079 / 094)

> **Subset selection:** these 6 were chosen *because the paper detects all 6*.
> This measures "do we match the paper where it succeeds."
> It is **NOT** a detection rate or FDR comparable to the paper's headline 52/120.

---

## TABLE 1 — Headline

| # | Configuration | Detected | Notes |
|---|---|---|---|
| 0 | **Paper** (IRIS + DeepSeekCoder-7B) | **6 / 6** | published per-project CSV |
| 1 | Ours — `6.7b-instruct`, IRIS as-shipped | 1 / 6 | wrong checkpoint |
| 2 | Ours — `6.7b-instruct` + JSON retry | 2 / 6 | retry recovered cron-utils |
| 3 | **Ours — `7b-instruct-v1.5` + JSON retry** | **5 / 6** | **reproduces** |
| 3b | Ours — `7b-instruct-v1.5`, *pre-filter* | **6 / 6** | exact match |

**Verdict: IRIS reproduces with DeepSeek-Coder-7B.**
The single miss (cron-utils) was caused by IRIS's own false-positive filter
deleting a correctly-found path — not by a detection failure.

---

## TABLE 2 — Per-project, all configurations (post-filter = official metric)

Format: `detected | paths / true-positive paths`

| Project | CWE | Paper | Cfg 1 (6.7b) | Cfg 2 (6.7b+retry) | **Cfg 3 (v1.5+retry)** |
|---|---|---|---|---|---|
| zt-zip CVE-2018-1002201 | 022 | ✅ 79/2 | ❌ 36/0 | ❌ 36/0 | ✅ **58/7** |
| plexus-utils CVE-2022-4244 | 022 | ✅ 111/8 | ❌ 14/0 | ❌ 15/0 | ✅ **131/8** |
| plexus-archiver CVE-2018-1002200 | 022 | ✅ 37/3 | ✅ 14/3 | ✅ 14/3 | ✅ **17/3** |
| cron-utils CVE-2021-41269 | 094 | ✅ 89/80 | ❌ 47/0 | ✅ 60/12 | ❌ **3/0** |
| antisamy CVE-2016-10006 | 079 | ✅ 61/9 | ❌ 21/0 | ❌ 21/0 | ✅ **21/4** |
| antisamy CVE-2017-14735 | 079 | ✅ 61/4 | ❌ 21/0 | ❌ 21/0 | ✅ **21/5** |
| **TOTAL DETECTED** | | **6/6** | **1/6** | **2/6** | **5/6** |

---

## TABLE 3 — v1.5: effect of the false-positive filter (Stage 8)

| Project | Pre-filter (Stage 6) | Post-filter (Stage 8) | Filter verdict |
|---|---|---|---|
| zt-zip | ✅ 71 paths / 7 TP | ✅ 58 / 7 | good — cut 13 noise, kept all TP |
| plexus-utils | ✅ 152 / 8 | ✅ 131 / 8 | good — cut 21 noise, kept all TP |
| plexus-archiver | ✅ 28 / 3 | ✅ 17 / 3 | good — cut 11 noise, kept all TP |
| **cron-utils** | ✅ 26 / 1 | ❌ **3 / 0** | **HARMFUL — deleted the real bug** |
| antisamy 2016 | ✅ 46 / 14 | ✅ 21 / 4 | cut noise, kept detection (lost 10 TP) |
| antisamy 2017 | ✅ 46 / 7 | ✅ 21 / 5 | cut noise, kept detection |
| **TOTAL** | **6 / 6** | **5 / 6** | net cost: **-1 detection** |

**Finding:** the filter is not monotonically beneficial. It removed real noise on
5 projects but destroyed the only true positive on cron-utils. Matches the paper's
own Fig. 9 (filtering helps strong models, hurts weaker ones).

---

## TABLE 4 — Why config 1 failed: silent JSON discards (6.7b run)

IRIS **silently deletes** any LLM batch whose reply is not parseable JSON —
no exception raised, only an ignored log line.

Measured across 81 batches, using IRIS's *actual* two-strategy parser:

| Stage | Prompt style | Batches | Discarded | Rate |
|---|---|---|---|---|
| `label_apis` (external APIs) | **few-shot** (3-4 examples) | 60 | 1 | **1.7 %** |
| `func_params` (internal APIs) | **zero-shot** | 21 | 6 | **28.6 %** |
| **Total** | | **81** | **7** | **8.6 %** |

A **17x** difference between the two prompts. The zero-shot internal-API prompt
(paper §A.3) is the weak point.

### Proven causal chain — cron-utils (the paper's own showcase example)

| Step | Evidence |
|---|---|
| 1. Ground truth | `fix_info.csv`: CVE patch is in `CronParser.parse(String)` |
| 2. Candidate offered? | ✅ yes, with JavaDoc *"Parse string with cron expression"* |
| 3. Sink labelled? | ✅ yes — `buildConstraintViolationWithTemplate` correctly labelled |
| 4. Source labelled? | ❌ **no** |
| 5. Why? | its batch returned **prose, not JSON** → discarded |
| 6. What the prose said | *"...such as the `parse` method in the `CronParser` class"* — **the model knew the answer** |
| 7. After retry fix | sources 11 → 26, `CronParser.parse` labelled, **0 TP → 12 TP, DETECTED** |

---

## TABLE 5 — Model comparison (same project: zt-zip)

| Metric | `6.7b-instruct` | `7b-instruct-v1.5` |
|---|---|---|
| APIs labelled | 67 | **104** |
| — sources | 18 | **39** |
| — sinks | 19 | **27** |
| func-param sources | 11 | **27** |
| unparseable batches | frequent | **~0** |
| Detected? | ❌ | ✅ |

v1.5 labels roughly 2x more specs and follows the JSON instruction far more reliably.

---

## TABLE 6 — Root causes of the original gap, ranked

| # | Cause | Impact | Status |
|---|---|---|---|
| 1 | **Wrong model checkpoint** (ours) | 1/6 → 5/6 | ✅ fixed |
| 2 | **Silent JSON discards** (IRIS bug) | +1 detection on 6.7b | ✅ fixed (retry) |
| 3 | Tokenizer corruption, transformers 5.15 (ours) | 0 labels parsed | ✅ fixed |
| 4 | `temperature=0.0` crash, transformers ≥5 (ours) | pipeline crash | ✅ fixed |
| 5 | Ollama backend ≠ HF transformers (ours) | not paper-faithful | ✅ abandoned |
| 6 | Model capability | **minor** — much less than first concluded | — |

### #1 in detail — the checkpoint trap

| Model | Released | Specified by |
|---|---|---|
| `deepseek-coder-6.7b-instruct` | Oct 2023 | Paper Table 6 (`deepseek-coder-7b-instruct`, HF **redirects** here) |
| `deepseek-coder-7b-instruct-v1.5` | Jan 2024 | **IRIS's own `src/models/deepseek.py`** |

Architecturally different models. **The authors' code is ground truth, not the
paper's table.** Do NOT "correct" IRIS's model map to match the paper.

---

## Environment

| Component | Value |
|---|---|
| Model | `deepseek-ai/deepseek-coder-7b-instruct-v1.5` (local, byte-verified) |
| Backend | HF transformers **4.56.1**, fp16, `device_map="auto"`, cuda:0 |
| torch | 2.5.1+cu121 (matches IRIS `environment.yml`) |
| huggingface_hub / accelerate / psutil | 0.34.5 / 1.10.1 / 7.0.0 |
| Generation | `do_sample=False` (greedy), `max_new_tokens=2048` |
| CodeQL | 2.15.5 (IRIS patched build; paper states 2.15.3) |
| JDK | Azul Zulu 8u202 (Oracle auth-gated) |

## Caveats

1. **6 projects, not 120** — selected as ones the paper detects. Not an
   overall detection rate; not comparable to the paper's 52/120 or its FDR.
2. **Single run per configuration** — no variance measured. Detection is
   all-or-nothing per project, so run-to-run variance may be material.
3. **JSON retry is a deviation** from the paper and must be disclosed in any write-up.
4. transformers 4.56.1 chosen for compatibility; the paper's exact version is unstated.

## Files

| File | Contents |
|---|---|
| `RESULTS_TABLE.md` | this file |
| `FINDINGS_HF_REPRODUCTION.md` | full narrative write-up + reproduce commands |
| `hf_run_logs/iris_hf2.log` | config 1 — 6.7b baseline |
| `hf_run_logs/iris_retry.log` | config 2 — 6.7b + retry |
| `hf_run_logs/iris_v15.log` | config 3 — v1.5 + retry |
