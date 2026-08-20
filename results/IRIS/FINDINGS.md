# IRIS Reproduction — Findings

Target: **IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities**,
ICLR 2025 (arXiv:2405.17238), repo pinned to branch `v1` (the 120-CVE ICLR configuration).

Run dates: 2026-08-17/18. Hardware: 1x NVIDIA A100-SXM4-80GB, 256 cores, Ubuntu 24.04
container, no sudo, shared 2.5TB volume at 99% capacity.

---

## Summary

| Stage | Reproduced? |
|---|---|
| CodeQL baseline (no LLM) | **Yes — exact**, on two independent subsets |
| IRIS LLM stage | **No.** 2/17 where the paper reports 10/17 |

The paper's central claim — that LLM-inferred taint specifications roughly double CodeQL's
detection rate — **did not reproduce** with an Ollama Q4_0 quantized 7B model. Our IRIS run
scores *below* its own CodeQL baseline (2 vs 3 detections), where the paper's roughly triples
it (10 vs 3).

## Scope

17-project subset of CWE-Bench-Java covering all four classes (CWE-022/078/079/094), chosen
for build speed and CWE balance. **Every row is computed on exactly the same 17 projects**:
paper rows are its own published per-project CSVs (`IRIS/results/*.csv`) filtered to that set.
Metrics follow the paper's Sec 3.6. Of 23 selected projects, 6 failed CodeQL database
creation and are excluded from all rows.

## Headline table

| configuration | #Det | rate % | AvgFDR % | AvgF1 |
|---|---|---|---|---|
| **OURS: CodeQL baseline** | **3/17** | 17.65 | 70.00 | 0.081 |
| **OURS: IRIS + deepseek-coder-6.7b (Q4_0)** | **2/17** | 11.76 | 96.48 | 0.036 |
| paper: CodeQL | 3/17 | 17.65 | 70.00 | 0.081 |
| paper: IRIS + DeepSeekCoder-7B | 10/17 | 58.82 | 81.31 | 0.202 |
| paper: IRIS + GPT-4 | 10/17 | 58.82 | 67.83 | 0.345 |

### 1. CodeQL baseline — REPRODUCED EXACTLY
Matched on all four metrics, independently on a 9-project and a 17-project subset, and
per-project (e.g. spark CVE-2018-9159: 36 paths / 12 TP paths, identical). This validates the
substituted toolchain: **Azul Zulu 8u202** for the auth-gated Oracle JDK, IRIS's patched
**CodeQL 2.15.5**, and locally rebuilt databases.

### 2. IRIS LLM stage — NOT REPRODUCED
Failure mode is **precision collapse, not silence**: AvgFDR 96.48% vs the paper's 81.31%.
The pipeline emits plenty of paths (ff4j: 360 paths / 24 TP; ESAPI: 131 paths / 42 TP) but
the specifications point at the wrong APIs, so most projects miss entirely.

Total path volume is comparable to the paper (443 vs 635 pre-filter), so this is **not
under-generation** — it is mis-targeting. IRIS detection is all-or-nothing per project: you
must label the one sink that matters. `cron-utils` is the clearest case — the paper's own
showcase example (Java EL injection via `buildConstraintViolationWithTemplate`), where the
paper gets 89 paths / 80 TP and we get 10 paths / 0 TP, because our model never labels that
sink. A handful of such misses swings the headline number.

Notably we *beat* the paper on some projects (ESAPI 42 TP vs their 4; ff4j detected), which
confirms the mechanism works when the right API happens to be labelled. It is a lottery over
spec quality.

**Attribution.** The paper served full-precision instruction-tuned weights via HuggingFace.
Full precision was impossible here (Gemma-2-27B ~54GB + Qwen2.5-Coder-32B ~64GB +
Llama-3-70B ~140GB against ~41GB free on a shared 99%-full volume; Llama-3 and Gemma-2 are
also license-gated). Ollama **Q4_0** was substituted. Quantization is the leading explanation,
but it **cannot be cleanly isolated** from the schema-constrained decoding we had to add
(below), which the paper did not use.

---

## Three silent failures that make quantized reproduction look like model weakness

Each produced a clean-looking run with **zero errors** and near-zero detections.

**1. Q4_0 ignores the JSON instruction.** IRIS's prompt says "DO NOT OUTPUT ANYTHING OTHER
THAN JSON"; the quantized model returned English prose, which IRIS's parser
(`re.findall(r"\[[\s\S]*\]", s)[0]`) discards silently. Fix: per-stage Ollama structured
output. Effect on spark/CWE-022: sources 4 -> 17, sinks 4 -> 17, propagators 6 -> 32.
Generic `format="json"` is *worse than nothing* (0/0/0): models emit a valid JSON **object**
but the parser requires a top-level **array**.

**2. Missing `sink_args` produces a dead CodeQL predicate.** `neusym_vul.py:837` only emits a
real sink clause when `"sink_args" in api`; otherwise IRIS generates
`predicate isGPTDetectedSink(...) { 1 = 0 or 1 = 0 }`. `1 = 0` is CodeQL for *false*, so **no
path can ever be found**. Fix: make `sink_args` schema-required. zt-zip 0 -> 4 paths.

**3. Unset `num_ctx` truncates the prompts.** IRIS batches 30 APIs per labelling call; Ollama's
default context silently cut them off so the model never saw most candidates. Fix: pin
`num_ctx=16384`. Spec counts rose ~an order of magnitude (4 sources -> up to 95) and the first
real detections appeared. Cost: ~4x slower per call (~11 min/project).

Also: `num_predict: -1` (unlimited) let a schema-constrained model run away generating a huge
`explanation` field — one path stalled 18+ minutes with GPU at 67% and ollama at 99% CPU.
Capped at 1024.

## Correction: the posthoc filter was NOT a meaningful cause of the gap

An earlier draft hypothesised that the broken false-positive filter (returning prose, and
cascading `source_is_false_positive` suppressions) was costing us detections. That was
**tested and disproved**. After schema-constraining the filter and re-running all 17:

- headline stayed at **2/17** (F1 0.027 -> 0.036, FDR 97.32 -> 96.48)
- it *reshuffled* rather than improved: gained `ff4j` (0 -> 360 paths, 24 TP, detected),
  lost `sling-servlets-resolver` (2 TP -> 0), `commons-text` still lost
- net effect on detections: ~zero

So the gap is genuinely spec quality, not our filter bug. Recorded because the wrong
hypothesis was plausible and someone else would likely form it too.

## Llama-3.3-70B: attempted, abandoned (see CHECKPOINT_llama33_70b.md)
95.9 GiB model + a defaulted 131072 context exceeded the 80GB A100, so Ollama silently split
it across CPU: `ollama runner` at 10135% CPU, GPU 0%, **0 completed LLM calls in 31 minutes**.
Retry requires `IRIS_NUM_CTX=8192` or a smaller quantisation.

## Six upstream bugs fixed to get this far
1. `setup_jdk.py` aborts the whole run on a missing JDK tarball instead of skipping.
2. Maven 3.9.8 URL in `mvn_version.json` is dead -> re-sourced from Maven Central.
3. `build_codeql_dbs.py` never puts Gradle on PATH, and Maven only when recorded.
4. `data/build_info.csv` disagrees with what actually builds; `build-info/<slug>.json` is authoritative.
5. `codeql_vul.py` uses a stale dataset schema (`cve`/`commits`/`db_name`).
6. **`codeql_vul.py` writes CSV over its own SARIF** (same `--output`), so the evaluator
   `json.load()`s a CSV and crashes. **The paper's CodeQL baseline is unreproducible as shipped.**

## Caveats
- 17 projects, not 120. Detection counts are small; treat rates as indicative.
- Single run per configuration; no variance measurement.
- Maven Central rate-limiting (HTTP 429, 58 events) caused build/DB failures unrelated to project quality.
- Schema-constrained decoding is a deviation from the paper's method, added out of necessity.
