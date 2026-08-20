# QASecClaw Reproduction — Findings

Reproduction attempt of *"QASecClaw: A Multi-Agent LLM Approach for False Positive
Reduction in Static Application Security Testing"* (arXiv:2605.01885v1).

Run date: 2026-08-12
Hardware: NVIDIA A100-SXM4-80GB, Ubuntu 24.04.3 LTS (no sudo, user-space installs only)

---

## Summary

| Stage | Reproduced? |
|---|---|
| Semgrep baseline | **Yes** — near-exact match |
| LLM false-positive filter | **Partially** — direction confirmed, ~half the magnitude |

The paper's core claim — that LLM contextual review substantially reduces SAST false
positives at small recall cost — **is directionally reproduced** with an open-weight
local model, but at roughly half the reported magnitude. Model choice dominates: one
open-weight model failed entirely, another recovered most of the gap.

---

## Headline comparison

| | Precision | Recall | F1 | FPR | Youden's J |
|---|---|---|---|---|---|
| Semgrep baseline (paper) | 0.695 | 0.900 | 0.784 | 0.423 | 0.477 |
| Semgrep baseline (**ours**) | 0.693 | 0.882 | 0.776 | 0.417 | 0.465 |
| QASecClaw (paper, Qwen 3.5 Plus) | **0.951** | 0.871 | **0.909** | **0.048** | **0.823** |
| QASecClaw (**ours**, qwen3.6:27b) | 0.817 | 0.849 | 0.833 | 0.203 | 0.646 |
| QASecClaw (**ours**, qwen3-coder:30b) | 0.697 | 0.852 | 0.766 | 0.396 | 0.455 |

### False-positive reduction — the paper's central claim

| | FPs before | FPs after | Reduction | TP loss |
|---|---|---|---|---|
| Paper | 560 | 64 | **88.6%** | 3.1% (40/1273) |
| Ours (qwen3.6:27b) | 552 | 269 | **51.3%** | **3.7%** (46/1248) |
| Ours (qwen3-coder:30b) | 552 | 525 | 4.9% | 3.7% (46/1248) |

**The recall cost reproduced almost exactly** (3.7% vs. the paper's 3.1%). The
false-positive reduction reached 51.3% vs. the reported 88.6% — the same mechanism
working, at reduced strength.

---

## 1. Semgrep Baseline — REPRODUCED

Semgrep 1.172.0 CE, `--config=auto`, 2,740 Java test cases, file-level CWE-matched scoring.
10 of 11 CWE categories matched the paper to ~3 decimal places.

### Deviation: CWE-326 → CWE-327 synonym mapping
Semgrep's current rules tag weak-crypto findings as **CWE-326** ("Inadequate Encryption
Strength"), not CWE-327. Strict equality scored CWE-327 as 0 TP / 0 FP despite Semgrep
correctly flagging every file. A synonym map (326→327) was applied; CWE-327 then scores
1.000/1.000/1.000, matching the paper. **This adjustment moves baseline F1 from 0.725 to
0.776** — it should be disclosed in any write-up, not folded in silently.

### Unresolved: CWE-501 gap
40 genuinely-vulnerable CWE-501 files received no CWE-501 finding. Paper reports Semgrep
recall 0.819 for CWE-501; we got 0.518. Attributed to **live Semgrep registry drift** —
the `auto`/community ruleset updates continuously and today's snapshot differs from the
paper's. This accounts for essentially the entire remaining baseline gap.

---

## 2. LLM Filter Stage — PARTIALLY REPRODUCED

### Model substitution (unavoidable)
The paper uses **Qwen 3.5 Plus**, which is **closed-weight and API-only** via Alibaba
DashScope. Verified directly: `ollama pull` of `qwen3.5-plus`, `qwen-plus`,
`qwen3.5:plus`, `qwen3.6-plus` all return `pull model manifest: file does not exist`.
It cannot be self-hosted at any price — only rented per-token.

Two open-weight substitutes were tested, both served via Ollama at temperature 0 with
structured JSON schema output, batch size 15, fail-open on failure — matching the paper's
described design. Both runs: 120/120 batches, **0 fail-opens**.

### Result: model choice dominates

| Model | Arch | Suppressed | Correctly | Suppression accuracy |
|---|---|---|---|---|
| qwen3-coder:30b | qwen3moe, 30.5B | 70 | 27 | **38.6%** (worse than chance) |
| qwen3.6:27b | qwen35, 27.8B | 329 | 283 | **86.0%** |

`qwen3-coder:30b` suppressed *fewer* findings and was *wrong more often than right* when
it did — net effect slightly worse than no filter at all (F1 0.766 vs. baseline 0.776).

`qwen3.6:27b` — despite being *smaller* — suppressed 4.7× more findings at 86% accuracy,
lifting F1 from 0.776 to 0.833 and halving FPR. Notably its architecture string is
`qwen35`, i.e. it belongs to the same Qwen 3.5 family as the paper's model, whereas
`qwen3-coder` is an older-generation coding specialist. **Coding specialization did not
help; generation and reasoning ability did.**

### Ruling out confounds
- **Batching:** balanced 40-file sample (20 vuln / 20 safe) at **batch size 1** — one file
  per call, no context dilution. qwen3-coder still only reached Precision 0.559 vs. a
  0.500 do-nothing baseline (+0.059). Not a batching artifact.
- **Reasoning mode:** an Ollama interaction bug — `think: true` (or thinking-capable
  models by default) combined with a constrained JSON `format` schema puts the answer in
  the `thinking` field and returns an empty `response`, causing 100% spurious fail-opens.
  Fixed by sending `think: false` explicitly plus a fallback that reads the `thinking`
  field when `response` is empty. **This silently cost an entire earlier run** and is
  worth knowing for anyone building on Ollama structured output.

### Where the remaining gap lives
Per-CWE, qwen3.6:27b trails the paper mainly on:
- **CWE-501 Trust Boundary** (F1 0.550) — but the paper *also* reports this as its worst
  category (0.593), so this is a shared weakness, compounded here by the baseline's
  CWE-501 rule drift.
- **CWE-78 Command Injection** (P 0.628) and **CWE-643 XPath** (P 0.571) — the filter
  leaves many FPs standing.
- Cryptographic categories (327/328/330) hold at precision 1.000, matching the paper.

---

## 3. Reproducibility Caveats

- **The paper's own code was not used.** `github.com/takrim1999/qasecqlaw` exists per the
  paper but the pipeline here was reimplemented from prose. The paper publishes no prompt
  template, JSON schema, or retry logic, so **the prompt wording is our own reconstruction
  and may differ materially.** Results below the paper's could reflect prompt differences
  rather than model capability. Fetching the original repo is the single highest-value
  next step for a tighter comparison.
- Implemented in Python, not the paper's TypeScript/Node — methodology is language-agnostic.
- Only the SAST-filtering pathway was implemented. The Test Planning Agent, Evidence
  Correlation Agent (dynamic-evidence path), and Report Generation Agent are inactive on a
  static benchmark by the paper's own admission.
- **Single run per model, no variance measurement.** The paper notes this limitation too.
- `qwen3.6:27b` post-dates the assistant's knowledge cutoff; its provenance was not
  independently verified against an authoritative Alibaba/Qwen source (corroborating web
  results were low-quality SEO pages). Confirm before citing it by name.

---

## 4. Files

| File | Contents |
|---|---|
| `semgrep_baseline.json` | 2,404 raw Semgrep findings |
| `filter_results.json` | 1,800 verdicts — qwen3-coder:30b |
| `filter_results_qwen36.json` | 1,800 verdicts — qwen3.6:27b |
| `sample_test.json` | Balanced 40-file comparison sample |
| `sample_qwen3coder.json` / `sample_qwen36.json` | batch=1 sample verdicts |
| `sample_qwen3think.json` | qwen3:30b thinking-mode attempt (all fail-open) |
| `results/metrics_semgrep_baseline.txt` | Baseline metrics + per-CWE |
| `results/metrics_qasecclaw_qwen36.txt` | qwen3.6:27b metrics + per-CWE |
| `results/metrics_qasecclaw_qwen3coder.txt` | qwen3-coder:30b metrics + per-CWE |
| `results/diagnostics_filter_accuracy.txt` | Verdict-accuracy breakdown |
| `sast_filter_agent.py` / `score_semgrep.py` / `score_qasecclaw.py` | Pipeline |

Regenerate all metrics from saved artifacts (seconds, no GPU needed):
```bash
python3 score_semgrep.py --gt owasp-benchmark/expectedresults-1.2.csv \
  --semgrep-json semgrep_baseline.json --label 'Semgrep Baseline'
python3 score_qasecclaw.py --gt owasp-benchmark/expectedresults-1.2.csv \
  --semgrep-json semgrep_baseline.json --filter-results filter_results_qwen36.json \
  --model-label 'qwen3.6:27b SAST Filter Agent'
```

---

## 5. Conclusion

The paper's mechanism is real and reproduces directionally on open weights: contextual LLM
review of SAST findings removes a majority of false positives while costing ~3.7% of true
positives — a recall cost matching the paper almost exactly. The magnitude does not fully
reproduce (51.3% vs. 88.6% FP reduction), and the gap is plausibly split between the
proprietary model's capability and our unknown prompt divergence from the original.

The sharpest practical finding is that **substitute model choice swings the result from
useless to useful**: two open-weight models of near-identical size differed by 47
percentage points in suppression accuracy. Anyone reproducing this should not treat
"a coding LLM" as a substitutable component.
