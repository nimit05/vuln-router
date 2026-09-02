# vuln-router


---

## LLMDFA

LLMDFA reproduced with a free open-weight model (Qwen2.5-Coder-7B-Instruct, bf16 via
vLLM on one A100-40GB) in place of the four paid APIs the paper used, which it reports
cost USD 1,622.

| bug | cases | precision | recall | F1 | paper gpt-3.5 |
|---|---|---|---|---|---|
| XSS | 666 | 97.47% | 92.64% | 0.950 | 100 / 92.31 / 0.96 |
| OSCI | 444 | 83.95% | 91.89% | 0.877 | 100 / 78.38 / 0.88 |
| DBZ | 1851 | 45.12% | 93.95% | 0.610 | 73.75 / 92.16 / 0.82 |

Recall exceeds the paper on all three bug types. XSS and OSCI reproduce within 0.01 F1;
DBZ does not, and the gap is localised to the Z3 path-feasibility stage rather than
spread across the pipeline. Details and deviations in `LLMDFA/reproduction/FINDINGS.md`.

Upstream LLMDFA is not vendored here — `LLMDFA/scripts/patch_llmdfa.sh` adapts a fresh
clone. Metrics regenerate offline with no GPU:

    python3 LLMDFA/scripts/score_llmdfa.py LLMDFA/reproduction/logs/*.out
