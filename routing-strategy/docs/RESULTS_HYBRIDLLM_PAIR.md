# Hybrid LLM on IRIS / CWE-Bench-Java

Router: `prob_2cls`, quality-gap tolerance t=0, backbone `linear`, leave-one-project-out.
small = `qwen-7b`, large = `granite-8b`. Metrics are IRIS Sec 3.6 via their own `score_subset.metrics`.

| Configuration | %large | GPU-s | #Det | AvgFDR% | AvgF1 |
|---|---|---|---|---|---|
| no filter (all paths) | 0.0 | 0.0 | 10/16 | 82.12 | 0.212 |
| always-qwen-7b | 0.0 | 569.2 | 4/16 | 79.60 | 0.142 |
| always-granite-8b | 100.0 | 1943.6 | 10/16 | 82.71 | 0.207 |
| label-following (perfect router) | 16.6 | 798.8 | 10/16 | 57.08 | 0.405 |
| pair oracle (upper bound) | -- | -- | 10/16 | 50.98 | 0.445 |
| | | | | | |
| hybrid tau=0.00 | 0.0 | 569.2 | 4/16 | 79.60 | 0.142 |
| hybrid tau=0.10 | 0.0 | 569.2 | 4/16 | 79.60 | 0.142 |
| hybrid tau=0.20 | 0.0 | 569.2 | 4/16 | 79.60 | 0.142 |
| hybrid tau=0.30 | 0.0 | 569.2 | 4/16 | 79.60 | 0.142 |
| hybrid tau=0.40 | 0.2 | 576.1 | 4/16 | 79.60 | 0.142 |
| hybrid tau=0.50 | 0.2 | 576.1 | 4/16 | 79.60 | 0.142 |
| hybrid tau=0.60 | 0.2 | 576.1 | 4/16 | 79.60 | 0.142 |
| hybrid tau=0.70 | 0.4 | 576.6 | 4/16 | 79.60 | 0.142 |
| hybrid tau=0.80 | 3.7 | 591.0 | 4/16 | 79.67 | 0.142 |
| hybrid tau=0.90 | 29.6 | 928.9 | 7/16 | 83.72 | 0.173 |
| hybrid tau=1.00 | 100.0 | 1943.6 | 10/16 | 82.71 | 0.207 |
