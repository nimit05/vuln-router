# Controls for the Hybrid LLM router + project floor

small = `qwen-7b`, large = `granite-8b`, router = `h5:struct+small+rank:linear`, loss `det_2cls` t=0, leave-one-project-out.

`floor-only` keeps the small model everywhere except the k riskiest paths per project.
`random, matched budget` spends the same number of large-model calls, floor included, on paths drawn uniformly at random, over 25 draws (mean +- sd).
Metrics are IRIS Sec 3.6 via their own `score_subset.metrics`.

| Configuration | %large | GPU-s | #Det | AvgFDR% | AvgF1 |
|---|---|---|---|---|---|
| always-qwen-7b | 0.0 | 569.2 | 4/16 | 79.60 | 0.142 |
| always-granite-8b | 100.0 | 1943.6 | 10/16 | 82.71 | 0.207 |
| | | | | | |
| router+floor  tau=0.85 k=10 | 47.9 | 1329.8 | 10/16 | 79.00 | 0.244 |
|   floor-only (always-small) k=10 | 5.1 | 590.4 | 9/16 | 79.14 | 0.237 |
|   random, matched budget | 47.9 | 1198.2 | 10.0/16 | -- | 0.216+-0.006 |
| **margin over floor-only +0.007, over random +0.028 (+4.5 sd)** | | | | | |
| | | | | | |
| router+floor  tau=0.90 k=5 | 60.5 | 1506.0 | 10/16 | 80.39 | 0.232 |
|   floor-only (always-small) k=5 | 2.7 | 578.6 | 7/16 | 78.03 | 0.241 |
|   random, matched budget | 60.5 | 1390.8 | 10.0/16 | -- | 0.209+-0.008 |
| **margin over floor-only -0.010, over random +0.023 (+2.8 sd)** | | | | | |
| | | | | | |
| router+floor  tau=0.95 k=5 | 79.7 | 1733.4 | 10/16 | 80.10 | 0.234 |
|   floor-only (always-small) k=5 | 2.7 | 578.6 | 7/16 | 78.03 | 0.241 |
|   random, matched budget | 79.7 | 1655.4 | 10.0/16 | -- | 0.206+-0.006 |
| **margin over floor-only -0.007, over random +0.028 (+5.0 sd)** | | | | | |
| | | | | | |
| router+floor  tau=0.70 k=10 | 26.3 | 1118.5 | 9/16 | 78.65 | 0.247 |
|   floor-only (always-small) k=10 | 5.1 | 590.4 | 9/16 | 79.14 | 0.237 |
|   random, matched budget | 26.3 | 893.2 | 10.0/16 | -- | 0.224+-0.006 |
| **margin over floor-only +0.010, over random +0.023 (+3.6 sd)** | | | | | |
| | | | | | |
| router+floor  tau=0.00 k=5 | 2.7 | 578.6 | 7/16 | 78.03 | 0.241 |
|   floor-only (always-small) k=5 | 2.7 | 578.6 | 7/16 | 78.03 | 0.241 |
|   random, matched budget | 2.7 | 578.6 | 7.0/16 | -- | 0.241+-0.000 |
| **margin over floor-only +0.000, over random +0.000 (+nan sd)** | | | | | |
| | | | | | |
