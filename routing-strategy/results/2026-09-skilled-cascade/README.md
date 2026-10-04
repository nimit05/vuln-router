# Results: skilled models, cascade, memorisation control (2026-09-26 → 27)

Raw outputs behind [`docs/10-graphrouter-v2.md`](../../docs/10-graphrouter-v2.md) and
[`docs/11-findings-summary.md`](../../docs/11-findings-summary.md). One JSON line per
IRIS path per model; scores only, no source code. All runs: vLLM 0.27.1, A100-SXM4-80GB,
16 CWE-Bench-Java projects, 2,257 paths, 344 real vulnerabilities.

| Path | What | Used for |
|---|---|---|
| `probes/<model>-<mode>__units_paths.jsonl` | Qwen3-32B, Qwen3-8B, gpt-oss-20b, phi-4, Mistral-Nemo-12B on every path: `p_vuln`, `pred_label`, tokens, seconds | skill gate (AUC), cascade, GraphRouter v2/v3 |
| `probes/qwen3-32b-think__gate.jsonl` | 32B thinking mode on the 358-path gate sample | thinking vs no-thinking |
| `probes/calib_*.jsonl` | small calibration runs per model | GPU-seconds per token |
| `probes/cost_per_token.json` | GPU-s per input and output token | cost column everywhere |
| `memorisation/qwen3-*__{orig,anonA,anonB}.jsonl` | names kept / renamed / strict-renamed | memorisation control (section 10) |
| `routers/d2_*.json` | GraphRouter v1/v2/v3 result rows per pool | sections 2, 4, 7 |
| `q6_budget_recall.csv`, `q6_perproject.csv` | fixed-budget recall vs matched random | E19 |
| `iris_truth.json` | per-project ground truth (paths, true-positive paths, CWE) | all IRIS metrics |
| `d1_gate_ids.json` | the 358 gate paths | thinking comparison |
| `logs/` | run logs of each chain | provenance |

The older 5-model pool (qwen-1.5b, phi-3.8b, qwen-7b, granite-8b, deepseek-6.7b) is in
`data/bench/` and `data/gr/probe*` (not tracked; regenerate with `graphrouter/b*`).

Reproduce the headline numbers (both scripts also need `data/gr/slices.jsonl`, the
path labels, which `graphrouter/b0..b3b` regenerate):

```
R=results/2026-09-skilled-cascade
# AUC 0.868 / 0.687, cascade table (recall@10 0.677 at f = 0.3), IRIS top-k table
python graphrouter/d4_skilled_combos.py --small $R/probes/qwen3-8b-nothink__units_paths.jsonl \
  --large $R/probes/qwen3-32b-nothink__units_paths.jsonl --cost $R/probes/cost_per_token.json
# memorisation control: 0.868 -> 0.795 -> 0.719 (32B), 0.687 -> 0.544 -> 0.511 (8B)
python graphrouter/d6_compare.py --d6 $R/memorisation --d1 $R/probes
```
