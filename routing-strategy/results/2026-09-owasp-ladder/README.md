# Results: OWASP model ladder and per-unit routers (2026-09-29)

Raw outputs behind [`docs/12-owasp-ladder.md`](../../docs/12-owasp-ladder.md). One JSON
line per OWASP unit per model; verdicts and scores only, no source code. All runs:
gpu7, vLLM 0.27.1, one A100-SXM4-80GB, 32 concurrent requests, OWASP Benchmark Java
units from `graphrouter/o1_owasp_units.py` (1,956 CodeQL alerts, 1,263 real).

| Path | What |
|---|---|
| `probes/<model>-<mode>__owasp.jsonl` | 12 models on every unit: `pred_label`, `p_vuln`, tokens, seconds |
| `probes/timing.jsonl` | wall time per run; cost = fastest run of a model / 1,956 |
| `memorisation/*__owasp_anon.jsonl` | Qwen3-8B, Qwen3-32B, gpt-oss-20b with benchmark names hidden (`ladder/l0_anonymise_owasp.py`) |
| `labels.jsonl` | unit_id, category, answer key (no code) |
| `split.json` | 70/30 train/test split, stratified by category x label, seed 0 |
| `emb_minilm.npy`, `.ids.json` | MiniLM embedding per unit (`ladder/l2_embed.py`), for the kNN router |
| `routers/*.txt` | outputs of `ladder/l1_score.py` and `ladder/l3_route.py` |
| `logs/` | runner and vLLM logs; `deepseek-6.7b-verbose__owasp.tokbug.jsonl` is the superseded run with the broken tokenizer |

Reproduce (from `routing-strategy/`):

```
R=results/2026-09-owasp-ladder; A="--dir $R/probes --units $R/labels.jsonl --split $R/split.json"
python3 ladder/l1_score.py $A                                  # routers/models.txt
python3 ladder/l3_route.py $A --emb $R/emb_minilm              # routers/pool12.txt
python3 ladder/l3_route.py $A --emb $R/emb_minilm \
  --models qwen3-1.7b-nothink,qwen3-8b-nothink,qwen3-32b-nothink,gpt-oss-120b-low   # routers/ladder4.txt
# add --constants for the always-yes / always-no control (routers/*_constants.txt, baserate_*.txt)
```

Run notes:
* gpt-oss-20b's first run had ~20 requests hang ~435 s (p99 433 s vs 4.6 s in the rerun),
  so its cost comes from the names-hidden rerun (same prompts, token totals within 4%).
  Repeat runs of the same model otherwise differ by ~25% in wall time.
* deepseek-6.7b needed `tokenizer_class: PreTrainedTokenizerFast` (transformers 5 decodes
  it as sentencepiece otherwise: `Ġ` in place of spaces, 53% of verdicts unreadable).
* Model weights were staged in `/tmp/nimit/models` on gpu7 (home disk full); runner
  `ladder/run_ladder.sh`, downloader `ladder/dl.sh`.
