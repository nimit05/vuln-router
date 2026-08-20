# RESUME — state as of 2026-08-17 (read this first)

## Where everything lives
- Server working tree: `~/nimit/vuln-pred-results/IRIS`
- Results (the things that matter): `~/nimit/vuln-pred-results/results/IRIS/`
- Local mirror on the Mac: `~/Documents/projects/vuln-pred-results/results/IRIS/`
  (auto-rsync every 5 min via `gpu-backup-sync.sh`)

## Status
| item | state |
|---|---|
| CodeQL baseline (LLM-free) | **DONE — reproduces the paper exactly** (3/17, FDR 70.00, F1 0.081) |
| IRIS + deepseek-coder-7b, first pass | DONE (2/17) — filter was broken, see below |
| IRIS + deepseek-coder-7b, filter re-run | 16/17 done; `apache__rocketmq` was last |
| Llama-3.3-70B | ABANDONED — see `CHECKPOINT_llama33_70b.md` |

## What is running unattended
`/tmp/finalize.sh` (launched with setsid, survives SSH disconnect). It waits up to 2h for
`ROCKET_DONE` in `/tmp/rocketmq.log`, then writes the scored comparison to
`results/IRIS/metrics_iris_deepseek7b.txt` and appends `FINALIZE_DONE` to
`results/IRIS/_run_status.txt`.

**To check if it finished:**
```bash
cat ~/nimit/vuln-pred-results/results/IRIS/_run_status.txt
cat ~/nimit/vuln-pred-results/results/IRIS/metrics_iris_deepseek7b.txt
```

**If it did NOT finish (e.g. rocketmq died), just score the other 16:**
```bash
cd ~/nimit/vuln-pred-results/IRIS
export PATH=$PWD/codeql:$PATH PYTHONPATH=$PWD:$PWD/src OLLAMA_HOST=http://localhost:11434
./.conda-iris/bin/python ../results/IRIS/score_subset.py --run-id dsc7bfull \
   --label "OURS: IRIS+deepseek-coder-7b" --compare IRIS+DeepSeekCoder-7B.csv CodeQL.csv
```
The scorer tolerates missing projects (it warns and scores what exists).

## How to re-run anything
Project list: `results/IRIS/subset_projects.txt` (17 usable of 23; 6 had no CodeQL db).

```bash
cd ~/nimit/vuln-pred-results/IRIS
export PATH=$PWD/codeql:$PATH PYTHONPATH=$PWD:$PWD/src OLLAMA_HOST=http://localhost:11434

# LLM-free baseline for one project
./.conda-iris/bin/python src/codeql_vul.py --query cwe-022wCodeQL --overwrite <SLUG>

# IRIS with an LLM for one project
./.conda-iris/bin/python src/neusym_vul.py --query cwe-022wLLM --run-id <ID> \
    --llm ollama-deepseekcoder-7b <SLUG>

# only redo the false-positive filter (specs + CodeQL results are cached — fast)
./.conda-iris/bin/python src/neusym_vul.py --query cwe-022wLLM --run-id <ID> \
    --llm <MODEL> --overwrite-posthoc-filter <SLUG>
```

## Adding another model (no Python editing)
Edit `IRIS/models.json`: `"ollama-yourname": "ollama-tag"`, after `ollama pull <tag>`.
`src/models/ollama.py` merges that file at import.

## Settings that MATTER (all learned the hard way — see FINDINGS.md)
- `IRIS_NUM_CTX` (default 16384) — lower it for big models; the default Ollama context
  truncates IRIS's long prompts and silently destroys spec quality.
- `IRIS_NUM_PREDICT` (default 1024) — unlimited caused an 18-minute stall on one path.
- Models >80GB (e.g. llama3.3:70b at 95.9GiB) silently offload to CPU and run ~100x slower.
  Check `curl -s localhost:11434/api/ps` and confirm `size_vram` >= total size.

## Known-good facts to compare against
Paper per-project CSVs are in `IRIS/results/*.csv`. The scorer restricts them to whatever
projects we actually ran, so comparisons are always like-for-like.
