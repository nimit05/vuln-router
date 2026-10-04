# Scaled IRIS run (E28), 2026-10-02/03, gpu7

Which IRIS stage should get the big model? Write-up: `../../docs/14-per-cve-and-stages.md`.

| file | what |
|---|---|
| `scaled.txt` | `ladder/l11_scaled.py` output: spec stage, whole pipelines, per-CVE comparisons |
| `projects.txt` | the 80 CWE-Bench-Java projects with a built CodeQL database, biggest first |
| `status/` | per-project exit codes and seconds: `s_q8.csv`/`s_q32.csv` LLM labelling, `*_cq.csv` CodeQL, `*_cq_retry.csv` CodeQL re-run with 64 GB; phase flags and start/end times |
| `probe/timing.jsonl` | filter-stage GPU seconds (`dedup` rows = measured; per-run rows = split by distinct slices, part estimated) |
| `data/<run>/iris_truth.json`, `validate.txt` | IRIS's own per-project accounting and the check that our path set equals IRIS's |

`s_q8` = Qwen3-8B spec, `s_q32` = Qwen3-32B spec; `*_full` adds the project recovered by the
CodeQL retry. Not copied here (5.8 GB): every path (`data/*/units_paths.jsonl`) and every score
(`probe/*.jsonl`), kept on gpu7 in `~/nimit/scaled_results/` (and `/tmp/nimit/scaled/`, which can vanish).

Reproduce: `ladder/run_scaled_iris.sh all` (prep, window 1, CPU worker), `retry` for CodeQL
failures, `ladder/run_window2.sh` (filter stage), then
`python ladder/l11_scaled.py --dir <run dir>`.
