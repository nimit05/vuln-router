# Checkpoint — Llama-3.3-70B attempt (halted, not a result)

Date: 2026-08-17. Project attempted: `zeroturnaround__zt-zip_CVE-2018-1002201_1.12`, CWE-022.
Run id: `l33_70b`. Output dir preserved at `IRIS/output/<project>/l33_70b/`.

## Outcome: NO usable result. Halted after 31 minutes with 0 completed LLM calls.

## Why — the model does not fit, so it silently ran on CPU

`GET /api/ps` during the run:

| field | value |
|---|---|
| model | `llama3.3:latest` (70.6B, Q4_K_M) |
| total size | 102,927,681,536 B = **95.9 GiB** |
| size_vram | 82,978,670,592 B = **77.3 GiB** |
| context_length | **131072** |

The A100 has 80 GB. A 95.9 GiB model cannot be resident, and Ollama additionally
defaulted to a 131k context, whose KV cache made it worse. Ollama silently split the
model across GPU and CPU instead of failing.

Symptoms:
- `ollama runner` at **10135% CPU** (~101 cores saturated)
- `nvidia-smi` utilisation **0%** while 63 GB VRAM stayed allocated
- python parent blocked in `futex_wait_queue_me`
- **0** raw LLM responses in 31 min (expected ~13 calls: 237 API candidates / batch 30,
  plus 99 func-param candidates / batch 20)

The trap: it looks like a hung job, and 0% GPU looks like nothing is running, but it is
grinding on CPU at roughly 1/100th speed.

## Fix applied for future large-model runs

`src/models/ollama.py` now pins `num_ctx` (default 16384, override `IRIS_NUM_CTX`),
bounding the KV cache instead of accepting the model's advertised maximum.

## To retry 70B properly, do ONE of:
1. `IRIS_NUM_CTX=8192`, then confirm via `/api/ps` that `size_vram` >= total size;
2. use a smaller quantisation that fits under 80 GB;
3. accept CPU offload and budget ~100x runtime (not advisable).

## Wider lesson
`num_ctx` was unset before this. IRIS batches 30 APIs per labelling prompt, which is long,
so Ollama's default context would TRUNCATE those prompts and the model never saw most of
the candidate list. That is a plausible contributor to the very sparse taint specs seen
with deepseek-coder-7b (4 sources / 3 sinks on zt-zip). All runs after this checkpoint
use num_ctx=16384.
