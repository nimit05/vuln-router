# LLMDFA reproduction on csecluster

Target: **LLMDFA — Analyzing Dataflow in Code with LLMs**, NeurIPS 2024
([repo](https://github.com/chengpeng-wang/LLMDFA)). Phase A of `PLAN.md`, second paper.

## Layout

```
LLMDFA/
  README.md                  this file: setup, landmines, results
  scripts/                   the pipeline - deploy to ~/cluster on csecluster
    patch_llmdfa.sh            adapts an upstream LLMDFA checkout to local models
    llm_local.py               drop-in for src/utility/llm.py (vLLM backend)
    stage_weights.sh           pre-download weights on the login node
    run_llmdfa.sbatch          one (model x bug-type) run
    gate_a.sbatch              protocol-compliance probe
    gate_a_probe.py            Phase II/IV probe items
    score_llmdfa.py            metrics + comparison table
    llmdfa.def                 reference container recipe (unused: no fakeroot)
  results/
    logs/                    SLURM stdout - score_llmdfa.py reads these
    job_artifacts/           per-job reports, vLLM logs, synthesized extractors
```

**All metrics regenerate offline, no GPU and no cluster required:**

```bash
python3 scripts/score_llmdfa.py reproduction/logs/1820.out reproduction/logs/1834.out reproduction/logs/1835.out
```

To redeploy after editing: `scp -r scripts/ csecluster:~/cluster/` then re-run
`~/cluster/patch_llmdfa.sh ~/LLMDFA` if the patches changed. Set `SCRIPT_DIR` if the
scripts live somewhere other than `~/cluster`.

## Changing the pipeline (per PLAN.md)

* **swap or add a model** - `stage_weights.sh <hf-id>`, then pass `MODEL=` to either
  sbatch. Models above ~30 GB need `TP=2 sbatch --gres=gpu:2`.
* **change the sample** - `CASE_LIMIT=` and `CASE_SEED=` on `run_llmdfa.sbatch`;
  `MODE=all` runs the full benchmark.
* **ablations** - drop `-syn-parser`, `-fscot`, or `-syn-solver` from the sbatch to get
  the paper's NoSynExt / NoCoT / NoSynVal rows.
* **route per call site (Phase B/C)** - the four LLM call sites are `SrcExtractor`,
  `SinkExtractor`, `IntraFlowPropagator`, and `InterFlowValidator`, each constructing its
  own `LLM` object. Giving each its own model means changing those four constructors in
  `src/LMAgent/**` and serving more than one endpoint.

Runs on the IIT Guwahati CSE cluster (`csecluster`, login `172.16.112.202`), SLURM,
Apptainer, no sudo, `gpu-A100` partition.

## Verified cluster state (checked 2026-08-20, account `w.nimit`, `mtech`)

| Fact | Value |
|---|---|
| Partitions available to `mtech` | `cpu`, `gpu-P100`, **`gpu-A100`** (no H100) |
| A100 nodes | `gpu-A100-01`, `gpu-A100-02` — **2 x A100-PCIE-40GB each**, 48 CPUs, 128 GB RAM |
| GPU visibility | `--gres=gpu:2` exposes both (`CUDA_VISIBLE_DEVICES=0,1`) |
| `$TMPDIR` on compute node | `/scratch/local/$USER/<jobid>`, 2.2 TB local SSD |
| Home filesystem | `storage:/userhome`, 18 TB, but the **50/200 GB user quota** is what binds |
| Login node internet | works (HTTPS 200) |
| **Compute node internet** | **broken for HTTPS** — TLS-intercepting proxy, certificate hostname mismatch |
| Apptainer | `/usr/bin/apptainer` on both login and compute nodes |

Two consequences: weights **must** be pre-staged and the job runs with
`HF_HUB_OFFLINE=1`; and the cards are **40 GB, not 80 GB**, so anything above ~30 GB of
weights needs `TP=2` across both GPUs.

## Cluster constraints that shaped these scripts

| Constraint | Consequence |
|---|---|
| Home quota 50 GB soft / **200 GB hard** | Stage **one** model at a time, delete before the next |
| Max 2 jobs queued, **1 running** | Model ladder runs strictly sequentially |
| Jobs must `cd "$TMPDIR"`, write to `./results` | Results auto-copy to `~/job_results/<jobid>/` |
| No sudo | Everything in one `.sif`; nothing installed on the node |
| Compute nodes may lack internet | Weights pre-staged on the login node; grammar compiled at container build time |

## Architecture (forced by the cluster)

The login node has **no fakeroot** (`/etc/subuid` has no entry), so an Apptainer `.def`
with a `%post` section cannot be built. The pipeline is therefore split:

* **`~/containers/vllm.sif`** — a plain pull of `docker://vllm/vllm-openai:v0.27.1`
  (no `%post`, so it builds unprivileged). Serves the model on `127.0.0.1:8000`.
* **`~/llmdfa-venv`** — a host virtualenv (python 3.12) with the LLMDFA client deps.
  The client needs no GPU, so it runs outside the container.
* **`~/LLMDFA/lib/build/my-languages.so`** — the tree-sitter Java grammar, compiled
  once on the login node with the system gcc.

Both halves live in the same SLURM job on the same node and talk over localhost.
`llmdfa.def` is kept for reference/portability but is **not** used here.

## Five environment landmines (each cost a failed job)

1. **No fakeroot** → no `%post`. Hence the split above.
2. **`openai==1.51.0` + `httpx>=0.28`** → `Client.__init__() got an unexpected keyword
   argument 'proxies'`. Pin `httpx==0.27.2`.
3. **`tree-sitter==0.20.2` uses `distutils`**, gone in python 3.12. Install `setuptools`,
   which ships the shim.
4. **`tiktoken` downloads its BPE file on first use** and compute-node HTTPS is
   TLS-intercepted. Pre-cache on the login node, export `TIKTOKEN_CACHE_DIR`.
5. **Four separate argparse/import traps in LLMDFA itself** — `synthesis/llm.py` imports
   `google.generativeai` and `anthropic` at module level, reads `OPENAI_API_KEY` at import
   time, and both `synthesis/main.py --model` and `run_llmdfa.py --model-name` have
   `choices=` whitelists that reject any local model name. All handled by
   `patch_llmdfa.sh`; the API key just needs a dummy value.

Also note: the repo README documents the flag as `-syn-solve`, but the code defines
**`-syn-solver`**.

## Results so far (Qwen2.5-Coder-7B-Instruct, bf16, 1x A100-40GB)

**Gate A — protocol compliance (job 1812): PASS**

| Phase | Metric | Result |
|---|---|---|
| I extractor synthesis | xss_src / xss_sink | both synthesized, **1 iteration, zero fixes** |
| II Yes/No | parse rate / accuracy | **6/6 / 6/6**, 1.2 s per call |
| IV Z3 script | well-fenced | 3/3 |
| IV Z3 script | runs first try | **1/3** — both failures were `Z3Exception: Symbolic expressions cannot be cast to concrete Boolean values` (python `and`/`not` applied to Z3 exprs) |

The Z3 weakness is the one real finding: LLMDFA's 3-round refine loop exists precisely for
this, so it is not disqualifying, but it is where a bigger model may earn its cost.

**Gate B — 10-case XSS demo (job 1814): PASS**

```
reported TPs=10 FPs=0   ground truth TPs=10 (19 easy-FP traps avoided)
precision=100%  recall=100%  F1=1.000
tokens/case: 2559 in, 335 out      time/case: 4.67 s (min 2.9, max 5.3)
empty_responses=0  failed_calls=0
```

These are the first 10 of 666 XSS cases, so this is a smoke test, not a result. But the
measured **4.67 s per case is 6-13x faster than the pre-run estimate**, which changes the
budget: a full single-model run is roughly XSS 52 min + OSCI ~37 min + DBZ ~8-10 h
(DBZ cases are ~3x heavier), i.e. **~10-12 h — one 24 h job, no sharding needed**.

## Gate C results — Qwen2.5-Coder-7B-Instruct (37-case seeded sample per bug type)

Jobs 1820 (XSS), 1834 (OSCI), 1835 (DBZ). `CASE_SEED=42`, bf16, 1x A100-40GB.
Zero empty responses and zero failed calls on all three.

| bug | TP | FP | dup traces | precision | recall | F1 | paper gpt-3.5 (full set) | dF1 |
|---|---|---|---|---|---|---|---|---|
| DBZ | 37 | 48 | 11 | 43.53% | 100.00% | 0.607 | 73.75% / 92.16% / 0.82 | -0.213 |
| XSS | 32 | 1 | 1 | 96.97% | 86.49% | 0.914 | 100.00% / 92.31% / 0.96 | -0.046 |
| OSCI | 31 | 10 | 1 | 75.61% | 83.78% | 0.795 | 100.00% / 78.38% / 0.88 | -0.085 |

Cost: DBZ 34,174 in / 3,035 out tokens per case at 41.5 s; XSS 7,479 / 796 at 10.4 s;
OSCI 4,995 / 469 at 6.2 s.

**Scoring caveat (important).** LLMDFA's `analysis_result["TPs"]` counts reported
*traces*, not distinct bugs: several paths can reach the same labelled sink, so a case
holding one ground-truth bug can report `TPs=2` (11 of 37 DBZ cases did). Summing raw
traces against per-case ground truth gives recall above 100%, which is how this was
caught. True positives are therefore **capped per case** at that case's ground truth,
and the discarded duplicates are reported in the `dup traces` column. The paper does not
state its aggregation, so this is a disclosed deviation, not a claimed match.

**The DBZ result has a diagnosed cause.** Gate A found this model writes Z3 that runs on
only 1 of 3 first attempts. DBZ is the one bug type where path feasibility does real work
(the paper's own NoSynVal ablation shows XSS/OSCI contain no infeasible bug paths), and
the run logs confirm the mechanism:

| bug | solver checks | fell back to LLM | refinement rounds |
|---|---|---|---|
| DBZ | 203 | **36 (17.7%)** | 267 |
| XSS | 42 | 0 | 1 |
| OSCI | 48 | 1 | 7 |

The paper reports **0.61%** of DBZ scripts exhausting three refinement rounds with
gpt-3.5; ours is 17.7%, roughly 29x higher. Weak Z3 synthesis means infeasible paths
survive validation, so false positives accumulate and precision falls to 43.53% against
the paper's 73.75% — while recall is a perfect 100%. The failure is localised to one
call site, which is precisely the per-call-site routing signal PLAN.md Phase C needs.

## Order of operations

```bash
# 0. verify (do this first, the manual's node table and account table disagree)
ssh csecluster
sacctmgr show user $USER withassoc          # confirm gpu-A100 access
sinfo -o "%P %N %G %m %t"                   # confirm the A100 node exists
srun -p gpu-A100 --gres=gpu:1 --pty nvidia-smi   # 40GB or 80GB card?

# 1. container (build on the LOGIN node: x86_64, matches the compute nodes;
#    an Apple-Silicon laptop would produce an arm64 image that cannot run here)
export APPTAINER_CACHEDIR=$TMPDIR/apptainer-cache   # keep ~10GB out of the quota
mkdir -p ~/containers
apptainer build ~/containers/llmdfa.sif cluster/llmdfa.def   # add --fakeroot if needed
apptainer cache clean -f

# 2. source + patches
git clone https://github.com/chengpeng-wang/LLMDFA.git ~/LLMDFA
cluster/patch_llmdfa.sh ~/LLMDFA

# 3. weights (one model at a time; compute nodes CANNOT download)
cluster/stage_weights.sh Qwen/Qwen2.5-Coder-7B-Instruct

# 4. smoke test: 10 XSS cases, ~minutes
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct BUG=xss MODE=single sbatch cluster/run_llmdfa.sbatch

# 5. full run, per bug type
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct BUG=xss MODE=all sbatch cluster/run_llmdfa.sbatch

# 6. the 32B rung needs both GPUs
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct BUG=xss MODE=all TP=2 \
    sbatch --gres=gpu:2 cluster/run_llmdfa.sbatch
```

Run order across bug types: **XSS -> OSCI -> DBZ**. XSS/OSCI are ~1/3 the size and the
paper reports 100% precision on both, so a broken pipeline shows up immediately. DBZ is
the expensive one and the only type where the Z3 stage materially matters.

## Model ladder

The paper used gpt-3.5-turbo-0125, gpt-4-turbo, gemini-1.0-pro, claude-3-opus — all
closed. **No open model is a 1:1 substitute**; this is a capability ladder, reported as a
substitution, not a claimed equivalence.

| Rung | Model | bf16 VRAM | Note |
|---|---|---|---|
| Rung | Model | bf16 weights | Fits how |
|---|---|---|---|
| small | `deepseek-ai/deepseek-coder-6.7b-instruct` | ~13 GB | 1 GPU — same model IRIS used at Q4, so this directly tests the quantization hypothesis |
| small | `Qwen/Qwen2.5-Coder-7B-Instruct` | ~15 GB | 1 GPU |
| mid | `Qwen/Qwen2.5-Coder-14B-Instruct` | ~29 GB | 1 GPU (tight — cap `MAX_MODEL_LEN`) |
| large | `Qwen/Qwen2.5-Coder-32B-Instruct` | ~62 GB | **`TP=2 sbatch --gres=gpu:2`**, ~31 GB/card |

All ungated. The 32B rung survives only via tensor parallelism over both 40 GB cards;
over PCIe (no NVLink) it will be slower per token, which is worth recording since
wall-clock is one of the cost axes.

## Deviations from the published code

Applied by `patch_llmdfa.sh`, original kept as `src/utility/llm.py.orig`:

1. **Local inference backend.** Upstream `infer()` dispatches on substring (`"gpt" in
   name`, `"gemini" in name`, `"claude" in name`); an open-weight model name matches none
   and it silently returns `""`. Replaced with an OpenAI-compatible client pointed at vLLM.
2. **Real token counts.** Upstream measures cost with `tiktoken`'s gpt-3.5 encoder applied
   to the prompt string — the wrong tokenizer for these models. Now taken from the
   server's `usage` field. The cost axis is the whole point of Phases B and C.
3. **Bounded retries.** `intra_flow_propagator.py` and `inter_flow_validator.py` both loop
   `while True` and `continue` whenever the response holds no bare `Yes`/`No`. At
   temperature 0 that is an infinite non-terminating loop. Now capped at
   `LLMDFA_MAX_RETRY` (default 5), failing closed, with counters printed.
4. **`<think>` stripping.** Reasoning models otherwise put the answer where the parser
   cannot see it — the failure that cost a full QASecClaw run.

## Sanity checks before trusting any number

- `results/empty_response_count.txt` and `failed_call_count.txt` must be ~0. Non-zero means
  silent fail-open, not model weakness.
- Phase I (source/sink extraction) should score **100% precision and recall** — the paper
  reports 100% for all 3 bug types on all 4 models, because the extractor is a synthesized
  tree-sitter script validated against labelled examples. Anything less means the synthesis
  or the grammar build is broken, and every later number is worthless.
