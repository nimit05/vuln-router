#!/bin/bash
#SBATCH --job-name=vrprobe
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1          # TP=2 sbatch --gres=gpu:2 ... for models >~35GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue            # node failure or preemption -> SLURM re-runs it;
#SBATCH --open-mode=append   # the client resumes from its own output
#
# One model's pass over one unit set -> one probe JSONL (protocol §5 schema).
#
#   MODEL=Qwen/Qwen2.5-Coder-7B-Instruct SHORT=qwen-7b UNITS=~/data/units/pilot.jsonl \
#       sbatch run_probe.sbatch
#
# Architecture is inherited from run_llmdfa.sbatch, which is known to work here:
#   * vLLM serves inside ~/containers/vllm.sif  (no fakeroot on this cluster,
#     so the image is a plain pull and cannot be rebuilt with a %post)
#   * the client runs on the HOST out of ~/llmdfa-venv, stdlib only
#   * they share the node and talk over 127.0.0.1
#
# Why SLURM and not the standalone GPU boxes: --gres gives an EXCLUSIVE GPU.
# `seconds` is the study's cost axis, so a GPU shared with another tenant makes
# every cost number a measurement of that tenant (problem doc §1.4).
set -euo pipefail

MODEL="${MODEL:?set MODEL=<hf-model-id>}"
SHORT="${SHORT:?set SHORT=<short name used in the table>}"
UNITS="${UNITS:?set UNITS=<path to a units jsonl>}"
TP="${TP:-1}"                       # must equal the requested GPU count
LIMIT="${LIMIT:-0}"                 # 0 = all units
NSAMPLES="${NSAMPLES:-1}"           # >1 populates sample_verdicts for S2
STYLE="${STYLE:-json}"              # json | cot
# CoT needs room to reason; the JSON-only prompt does not. Truncating a CoT
# completion at 64 tokens cuts off the verdict line and reads as a parse failure.
MAXTOK="${MAXTOK:-64}"
# Per-job port. SLURM can co-schedule two of these on one node; a hardcoded
# port makes the second job's health check succeed against the first job's
# server, which looks like a fast startup and produces no rows.
PORT="${PORT:-$(( 20000 + (SLURM_JOB_ID % 20000) ))}"
# vLLM's internal distributed-init socket is separate from --port and defaults
# to a fixed value; two co-scheduled jobs collide on it. Give it its own range.
export VLLM_PORT="${VLLM_PORT:-$(( 45000 + (SLURM_JOB_ID % 15000) ))}"
SERVED="$(basename "$MODEL")"
OUTDIR="$HOME/job_results/$SLURM_JOB_ID"

export HF_HOME="${HF_HOME:-$HOME/models/hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # compute nodes: TLS-intercepted
export PATH="$HOME/llmdfa-venv/bin:$PATH"
mkdir -p "$OUTDIR"

cd "$TMPDIR" || exit 1               # cluster manual: run on the node-local SSD
mkdir -p results

echo "job $SLURM_JOB_ID on $(hostname)  model=$MODEL short=$SHORT tp=$TP"
echo "units=$UNITS limit=$LIMIT n_samples=$NSAMPLES style=$STYLE max_tokens=$MAXTOK"
HW="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -s ' ' '-')"
echo "hardware=$HW  api_port=$PORT  vllm_port=$VLLM_PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- 1. vLLM, in the container, background ---------------------------------
apptainer exec --nv \
    --bind "$HF_HOME:$HF_HOME" \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env VLLM_PORT="$VLLM_PORT" \
    "$HOME/containers/vllm.sif" \
    vllm serve "$MODEL" \
        --served-model-name "$SERVED" \
        --dtype bfloat16 \
        --tensor-parallel-size "$TP" \
        --max-model-len "${MAX_MODEL_LEN:-8192}" \
        --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
        --port "$PORT" ${VLLM_EXTRA:-} \
    > "results/vllm_${SHORT}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# --- 2. wait for health, failing loudly rather than hanging the allocation --
echo "waiting for vLLM..."
UP=0
for i in $(seq 1 120); do
    if python3 -c "
import sys,urllib.request
try: urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5); sys.exit(0)
except Exception: sys.exit(1)" 2>/dev/null; then UP=1; break; fi
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died during startup:"; tail -40 "results/vllm_${SHORT}.log"; exit 1; }
    sleep 15
done
[ "$UP" = 1 ] || { echo "vLLM never became healthy:"; tail -40 "results/vllm_${SHORT}.log"; exit 1; }
echo "vLLM up after ~$((i*15))s"

# the engine version is part of the run configuration (protocol §6): two engines
# on identical weights and greedy decoding still differ, so it is recorded, not assumed
EV="$(grep -oE "LLM engine \(v[0-9]+\.[0-9]+\.[0-9]+\)" "results/vllm_${SHORT}.log" \
        | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)"
[ -n "$EV" ] || EV="$(apptainer exec "$HOME/containers/vllm.sif" vllm --version 2>/dev/null \
        | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)"
[ -n "$EV" ] || EV=unknown
echo "engine_version=$EV  served_root=$SERVED"

# --- 3. probe, sequentially, on the host -----------------------------------
OUT="$HOME/probe_out/${SHORT}__$(basename "$UNITS" .jsonl).jsonl"
mkdir -p "$HOME/probe_out"
python3 "$HOME/routing/cluster/probe_client.py" \
    --units "$UNITS" --out "$OUT" --resume \
    --model "$SHORT" --served "$SERVED" \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --limit "$LIMIT" --n-samples "$NSAMPLES" --prompt-style "$STYLE" \
    --max-tokens "$MAXTOK" \
    --engine vllm --engine-version "$EV" --hardware "$HW" \
    2>&1 | tee "results/client_${SHORT}.log"

# --- 4. collect ------------------------------------------------------------
cp -r results/. "$OUTDIR"/          # vLLM + client logs, per job
echo "probe rows: $(wc -l < "$OUT") in $OUT"
echo "job $SLURM_JOB_ID done $(date). logs -> $OUTDIR"
