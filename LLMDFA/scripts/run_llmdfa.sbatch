#!/bin/bash
#SBATCH --job-name=llmdfa
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1        # override for TP=2: sbatch --gres=gpu:2 ...
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#
# One (model x bug-type) LLMDFA run.
#
#   MODEL=Qwen/Qwen2.5-Coder-7B-Instruct BUG=xss MODE=single sbatch run_llmdfa.sbatch
#
# Architecture (constrained by the cluster: no fakeroot, so no %post in a .def):
#   * vLLM runs INSIDE ~/containers/vllm.sif, a plain pull of vllm/vllm-openai
#   * the LLMDFA client runs on the HOST out of ~/llmdfa-venv (no GPU needed)
#   * both live in the same job on the same node and talk over 127.0.0.1
#
# GPUs are A100-PCIE-40GB, 2 per node. Models >~30GB need both:
#   TP=2 sbatch --gres=gpu:2 run_llmdfa.sbatch
set -euo pipefail

MODEL="${MODEL:?set MODEL=<hf-model-id>}"
BUG="${BUG:-xss}"                  # xss | osci | dbz
MODE="${MODE:-single}"             # single (10 cases) | all
TP="${TP:-1}"                      # must equal the requested GPU count
SERVED="$(basename "$MODEL")"

export HF_HOME="${HF_HOME:-$HOME/models/hf}"
# LLMDFA's Z3 stage shells out to `python`; it must be an interpreter that has
# z3-solver installed, or every path check silently falls back to the LLM.
export PATH="$HOME/llmdfa-venv/bin:$PATH"
# Compute nodes reach HTTPS only through a TLS-intercepting proxy (cert hostname
# mismatch), so weights must be pre-staged and the libraries told not to phone home.
# synthesis/llm.py reads OPENAI_API_KEY at import time; a dummy value is enough
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
# tiktoken downloads its BPE file on first use; compute-node HTTPS is intercepted,
# so point it at a cache populated on the login node.
export TIKTOKEN_CACHE_DIR="$HOME/models/tiktoken"
export LLMDFA_CASE_LIMIT="${CASE_LIMIT:-10}"   # cases when MODE=single
# an empty LLMDFA_CASE_SEED would reach int("") and crash, so only export it when set
if [ -n "${CASE_SEED:-}" ]; then export LLMDFA_CASE_SEED="$CASE_SEED"; fi
# sharding for long runs: cases are independent, so a bug type can be split
if [ -n "${CASE_OFFSET:-}" ]; then export LLMDFA_CASE_OFFSET="$CASE_OFFSET"; fi
if [ -n "${CASE_COUNT:-}" ]; then export LLMDFA_CASE_COUNT="$CASE_COUNT"; fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# === CRITICAL (cluster manual): run on the node-local SSD ===
cd "$TMPDIR" || exit 1
mkdir -p ./results

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "model=$MODEL bug=$BUG mode=$MODE tp=$TP cases=$LLMDFA_CASE_LIMIT seed=${CASE_SEED:-none} shard=${CASE_OFFSET:-0}+${CASE_COUNT:-all}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import z3,tree_sitter; print('client deps ok')"

cp -r "${LLMDFA_SRC:-$HOME/LLMDFA}" ./LLMDFA
test -f ./LLMDFA/lib/build/my-languages.so || { echo "grammar .so missing"; exit 1; }

# --- 1. vLLM server, in the container, background ---------------------------
apptainer exec --nv \
    --bind "$HF_HOME:$HF_HOME" \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$HOME/containers/vllm.sif" \
    vllm serve "$MODEL" \
        --served-model-name "$SERVED" \
        --dtype bfloat16 \
        --tensor-parallel-size "$TP" \
        --max-model-len "${MAX_MODEL_LEN:-16384}" \
        --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
        --port 8000 ${VLLM_EXTRA:-} \
    > "./results/vllm_${SERVED}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

# --- 2. wait for it to load (30 min ceiling) --------------------------------
echo "waiting for vLLM..."
UP=0
for i in $(seq 1 120); do
    if python -c "
import sys,urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5); sys.exit(0)
except Exception: sys.exit(1)
" 2>/dev/null; then UP=1; break; fi
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died during startup:"; tail -40 "./results/vllm_${SERVED}.log"; exit 1; }
    sleep 15
done
[ "$UP" = 1 ] || { echo "vLLM never became healthy:"; tail -40 "./results/vllm_${SERVED}.log"; exit 1; }
echo "vLLM up after ~$((i*15))s"

# --- 3. run LLMDFA on the host against it ----------------------------------
cd ./LLMDFA/src
LLMDFA_BASE_URL=http://127.0.0.1:8000/v1 \
LLMDFA_MAX_RETRY="${LLMDFA_MAX_RETRY:-5}" \
    python run_llmdfa.py \
        --bug-type "$BUG" \
        --model-name "$SERVED" \
        -syn-parser -fscot -syn-solver \
        --solving-refine-number 3 \
        --analysis-mode "$MODE" \
    2>&1 | tee "$TMPDIR/results/llmdfa_${SERVED}_${BUG}_${MODE}.log"
cd "$TMPDIR"

# --- 4. collect ------------------------------------------------------------
cp -r ./LLMDFA/log "./results/llmdfa_log_${SERVED}_${BUG}_${MODE}" 2>/dev/null || true
LOG="./results/llmdfa_${SERVED}_${BUG}_${MODE}${CASE_OFFSET:+_off$CASE_OFFSET}.log"
{
  echo "empty_responses=$(grep -c 'empty response' "$LOG" || true)"
  echo "failed_calls=$(grep -c 'inference FAILED' "$LOG" || true)"
} > ./results/sanity_counters.txt
cat ./results/sanity_counters.txt

echo "Job completed at $(date). Results -> ~/job_results/$SLURM_JOB_ID/"
