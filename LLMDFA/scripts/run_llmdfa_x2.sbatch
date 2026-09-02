#!/bin/bash
#SBATCH --job-name=llmdfa2
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#
# Two-GPU variant of run_llmdfa.sbatch: one (model x bug-type) run split across
# BOTH GPUs of a node as two independent shards.
#
# Why: csecluster's mtech QOS allows only 2 SUBMITTED JOBS per user, but each
# A100 node has 2 GPUs. Requesting gpu:1 therefore leaves half the node's
# capacity unreachable - the cap is on jobs, not on GPUs. Asking for gpu:2 and
# running two shards inside one job doubles throughput within the same cap.
#
# Why not just raise concurrency against one server: per-case wall time is the
# cost axis for the router comparison. Batching many concurrent requests into
# one vLLM server raises throughput but changes what "seconds per case" means,
# making these runs incomparable to the sequential ones already measured. Two
# servers on two GPUs, each serving ONE sequential client, keeps per-case timing
# identical to run_llmdfa.sbatch while halving wall clock.
#
#   MODEL=Qwen/Qwen2.5-Coder-7B-Instruct BUG=dbz TOTAL=300 sbatch run_llmdfa_x2.sbatch
#
# TOTAL is the number of cases to analyse, taken from the SORTED case list and
# split in half. Sorted order groups a form's variants contiguously, so a
# prefix of N cases covers ~N/37 complete forms with ALL 37 control-flow
# variants present - which is the axis that carries difficulty (paper Sec 4.2:
# other programs "only differ ... in terms of sources and sinks"). That is a
# better sample than a seeded shuffle, and it shards cleanly.
set -euo pipefail

MODEL="${MODEL:?set MODEL=<hf-model-id>}"
BUG="${BUG:-xss}"
TOTAL="${TOTAL:?set TOTAL=<number of cases to analyse>}"
SERVED="$(basename "$MODEL")"

HALF=$(( TOTAL / 2 ))
REST=$(( TOTAL - HALF ))

export HF_HOME="${HF_HOME:-$HOME/models/hf}"
export PATH="$HOME/llmdfa-venv/bin:$PATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export TIKTOKEN_CACHE_DIR="$HOME/models/tiktoken"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LLMDFA_ABORT_AFTER_FAILURES="${LLMDFA_ABORT_AFTER_FAILURES:-20}"

cd "$TMPDIR" || exit 1
mkdir -p ./results

# Rename the job after the fact so `squeue` distinguishes concurrent runs.
# A hardcoded name made two different models look like duplicate submissions to
# other users sharing the partition.
scontrol update JobId="$SLURM_JOB_ID" JobName="dfa2-${SERVED:0:12}-${BUG}" 2>/dev/null || true

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "model=$MODEL bug=$BUG mode=shardx2 tp=1 cases=$TOTAL seed=none shard=0+$HALF,$HALF+$REST"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c "import z3,tree_sitter; print('client deps ok')"

# --- 1. one vLLM server per GPU -------------------------------------------
PIDS=()
for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i apptainer exec --nv \
        --bind "$HF_HOME:$HF_HOME" \
        --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        --env CUDA_VISIBLE_DEVICES=$i \
        "$HOME/containers/vllm.sif" \
        vllm serve "$MODEL" \
            --served-model-name "$SERVED" \
            --dtype bfloat16 \
            --tensor-parallel-size 1 \
            --max-model-len "${MAX_MODEL_LEN:-16384}" \
            --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
            --port $(( 8000 + i )) ${VLLM_EXTRA:-} \
        > "./results/vllm_${SERVED}_gpu${i}.log" 2>&1 &
    PIDS+=($!)
done
trap 'for p in "${PIDS[@]}"; do kill $p 2>/dev/null || true; done' EXIT

# --- 2. wait for both to load ---------------------------------------------
for i in 0 1; do
    port=$(( 8000 + i ))
    echo "waiting for vLLM on :$port ..."
    UP=0
    for t in $(seq 1 160); do
        if python -c "
import sys,urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:$port/health', timeout=5); sys.exit(0)
except Exception: sys.exit(1)
" 2>/dev/null; then UP=1; break; fi
        kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "vLLM gpu$i died:"; tail -40 "./results/vllm_${SERVED}_gpu${i}.log"; exit 1; }
        sleep 15
    done
    [ "$UP" = 1 ] || { echo "vLLM gpu$i never healthy:"; tail -40 "./results/vllm_${SERVED}_gpu${i}.log"; exit 1; }
    echo "vLLM :$port up"
done

# --- 3. two client shards, one per server ---------------------------------
# Separate working copies: each client writes into its own LLMDFA/log tree.
run_shard () {
    local idx="$1" off="$2" cnt="$3" port=$(( 8000 + $1 ))
    cp -r "${LLMDFA_SRC:-$HOME/LLMDFA}" "./LLMDFA_$idx"
    test -f "./LLMDFA_$idx/lib/build/my-languages.so" || { echo "grammar .so missing"; return 1; }
    cd "./LLMDFA_$idx/src"
    # Both shards also stream to the job's stdout, where they INTERLEAVE. The
    # scorer pairs each result dict with the .java path logged just above it, so
    # an interleaved %j.out would mis-attribute cases across shards. Score the
    # per-shard logs instead - which means each needs its own header line.
    {
      echo "model=$MODEL bug=$BUG mode=shardx2 tp=1 cases=$cnt seed=none shard=$off+$cnt"
      LLMDFA_BASE_URL="http://127.0.0.1:$port/v1" \
      LLMDFA_MAX_RETRY="${LLMDFA_MAX_RETRY:-5}" \
      LLMDFA_CASE_OFFSET="$off" \
      LLMDFA_CASE_COUNT="$cnt" \
          python run_llmdfa.py \
              --bug-type "$BUG" \
              --model-name "$SERVED" \
              -syn-parser -fscot -syn-solver \
              --solving-refine-number 3 \
              --analysis-mode all
    } 2>&1 | tee "$TMPDIR/results/llmdfa_${SERVED}_${BUG}_shard${idx}.log"
    cd "$TMPDIR"
}

run_shard 0 0      "$HALF" &
C0=$!
run_shard 1 "$HALF" "$REST" &
C1=$!
# `set -e` must not abort before collection runs: a shard that trips the
# dead-server watchdog exits non-zero, and that is exactly when the partial
# logs and counters are most worth keeping.
wait $C0 && R0=0 || R0=$?
wait $C1 && R1=0 || R1=$?
echo "shard exit codes: 0->$R0 1->$R1"

# --- 4. collect ------------------------------------------------------------
for i in 0 1; do
    cp -r "./LLMDFA_$i/log" "./results/llmdfa_log_${SERVED}_${BUG}_shard$i" 2>/dev/null || true
done
{
  for i in 0 1; do
    L="./results/llmdfa_${SERVED}_${BUG}_shard${i}.log"
    echo "shard$i cases=$(grep -c "^{'input_token_cost'" "$L" || true)" \
         "empty=$(grep -c 'empty response' "$L" || true)" \
         "failed=$(grep -c 'inference FAILED' "$L" || true)"
  done
} > ./results/sanity_counters.txt
cat ./results/sanity_counters.txt

echo "Job completed at $(date). Results -> ~/job_results/$SLURM_JOB_ID/"
