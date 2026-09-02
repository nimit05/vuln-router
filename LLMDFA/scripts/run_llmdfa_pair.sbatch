#!/bin/bash
#SBATCH --job-name=llmdfapair
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#
# TWO MODELS, ONE CASE BLOCK. Sibling of run_llmdfa_x2.sbatch, which runs one
# model as two shards. Here each GPU serves a DIFFERENT model and both clients
# analyse the SAME cases, so the two models are matched case-for-case inside a
# single job -- no second queue wait, and no risk of the two halves landing on
# different hardware.
#
#   MODEL0=microsoft/Phi-4-mini-instruct \
#   MODEL1=mistralai/Mistral-Nemo-Instruct-2407 \
#   BUG=dbz OFFSET=815 COUNT=111 sbatch run_llmdfa_pair.sbatch
#
# Partition is pinned to gpu-A100 deliberately: the cost axis of the router
# comparison is GPU-seconds measured on A100-PCIE-40GB. Running elsewhere would
# produce timings incomparable with every run already recorded.
#
# OFFSET/COUNT index the SORTED case list (LLMDFA_CASE_OFFSET/COUNT, added in
# patch_llmdfa.sh). Juliet forms are exactly 37 cases wide and contiguous in
# sorted order, so a form-aligned block gives complete 37-variant coverage of
# whole forms -- which is what a held-out-FORM generalisation test needs.
set -euo pipefail

MODEL0="${MODEL0:?set MODEL0=<hf-model-id>}"
MODEL1="${MODEL1:?set MODEL1=<hf-model-id>}"
BUG="${BUG:-dbz}"
OFFSET="${OFFSET:?set OFFSET=<sorted-list offset>}"
COUNT="${COUNT:?set COUNT=<number of cases>}"
MODELS=("$MODEL0" "$MODEL1")

export HF_HOME="${HF_HOME:-$HOME/models/hf}"
export PATH="$HOME/llmdfa-venv/bin:$PATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export TIKTOKEN_CACHE_DIR="$HOME/models/tiktoken"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LLMDFA_ABORT_AFTER_FAILURES="${LLMDFA_ABORT_AFTER_FAILURES:-20}"

cd "$TMPDIR" || exit 1
mkdir -p ./results

scontrol update JobId="$SLURM_JOB_ID" \
    JobName="pair-${BUG}-off${OFFSET}" 2>/dev/null || true

echo "Job $SLURM_JOB_ID on $(hostname)"
echo "PAIR bug=$BUG offset=$OFFSET count=$COUNT m0=$MODEL0 m1=$MODEL1"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python -c "import z3,tree_sitter; print('client deps ok')"

# --- 1. one vLLM server per GPU, a different model on each ------------------
PIDS=()
for i in 0 1; do
    M="${MODELS[$i]}"; S="$(basename "$M")"
    CUDA_VISIBLE_DEVICES=$i apptainer exec --nv \
        --bind "$HF_HOME:$HF_HOME" \
        --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        --env CUDA_VISIBLE_DEVICES=$i \
        "$HOME/containers/vllm.sif" \
        vllm serve "$M" \
            --served-model-name "$S" \
            --dtype bfloat16 \
            --tensor-parallel-size 1 \
            --max-model-len "${MAX_MODEL_LEN:-16384}" \
            --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
            --port $(( 8000 + i )) ${VLLM_EXTRA:-} \
        > "./results/vllm_${S}_gpu${i}.log" 2>&1 &
    PIDS+=($!)
done
trap 'for p in "${PIDS[@]}"; do kill $p 2>/dev/null || true; done' EXIT

# --- 2. wait for both ------------------------------------------------------
for i in 0 1; do
    port=$(( 8000 + i )); S="$(basename "${MODELS[$i]}")"
    echo "waiting for vLLM on :$port ($S) ..."
    UP=0
    for t in $(seq 1 160); do
        if python -c "
import sys,urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:$port/health', timeout=5); sys.exit(0)
except Exception: sys.exit(1)
" 2>/dev/null; then UP=1; break; fi
        kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "vLLM gpu$i died:"; tail -40 "./results/vllm_${S}_gpu${i}.log"; exit 1; }
        sleep 15
    done
    [ "$UP" = 1 ] || { echo "vLLM gpu$i never healthy:"; tail -40 "./results/vllm_${S}_gpu${i}.log"; exit 1; }
    echo "vLLM :$port up ($S)"
done

# --- 3. one client per server, SAME cases ----------------------------------
run_one () {
    local idx="$1" M="${MODELS[$1]}" port=$(( 8000 + $1 ))
    local S="$(basename "$M")"
    cp -r "${LLMDFA_SRC:-$HOME/LLMDFA}" "./LLMDFA_$idx"
    test -f "./LLMDFA_$idx/lib/build/my-languages.so" || { echo "grammar .so missing"; return 1; }
    cd "./LLMDFA_$idx/src"
    # Per-model header: both clients stream to the job's stdout and interleave
    # there, so the scorer must read these per-model logs, not %j.out.
    {
      echo "model=$M bug=$BUG mode=pair tp=1 cases=$COUNT seed=none shard=$OFFSET+$COUNT"
      LLMDFA_BASE_URL="http://127.0.0.1:$port/v1" \
      LLMDFA_MAX_RETRY="${LLMDFA_MAX_RETRY:-5}" \
      LLMDFA_CASE_OFFSET="$OFFSET" \
      LLMDFA_CASE_COUNT="$COUNT" \
          python run_llmdfa.py \
              --bug-type "$BUG" \
              --model-name "$S" \
              -syn-parser -fscot -syn-solver \
              --solving-refine-number 3 \
              --analysis-mode all
    } 2>&1 | tee "$TMPDIR/results/llmdfa_${S}_${BUG}_off${OFFSET}.log"
    cd "$TMPDIR"
}

run_one 0 & C0=$!
run_one 1 & C1=$!
wait $C0 && R0=0 || R0=$?
wait $C1 && R1=0 || R1=$?
echo "client exit codes: 0->$R0 1->$R1"

# --- 4. collect ------------------------------------------------------------
{
  for i in 0 1; do
    S="$(basename "${MODELS[$i]}")"
    L="./results/llmdfa_${S}_${BUG}_off${OFFSET}.log"
    cp -r "./LLMDFA_$i/log" "./results/llmdfa_log_${S}_${BUG}_off${OFFSET}" 2>/dev/null || true
    echo "$S cases=$(grep -c "^{'input_token_cost'" "$L" || true)" \
         "empty=$(grep -c 'empty response' "$L" || true)" \
         "failed=$(grep -c 'inference FAILED' "$L" || true)"
  done
} > ./results/sanity_counters.txt
cat ./results/sanity_counters.txt
echo "Job completed at $(date). Results -> ~/job_results/$SLURM_JOB_ID/"
