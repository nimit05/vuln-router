#!/bin/bash
#SBATCH --job-name=llmdfa-gateA
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=gpu-A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#
# Gate A - protocol compliance. Does not touch the benchmark.
#   Phase I  : run LLMDFA's real extractor synthesizer (xss src + sink)
#   Phase II : Yes/No parse rate + accuracy on 6 known-answer items
#   Phase IV : Z3 script fencing, runnability, verdict correctness
#
#   MODEL=Qwen/Qwen2.5-Coder-7B-Instruct sbatch gate_a.sbatch
set -euo pipefail

MODEL="${MODEL:?set MODEL=<hf-model-id>}"
SERVED="$(basename "$MODEL")"
export HF_HOME="${HF_HOME:-$HOME/models/hf}"
SCRIPT_DIR="${SCRIPT_DIR:-$HOME/cluster}"   # where these scripts were deployed
export PATH="$HOME/llmdfa-venv/bin:$PATH"
# synthesis/llm.py reads OPENAI_API_KEY at import time; a dummy value is enough
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
# tiktoken downloads its BPE file on first use; compute-node HTTPS is intercepted,
# so point it at a cache populated on the login node.
export TIKTOKEN_CACHE_DIR="$HOME/models/tiktoken"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

cd "$TMPDIR" || exit 1
mkdir -p ./results

echo "Job $SLURM_JOB_ID on $(hostname); model=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cp -r "${LLMDFA_SRC:-$HOME/LLMDFA}" ./LLMDFA

apptainer exec --nv \
    --bind "$HF_HOME:$HF_HOME" \
    --env HF_HOME="$HF_HOME" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$HOME/containers/vllm.sif" \
    vllm serve "$MODEL" --served-model-name "$SERVED" \
        --dtype bfloat16 --tensor-parallel-size "${TP:-1}" \
        --max-model-len "${MAX_MODEL_LEN:-16384}" \
        --gpu-memory-utilization "${GPU_UTIL:-0.90}" --port 8000 \
        ${VLLM_EXTRA:-} \
    > "./results/vllm_${SERVED}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "waiting for vLLM..."
UP=0
for i in $(seq 1 120); do
    if python -c "
import sys,urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5); sys.exit(0)
except Exception: sys.exit(1)
" 2>/dev/null; then UP=1; break; fi
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died:"; tail -40 "./results/vllm_${SERVED}.log"; exit 1; }
    sleep 15
done
[ "$UP" = 1 ] || { echo "vLLM never healthy:"; tail -40 "./results/vllm_${SERVED}.log"; exit 1; }
echo "vLLM up after ~$((i*15))s"
export LLMDFA_BASE_URL=http://127.0.0.1:8000/v1

# --- Phase I: the real extractor synthesizer -------------------------------
echo "=== Phase I: extractor synthesis (xss_src, xss_sink) ==="
cd ./LLMDFA/src
for EX in xss_src xss_sink; do
    echo "--- $EX ---"
    ( time python TSAgent/synthesis/main.py --extractor "$EX" --model "$SERVED" \
        --output-file "$TMPDIR/results/_${EX}_extractor.py" ) \
        2>&1 | tee "$TMPDIR/results/synth_${EX}.log" || echo "SYNTHESIS FAILED: $EX"
done
cd "$TMPDIR"

# --- Phases II and IV: prompt-level protocol probe --------------------------
echo "=== Phases II/IV: protocol probe ==="
python "$SCRIPT_DIR/gate_a_probe.py" \
    --llmdfa "$TMPDIR/LLMDFA" --model "$SERVED" \
    --out ./results/gate_a_${SERVED}.json 2>&1 | tee ./results/gate_a_probe.log

echo "Gate A done at $(date). Results -> ~/job_results/$SLURM_JOB_ID/"
