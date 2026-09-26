#!/bin/bash
# B6 on gpu0's A100-SXM4-80GB: all five models, one after another, one card.
#
# WHY THIS EXISTS, AND WHAT IT COSTS
#
# run_probe.sbatch deliberately uses SLURM because --gres gives an EXCLUSIVE
# GPU, and `seconds` is this study's cost axis: a GPU shared with another tenant
# measures that tenant, not our model. gpu0 is a SHARED box, so that guarantee
# is gone here. This script is the considered trade, not an oversight:
#
#   * csecluster has no free GPU until ~13:00 tomorrow (one job holds both GPUs
#     on node 01 for 2 days, another holds a node-02 GPU for 15 days).
#   * The keep/drop verdicts that produce the FDR table are hardware-independent,
#     so the deliverable is safe either way.
#   * Only the cost column is exposed, and only if a tenant appears mid-run.
#
# So: sample GPU contention throughout and write it beside the rows. A cost
# number from a contended window can then be found and re-timed, instead of
# quietly poisoning the cost axis. That is the difference between a measurement
# with a caveat and a wrong number.
#
# All five models share ONE card and ONE token budget, which is the constraint
# that actually matters for comparing costs ACROSS models. Absolute seconds are
# not comparable to the csecluster A100-PCIE-40GB runs; `hardware` records which.
set -uo pipefail

# units_paths.jsonl, NOT units_slices.jsonl. The output filename is derived from
# this basename, and units_slices is the name the SUPERSEDED per-alert probe run
# already wrote under. Same name, incompatible id space, silent empty joins.
UNITS="${UNITS:-$HOME/nimit/graphrouter/data/units_paths.jsonl}"
STYLE="${STYLE:-slice}"
MAXTOK="${MAXTOK:-256}"
PORT="${PORT:-8713}"
# Modest by default: this is a shared box and vLLM's 0.90 default would claim
# ~72 GB of 80 and lock everyone else out. The largest model here is 8B in
# bf16 (~16 GB of weights), so 0.30 leaves ample KV cache.
GPU_UTIL="${GPU_UTIL:-0.30}"
# vllmenv, NOT the llmdfa env. llmdfa is Python 3.11 and vLLM 0.27.1 dies there
# with "TypeError: type array.array is not subscriptable" -- it annotates
# array.array[int], valid only from 3.12. Built by setup_vllm_gpu0.sh, pinned to
# 0.27.1 to match the cluster container so engine_version is identical on both.
PY="${PY:-$HOME/nimit/vllmenv/bin/python}"
VLLM="${VLLM:-$HOME/nimit/vllmenv/bin/vllm}"
OUTDIR="$HOME/nimit/probe_out"
LOGDIR="$HOME/nimit/logs/probe_gpu0"
export HF_HOME="${HF_HOME:-$HOME/nimit/hf}"

# The env's bin/ must be on PATH, not just used for absolute-path launching.
# vLLM's inductor backend shells out to `ninja` to build kernels, and the
# EngineCore subprocess inherits PATH: without this it loads the model, captures
# CUDA graphs, allocates KV cache, and only THEN dies with
#   FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
# even though the ninja package is installed in that very env.
export PATH="$HOME/nimit/vllmenv/bin:$PATH"

mkdir -p "$OUTDIR" "$LOGDIR"

ALL_MODELS=(
  "qwen-1.5b:Qwen/Qwen2.5-Coder-1.5B-Instruct"
  "phi-3.8b:microsoft/Phi-4-mini-instruct"
  "qwen-7b:Qwen/Qwen2.5-Coder-7B-Instruct"
  "granite-8b:ibm-granite/granite-3.1-8b-instruct"
  "deepseek-6.7b:deepseek-ai/deepseek-coder-6.7b-instruct"
)
# ONLY="deepseek-6.7b granite-8b" re-runs a subset. Used when a budget change
# affects some models and provably not others: at MAXTOK=256, deepseek truncated
# 45.8% of attempts and granite 14.9%, while qwen-1.5b, qwen-7b and phi-3.8b
# never exceeded 26 output tokens. A cap only changes a model that reaches it,
# so those three give byte-identical answers at any larger budget and re-running
# them would burn an hour to reproduce the file we already have.
MODELS=()
if [ -n "${ONLY:-}" ]; then
    for e in "${ALL_MODELS[@]}"; do
        for want in $ONLY; do
            [ "${e%%:*}" = "$want" ] && MODELS+=("$e")
        done
    done
else
    MODELS=("${ALL_MODELS[@]}")
fi

HW="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -s ' ' '-')"
NEED="$(wc -l < "$UNITS")"
log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOGDIR/driver.log"; }

log "gpu0 probe driver up: units=$UNITS need=$NEED hw=$HW util=$GPU_UTIL maxtok=$MAXTOK"

# Contention sampler. Records every process on the GPU once a minute for the
# whole run, so a suspicious cost window can be checked against who else was on
# the card at that moment.
( while true; do
    echo "$(date '+%Y-%m-%dT%H:%M:%S') $(nvidia-smi --query-compute-apps=pid,used_memory \
        --format=csv,noheader | tr '\n' ';')"
    sleep 60
  done ) >> "$LOGDIR/contention.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

rows_for() {   # only rows with in_tokens>0 count: a failed request is still
    local f="$OUTDIR/$1__$(basename "$UNITS" .jsonl).jsonl"   # written as data
    [ -f "$f" ] || { echo 0; return; }
    "$PY" - "$f" <<'PY' 2>/dev/null || wc -l < "$f"
import json,sys
n=0
for line in open(sys.argv[1]):
    try:
        if json.loads(line).get('in_tokens',0) > 0: n+=1
    except Exception: pass
print(n)
PY
}

for entry in "${MODELS[@]}"; do
    SHORT="${entry%%:*}"; MODEL="${entry#*:}"
    HAVE="$(rows_for "$SHORT")"
    if [ "$HAVE" -ge "$NEED" ]; then log "SKIP $SHORT: already $HAVE/$NEED"; continue; fi

    SERVED="$(basename "$MODEL")"
    log "=== $SHORT ($MODEL), resuming from $HAVE/$NEED rows"

    "$VLLM" serve "$MODEL" \
        --served-model-name "$SERVED" \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization "$GPU_UTIL" \
        --port "$PORT" \
        > "$LOGDIR/vllm_${SHORT}.log" 2>&1 &
    VPID=$!

    UP=0
    for i in $(seq 1 160); do
        if "$PY" -c "
import sys,urllib.request
try: urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5); sys.exit(0)
except Exception: sys.exit(1)" 2>/dev/null; then UP=1; break; fi
        kill -0 $VPID 2>/dev/null || { log "vLLM died starting $SHORT"; tail -30 "$LOGDIR/vllm_${SHORT}.log"; break; }
        sleep 15
    done
    if [ "$UP" != 1 ]; then
        log "SKIP $SHORT: vLLM never healthy"
        kill $VPID 2>/dev/null; wait $VPID 2>/dev/null
        continue
    fi
    log "$SHORT: vLLM up after ~$((i*15))s"

    EV="$(grep -oE "LLM engine \(v[0-9]+\.[0-9]+\.[0-9]+\)" "$LOGDIR/vllm_${SHORT}.log" \
            | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)"
    [ -n "$EV" ] || EV="$("$VLLM" --version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -1)"
    [ -n "$EV" ] || EV=unknown

    "$PY" "$HOME/nimit/graphrouter/scripts/probe_client.py" \
        --units "$UNITS" --out "$OUTDIR/${SHORT}__$(basename "$UNITS" .jsonl).jsonl" --resume \
        --model "$SHORT" --served "$SERVED" \
        --base-url "http://127.0.0.1:$PORT/v1" \
        --prompt-style "$STYLE" --max-tokens "$MAXTOK" \
        --engine vllm --engine-version "$EV" --hardware "$HW" \
        2>&1 | tee "$LOGDIR/client_${SHORT}.log" | tail -3

    kill $VPID 2>/dev/null; wait $VPID 2>/dev/null
    log "$SHORT done: $(rows_for "$SHORT")/$NEED rows"
    sleep 5
done

log "gpu0 sweep finished"
for entry in "${MODELS[@]}"; do
    SHORT="${entry%%:*}"
    log "  $SHORT $(rows_for "$SHORT")/$NEED"
done
