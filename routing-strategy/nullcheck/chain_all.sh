#!/bin/bash
# gpu0 driver: Qwen3-8B (think vs no-think) then phi-4 (direct vs CoT). One server at a time.
set -uo pipefail
cd ~/nimit/vr_explore
export PATH=~/nimit/vllmenv/bin:$PATH          # vLLM JIT needs ninja, which lives here
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=~/nimit/vllmenv/bin/python

serve () {  # $1=model $2=served $3=port $4=util $5=hf_home $6=log
    HF_HOME=$5 vllm serve "$1" --served-model-name "$2" --dtype bfloat16 \
        --max-model-len 16384 --gpu-memory-utilization "$4" --port "$3" > "$6" 2>&1 &
    SRV=$!
    for i in $(seq 1 150); do
        curl -sf "http://127.0.0.1:$3/health" >/dev/null && { echo "up: $2 $(date)"; return 0; }
        kill -0 $SRV 2>/dev/null || { echo "DIED: $2"; tail -25 "$6"; return 1; }
        sleep 10
    done
    echo "TIMEOUT: $2"; return 1
}
stop () { kill $SRV 2>/dev/null; for i in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 2; done; kill -9 $SRV 2>/dev/null; sleep 5; echo "stopped $(date)"; }
trap 'stop' EXIT

echo "=== QWEN3-8B $(date)"
if serve Qwen/Qwen3-8B qwen3-8b 18080 0.40 ~/nimit/hf vllm_qwen3.log; then
    B=http://127.0.0.1:18080/v1
    $PY think_probe.py --units pairs.jsonl --out qwen3_nothink.jsonl --mode nothink --pairs 300 --served qwen3-8b --base $B --workers 64
    $PY think_probe.py --units pairs.jsonl --out qwen3_think.jsonl   --mode think   --pairs 300 --served qwen3-8b --base $B --workers 96
fi
stop

echo "=== PHI-4 $(date)"
if serve microsoft/phi-4 phi-4 18081 0.45 ~/.cache/huggingface vllm_phi4.log; then
    B=http://127.0.0.1:18081/v1
    $PY think_probe.py --units pairs.jsonl --out phi4_direct.jsonl --mode direct --pairs 300 --served phi-4 --base $B --workers 64
    $PY think_probe.py --units pairs.jsonl --out phi4_cot.jsonl    --mode cot    --pairs 300 --served phi-4 --base $B --workers 96
fi
echo "=== ALL DONE $(date)"
