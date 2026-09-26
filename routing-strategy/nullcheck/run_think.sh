#!/bin/bash
# gpu0: serve Qwen3-8B (40% of the A100-80GB), probe nothink on all pairs then think on 300 pairs, shut down.
set -uo pipefail
cd ~/nimit/vr_explore
export HF_HOME=~/nimit/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=~/nimit/vllmenv/bin/python
~/nimit/vllmenv/bin/vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b \
    --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.40 \
    --port 18080 > vllm.log 2>&1 &
VP=$!
trap 'kill $VP 2>/dev/null; sleep 10; kill -9 $VP 2>/dev/null; echo server stopped $(date)' EXIT
for i in $(seq 1 120); do
    curl -sf http://127.0.0.1:18080/health >/dev/null && break
    kill -0 $VP 2>/dev/null || { echo "vllm died"; tail -30 vllm.log; exit 1; }
    sleep 10
done
echo "server up $(date)"
$PY think_probe.py --units pairs.jsonl --out nothink.jsonl --mode nothink --workers 64
$PY think_probe.py --units pairs.jsonl --out think.jsonl --mode think --pairs ${THINK_PAIRS:-300} --workers 96
echo "all done $(date)"
