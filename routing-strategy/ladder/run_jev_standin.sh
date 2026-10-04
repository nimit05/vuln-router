#!/bin/bash
# Jev stand-in on gpu7: one Qwen3-8B server, loaded once and held for the whole run
# (no gaps), both description sets on IRIS paths and OWASP units.
set -uo pipefail
T=/tmp/nimit; O=~/nimit/jev_standin; mkdir -p $O; cd $O
export PATH=$T/vllmenv/bin:$PATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=$T/vllmenv/bin/python; PORT=18091
m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$m" -lt 1000 ] || { echo "GPU busy ($m MiB), not starting"; exit 1; }
echo "=== serve $(date)"
vllm serve $T/models/Qwen3-8B --served-model-name qwen3-8b --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.5 --max-num-seqs 32 --port $PORT > vllm.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; sleep 5; kill -9 $SRV 2>/dev/null; echo "server stopped $(date)"' EXIT
for i in $(seq 1 90); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 10; done
for v in generic informed; do
  for d in iris owasp; do
    U=$([ $d = iris ] && echo ~/nimit/iris_ladder/units_iris.jsonl || echo ~/nimit/owasp_ladder/units_owasp.jsonl)
    t0=$(date +%s)
    $PY l7_jev_standin.py --units $U --variant $v --out jev_${v}__$d.jsonl --workers 32 2>&1 | tail -1
    echo "{\"judge\": \"qwen3-8b\", \"variant\": \"$v\", \"data\": \"$d\", \"wall_s\": $(( $(date +%s) - t0 )), \"units\": $(wc -l < jev_${v}__$d.jsonl)}" >> timing.jsonl
  done
done
echo "=== JEV STANDIN DONE $(date)"
