#!/bin/bash
# IRIS paths, same 12-model ladder and settings as OWASP, on gpu7: small -> medium -> large -> extra large, one vLLM server at a time.
# Waits for weights + vLLM, and waits for the GPU to be empty (no other user) before each load.
set -uo pipefail
T=/tmp/nimit; O=~/nimit/iris_ladder; mkdir -p $O
U=$O/units_iris.jsonl; G=~/nimit/owasp_ladder/code
export PATH=$T/vllmenv/bin:$PATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=$T/vllmenv/bin/python; PORT=18091; B=http://127.0.0.1:$PORT/v1; SRV=
stop() { [ -n "$SRV" ] && { kill $SRV 2>/dev/null; for i in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 2; done; kill -9 $SRV 2>/dev/null; }; SRV=; sleep 2; echo "server stopped $(date)"; }
trap stop EXIT
# SHARE=1: another user's job may stay on the GPU; start as long as ~40 GB is free and
# take only 45% of the memory (small models only)
SHARE=${SHARE:-0}; UTIL=0.9; [ "$SHARE" = 1 ] && UTIL=0.45
gpu_empty() {
  local used free
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$SHARE" = 1 ]; then [ "$free" -gt 40000 ]; else [ "$used" -lt 1000 ] && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; fi
}
weights_ok() { [ -s $1/config.json ] && ! ls $1/*.part >/dev/null 2>&1 && python3 - "$1" <<'PY'
import json,os,sys
d=sys.argv[1]; ix=os.path.join(d,"model.safetensors.index.json")
need=set(json.load(open(ix))["weight_map"].values()) if os.path.exists(ix) else {"model.safetensors"}
sys.exit(0 if all(os.path.exists(os.path.join(d,f)) for f in need) else 1)
PY
}
until $PY -c "import vllm" 2>/dev/null; do echo "waiting for vllm install $(date +%T)"; sleep 120; done
# short|weights|mode|flags|units variant ("" = original, "anon" = benchmark names hidden)
MODELS=(
  "qwen3-8b|$T/models/Qwen3-8B|nothink||"
  "qwen3-32b|$T/models/Qwen3-32B|nothink||"
  "gpt-oss-20b|$T/models/gpt-oss-20b|low|--plain|"
  "gpt-oss-120b|$T/models/gpt-oss-120b|low|--plain|"
  "qwen3-1.7b|$T/models/Qwen3-1.7B|nothink||"
  "phi-4|$T/models/phi-4|verbose|--plain|"
  "nemo-12b|$T/models/Mistral-Nemo-Instruct-2407|nothink|--plain|"
  "qwen-7b|$T/models/Qwen2.5-Coder-7B-Instruct|verbose|--plain|"
  "granite-8b|$T/models/granite-3.1-8b-instruct|verbose|--plain|"
  "deepseek-6.7b|$T/models/deepseek-coder-6.7b-instruct|verbose|--plain|"
  "phi-3.8b|$T/models/Phi-4-mini-instruct|verbose|--plain|"
  "qwen-1.5b|$T/models/Qwen2.5-Coder-1.5B-Instruct|verbose|--plain|"
)
is_done() { [ -s $1 ] && [ $(wc -l < $1) -ge $(wc -l < $U) ]; }
# run whichever pending model has its weights ready; loop until all are done
while :; do
 pending=0; ran=0
 for spec in "${MODELS[@]}"; do
  IFS='|' read -r short w mode flags var <<< "$spec"
  sfx=iris${var:+_$var}; UU=$O/units_$sfx.jsonl
  out=$O/$short-${mode}__$sfx.jsonl
  is_done $out && continue
  [ -e $O/failed_$short ] && continue
  pending=1
  weights_ok $w || continue
  ran=1
  until gpu_empty; do echo "GPU busy (someone else), waiting $(date +%T)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; sleep 30; done
  dt="--dtype bfloat16"; [ "$short" = gpt-oss-120b ] && dt=""
  echo "=== $short serve $(date)"
  vllm serve $w --served-model-name $short $dt --max-model-len 8192 --gpu-memory-utilization $UTIL \
      --max-num-seqs 32 --port $PORT > $O/vllm_$short.log 2>&1 &
  SRV=$!; up=0
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && { up=1; break; }; kill -0 $SRV 2>/dev/null || break; sleep 10; done
  [ $up = 1 ] || { echo "DIED $short"; touch $O/failed_$short; tail -20 $O/vllm_$short.log; stop; continue; }
  t0=$(date +%s)
  $PY $G/d1_probe_reasoner.py --units $UU --base $B --served $short --model $short --mode $mode $flags \
      --out $out --workers 32 2>&1 | tail -2
  t1=$(date +%s)
  echo "{\"model\": \"$short-$mode\", \"units_file\": \"$sfx\", \"wall_s\": $((t1-t0)), \"units\": $(wc -l < $out), \"gpu\": \"A100-SXM4-80GB x1\", \"workers\": 32}" >> $O/timing.jsonl
  echo "=== $short done in $((t1-t0)) s $(date)"
  stop
 done
 [ $pending = 0 ] && break
 [ $ran = 0 ] && { echo "waiting for weights $(date +%T)"; sleep 120; }
done
echo "=== LADDER DONE $(date)"
