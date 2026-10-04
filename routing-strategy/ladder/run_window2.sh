#!/bin/bash
# Window 2 (GPU) of the scaled IRIS run: the filter stage, Qwen3-32B then Qwen3-8B (no-think, the
# E20 prompt), once per distinct code slice over all spec runs (filter_dedup.py), then copied back
# to every path. Starts when the CPU worker is done (cpu.done) and the booking is posted
# (window2_dedup.go). The GPU is held between the two models by a placeholder.
set -uo pipefail
T=/tmp/nimit; R=$T/scaled; GR=$T/gr
VPY=$T/vllmenv/bin/python; VLLM=$T/vllmenv/bin/vllm
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PATH=$PATH:$T/vllmenv/bin
PORT=18095; B=http://127.0.0.1:$PORT/v1
Q32="--dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.95 --max-num-seqs 64"
Q8="--dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 128"
U=$R/data/filter_units.jsonl
log() { echo "[$(date '+%F %T')] $*"; }
SRV=; HOLD=
gpu_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
hold_on()  { [ -n "$HOLD" ] && return 0
  setsid nohup $VPY -c "import torch,time; x=torch.empty(int(5e8),dtype=torch.uint8,device='cuda'); time.sleep(36000)" \
      > $R/logs/hold2.log 2>&1 < /dev/null & HOLD=$!; sleep 25; log "placeholder holds the GPU (pid $HOLD)"; }
hold_off() { [ -n "$HOLD" ] && { kill $HOLD 2>/dev/null; log "placeholder released"; }; HOLD=; }
stop_srv() { [ -z "$SRV" ] && return 0
  kill -- -$SRV 2>/dev/null; for i in $(seq 1 60); do kill -0 $SRV 2>/dev/null || break; sleep 2; done
  kill -9 -- -$SRV 2>/dev/null; SRV=; sleep 10; log "server stopped, GPU now $(gpu_mb) MiB"; }
serve() {      # weights name served flags...
  local w=$1 name=$2 served=$3; shift 3
  [ -n "$SRV" ] && hold_on
  stop_srv; log "serve $name"
  setsid nohup $VLLM serve $w --served-model-name $served --port $PORT "$@" > $R/logs/vllm_$name.log 2>&1 < /dev/null &
  SRV=$!
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null && { log "server $name up"; hold_off; return 0; }
    kill -0 $SRV 2>/dev/null || break; sleep 10
  done
  log "server $name FAILED"; tail -20 $R/logs/vllm_$name.log; hold_off; return 1
}
cleanup() { stop_srv; hold_off; log "GPU released"; }
probe() {      # short served
  local short=$1 served=$2 O=$R/probe/$1-nothink__dedup.jsonl t0=$(date +%s)
  log "filter $short on $(wc -l < $U) distinct slices"
  $VPY $GR/d1_probe_reasoner.py --units $U --base $B --served $served --model $short --mode nothink \
      --out $O --workers 32 > $R/logs/filter_${short}_dedup.log 2>&1
  echo "{\"model\": \"$short-nothink\", \"units_file\": \"dedup\", \"wall_s\": $(( $(date +%s)-t0 )), \"units\": $(wc -l < $O), \"workers\": 32}" >> $R/probe/timing.jsonl
  log "filter $short done in $(( $(date +%s)-t0 )) s, $(grep -c '"error": null' $O) of $(wc -l < $O) without error"
}

until [ -f $R/status/cpu.done ] && [ -f $R/status/window2_dedup.go ]; do sleep 30; done
$VPY $T/ladder/filter_dedup.py make --dir $R
until [ "$(gpu_mb)" -lt 1000 ] && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
  log "GPU busy (someone else), waiting"; sleep 30; done
trap cleanup EXIT; log "WINDOW 2 START"; date +%s > $R/status/window2.t0
serve $T/models/Qwen3-32B qwen3-32b_w2 qwen3-32b $Q32 || exit 1
probe qwen3-32b qwen3-32b
serve $T/models/Qwen3-8B qwen3-8b_w2 qwen3-8b $Q8 || exit 1
probe qwen3-8b qwen3-8b
cleanup; trap - EXIT; date +%s > $R/status/window2.t1
$VPY $T/ladder/filter_dedup.py expand --dir $R
touch $R/status/window2.done; log "WINDOW 2 DONE, GPU released"
