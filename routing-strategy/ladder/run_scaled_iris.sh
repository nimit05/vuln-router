#!/bin/bash
# Scaled IRIS on gpu7 (2026-10-02): which of IRIS's two LLM stages should get the big model?
#   spec stage   an LLM labels library APIs as taint sources/sinks; CodeQL then finds dataflow paths.
#                Runs: s_q8 (Qwen3-8B), s_dsc (deepseek-coder-7b-v1.5, the paper's model), s_q32 (Qwen3-32B)
#   filter stage Qwen3-8B and Qwen3-32B score every path of every spec run (d1_probe_reasoner, no-think)
# GPU only in two dense windows; all CodeQL work is CPU and runs outside them:
#   prep     CPU  IRIS stages 1-2 (candidate APIs) for every project, once, shared by all runs
#   window1  GPU  IRIS stages 3-4 (the LLM labelling) per spec model; booked from START_UTC
#   cpu      CPU  per spec run, as soon as its labels exist: IRIS stages 5-9 (CodeQL), then paths
#   window2  GPU  the filter stage; starts only when $R/status/window2.go exists (booking posted)
# Usage: scaled.sh all | prep | window1 | cpu | window2 | analysis      (steps skip finished work)
set -uo pipefail
T=/tmp/nimit; I=$T/IRIS; R=$T/scaled; GR=$T/gr
mkdir -p $R/logs $R/usage $R/data $R/probe $R/status
IPY=$I/.conda-iris/bin/python; VPY=$T/vllmenv/bin/python; VLLM=$T/vllmenv/bin/vllm
export PYTHONPATH=$I:$I/src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PATH=$PATH:$T/vllmenv/bin      # vLLM JIT-builds kernels with ninja; without it the engine dies at start
export IRIS_CODEQL_THREADS=${IRIS_CODEQL_THREADS:-3} IRIS_CODEQL_RAM=${IRIS_CODEQL_RAM:-16000} IRIS_CSV_NO_RERUN=1
PORT=18095; B=http://127.0.0.1:$PORT/v1; DEAD=http://127.0.0.1:9/v1
CPU_PAR=${CPU_PAR:-16}; LAB_PAR=${LAB_PAR:-24}
START_UTC=${START_UTC:-2026-10-02 12:00}          # 5:30 PM IST, the posted booking
NOTHINK='{"chat_template_kwargs": {"enable_thinking": false}}'
RUNS="s_q8 s_dsc s_q32"
log() { echo "[$(date '+%F %T')] $*"; }
PIDS=()        # only the jobs a step launched; a bare `wait` would also wait for the vLLM server
throttle() { while :; do local n=0 p; for p in "${PIDS[@]}"; do kill -0 $p 2>/dev/null && n=$((n+1)); done
  [ $n -lt $1 ] && return 0; sleep 2; done; }
wait_all() { local p; for p in "${PIDS[@]}"; do wait $p 2>/dev/null; done; PIDS=(); }

declare -A CWE
while IFS=, read -r id slug cve cwe rest; do CWE[$slug]=${cwe#CWE-}; done < <(tail -n +2 $I/data/cwe-bench-java/data/project_info.csv)
if [ ! -s $R/projects.txt ]; then      # every project with a database, biggest first (shorter tail)
  for d in $I/data/codeql-dbs/*/db-java; do
    s=$(basename $(dirname $d)); [ -n "${CWE[$s]:-}" ] && echo "$(du -sk $d | cut -f1) $s"
  done | sort -rn | awk '{print $2}' > $R/projects.txt
fi

iris_one() {   # slug run-id tag ; the environment decides the LLM endpoint and the stages
  local s=$1 run=$2 tag=$3 c=${CWE[$1]} t0=$(date +%s)
  mkdir -p $R/logs/$tag
  ( cd $I && timeout ${TMO:-7200} $IPY src/neusym_vul.py --query "cwe-${c}wLLM" --run-id $run --llm gpt-4 \
      --skip-posthoc-filter --num-threads 16 $s > $R/logs/$tag/$s.log 2>&1 )
  echo "$s,$?,$(( $(date +%s)-t0 ))" >> $R/status/$tag.csv
}
cand_ok() { [ -f $I/output/$1/prep/cwe-${CWE[$1]}/candidate_apis.csv ]; }
seed() {       # LLM-free candidate lists from the prep run, so stages 1-2 are not redone
  local s=$1 run=$2 c=${CWE[$1]}; local src=$I/output/$s/prep dst=$I/output/$s/$run
  mkdir -p $dst/cwe-$c $dst/common
  for f in external_apis.csv candidate_apis.csv; do [ -f $src/cwe-$c/$f ] && cp -n $src/cwe-$c/$f $dst/cwe-$c/; done
  for f in func_params.csv source_func_param_candidates.csv; do [ -f $src/common/$f ] && cp -n $src/common/$f $dst/common/; done
  return 0
}

prep() {       # the LLM endpoint is dead on purpose: each run stops at stage 3 with its candidates saved
  [ -f $R/status/prep.done ] && return 0
  log "prep start, $(wc -l < $R/projects.txt) projects"; : > $R/status/prep.csv
  for s in $(cat $R/projects.txt); do
    throttle $CPU_PAR
    OPENAI_BASE_URL=$DEAD OPENAI_API_KEY=EMPTY TMO=3600 iris_one $s prep prep & PIDS+=($!)
  done
  wait_all
  local ok=0; for s in $(cat $R/projects.txt); do cand_ok $s && ok=$((ok+1)); done
  log "prep done: candidates for $ok of $(wc -l < $R/projects.txt) projects"
  touch $R/status/prep.done
}

labels() {     # run-id: IRIS stages 3-4 only, every project at once-ish; server already up
  local run=$1
  [ -f $R/status/labels_$run.done ] && { log "labels $run already done"; return 0; }
  mkdir -p $R/usage/$run; : > $R/status/$run.csv
  log "labels $run start"; date +%s > $R/status/$run.t0
  for s in $(cat $R/projects.txt); do
    cand_ok $s || continue
    throttle $LAB_PAR
    seed $s $run
    IRIS_STOP_AFTER_LABELS=1 IRIS_USAGE_LOG=$R/usage/$run/$s.jsonl iris_one $s $run $run & PIDS+=($!)
  done
  wait_all
  date +%s > $R/status/$run.t1; touch $R/status/labels_$run.done
  log "labels $run done: $(awk -F, '$2==0' $R/status/$run.csv | wc -l) of $(wc -l < $R/status/$run.csv) exit 0, $(( $(cat $R/status/$run.t1)-$(cat $R/status/$run.t0) )) s"
}

codeql_pass() {   # run-id: IRIS stages 5-9 with the labels on disk; no LLM (dead endpoint)
  local run=$1
  [ -f $R/status/cq_$run.done ] && return 0
  log "codeql $run start"; : > $R/status/${run}_cq.csv
  for s in $(cat $R/projects.txt); do
    [ -f $I/output/$s/$run/cwe-${CWE[$s]}/llm_labelled_source_apis.json ] || continue
    throttle $CPU_PAR
    OPENAI_BASE_URL=$DEAD OPENAI_API_KEY=EMPTY iris_one $s $run ${run}_cq & PIDS+=($!)
  done
  wait_all; touch $R/status/cq_$run.done
  log "codeql $run done: $(awk -F, '$2==0' $R/status/${run}_cq.csv | wc -l) of $(wc -l < $R/status/${run}_cq.csv) exit 0"
}

retry() {      # run-id: re-run the projects whose CodeQL step died (heap) with more RAM, few at a time
  local run=$1; local st=$R/status/${run}_cq.csv
  export IRIS_CODEQL_RAM=${RETRY_RAM:-64000}
  local todo=$(awk -F, '$2!=0 {print $1}' $st | sort -u)
  [ -f $R/status/${run}_cq_retry.csv ] && todo=$(comm -23 <(echo "$todo") <(awk -F, '$2==0 {print $1}' $R/status/${run}_cq_retry.csv | sort -u))
  log "retry $run: $(echo "$todo" | grep -c .) projects, CodeQL RAM $IRIS_CODEQL_RAM MB"
  for s in $todo; do
    throttle ${RETRY_PAR:-3}
    OPENAI_BASE_URL=$DEAD OPENAI_API_KEY=EMPTY iris_one $s $run ${run}_cq_retry & PIDS+=($!)
  done
  wait_all
  log "retry $run done: $(awk -F, '$2==0' $R/status/${run}_cq_retry.csv | wc -l) of $(wc -l < $R/status/${run}_cq_retry.csv) exit 0"
}

paths() {      # run-id [out-dir] (CPU): IRIS's own evaluator -> one row per dataflow path -> labels -> slices
  local run=$1 D=${2:-$R/data/$1}; mkdir -p $D
  [ -s $D/units_paths.jsonl ] && return 0
  log "paths $run start"
  ( cd $GR
    $IPY b0_iris_truth.py --iris $I --variant $run --out $D/iris_truth.json 2>&1 | tail -2
    $IPY b1_extract_paths.py --iris $I --variant $run --out $D/paths.jsonl 2>&1 | tail -2
    $IPY b2_label_paths.py --paths $D/paths.jsonl --iris $I --variant $run --out $D/paths_labelled.jsonl 2>&1 | tail -2
    $IPY b1_validate.py --labelled $D/paths_labelled.jsonl --truth $D/iris_truth.json > $D/validate.txt 2>&1; tail -4 $D/validate.txt
    $IPY b3_slice.py --alerts $D/paths_labelled.jsonl --variant $run --iris $I --out $D/slices.jsonl 2>&1 | tail -2
    $IPY b3b_make_units.py --slices $D/slices.jsonl --out $D/units_paths.jsonl 2>&1 | tail -2 ) > $R/logs/paths_$run.log 2>&1
  log "paths $run done ($D): $(wc -l < $D/units_paths.jsonl 2>/dev/null) paths"
}

cpu() {        # follows window1: each spec run gets its CodeQL pass and paths once its labels exist
  for run in $RUNS; do
    until [ -f $R/status/labels_$run.done ]; do
      [ -f $R/status/window1.failed ] && { log "window 1 failed, CPU worker stops"; exit 1; }; sleep 30; done
    codeql_pass $run; paths $run
  done
  touch $R/status/cpu.done; log "CPU WORK DONE"
}

filt() {       # short served run-id
  local short=$1 served=$2 run=$3; local U=$R/data/$run/units_paths.jsonl O=$R/probe/$1-nothink__$3.jsonl
  [ -s $U ] || { log "no units for $run"; return 0; }
  [ -s $O ] && [ $(wc -l < $O) -ge $(wc -l < $U) ] && return 0
  log "filter $short on $run ($(wc -l < $U) paths)"; local t0=$(date +%s)
  $VPY $GR/d1_probe_reasoner.py --units $U --base $B --served $served --model $short --mode nothink \
      --out $O --workers 32 > $R/logs/filter_${short}_$run.log 2>&1
  echo "{\"model\": \"$short-nothink\", \"units_file\": \"$run\", \"wall_s\": $(( $(date +%s)-t0 )), \"units\": $(wc -l < $O), \"workers\": 32}" >> $R/probe/timing.jsonl
  log "filter $short on $run done in $(( $(date +%s)-t0 )) s"
}

SRV=; HOLD=
gpu_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
gpu_free_wait() {
  until [ "$(gpu_mb)" -lt 1000 ] && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
    log "GPU busy (someone else), waiting"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; sleep 30; done; }
hold_on()  { [ -n "$HOLD" ] && return 0
  setsid nohup $VPY -c "import torch,time; x=torch.empty(int(5e8),dtype=torch.uint8,device='cuda'); time.sleep(36000)" \
      > $R/logs/hold.log 2>&1 < /dev/null & HOLD=$!; sleep 25; log "placeholder holds the GPU (pid $HOLD)"; }
hold_off() { [ -n "$HOLD" ] && { kill $HOLD 2>/dev/null; log "placeholder released"; }; HOLD=; }
stop_srv() { [ -z "$SRV" ] && return 0
  kill -- -$SRV 2>/dev/null; for i in $(seq 1 60); do kill -0 $SRV 2>/dev/null || break; sleep 2; done
  kill -9 -- -$SRV 2>/dev/null; SRV=; sleep 10; log "server stopped, GPU now $(gpu_mb) MiB"; }
serve() {      # weights log-name extra-flags... (served names in SNAMES)
  local w=$1 name=$2; shift 2
  [ -n "$SRV" ] && hold_on
  stop_srv
  log "serve $name"
  setsid nohup $VLLM serve $w --served-model-name $SNAMES --port $PORT "$@" > $R/logs/vllm_$name.log 2>&1 < /dev/null &
  SRV=$!
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null && { log "server $name up"; hold_off; return 0; }
    kill -0 $SRV 2>/dev/null || break; sleep 10
  done
  log "server $name FAILED"; tail -20 $R/logs/vllm_$name.log; hold_off; return 1
}
cleanup() { stop_srv; hold_off; log "GPU released"; }
Q8="--dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 128"
Q32="--dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.95 --max-num-seqs 64"
DSC="--dtype float16 --max-model-len 20480 --gpu-memory-utilization 0.90 --max-num-seqs 128"

window1() {
  [ -f $R/status/window1.done ] && return 0
  local t=$(date -d "$START_UTC UTC" +%s)
  while [ $(date +%s) -lt $t ]; do sleep 20; done
  gpu_free_wait; trap cleanup EXIT; log "WINDOW 1 START"; date +%s > $R/status/window1.t0
  export OPENAI_BASE_URL=$B OPENAI_API_KEY=EMPTY IRIS_DROP_EMPTY_STOP=1
  rm -f $R/status/window1.failed
  SNAMES="gpt-4-0125-preview" serve $T/models/Qwen3-8B qwen3-8b_w1 $Q8 || { touch $R/status/window1.failed; return 1; }
  export IRIS_EXTRA_BODY="$NOTHINK"; labels s_q8; unset IRIS_EXTRA_BODY
  SNAMES="gpt-4-0125-preview" serve $T/models/dsc7b-v15-tokfix dsc7b_w1 $DSC || { touch $R/status/window1.failed; return 1; }
  export IRIS_LOCAL_MERGE=1; labels s_dsc; unset IRIS_LOCAL_MERGE
  SNAMES="gpt-4-0125-preview" serve $T/models/Qwen3-32B qwen3-32b_w1 $Q32 || { touch $R/status/window1.failed; return 1; }
  export IRIS_EXTRA_BODY="$NOTHINK"; labels s_q32; unset IRIS_EXTRA_BODY
  cleanup; trap - EXIT; date +%s > $R/status/window1.t1; touch $R/status/window1.done
  log "WINDOW 1 DONE, GPU released"
}

window2() {
  until [ -f $R/status/cpu.done ] && [ -f $R/status/window2.go ]; do sleep 30; done
  gpu_free_wait; trap cleanup EXIT; log "WINDOW 2 START"; date +%s > $R/status/window2.t0
  SNAMES="qwen3-32b" serve $T/models/Qwen3-32B qwen3-32b_w2 $Q32 || return 1
  for run in $RUNS; do filt qwen3-32b qwen3-32b $run; done
  SNAMES="qwen3-8b" serve $T/models/Qwen3-8B qwen3-8b_w2 $Q8 || return 1
  for run in $RUNS; do filt qwen3-8b qwen3-8b $run; done
  cleanup; trap - EXIT; date +%s > $R/status/window2.t1; touch $R/status/window2.done
  log "WINDOW 2 DONE, GPU released"
}

analysis() {
  cd $T/ladder && $VPY l11_scaled.py --dir $R > $R/scaled.txt 2>&1; log "analysis written to $R/scaled.txt"
}

case "${1:-all}" in
  all) prep                    # the CPU worker is its own process, so no step here waits for it
       ( setsid nohup $0 cpu > $R/logs/cpu.log 2>&1 < /dev/null & )
       window1 || { log "WINDOW 1 FAILED, stopping"; cleanup; exit 1; }
       window2 || { log "WINDOW 2 FAILED, stopping"; cleanup; exit 1; }
       until [ -f $R/status/cpu.done ]; do sleep 30; done; analysis ;;
  prep) prep ;; window1) window1 ;; cpu) cpu ;; window2) window2 ;; analysis) analysis ;;
  paths) paths $2 ${3:-} ;;
  retry) retry $2 ;;
esac
