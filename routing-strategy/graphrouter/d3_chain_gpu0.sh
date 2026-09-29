#!/bin/bash
# D3 on gpu0: probe candidate SKILLED nodes for GraphRouter v2, no-think mode.
#
# D2 showed the router picks Qwen3-32B for every path and ties it (0.213): with
# one skilled model and four at chance there is nothing to route between. A
# routing gain needs a second skilled model that fails on DIFFERENT paths, so
# each candidate here gets the SAME prompt, mode and engine as the D1 no-think
# pass, then a batch-1 calibration for cost. One model at a time; each server is
# stopped before the next starts.
#
# usage: d3_chain_gpu0.sh "short|model_path|gpu_util|probe flags|serve flags|mode" ...
#   e.g. d3_chain_gpu0.sh "nemo-12b|/path/to/snapshot|0.55|--plain"
# (run 2026-09-26 first as qwen3-8b + nemo-12b; nemo needed --plain, see below)
set -uo pipefail
W=~/nimit/graphrouter
cd $W
export PATH=~/nimit/vllmenv/bin:$PATH
export HF_HOME=~/nimit/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=~/nimit/vllmenv/bin/python
PORT=18084
B=http://127.0.0.1:$PORT/v1
U=$W/data/units_paths.jsonl
O=$W/d1_out
SRV=
stop() { [ -n "$SRV" ] && { kill $SRV 2>/dev/null; for i in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 2; done; kill -9 $SRV 2>/dev/null; }; SRV=; sleep 5; echo "server stopped $(date)"; }
trap stop EXIT

run_model() {   # short  path  util  extra-probe-flags  extra-serve-flags  mode
    local short=$1 path=$2 util=$3 extra=${4:-} sextra=${5:-} mode=${6:-nothink}
    echo "=== $short: serve $(date)"
    vllm serve $path --served-model-name $short \
        --max-model-len 8192 --gpu-memory-utilization $util --max-num-seqs 32 \
        --port $PORT $sextra > $O/vllm_$short.log 2>&1 &
    SRV=$!
    for i in $(seq 1 120); do
        curl -sf http://127.0.0.1:$PORT/health >/dev/null && { echo "up $(date)"; break; }
        kill -0 $SRV 2>/dev/null || { echo "DIED $short"; tail -20 $O/vllm_$short.log; SRV=; return 1; }
        sleep 10
    done
    local P="$PY $W/scripts/d1_probe_reasoner.py --units $U --base $B --served $short --model $short"
    rm -f $O/smoke_$short.jsonl
    $P --limit 2 --out $O/smoke_$short.jsonl --mode $mode --workers 2 $extra
    $PY -c "
import json,sys
rows=[json.loads(l) for l in open('$O/smoke_$short.jsonl')]
bad=[r for r in rows if r.get('error') or r.get('pred_label') is None or r.get('p_vuln') is None]
print('$short smoke', 'FAIL' if bad else 'OK', json.dumps(rows[0])[:300]); sys.exit(1 if bad else 0)" || { echo "smoke failed $short"; grep -iE 'error|400' $O/vllm_$short.log | tail -5; stop; return 1; }
    echo "=== $short: all paths $(date)"
    $P --out $O/${short}-${mode}__units_paths.jsonl --mode $mode --workers 32 $extra
    echo "=== $short: calibration $(date)"
    rm -f $O/calib_$short.jsonl
    $P --limit 8 --out $O/calib_$short.jsonl --mode $mode --workers 1 $extra
    stop
}

for spec in "$@"; do
    IFS='|' read -r short path util extra sextra mode <<< "$spec"
    run_model "$short" "$path" "$util" "$extra" "$sextra" "${mode:-nothink}"
done
echo "=== ALL DONE $(date)"
