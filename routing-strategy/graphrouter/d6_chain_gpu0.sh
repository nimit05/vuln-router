#!/bin/bash
# D6 on gpu0: memorisation control. Same models, prompt, mode and engine as D1/D3;
# only the code text changes (d6_anonymise.py: project names hidden).
#
#   Qwen3-32B no-think : original again (run-to-run noise), renamed (a), strict (b)
#   Qwen3-8B  no-think : renamed (a), strict (b)
#
# If the 32B's within-project AUC holds on (a) and (b), it is reading the code,
# not recognising known CVEs. One server at a time; each is stopped on exit.
set -uo pipefail
W=~/nimit/graphrouter
cd $W
export PATH=~/nimit/vllmenv/bin:$PATH
export HF_HOME=~/nimit/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=~/nimit/vllmenv/bin/python
PORT=18086
B=http://127.0.0.1:$PORT/v1
D=$W/data
O=$W/d6_out
mkdir -p $O
SRV=
stop() { [ -n "$SRV" ] && { kill $SRV 2>/dev/null; for i in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 2; done; kill -9 $SRV 2>/dev/null; }; SRV=; sleep 5; echo "server stopped $(date)"; }
trap stop EXIT

serve() {   # short path util
    echo "=== $1: serve $(date)"
    vllm serve $2 --served-model-name $1 --dtype bfloat16 --max-model-len 8192 \
        --gpu-memory-utilization $3 --max-num-seqs 32 --port $PORT > $O/vllm_$1.log 2>&1 &
    SRV=$!
    for i in $(seq 1 120); do
        curl -sf http://127.0.0.1:$PORT/health >/dev/null && { echo "up $(date)"; return 0; }
        kill -0 $SRV 2>/dev/null || { echo "DIED $1"; tail -20 $O/vllm_$1.log; SRV=; return 1; }
        sleep 10
    done
    return 1
}

probe() {   # short variant units-file
    local P="$PY $W/scripts/d1_probe_reasoner.py --base $B --served $1 --model $1 --mode nothink"
    rm -f $O/smoke_$1_$2.jsonl
    $P --units $3 --limit 2 --out $O/smoke_$1_$2.jsonl --workers 2
    $PY -c "
import json,sys
rows=[json.loads(l) for l in open('$O/smoke_$1_$2.jsonl')]
bad=[r for r in rows if r.get('error') or r.get('pred_label') is None or r.get('p_vuln') is None]
print('$1 $2 smoke', 'FAIL' if bad else 'OK'); sys.exit(1 if bad else 0)" || { echo "smoke failed $1 $2"; return 1; }
    echo "=== $1 $2: all paths $(date)"
    $P --units $3 --out $O/$1-nothink__$2.jsonl --workers 32
}

Q32=$HOME/nimit/models/Qwen3-32B
Q8=$(ls -d ~/nimit/hf/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)

if serve qwen3-32b $Q32 0.95; then
    probe qwen3-32b orig  $D/units_paths.jsonl
    probe qwen3-32b anonA $D/units_paths_anonA.jsonl
    probe qwen3-32b anonB $D/units_paths_anonB.jsonl
fi
stop
if serve qwen3-8b $Q8 0.45; then
    probe qwen3-8b anonA $D/units_paths_anonA.jsonl
    probe qwen3-8b anonB $D/units_paths_anonB.jsonl
fi
stop
echo "=== ALL DONE $(date)"
