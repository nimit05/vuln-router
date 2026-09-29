#!/bin/bash
# D1 on gpu0: Qwen3-32B over IRIS's dataflow paths -- the positive-control gate.
#
#   1. smoke: 2 paths through both modes, so a template/parse bug fails in seconds
#   2. no-think over ALL 2,257 paths (short outputs, ~10 min)
#   3. think over the 358-path gate sample (<=30 TP + <=30 FP per project)
#   4. batch-1 calibration: a few paths at --workers 1, to convert tokens to the
#      sequential GPU-seconds the B6 pool rows carry
#
# Full think over all paths is NOT run here: it is ~3 more hours and is only
# worth it if step 3 shows the model can rank real bugs above false alarms
# within a project. That decision is made on the laptop, after this exits.
set -uo pipefail
W=~/nimit/graphrouter
cd $W
export PATH=~/nimit/vllmenv/bin:$PATH          # vLLM JIT needs ninja, which lives here
export HF_HOME=~/nimit/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=~/nimit/vllmenv/bin/python
PORT=18083
B=http://127.0.0.1:$PORT/v1
U=$W/data/units_paths.jsonl
O=$W/d1_out
mkdir -p $O

vllm serve $HOME/nimit/models/Qwen3-32B --served-model-name qwen3-32b --dtype bfloat16 \
    --max-model-len 12288 --gpu-memory-utilization 0.95 --max-num-seqs 16 \
    --port $PORT > $O/vllm_d1.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; for i in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 2; done; kill -9 $SRV 2>/dev/null; sleep 5; echo "server stopped $(date)"' EXIT

for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null && { echo "up $(date)"; break; }
    kill -0 $SRV 2>/dev/null || { echo "DIED"; tail -30 $O/vllm_d1.log; exit 1; }
    sleep 10
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "TIMEOUT waiting for server"; exit 1; }

P="$PY $W/scripts/d1_probe_reasoner.py --units $U --base $B"
rm -f $O/smoke_*.jsonl
$P --limit 2 --out $O/smoke_nothink.jsonl --mode nothink --workers 2
$P --limit 2 --out $O/smoke_think.jsonl   --mode think   --workers 2
$PY - "$O" <<'PYEOF'
import json, sys, os
ok = True
for m in ("nothink", "think"):
    rows = [json.loads(l) for l in open(os.path.join(sys.argv[1], f"smoke_{m}.jsonl"))]
    bad = [r for r in rows if r.get("error") or r.get("pred_label") is None or r.get("p_vuln") is None]
    print(m, "FAIL" if bad else "OK", json.dumps(rows[0])[:300])
    ok &= not bad
open(os.path.join(sys.argv[1], "smoke_ok"), "w").write("ok" if ok else "")
PYEOF
[ -s $O/smoke_ok ] || { echo "smoke failed, stopping before the expensive run"; exit 1; }

echo "=== NO-THINK, all paths $(date)"
$P --out $O/qwen3-32b-nothink__units_paths.jsonl --mode nothink --workers 16
echo "=== THINK, gate sample $(date)"
$P --ids $W/data/d1_gate_ids.json --out $O/qwen3-32b-think__gate.jsonl --mode think --workers 16
echo "=== CALIBRATION, batch-1 $(date)"
$P --limit 8 --out $O/calib_nothink.jsonl --mode nothink --workers 1
$P --ids $W/data/d1_gate_ids.json --limit 5 --out $O/calib_think.jsonl --mode think --workers 1
echo "=== ALL DONE $(date)"
