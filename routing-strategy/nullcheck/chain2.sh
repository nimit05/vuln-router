#!/bin/bash
# gpu0: Qwen3-8B thinking-budget frontier + self-consistency. One server, stopped at the end.
set -uo pipefail
cd ~/nimit/vr_explore
export PATH=~/nimit/vllmenv/bin:$PATH HF_HOME=~/nimit/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=~/nimit/vllmenv/bin/python
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --dtype bfloat16 \
    --max-model-len 16384 --gpu-memory-utilization 0.40 --port 18080 > vllm_b.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; sleep 8; kill -9 $SRV 2>/dev/null; echo "server stopped $(date)"' EXIT
for i in $(seq 1 150); do
    curl -sf http://127.0.0.1:18080/health >/dev/null && { echo "up $(date)"; break; }
    kill -0 $SRV 2>/dev/null || { echo "DIED"; tail -25 vllm_b.log; exit 1; }
    sleep 10
done
B=http://127.0.0.1:18080/v1
# smoke: one unit at budget 512, so a template/parse bug fails in seconds not an hour
head -2 pairs.jsonl > smoke.jsonl
$PY probe2.py --units smoke.jsonl --out smoke_out.jsonl --mode budget:512 --base $B --workers 2
python3 - <<'PYEOF'
import json
rows=[json.loads(l) for l in open('smoke_out.jsonl')]
bad=[r for r in rows if r.get('error') or r.get('pred') is None]
print('SMOKE', 'FAIL' if bad else 'OK', json.dumps(rows[0], default=str)[:400])
open('smoke_ok','w').write('' if bad else 'ok')
PYEOF
[ -s smoke_ok ] || { echo "smoke failed, stopping"; exit 1; }
for BUD in 0 256 512 1024 2048 4096; do
    echo "=== budget $BUD $(date)"
    $PY probe2.py --units pairs.jsonl --out budget_$BUD.jsonl --mode budget:$BUD --pairs 299 --base $B --workers 64
done
echo "=== self-consistency k=5 @1024 $(date)"
$PY probe2.py --units pairs.jsonl --out sc5_1024.jsonl --mode sc:5@1024 --pairs 299 --base $B --workers 48
echo "=== ALL DONE $(date)"
