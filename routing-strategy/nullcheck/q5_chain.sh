#!/bin/bash
# Q5 positive control, gpu0: Qwen3-32B no-think vs think on the SAME 300 PrimeVul
# pairs E6 used, so the only variable against the measured Qwen3-8B rows is scale.
#
# Same box (A100-SXM4-80GB), same engine (vLLM 0.27.1), same prompts, same
# think_probe.py, same `--pairs 300` selection -> the two new rows drop straight
# into the E12 table.
#
# 32B in bf16 is ~66 GB of weights, so this takes most of the card. E6 ran at
# util 0.40 to stay polite; that is not available here.
#
# max-model-len stays at E6's 16384: the longest PrimeVul prompt in this set is
# 8,568 input tokens and think mode generates up to 6,144, so a smaller window
# would reject exactly the largest functions and bias the sample.
set -uo pipefail
cd ~/nimit/vr_explore
export PATH=~/nimit/vllmenv/bin:$PATH          # vLLM JIT needs ninja, which lives here
export HF_HOME=~/nimit/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
# Weights are a plain directory, not an HF cache entry: gpu0's own link to
# huggingface.co collapsed to ~21 MB/min, so the 65 GB was streamed in through
# the laptop instead. vLLM serves a local path the same way it serves a repo id.
PY=~/nimit/vllmenv/bin/python
PORT=18082
B=http://127.0.0.1:$PORT/v1

vllm serve $HOME/nimit/models/Qwen3-32B --served-model-name qwen3-32b --dtype bfloat16 \
    --max-model-len 16384 --gpu-memory-utilization 0.95 --max-num-seqs 16 \
    --port $PORT > vllm_q5.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; for i in $(seq 1 30); do kill -0 $SRV 2>/dev/null || break; sleep 2; done; kill -9 $SRV 2>/dev/null; sleep 5; echo "server stopped $(date)"' EXIT

for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null && { echo "up $(date)"; break; }
    kill -0 $SRV 2>/dev/null || { echo "DIED"; tail -30 vllm_q5.log; exit 1; }
    sleep 10
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "TIMEOUT waiting for server"; exit 1; }

# smoke: one pair through BOTH modes, so a template or parse bug fails in
# seconds rather than after an hour of thinking.
head -2 pairs.jsonl > q5_smoke.jsonl
rm -f q5_smoke_nothink.jsonl q5_smoke_think.jsonl
$PY think_probe.py --units q5_smoke.jsonl --out q5_smoke_nothink.jsonl --mode nothink --served qwen3-32b --base $B --workers 2
$PY think_probe.py --units q5_smoke.jsonl --out q5_smoke_think.jsonl   --mode think   --served qwen3-32b --base $B --workers 2
python3 - <<'PYEOF'
import json, sys
ok = True
for f in ("q5_smoke_nothink.jsonl", "q5_smoke_think.jsonl"):
    rows = [json.loads(l) for l in open(f)]
    bad = [r for r in rows if r.get("error") or r.get("pred") is None or r.get("p_vuln") is None]
    print(f, "FAIL" if bad else "OK", json.dumps(rows[0], default=str)[:300])
    if bad:
        print("  first bad:", json.dumps(bad[0], default=str)[:400]); ok = False
open("q5_smoke_ok", "w").write("ok" if ok else "")
PYEOF
[ -s q5_smoke_ok ] || { echo "smoke failed, stopping before the expensive run"; exit 1; }

echo "=== Qwen3-32B NO-THINK, 300 pairs $(date)"
$PY think_probe.py --units pairs.jsonl --out qwen32_nothink.jsonl --mode nothink --pairs 300 --served qwen3-32b --base $B --workers 24
echo "=== Qwen3-32B THINK, 300 pairs $(date)"
$PY think_probe.py --units pairs.jsonl --out qwen32_think.jsonl   --mode think   --pairs 300 --served qwen3-32b --base $B --workers 24
echo "=== ALL DONE $(date)"
