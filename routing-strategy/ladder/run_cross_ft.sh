#!/bin/bash
# after the IRIS ladder: fine-tuned Jev-style router + direct classifier, fitted on ALL OWASP,
# applied to every IRIS path; 3 seeds each. Waits for LADDER DONE and ~20 GB free (UniXcoder needs ~12 GB).
set -uo pipefail
I=~/nimit/iris_ladder; cd /tmp/nimit/jevlike
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
PY=/tmp/nimit/vllmenv/bin/python; ENC=/tmp/nimit/models/unixcoder-base
until tr -d '\000' < $I/run.log | grep -q "LADDER DONE"; do sleep 3; done
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 20000 ]; do echo "GPU busy, waiting $(date +%T)"; sleep 30; done
echo "=== start $(date)"
for s in 0 1 2; do
  $PY ladder/l4_learned_router.py --encoder finetune --encoder-path $ENC --seed $s --tag ft-s$s \
      --apply $I/units_iris.jsonl --out results/cross/apply_ft-s${s}_models.json || echo "FAILED ft s$s"
  $PY ladder/l4_learned_router.py --encoder finetune --encoder-path $ENC --seed $s --tag direct-ft-s$s --target label \
      --apply $I/units_iris.jsonl --out results/cross/apply_ft-s${s}_label.json || echo "FAILED direct s$s"
done
echo "=== CROSS FT DONE $(date)"
