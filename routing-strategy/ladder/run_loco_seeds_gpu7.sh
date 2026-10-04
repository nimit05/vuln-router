#!/bin/bash
# leave-one-category-out for fine-tune seeds 1 and 2 on gpu7 (seed 0 ran on gpu0 in run_finetune.sh)
set -uo pipefail
cd /tmp/nimit/jevlike
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
PY=/tmp/nimit/vllmenv/bin/python; ENC=/tmp/nimit/models/unixcoder-base
m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$m" -lt 1000 ] || { echo "GPU busy ($m MiB), not starting"; exit 1; }
echo "=== start $(date)"
for s in 1 2; do
  $PY ladder/l4_learned_router.py --encoder finetune --encoder-path $ENC --seed $s --tag ft-s$s --out results/pred_ft-s${s}_loco.json || echo "FAILED s$s"
done
echo "=== LOCO SEEDS DONE $(date)"
