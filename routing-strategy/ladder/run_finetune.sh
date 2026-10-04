#!/bin/bash
# UniXcoder Jev-style router on gpu0: 3 seeds on the main split, LOCO once (seed 0)
set -uo pipefail
cd ~/nimit/jevlike
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
PY=~/nimit/vllmenv/bin/python; ENC=~/nimit/models/unixcoder-base
echo "=== start $(date)"
$PY ladder/l4_learned_router.py --encoder finetune --encoder-path $ENC --seed 0 --tag ft-s0 --out results/pred_ft-s0.json || echo "FAILED s0"
for s in 1 2; do
  $PY ladder/l4_learned_router.py --encoder finetune --encoder-path $ENC --seed $s --tag ft-s$s --no-loco --out results/pred_ft-s$s.json || echo "FAILED s$s"
done
echo "=== FT DONE $(date)"
