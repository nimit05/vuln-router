#!/bin/bash
# control: UniXcoder fine-tuned on the answer key directly (no LLM), main split + LOCO;
# then save the Jev-style routers fitted on train for jevlike_route.py
set -uo pipefail
cd ~/nimit/jevlike
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
PY=~/nimit/vllmenv/bin/python; ENC=~/nimit/models/unixcoder-base
echo "=== start $(date)"
$PY ladder/l4_learned_router.py --encoder finetune --encoder-path $ENC --target label --tag direct-ft --out results/pred_direct-ft.json || echo "FAILED direct"
$PY ladder/l4_learned_router.py --encoder emb-lr --target label --tag direct-lr --out results/pred_direct-lr.json || echo "FAILED direct-lr"
$PY ladder/l4_learned_router.py --encoder emb-lr --emb-model ~/nimit/hf/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/$(ls ~/nimit/hf/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots | head -1) --no-loco --save results/router_emb-lr.pt --out results/pred_emb-lr_saved.json || echo "FAILED save lr"
$PY ladder/l4_learned_router.py --encoder finetune --encoder-path $ENC --seed 0 --no-loco --save results/router_ft.pt --out results/pred_ft_saved.json || echo "FAILED save ft"
echo "=== DIRECT DONE $(date)"
