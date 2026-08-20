#!/bin/bash
# Pre-download model weights on the LOGIN node.
# Compute nodes may have no outbound network, and vLLM must not stall mid-job
# fetching from HuggingFace. Run this before submitting anything.
#
# Usage: ./stage_weights.sh Qwen/Qwen2.5-Coder-7B-Instruct
#
# HOME QUOTA: 50 GB soft / 200 GB hard. bf16 weights are ~2 bytes/param:
#   deepseek-coder-6.7b-instruct   ~13 GB
#   Qwen2.5-Coder-7B-Instruct      ~15 GB
#   Qwen2.5-Coder-14B-Instruct     ~29 GB
#   Qwen2.5-Coder-32B-Instruct     ~62 GB   (needs an 80GB A100)
# Stage ONE model at a time and delete it before staging the next.
set -euo pipefail

MODEL="${1:?usage: stage_weights.sh <hf-model-id>}"
export HF_HOME="${HF_HOME:-$HOME/models/hf}"
SIF="${SIF:-$HOME/containers/llmdfa.sif}"

mkdir -p "$HF_HOME"
echo "quota before:"; quota -s 2>/dev/null || df -h "$HOME" | tail -1

apptainer exec --bind "$HF_HOME:$HF_HOME" --env HF_HOME="$HF_HOME" "$SIF" \
    python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$MODEL', allow_patterns=['*.json','*.safetensors','*.model','*.txt'])
print('staged ->', p)
"

echo "quota after:"; quota -s 2>/dev/null || df -h "$HOME" | tail -1
echo
echo "To free it later:  rm -rf $HF_HOME/hub/models--${MODEL//\//--}"
