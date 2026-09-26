#!/bin/bash
# Stand up vLLM and the model pool on gpu0.
#
# gpu0 is a SHARED box: the account holds other people's directories, so
# everything here lives under ~/nimit/ and touches nothing else (lab operating
# procedure, point 2). The venv is named for project + owner (point 3).
#
# tmux is not installed on this node, so point 4 cannot be followed literally;
# work runs under setsid with logs under ~/nimit/graphrouter/logs so the owner
# is identifiable from the process list.
set -uo pipefail
ROOT="$HOME/nimit/graphrouter"
VENV="$HOME/nimit/graphrouter_nimit"        # <project>_<identifier>
export HF_HOME="$HOME/nimit/models/hf"
LOG="$ROOT/logs/setup.log"
mkdir -p "$ROOT/logs" "$HF_HOME"
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$LOG"; }

log "setup starting on $(hostname)"

# python3-venv is not installed on this node and there is no sudo, so the
# stdlib venv module cannot bootstrap pip. miniconda3 is present, so the env is
# created with conda at OUR OWN prefix -- inside ~/nimit, not in the shared
# envs directory, so it is attributable and removable (operating procedure 2/3).
CONDA="$HOME/miniconda3/bin/conda"
if [ ! -x "$VENV/bin/python" ]; then
    [ -x "$CONDA" ] || { log "no conda at $CONDA and no python3-venv -- cannot build an env"; exit 1; }
    log "creating conda env at $VENV"
    "$CONDA" create -y -p "$VENV" python=3.12 2>&1 | tail -3 \
        || { log "conda env creation FAILED"; exit 1; }
fi
"$VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -1

if ! "$VENV/bin/python" -c "import vllm" 2>/dev/null; then
    log "installing vllm (this pulls torch + CUDA wheels, several GB)"
    "$VENV/bin/pip" install -q vllm 2>&1 | tail -5 || { log "vllm install FAILED"; exit 1; }
fi
log "vllm: $("$VENV/bin/python" -c 'import vllm;print(vllm.__version__)' 2>&1 | tail -1)"
log "torch: $("$VENV/bin/python" -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1)"

# --- weights -------------------------------------------------------------
# Staged one at a time so a failure is attributable and disk stays predictable.
MODELS=(
  "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  "microsoft/Phi-4-mini-instruct"
  "deepseek-ai/deepseek-coder-6.7b-instruct"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "ibm-granite/granite-3.1-8b-instruct"
)
for M in "${MODELS[@]}"; do
    log "staging $M"
    HF_HOME="$HF_HOME" "$VENV/bin/python" - "$M" <<'PY' 2>&1 | tail -2
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], ignore_patterns=["*.pth", "*.msgpack", "*.h5"])
print("  ->", p)
PY
done

log "HF cache size: $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
log "setup complete"
