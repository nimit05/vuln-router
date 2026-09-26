#!/bin/bash
# Stand up the environment GraphRouter's own code needs, on the LOGIN node
# (compute nodes have no clean HTTPS, so everything is fetched here first).
#
# Two deviations from their requirements.txt, both forced and both recorded:
#
#  * torch is NOT pinned to 2.1.0. That release has no wheel for Python 3.12,
#    which is what this cluster runs. Installing the current CPU build instead.
#  * torch_geometric is IMPORTED by model/graph_nn.py but missing from
#    requirements.txt, so it is added explicitly.
#
# CPU-only torch on purpose: the router is two GeneralConv layers at
# embedding_dim 8 over ~2.5k nodes. It does not need a GPU, and keeping it off
# the GPU partition means it never waits in that queue.
set -uo pipefail
VENV="$HOME/grvenv"
LOG="$HOME/logs/stage_env.log"
mkdir -p "$HOME/logs"
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$LOG"; }

if [ ! -x "$VENV/bin/python" ]; then
    log "creating venv at $VENV"
    python3 -m venv "$VENV" || { log "venv creation FAILED"; exit 1; }
fi
"$VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -2

log "installing torch (CPU build)"
"$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu \
    2>&1 | tail -3 || { log "torch install FAILED"; exit 1; }

log "installing torch_geometric + the rest"
"$VENV/bin/pip" install -q torch_geometric pandas scikit-learn scipy pyyaml \
    sentence-transformers tqdm 2>&1 | tail -3 \
    || { log "deps install FAILED"; exit 1; }

log "versions:"
"$VENV/bin/python" - <<'PY' 2>&1 | tee -a "$LOG"
import torch, torch_geometric, pandas, sklearn, sentence_transformers as st
print(f"  torch            {torch.__version__}")
print(f"  torch_geometric  {torch_geometric.__version__}")
print(f"  pandas           {pandas.__version__}")
print(f"  sklearn          {sklearn.__version__}")
print(f"  sentence-transf. {st.__version__}")
from torch_geometric.nn import GeneralConv
print("  GeneralConv importable: yes")
PY

# The encoder GraphRouter uses. Pull it here; compute nodes cannot reach HF.
log "pre-staging all-MiniLM-L6-v2 into the HF cache"
HF_HOME="$HOME/models/hf" "$VENV/bin/python" - <<'PY' 2>&1 | tail -3
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
v = m.encode(["a smoke-test sentence"])
print("  embedding dim:", v.shape)
PY

log "environment ready"
