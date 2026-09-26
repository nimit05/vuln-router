#!/bin/bash
# Everything from a complete 5-model probe table to the deliverable, in one run.
# Intended for the csecluster login node, where ~/grvenv has torch + PyG.
#
#   ~/routing/cluster/finish_chain.sh
#
# Refuses to start unless all five models have a full set of VALID rows. A
# partial probe silently becomes "kept" for every missing path in the scorer,
# which would read as a permissive filter rather than as missing data.
set -uo pipefail

GR="$HOME/routing/graphrouter"
PROBE="${PROBE:-$HOME/probe_gpu0}"
UNITS="${UNITS:-$HOME/data/units_v2/units_paths.jsonl}"
SUFFIX="units_paths"
EPOCHS="${EPOCHS:-1000}"
PY="$HOME/grvenv/bin/python"
NEED="$(wc -l < "$UNITS")"

cd "$GR" || exit 1

echo "==> checking the probe table is complete"
MISSING=0
for m in qwen-1.5b phi-3.8b qwen-7b granite-8b deepseek-6.7b; do
    f="$PROBE/${m}__${SUFFIX}.jsonl"
    if [ ! -f "$f" ]; then echo "  MISSING $m"; MISSING=1; continue; fi
    V="$($PY - "$f" <<'PY'
import json,sys
n=0
for line in open(sys.argv[1]):
    try:
        if json.loads(line).get('in_tokens',0) > 0: n+=1
    except Exception: pass
print(n)
PY
)"
    echo "  $m: $V/$NEED valid"
    [ "$V" -ge "$NEED" ] || MISSING=1
done
[ "$MISSING" = 0 ] || { echo "probe table incomplete -- refusing to build a table from it"; exit 1; }

echo "==> A3: LLM nodes, with measured GPU-seconds as cost"
$PY a3_llm_nodes.py --probe-dir "$PROBE" --unit-set "$SUFFIX" \
    --out-json data/LLM_Descriptions.json \
    --out-pkl data/llm_description_embedding.pkl || exit 1

echo "==> B6b: router_data.csv (reports model disagreement -- read it)"
$PY b6b_router_data.py --probe-dir "$PROBE" --units "$UNITS" \
    --unit-set "$SUFFIX" --llm-json data/LLM_Descriptions.json \
    --out data/router_data.csv --groups-out data/split_groups.json || exit 1

echo "==> B7: leave-one-project-out, $EPOCHS epochs"
$PY b7_train_router.py --gr-root "$HOME/routing/third_party/GraphRouter" \
    --router-data data/router_data.csv --groups data/split_groups.json \
    --llm-json data/LLM_Descriptions.json \
    --llm-emb data/llm_description_embedding.pkl \
    --epochs "$EPOCHS" --out data/routes.json \
    --model-path data/router_fold.pth || exit 1

echo "==> done. Pull data/routes.json and the probe files to score locally,"
echo "    or run c1_score_table.py here if the IRIS repo is reachable."
