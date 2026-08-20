#!/bin/bash
# ==============================================================================
# Run IRIS over a list of LLMs and produce one comparison table.
#
#   ./run_models.sh                          # every model in models.json
#   ./run_models.sh ollama-llama3-8b         # just these
#   PROJECTS=/tmp/subset.txt ./run_models.sh # different project list
#
# Results -> ../results/IRIS/metrics_<model>.txt  and  comparison.txt
# ==============================================================================
set -u
cd "$(dirname "$0")"
IRIS=$PWD
RES=$IRIS/../results/IRIS
PY=$IRIS/.conda-iris/bin/python
export PATH=$IRIS/codeql:$PATH
export PYTHONPATH=$IRIS:$IRIS/src
export OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}
PROJECTS=${PROJECTS:-/tmp/subset.txt}
PARALLEL=${PARALLEL:-3}          # projects analysed at once
mkdir -p "$RES"

if [ $# -gt 0 ]; then
  MODELS=("$@")
else
  mapfile -t MODELS < <($PY -c "
import json;print('\n'.join(k for k in json.load(open('models.json')) if not k.startswith('_')))")
fi

declare -A CWE
while IFS=, read -r id slug cve cwe rest; do CWE[$slug]=$cwe; done \
  < <(tail -n +2 data/cwe-bench-java/data/project_info.csv)

echo "models:   ${MODELS[*]}"
echo "projects: $(grep -c . "$PROJECTS") from $PROJECTS"

for MODEL in "${MODELS[@]}"; do
  RUNID=$(echo "$MODEL" | sed 's/^ollama-//; s/[^a-zA-Z0-9]/_/g')
  LOG=/tmp/iris_$RUNID.log; : > "$LOG"
  echo ""; echo "=========== $MODEL (run-id=$RUNID) ==========="
  N=0
  while read -r slug; do
    [ -z "$slug" ] && continue
    if [ ! -d "data/codeql-dbs/$slug/db-java" ]; then
      echo "  skip (no CodeQL db): $slug"; continue
    fi
    c=${CWE[$slug]}; c=${c#CWE-}
    ( timeout 7200 $PY src/neusym_vul.py --query "cwe-${c}wLLM" \
        --run-id "$RUNID" --llm "$MODEL" "$slug" >> "$LOG" 2>&1 ) &
    N=$((N+1)); [ $((N % PARALLEL)) -eq 0 ] && wait
  done < "$PROJECTS"
  wait
  echo "  -> scoring"
  $PY "$RES/score_subset.py" --run-id "$RUNID" --subset "$PROJECTS" \
      --label "OURS: $MODEL" --compare CodeQL.csv IRIS+DeepSeekCoder-7B.csv IRIS+GPT-4.csv \
      > "$RES/metrics_$RUNID.txt" 2>/dev/null
  sed -n '2,8p' "$RES/metrics_$RUNID.txt"
done

# ---- combined table ----
{
  echo "# IRIS multi-LLM comparison — $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "# projects: $PROJECTS"
  echo ""
  printf "%-40s %6s %8s %10s %8s\n" configuration "#Det" "rate%" "AvgFDR%" "AvgF1"
  printf -- "------------------------------------------------------------------------\n"
  grep -h "^OURS:"  "$RES"/metrics_*.txt 2>/dev/null | sort -u
  grep -h "^paper:" "$RES"/metrics_*.txt 2>/dev/null | sort -u
} > "$RES/comparison.txt"
echo ""; echo "===== COMPARISON ====="; cat "$RES/comparison.txt"
