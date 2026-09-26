#!/bin/bash
# Drive the probe sweep into SLURM, unattended, and survive its own death.
#
# Three things this has to tolerate, because a 3-hour sweep will meet at least
# one of them:
#   * the 2-submitted-jobs QOS cap  -- so it submits only when a slot frees;
#     --dependency does not help, a held job still occupies a slot
#   * a job dying (node failure, walltime, OOM) -- so completion is judged by
#     ROWS ON DISK, never by exit status, and an incomplete model is resubmitted
#   * this script dying (login-node reboot, admin cleanup) -- so it is fully
#     idempotent and a cron watchdog restarts it
#
# Nothing here depends on a laptop being connected. Re-running it at any time is
# safe and is the intended recovery action.
set -uo pipefail

UNITS="${UNITS:-$HOME/data/units/all.jsonl}"
BASE="$(basename "$UNITS" .jsonl)"
SBATCH="$HOME/routing/cluster/run_probe.sbatch"
STYLE="${STYLE:-json}"
MAXTOK="${MAXTOK:-64}"
OUTDIR="$HOME/probe_out"
STATE="$HOME/logs/sweep_attempts"
LOG="$HOME/logs/sweep.log"
MAX_ATTEMPTS=3          # a model that dies 3 times is broken, not unlucky
mkdir -p "$HOME/logs" "$OUTDIR" "$STATE"

NEED="$(wc -l < "$UNITS")"

# cheapest first, so a pipeline fault surfaces on the model that costs least
MODELS=(
  "qwen-1.5b:Qwen/Qwen2.5-Coder-1.5B-Instruct"
  "phi-3.8b:microsoft/Phi-4-mini-instruct"
  "deepseek-6.7b:deepseek-ai/deepseek-coder-6.7b-instruct"
  "qwen-7b:Qwen/Qwen2.5-Coder-7B-Instruct"
  "granite-8b:ibm-granite/granite-3.1-8b-instruct"
)

# Per-model walltime, and it is a SCHEDULING lever, not a safety margin.
#
# A single --time=12:00:00 for every model kept the whole sweep out of SLURM's
# backfill: a 12-hour request cannot be slotted into the 30-minute gap that
# opens when someone else's job ends, so our jobs sat behind a higher-priority
# reservation with an estimated start 20 HOURS out while a GPU stood idle.
# Asking for what a model actually needs lets backfill run it almost at once.
#
# Measured live: dropping qwen-1.5b from 45 to 28 minutes moved it from PENDING
# to RUNNING within 20 seconds, because the gap ahead of the next reservation was
# 32 minutes. Keep these close to the measured runtime, not padded.
#
# Asking too LITTLE is cheap here and that is the point: the job is --requeue,
# the probe client appends and resumes from its own output, and the driver
# resubmits from the row count on disk. A walltime kill costs one restart, not
# a rerun. Measured on 2,463 units, so these carry roughly 2x headroom.
wall_for() {
    case "$1" in
        qwen-1.5b)     echo "00:30:00" ;;   # ~18 min measured
        phi-3.8b)      echo "00:25:00" ;;   # ~13 min
        qwen-7b)       echo "00:30:00" ;;   # ~18 min
        granite-8b)    echo "00:50:00" ;;   # ~30 min
        deepseek-6.7b) echo "03:00:00" ;;   # ~2.4 h, the slow one by 8x
        *)             echo "01:00:00" ;;
    esac
}

log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$LOG"; }

rows_for() {                       # USABLE rows already probed for a model
    # Counting lines is not enough. A job whose every HTTP request failed still
    # writes one row per unit -- the client records a failure as data on purpose
    # (gate G2 needs retry cost) -- so a completely dead run looks 100% done.
    # That happened to phi-3.8b: 2463 rows, 0 real calls, all junk. A row only
    # counts if the model actually answered, i.e. in_tokens > 0.
    local f="$OUTDIR/$1__$BASE.jsonl"
    [ -f "$f" ] || { echo 0; return; }
    python3 -c "
import json,sys
n=0
for line in open(sys.argv[1]):
    line=line.strip()
    if not line: continue
    try:
        if json.loads(line).get('in_tokens',0) > 0: n+=1
    except Exception: pass
print(n)" "$f" 2>/dev/null || wc -l < "$f"
}
queued() {                         # is this model already in the queue?
    squeue -h -u "$USER" -o "%j" 2>/dev/null | grep -qx "vrp_$1"
}
slots_used() { squeue -h -u "$USER" 2>/dev/null | wc -l; }

log "sweep driver up (pid $$), units=$UNITS need=$NEED models=${#MODELS[@]}"

while true; do
    pending=0
    for entry in "${MODELS[@]}"; do
        SHORT="${entry%%:*}"; MODEL="${entry#*:}"
        HAVE="$(rows_for "$SHORT")"
        if [ "$HAVE" -ge "$NEED" ]; then continue; fi
        if queued "$SHORT"; then pending=1; continue; fi

        ATT_FILE="$STATE/${BASE}_$SHORT"
        ATT="$(cat "$ATT_FILE" 2>/dev/null || echo 0)"
        if [ "$ATT" -ge "$MAX_ATTEMPTS" ]; then
            log "SKIP $SHORT: $ATT attempts, $HAVE/$NEED rows -- needs a human"
            continue
        fi
        if [ "$(slots_used)" -ge 2 ]; then pending=1; continue; fi

        echo $((ATT + 1)) > "$ATT_FILE"
        cd "$HOME/cluster" || exit 1
        WALL="$(wall_for "$SHORT")"
        OUT="$(MODEL="$MODEL" SHORT="$SHORT" UNITS="$UNITS" STYLE="$STYLE" MAXTOK="$MAXTOK" \
               sbatch -J "vrp_$SHORT" --time="$WALL" "$SBATCH" 2>&1)"
        log "submit $SHORT (attempt $((ATT + 1)), resuming from $HAVE rows, wall=$WALL) -> $OUT"
        pending=1
        sleep 20
    done
    if [ "$pending" -eq 0 ]; then break; fi
    sleep 120
done

log "sweep complete: all models at $NEED rows"
for entry in "${MODELS[@]}"; do
    SHORT="${entry%%:*}"
    log "  $SHORT $(rows_for "$SHORT")/$NEED"
done
