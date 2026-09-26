#!/bin/bash
# One-shot bridge from the first (pre-resume) probe job to the hardened driver.
#
# Job 2847 was submitted before probe output moved to a stable, resumable path,
# so it writes into ~/job_results/<jobid>/ and only at the end. This waits for
# it, files its rows under the canonical name the driver looks for, lifts the
# stand-down guard, and becomes the driver.
#
# Runs detached on the login node: once launched it needs no connection.
set -uo pipefail
JOB="${JOB:-2847}"
SHORT="${SHORT:-qwen-1.5b}"
BASE="${BASE:-all}"
LOG="$HOME/logs/sweep.log"
mkdir -p "$HOME/probe_out" "$HOME/logs"

echo "$(date '+%m-%d %H:%M') handoff: waiting on job $JOB" >> "$LOG"
while squeue -h -j "$JOB" 2>/dev/null | grep -q .; do sleep 60; done

SRC="$HOME/job_results/$JOB/probe_${SHORT}_${BASE}.jsonl"
DST="$HOME/probe_out/${SHORT}__${BASE}.jsonl"
if [ -f "$SRC" ] && [ ! -f "$DST" ]; then
    cp "$SRC" "$DST"
    echo "$(date '+%m-%d %H:%M') handoff: filed $(wc -l < "$DST") rows -> $DST" >> "$LOG"
else
    echo "$(date '+%m-%d %H:%M') handoff: no usable output at $SRC; the driver will re-run $SHORT" >> "$LOG"
fi

rm -f "$HOME/logs/sweep_stop"          # lift the guard; watchdog now covers us
echo "$(date '+%m-%d %H:%M') handoff: starting driver" >> "$LOG"
exec "$HOME/routing/cluster/queue_sweep.sh"
