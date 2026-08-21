#!/bin/bash
# Feeds queued submissions in as SLURM slots free up.
# csecluster enforces QOSMaxSubmitJobPerUserLimit = 2 submitted jobs per user, so
# long multi-shard runs cannot all be queued at once.
#
#   nohup ~/cluster/queue_runner.sh ~/cluster/pending_jobs.txt > ~/logs/queue_runner.log 2>&1 &
#
# pending_jobs.txt holds one full sbatch command per line. Lines are consumed as
# they are submitted, so the file doubles as the remaining-work list.
set -u
PENDING="${1:?usage: queue_runner.sh <pending-file>}"
MAXQ="${MAXQ:-2}"
POLL="${POLL:-120}"

while :; do
    [ -s "$PENDING" ] || { echo "$(date '+%F %T') all submissions done"; break; }
    n=$(squeue -u "$USER" -h 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -lt "$MAXQ" ]; then
        cmd=$(head -1 "$PENDING")
        tail -n +2 "$PENDING" > "$PENDING.tmp" && mv "$PENDING.tmp" "$PENDING"
        echo "$(date '+%F %T') slots=$n submitting: $cmd"
        ( cd "$(dirname "$PENDING")" && eval "$cmd" ) || echo "  submission FAILED, continuing"
    else
        echo "$(date '+%F %T') slots=$n full, waiting"
    fi
    sleep "$POLL"
done
