#!/bin/bash
# Move the B6 sweep off the superseded slice set and onto the rebuilt path set.
#
# The old units (units_slices.jsonl, 2463 rows) were keyed on SARIF *results* and
# do not reconcile with IRIS's accounting. The new ones (units_paths.jsonl, 2257
# rows) are one row per dataflow path and pass b1_validate.py on all 16 projects.
# Probe rows join on unit_id, and the two id spaces do not overlap, so the old
# output cannot be reused -- it is archived, not deleted.
#
# Run on the csecluster login node:  ~/routing/cluster/swap_to_v2.sh
set -uo pipefail

NEW="$HOME/data/units_v2/units_paths.jsonl"
ARCHIVE="$HOME/probe_out/archive_units_v1_stale"

[ -f "$NEW" ] || { echo "missing $NEW -- stage it first"; exit 1; }

echo "==> stopping the sweep driver"
touch "$HOME/logs/sweep_stop"
sleep 12

echo "==> cancelling my queued and running probe jobs"
# scancel by user+name, never pkill: a pkill pattern sent over ssh matches the
# ssh command line carrying it and kills the session (this has happened 3x).
scancel -u "$USER" --name=vrp_qwen-1.5b,vrp_phi-3.8b,vrp_deepseek-6.7b,vrp_qwen-7b,vrp_granite-8b 2>/dev/null
scancel -u "$USER" 2>/dev/null
sleep 5
squeue -u "$USER"

echo "==> archiving probe output from the stale unit set"
mkdir -p "$ARCHIVE"
for f in "$HOME"/probe_out/*__units_slices.jsonl; do
    [ -e "$f" ] || continue
    mv -v "$f" "$ARCHIVE/"
done

echo "==> clearing attempt counters so every model starts fresh"
rm -f "$HOME"/logs/sweep_attempts/*__units_paths 2>/dev/null
rm -f "$HOME"/logs/sweep_attempts/*__units_slices 2>/dev/null

echo "==> launching the sweep on the rebuilt path set"
# max_tokens stays 256 for every model: the cost column is measured GPU-seconds,
# so a different budget on one model makes its cost incomparable to the others.
"$HOME/routing/cluster/launch_sweep.sh" "$NEW" slice 256

echo
echo "watch it with:  cat ~/STATUS.txt   or   tail -f ~/logs/sweep.log"
