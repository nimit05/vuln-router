#!/bin/bash
# Launch a probe sweep and record it, so the cron watchdog can restart the RIGHT
# one if the driver dies.
#
#   ./launch_sweep.sh ~/data/units/units_slices.jsonl slice 64
#
# Writes ~/.graphrouter_sweep.env, which watchdog.sh sources. Without that file
# a watchdog restart falls back to the default unit set and silently runs the
# wrong sweep.
set -uo pipefail
UNITS="${1:?usage: launch_sweep.sh <units.jsonl> [style] [max_tokens]}"
STYLE="${2:-json}"
MAXTOK="${3:-64}"

[ -f "$UNITS" ] || { echo "no such unit file: $UNITS"; exit 1; }

cat > "$HOME/.graphrouter_sweep.env" <<EOF
UNITS="$UNITS"
STYLE="$STYLE"
MAXTOK="$MAXTOK"
EOF

rm -f "$HOME/logs/sweep_stop"
mkdir -p "$HOME/logs"

if pgrep -f "queue_sweep.sh" >/dev/null 2>&1; then
    echo "a driver is already running; not starting a second one"
    exit 0
fi

cd "$HOME" || exit 1
UNITS="$UNITS" STYLE="$STYLE" MAXTOK="$MAXTOK" \
    setsid nohup "$HOME/routing/cluster/queue_sweep.sh" \
    >> "$HOME/logs/sweep_driver.out" 2>&1 < /dev/null &
sleep 2
echo "launched: $(wc -l < "$UNITS") units, style=$STYLE, max_tokens=$MAXTOK"
echo "watchdog will restart this exact sweep from ~/.graphrouter_sweep.env"
