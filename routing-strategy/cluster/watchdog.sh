#!/bin/bash
# Restart the sweep driver if it is not running. Installed as a cron entry on
# the LOGIN node, so the sweep resumes after a reboot or an admin cleanup with
# nobody connected.
#
# It must restart the sweep that was actually launched, not the default one.
# `launch_sweep.sh` records the unit set, prompt style and token budget in
# ~/.graphrouter_sweep.env; without sourcing that, a watchdog restart would
# silently re-run whichever sweep is hardcoded as the default and report it as
# complete -- which is exactly what happened the first time this ran.
#
# Safe to fire against a healthy driver: queue_sweep.sh judges completion by
# rows on disk and skips anything already queued, so a duplicate copy finds no
# work and exits.
#
# To stand the whole thing down:  touch ~/logs/sweep_stop
pgrep -f "queue_sweep.sh|handoff.sh" >/dev/null 2>&1 && exit 0
[ -f "$HOME/logs/sweep_stop" ] && exit 0          # manual kill switch only
[ -f "$HOME/.graphrouter_sweep.env" ] || exit 0   # no sweep has been launched

# shellcheck disable=SC1090
. "$HOME/.graphrouter_sweep.env"
export UNITS STYLE MAXTOK

cd "$HOME" || exit 1
setsid nohup "$HOME/routing/cluster/queue_sweep.sh" \
    >> "$HOME/logs/sweep_driver.out" 2>&1 < /dev/null &
echo "$(date '+%m-%d %H:%M') watchdog restarted the driver (UNITS=$UNITS STYLE=$STYLE)" \
    >> "$HOME/logs/sweep.log"
