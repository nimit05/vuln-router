#!/bin/bash
# Write a human-readable status snapshot to ~/STATUS.txt every few minutes.
#
# The point is that the state of the work should be answerable without a laptop
# connected and without anyone reading a log: `cat ~/STATUS.txt` on the login
# node says what is running, how far it has got, and whether anything is stuck.
#
# Installed as a cron entry alongside watchdog.sh.
OUT="$HOME/STATUS.txt"
TMP="$OUT.tmp"
{
  echo "GraphRouter / routing-strategy  --  $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "=============================================================="
  echo
  echo "SLURM queue (mine):"
  squeue -u "$USER" -o "  %.8i %.16j %.9P %.2t %.10M %.10L %R" 2>/dev/null | sed 1d \
      || echo "  (squeue unavailable)"
  [ -z "$(squeue -h -u "$USER" 2>/dev/null)" ] && echo "  (nothing queued or running)"
  echo
  echo "Probe progress (rows written / units required):"
  for f in "$HOME"/probe_out/*.jsonl; do
      [ -e "$f" ] || { echo "  (no probe output yet)"; break; }
      base="$(basename "$f" .jsonl)"
      set="${base##*__}"
      need=0
      [ -f "$HOME/data/units/$set.jsonl" ] && need=$(wc -l < "$HOME/data/units/$set.jsonl")
      have=$(wc -l < "$f")
      pct=0; [ "$need" -gt 0 ] && pct=$(( 100 * have / need ))
      printf "  %-34s %6d / %-6d  %3d%%\n" "$base" "$have" "$need" "$pct"
  done
  echo
  echo "Drivers:"
  pgrep -f "queue_sweep.sh" >/dev/null 2>&1 \
      && echo "  sweep driver: RUNNING" || echo "  sweep driver: not running"
  pgrep -f "joern" >/dev/null 2>&1 && echo "  joern: RUNNING"
  [ -f "$HOME/logs/sweep_stop" ] && echo "  ** stand-down flag set (~/logs/sweep_stop) **"
  echo
  echo "Recent sweep log:"
  tail -6 "$HOME/logs/sweep.log" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
  echo
  echo "Storage:"
  printf "  home used: %s\n" "$(du -sh "$HOME" 2>/dev/null | cut -f1)"
  df -h "$HOME" 2>/dev/null | tail -1 | sed 's/^/  /'
  echo
  echo "Cluster GPU partition:"
  sinfo -p gpu-A100 -o "  %14N %6t %20G %.16C" 2>/dev/null | sed 1d
} > "$TMP" 2>&1
mv "$TMP" "$OUT"
