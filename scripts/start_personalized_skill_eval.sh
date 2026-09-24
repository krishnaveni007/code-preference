#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$ROOT/outputs/longitudinal_pilot/personalized_skill_run"
LOG="$RUN_DIR/run.log"
PID_FILE="$RUN_DIR/run.pid"

mkdir -p "$RUN_DIR"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Run is already active (PID $old_pid)."
    echo "Log: $LOG"
    exit 0
  fi
fi

# The Python runner uses --skip-existing for every cell, so this launcher is
# safe to resume after interruption. caffeinate prevents macOS sleep while the
# detached process is active.
nohup caffeinate -dimsu \
  python3 "$ROOT/scripts/run_personalized_skill_eval.py" run \
    --execute \
    --parallel 3 \
  >>"$LOG" 2>&1 &

run_pid=$!
printf '%s\n' "$run_pid" > "$PID_FILE"

echo "Started personalized-skill evaluation (PID $run_pid)."
echo "Planned new cells: 9 (existing baseline traces are reused as controls)."
echo "Log: $LOG"
echo "Status: ps -p $run_pid"
echo "Follow: tail -f '$LOG'"
