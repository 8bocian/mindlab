#!/usr/bin/env bash
# Runs ON THE VM: (re)starts the MindLab server and the idle watchdog in the background.
set -euo pipefail
LAB="$HOME/mindlab"
cd "$LAB/server"
set -a; source "$LAB/server.env"; [ -f "$LAB/runtime.env" ] && source "$LAB/runtime.env"; set +a
export HEARTBEAT_FILE="$LAB/heartbeat"
touch "$HEARTBEAT_FILE"

pkill -f "uvicorn app:app" 2>/dev/null || true
sleep 1
nohup "$LAB/.venv/bin/python" -m uvicorn app:app --host 127.0.0.1 --port 8000 > "$LAB/server.log" 2>&1 &
echo "serwer uruchomiony (log: $LAB/server.log)"

if [ -f "$LAB/watchdog.env" ]; then
  pkill -f "mindlab/watchdog.py" 2>/dev/null || true
  nohup python3 "$LAB/watchdog.py" > "$LAB/watchdog.log" 2>&1 &
  echo "strażnik bezczynności uruchomiony (log: $LAB/watchdog.log)"
fi
