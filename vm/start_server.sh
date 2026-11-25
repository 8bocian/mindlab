#!/usr/bin/env bash
# Runs ON THE VM. Installs MindLab as systemd services, so the server, the Cloudflare tunnel and the
# idle watchdog start by themselves every time the VM boots - also after "Restore" from hibernation
# in the Hyperstack panel, without lab.py. Then (re)starts them now. Safe to run again.
set -euo pipefail
LAB="$HOME/mindlab"
ME="$(id -un)"

# ---- launchers (generated here so this script is self-contained) ----
cat > "$LAB/run_server.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
LAB="$HOME/mindlab"
cd "$LAB/server"
set -a; source "$LAB/server.env"; [ -f "$LAB/runtime.env" ] && source "$LAB/runtime.env"; set +a
export HEARTBEAT_FILE="$LAB/heartbeat"
exec "$LAB/.venv/bin/python" -m uvicorn app:app --host 127.0.0.1 --port 8000
EOF
cat > "$LAB/run_tunnel.sh" <<'EOF'
#!/usr/bin/env bash
# TUNNEL_TOKEN is read from the environment by cloudflared (keeps it out of `ps` output)
set -euo pipefail
set -a; source "$HOME/mindlab/tunnel.env"; set +a
exec "$HOME/.local/bin/cloudflared" tunnel --no-autoupdate run
EOF
chmod +x "$LAB/run_server.sh" "$LAB/run_tunnel.sh"

# ---- stop processes started the old way (nohup) and the temporary setup-time watchdog ----
pkill -f "uvicorn app:app" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true
pkill -f "mindlab/watchdog.py" 2>/dev/null || true
sleep 1

want_tunnel=0; [ -f "$LAB/tunnel.env" ] && want_tunnel=1
want_watchdog=0; [ -f "$LAB/watchdog.env" ] && want_watchdog=1

if sudo -n true 2>/dev/null; then
  unit(){ # name description command log restart
    sudo tee "/etc/systemd/system/$1.service" >/dev/null <<EOF
[Unit]
Description=$2
After=network-online.target
Wants=network-online.target

[Service]
User=$ME
Environment=HOME=$HOME
ExecStart=$3
Restart=$5
RestartSec=5
StandardOutput=append:$4
StandardError=append:$4

[Install]
WantedBy=multi-user.target
EOF
  }
  unit mindlab          "MindLab server"            "$LAB/run_server.sh"               "$LAB/server.log"   always
  unit mindlab-tunnel   "MindLab Cloudflare tunnel" "$LAB/run_tunnel.sh"               "$LAB/tunnel.log"   always
  unit mindlab-watchdog "MindLab idle watchdog"     "/usr/bin/python3 $LAB/watchdog.py" "$LAB/watchdog.log" on-failure
  sudo systemctl daemon-reload
  # remove the old cron fallback if a previous run installed it
  (crontab -l 2>/dev/null | grep -v "mindlab/start_server.sh") | crontab - 2>/dev/null || true

  : > "$LAB/server.log"
  sudo systemctl enable --quiet mindlab
  sudo systemctl restart mindlab
  echo "serwer: usługa systemd 'mindlab' (log: $LAB/server.log)"
  if [ $want_tunnel = 1 ]; then
    sudo systemctl enable --quiet mindlab-tunnel; sudo systemctl restart mindlab-tunnel
    echo "tunel Cloudflare: usługa 'mindlab-tunnel' (log: $LAB/tunnel.log)"
  else
    sudo systemctl disable --now --quiet mindlab-tunnel 2>/dev/null || true
  fi
  if [ $want_watchdog = 1 ]; then
    sudo systemctl enable --quiet mindlab-watchdog; sudo systemctl restart mindlab-watchdog
    echo "strażnik bezczynności: usługa 'mindlab-watchdog' (log: $LAB/watchdog.log)"
  else
    sudo systemctl disable --now --quiet mindlab-watchdog 2>/dev/null || true
  fi
  echo "usługi startują same przy każdym uruchomieniu VM"
else
  # no passwordless sudo: plain background processes + start again on every boot via cron
  (crontab -l 2>/dev/null | grep -v "mindlab/start_server.sh"; echo "@reboot bash $LAB/start_server.sh >> $LAB/boot.log 2>&1") | crontab -
  nohup "$LAB/run_server.sh" > "$LAB/server.log" 2>&1 &
  echo "serwer uruchomiony (log: $LAB/server.log)"
  if [ $want_tunnel = 1 ]; then nohup "$LAB/run_tunnel.sh" >> "$LAB/tunnel.log" 2>&1 & echo "tunel Cloudflare uruchomiony"; fi
  if [ $want_watchdog = 1 ]; then nohup python3 "$LAB/watchdog.py" >> "$LAB/watchdog.log" 2>&1 & echo "strażnik uruchomiony"; fi
  echo "brak sudo bez hasła: autostart przez cron @reboot"
fi
