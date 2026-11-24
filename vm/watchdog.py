#!/usr/bin/env python3
"""
Runs ON THE VM (stdlib only). Hibernates or deletes this VM via the Hyperstack API when:
  * nobody used the dashboard / setup for IDLE_MINUTES, or
  * the VM has been up for more than MAX_HOURS.
This protects you if your laptop sleeps or loses connection while the VM is running.
Reads ~/mindlab/watchdog.env (chmod 600): HYPERSTACK_API_KEY, VM_ID, ACTION, IDLE_MINUTES, MAX_HOURS
"""
import json
import os
import pathlib
import time
import urllib.request

LAB = pathlib.Path.home() / "mindlab"
BASE = "https://infrahub-api.nexgencloud.com/v1"


def load_env(p):
    env = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def call(method, path, key):
    req = urllib.request.Request(BASE + path, method=method, headers={"api_key": key, "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")


def last_activity():
    ts = []
    for name in ("heartbeat", "setup.log"):
        f = LAB / name
        if f.exists():
            ts.append(f.stat().st_mtime)
    return max(ts) if ts else time.time()


def main():
    env = load_env(LAB / "watchdog.env")
    key, vm_id = env["HYPERSTACK_API_KEY"], env["VM_ID"]
    action = env.get("ACTION", "hibernate")
    idle_s = float(env.get("IDLE_MINUTES", "30")) * 60
    max_s = float(env.get("MAX_HOURS", "8")) * 3600
    started = float(env.get("STARTED_AT", time.time()))
    print(f"watchdog: vm={vm_id} action={action} idle={idle_s/60:.0f}min max={max_s/3600:.1f}h", flush=True)
    while True:
        now = time.time()
        idle, up = now - last_activity(), now - started
        reason = None
        if idle > idle_s:
            reason = f"bezczynność {idle/60:.0f} min"
        elif up > max_s:
            reason = f"przekroczono limit {max_s/3600:.1f} h"
        if reason:
            print(f"{time.ctime()}: {reason} -> {action}", flush=True)
            for attempt in range(5):
                try:
                    if action == "delete":
                        call("DELETE", f"/core/virtual-machines/{vm_id}", key)
                    else:
                        call("GET", f"/core/virtual-machines/{vm_id}/hibernate", key)
                    print("wysłano żądanie, kończę", flush=True)
                    return
                except Exception as e:  # noqa: BLE001
                    print(f"błąd API ({e}); próba {attempt + 1}/5", flush=True)
                    time.sleep(30)
            # last resort: if hibernation keeps failing, delete (stopped VMs are billed in full)
            if action != "delete":
                try:
                    call("DELETE", f"/core/virtual-machines/{vm_id}", key)
                    print("hibernacja nieudana -> usunięto VM", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"nie udało się nawet usunąć VM: {e}", flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
