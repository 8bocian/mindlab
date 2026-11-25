#!/usr/bin/env python3
"""
MindLab – wynajmij GPU na Hyperstacku, uruchom dashboard, posprzątaj.

  python lab.py check       # sprawdź klucz API, regiony, klucze SSH, dostępne GPU i obrazy
  python lab.py up          # wynajmij/wybudź VM, zainstaluj, uruchom, otwórz dashboard
  python lab.py up --keep   # jak wyżej, ale po Ctrl+C NIE hibernuj/usuwaj
  python lab.py start       # tryb publiczny: zainstaluj/zaktualizuj VM (serwer + tunel jako usługi) i zakończ;
                            #   potem włączasz/wyłączasz VM w panelu Hyperstack (Restore / Hibernate),
                            #   serwer i tunel startują same; lab.py tylko przy zmianie kodu/konfiguracji
  python lab.py status      # stan VM i szacowany koszt
  python lab.py hibernate   # zahibernuj (płacisz tylko za dysk)
  python lab.py down        # usuń VM całkowicie (nic nie płacisz)
  python lab.py logs        # ostatnie linie logu serwera z VM
  python lab.py ssh         # wypisz komendę SSH do VM

Wymaga: Python 3.9+, `pip install requests paramiko`, klient `ssh` w systemie.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shlex
import socket
import subprocess
import sys
import time
import webbrowser

try:
    # use the OS certificate store (works behind antivirus/corporate HTTPS scanning, e.g. Norton)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
try:
    import requests
except ImportError:
    sys.exit("Brak pakietu requests:  pip install requests paramiko")

ROOT = pathlib.Path(__file__).resolve().parent
BASE = os.environ.get("HYPERSTACK_API_BASE", "https://infrahub-api.nexgencloud.com/v1")
SERVER_ENV_KEYS = ["MODEL", "DTYPE", "TRUST_REMOTE_CODE", "HF_TOKEN", "LAYERS_PATH", "NORM_PATH",
                   "DEFAULT_CONCEPTS", "EXTRA_PIP", "ACCESS_PASSWORD", "ALLOWED_ORIGINS", "MAX_NEW_TOKENS_CAP"]


# ----------------------------------------------------------------------------- config
def load_cfg(path: pathlib.Path) -> dict:
    if not path.exists():
        sys.exit(f"Brak {path.name}. Skopiuj lab.env.example do lab.env i uzupełnij.")
    cfg = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #", 1)[0].strip().strip('"').strip("'")
        cfg[k.strip()] = v
    defaults = dict(ENVIRONMENT="", KEYPAIR_NAME="mindlab-key", SSH_PRIVATE_KEY="~/.ssh/id_ed25519",
                    SSH_PUBLIC_KEY="~/.ssh/id_ed25519.pub", SSH_USER="ubuntu", SSH_ALLOW_CIDR="0.0.0.0/0",
                    GPU_MATCH="RTXPRO6000", GPU_COUNT="1", FLAVOR_NAME="", IMAGE_NAME="", IMAGE_PREFER="24.04",
                    VM_NAME="mindlab", PRICE_PER_HOUR="0", END_ACTION="hibernate", WATCHDOG="1",
                    IDLE_MINUTES="30", MAX_HOURS="8", MODEL="Qwen/Qwen3-8B", DTYPE="auto",
                    TRUST_REMOTE_CODE="0", LOCAL_PORT="8000", CLOUDFLARE_TUNNEL_TOKEN="", ACCESS_PASSWORD="",
                    ALLOWED_ORIGINS="*", PUBLIC_URL="", MAX_NEW_TOKENS_CAP="400")
    for k, v in defaults.items():
        if not cfg.get(k):
            cfg[k] = v
    if not cfg.get("HYPERSTACK_API_KEY") or "wklej" in cfg["HYPERSTACK_API_KEY"]:
        sys.exit("Uzupełnij HYPERSTACK_API_KEY w lab.env")
    for k in ("SSH_PRIVATE_KEY", "SSH_PUBLIC_KEY"):
        cfg[k] = str(pathlib.Path(cfg[k]).expanduser())
    return cfg


def say(msg, color="34"):
    print(f"\033[1;{color}m[mindlab]\033[0m {msg}", flush=True)


def warn(msg):
    say(msg, "33")


# ----------------------------------------------------------------------------- API
class HSError(RuntimeError):
    pass


class Hyperstack:
    def __init__(self, key: str):
        self.s = requests.Session()
        self.s.headers.update({"api_key": key, "accept": "application/json"})

    def req(self, method, path, **kw):
        for attempt in range(4):   # transient network errors happen; retry with backoff
            try:
                r = self.s.request(method, BASE + path, timeout=90, **kw)
                break
            except requests.RequestException as e:
                if attempt == 3:
                    raise HSError(f"{method} {path}: brak połączenia ({e})")
                time.sleep(3 * (attempt + 1))
        try:
            j = r.json()
        except ValueError:
            j = {"message": r.text[:300]}
        if r.status_code >= 400 or (isinstance(j, dict) and j.get("status") is False):
            msg = (j.get("message") or j.get("detail") or r.text[:300]) if isinstance(j, dict) else j
            hint = " (sprawdź HYPERSTACK_API_KEY w lab.env)" if r.status_code in (401, 403) else ""
            raise HSError(f"{method} {path} -> HTTP {r.status_code}: {msg}{hint}")
        return j

    # resources
    def environments(self):
        return self.req("GET", "/core/environments", params={"page_size": 100}).get("environments", [])

    def keypairs(self):
        return self.req("GET", "/core/keypairs", params={"page_size": 100}).get("keypairs", [])

    def import_keypair(self, env, name, pub):
        return self.req("POST", "/core/keypairs", json={"environment_name": env, "name": name, "public_key": pub})

    def flavors(self, region):
        out = []
        for group in self.req("GET", "/core/flavors", params={"region": region}).get("data", []):
            out.extend(group.get("flavors", []) or [])
        return out

    def images(self, region):
        out = []
        for group in self.req("GET", "/core/images", params={"region": region}).get("images", []):
            out.extend(group.get("images", []) or [])
        return out

    def vms(self, search=None):
        p = {"page_size": 100}
        if search:
            p["search"] = search
        return self.req("GET", "/core/virtual-machines", params=p).get("instances", [])

    def vm(self, vm_id):
        return self.req("GET", f"/core/virtual-machines/{vm_id}").get("instance", {})

    def create_vm(self, payload):
        return self.req("POST", "/core/virtual-machines", json=payload).get("instances", [])

    def delete_vm(self, vm_id):
        return self.req("DELETE", f"/core/virtual-machines/{vm_id}")

    def hibernate(self, vm_id):
        return self.req("GET", f"/core/virtual-machines/{vm_id}/hibernate")

    def restore(self, vm_id):
        return self.req("GET", f"/core/virtual-machines/{vm_id}/hibernate-restore")

    def start(self, vm_id):
        return self.req("GET", f"/core/virtual-machines/{vm_id}/start")


norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())  # noqa: E731
status_of = lambda vm: (vm.get("status") or "").upper()  # noqa: E731


def find_env(hs, cfg):
    envs = hs.environments()
    for e in envs:
        if e.get("name") == cfg["ENVIRONMENT"]:
            return e
    names = ", ".join(e.get("name", "?") for e in envs) or "(brak - utwórz environment w panelu Hyperstack)"
    sys.exit(f"Nie znaleziono environment '{cfg['ENVIRONMENT']}'. Dostępne: {names}")


def find_vm(hs, cfg):
    for v in hs.vms(cfg["VM_NAME"]):
        if v.get("name") == cfg["VM_NAME"] and status_of(v) not in ("DELETED", "DELETING"):
            return v
    return None


def flavor_matches(f, cfg):
    """GPU name contains GPU_MATCH, right GPU count; spot (interruptible) only if GPU_MATCH asks for it."""
    m = norm(cfg["GPU_MATCH"])
    want_spot = "spot" in m
    m = m.replace("spot", "")
    is_spot = "spot" in norm(f.get("name")) + norm(f.get("gpu"))
    if is_spot != want_spot:
        return False
    return (m in norm(f.get("gpu")) or m in norm(f.get("name"))) and int(f.get("gpu_count") or 0) == int(cfg["GPU_COUNT"])


def pick_flavor(hs, cfg, region):
    fl = hs.flavors(region)
    if cfg["FLAVOR_NAME"]:
        for f in fl:
            if f.get("name") == cfg["FLAVOR_NAME"]:
                return f, fl
        sys.exit(f"Flavor {cfg['FLAVOR_NAME']} nie istnieje w regionie {region}. Uruchom: python lab.py check")
    m = norm(cfg["GPU_MATCH"])
    cands = [f for f in fl if flavor_matches(f, cfg)]
    in_stock = [f for f in cands if f.get("stock_available")]
    if in_stock:
        return sorted(in_stock, key=lambda f: (f.get("ram") or 0))[0], fl
    if cands:
        sys.exit(f"GPU {cfg['GPU_MATCH']} x{cfg['GPU_COUNT']} istnieje w {region}, ale chwilowo brak dostępności. "
                 "Spróbuj później, inny region (ENVIRONMENT) albo inne GPU (GPU_MATCH).")
    sys.exit(f"Brak GPU pasującego do '{cfg['GPU_MATCH']}' x{cfg['GPU_COUNT']} w {region}. Uruchom: python lab.py check")


def pick_image(hs, cfg, region):
    ims = hs.images(region)
    if cfg["IMAGE_NAME"]:
        for i in ims:
            if i.get("name") == cfg["IMAGE_NAME"]:
                return i, ims
        sys.exit(f"Obraz {cfg['IMAGE_NAME']} nie istnieje w {region}. Uruchom: python lab.py check")
    ubuntu = [i for i in ims if "ubuntu" in (i.get("name") or "").lower()]
    cuda = [i for i in ubuntu if "cuda" in (i.get("name") or "").lower()]
    pool = cuda or ubuntu
    if not pool:
        sys.exit("Nie znalazłem obrazu Ubuntu. Ustaw IMAGE_NAME (lista: python lab.py check)")
    pref = [i for i in pool if cfg["IMAGE_PREFER"] in (i.get("name") or "")]
    pool = pref or pool
    # prefer images without Docker in the name, then newest-looking name
    pool.sort(key=lambda i: ("docker" in (i.get("name") or "").lower(), [-int(x) for x in re.findall(r"\d+", i.get("name") or "")[:4]]))
    return pool[0], ims


# ----------------------------------------------------------------------------- SSH
def ssh_connect(cfg, ip, timeout_s=600):
    try:
        import paramiko
    except ImportError:
        sys.exit("Brak pakietu paramiko:  pip install paramiko")
    import logging
    logging.getLogger("paramiko").setLevel(logging.CRITICAL)   # hide noisy tracebacks while sshd boots
    t0 = time.time()
    last = None
    shown = False
    shown_pw = False
    auth_fail = 0
    while time.time() - t0 < timeout_s:
        # 1) wait until the SSH server actually answers with its banner (VM may be ACTIVE but still booting)
        try:
            with socket.create_connection((ip, 22), timeout=10) as sock:
                sock.settimeout(10)
                if not sock.recv(64).startswith(b"SSH-"):
                    raise OSError("no SSH banner yet")
        except OSError as e:
            last = e
            if not shown:
                say("system na VM jeszcze się uruchamia, czekam na SSH (zwykle 1-3 min)…")
                shown = True
            time.sleep(10)
            continue
        # 2) real login
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(ip, username=cfg["SSH_USER"], key_filename=cfg["SSH_PRIVATE_KEY"], timeout=15,
                      banner_timeout=30, auth_timeout=30)
            c.get_transport().set_keepalive(30)
            return c
        except paramiko.PasswordRequiredException:
            if not shown_pw:
                warn(f"Klucz {cfg['SSH_PRIVATE_KEY']} ma hasło (passphrase), a lab.py nie umie go podać.\n"
                     f"   W drugim oknie usuń hasło z klucza:  ssh-keygen -p -f {cfg['SSH_PRIVATE_KEY']}\n"
                     "   (podaj stare hasło, a przy nowym wciśnij Enter dwa razy). Skrypt sam pójdzie dalej.")
                shown_pw = True
            last = "klucz ma hasło"
            time.sleep(10)
        except paramiko.AuthenticationException as e:
            auth_fail += 1
            last = e
            if auth_fail == 3:
                warn("SSH odrzuca klucz. Sprawdź SSH_PRIVATE_KEY/SSH_USER w lab.env i czy KEYPAIR_NAME w Hyperstacku "
                     "to ten sam klucz (.pub). Próbuję dalej…")
            time.sleep(10)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(10)
    sys.exit(f"Nie mogę połączyć się przez SSH z {ip}: {last}")


def run(ssh, cmd, show=True):
    chan = ssh.get_transport().open_session()
    chan.get_pty()
    chan.exec_command(cmd)
    buf = b""
    while True:
        if chan.recv_ready():
            data = chan.recv(65536)
            buf += data
            if show:
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()
        elif chan.exit_status_ready():
            while chan.recv_ready():
                data = chan.recv(65536)
                buf += data
                if show:
                    sys.stdout.write(data.decode("utf-8", errors="replace"))
            break
        else:
            time.sleep(0.1)
    return chan.recv_exit_status(), buf.decode("utf-8", errors="replace")


def upload(ssh, cfg, vm_id):
    sftp = ssh.open_sftp()
    home = run(ssh, "echo $HOME", show=False)[1].strip()
    lab = f"{home}/mindlab"

    def mkdir(p):
        try:
            sftp.mkdir(p)
        except OSError:
            pass

    mkdir(lab)
    mkdir(f"{lab}/server")
    mkdir(f"{lab}/server/static")
    for rel in ["server/app.py", "server/engine.py", "server/requirements.txt", "server/static/index.html",
                "server/static/config.js"]:
        sftp.put(str(ROOT / rel), f"{lab}/{rel}")
    for name in ["setup_vm.sh", "start_server.sh", "watchdog.py"]:
        sftp.put(str(ROOT / "vm" / name), f"{lab}/{name}")
    env_lines = [f"{k}={shlex.quote(cfg.get(k, ''))}" for k in SERVER_ENV_KEYS]
    with sftp.open(f"{lab}/server.env", "w") as f:
        f.write("\n".join(env_lines) + "\n")
    sftp.chmod(f"{lab}/server.env", 0o600)
    tn = f"{lab}/tunnel.env"
    if cfg["CLOUDFLARE_TUNNEL_TOKEN"]:
        with sftp.open(tn, "w") as f:
            f.write(f"TUNNEL_TOKEN={shlex.quote(cfg['CLOUDFLARE_TUNNEL_TOKEN'])}\n")
        sftp.chmod(tn, 0o600)
    else:
        try:
            sftp.remove(tn)
        except OSError:
            pass
    wd = f"{lab}/watchdog.env"
    if cfg["WATCHDOG"] == "1" and cfg["END_ACTION"] in ("hibernate", "delete"):
        with sftp.open(wd, "w") as f:
            f.write(f"HYPERSTACK_API_KEY={cfg['HYPERSTACK_API_KEY']}\nVM_ID={vm_id}\nACTION={cfg['END_ACTION']}\n"
                    f"IDLE_MINUTES={cfg['IDLE_MINUTES']}\nMAX_HOURS={cfg['MAX_HOURS']}\nSTARTED_AT={time.time():.0f}\n")
        sftp.chmod(wd, 0o600)
    else:
        try:
            sftp.remove(wd)
        except OSError:
            pass
    sftp.close()
    return lab


def free_port(start):
    p = start
    while p < start + 50:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                p += 1
    sys.exit("Brak wolnego portu lokalnego")


def ssh_base(cfg, ip):
    return ["ssh", "-i", cfg["SSH_PRIVATE_KEY"], "-o", "StrictHostKeyChecking=no", "-o", f"UserKnownHostsFile={'NUL' if os.name == 'nt' else '/dev/null'}",
            "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30", f"{cfg['SSH_USER']}@{ip}"]


# ----------------------------------------------------------------------------- lifecycle
def wait_active(hs, vm_id, timeout_s=1800):
    t0, last, asked_ip, active_since = time.time(), None, False, 0.0
    while time.time() - t0 < timeout_s:
        vm = hs.vm(vm_id)
        st = status_of(vm)
        if st != last:
            say(f"stan VM: {st or '?'}")
            last = st
        if st == "ACTIVE" and vm.get("floating_ip"):
            return vm
        if st == "ACTIVE" and not active_since:
            active_since = time.time()
        if st == "ACTIVE" and not asked_ip and time.time() - active_since > 60:
            # restoring from hibernation releases the public IP; ask for a new one
            asked_ip = True
            say("VM nie ma publicznego IP (po hibernacji) - podłączam nowy…")
            try:
                hs.req("POST", f"/core/virtual-machines/{vm_id}/attach-floatingip")
            except HSError as e:
                warn(f"nie udało się podłączyć IP ({e}); podłącz je w panelu Hyperstack")
        if st == "HIBERNATED":          # e.g. we arrived while it was still HIBERNATING
            say("VM zahibernowana - wybudzam…")
            hs.restore(vm_id)
        elif st in ("SHUTOFF", "STOPPED"):
            say("VM zatrzymana - uruchamiam…")
            hs.start(vm_id)
        if st == "ERROR":
            sys.exit("VM przeszła w stan ERROR - sprawdź panel Hyperstack (i usuń ją: python lab.py down)")
        time.sleep(10)
    sys.exit("VM nie wystartowała w 30 min - sprawdź panel Hyperstack")


def ensure_keypair(hs, cfg):
    for k in hs.keypairs():
        env = (k.get("environment") or {}).get("name")
        if k.get("name") == cfg["KEYPAIR_NAME"] and env == cfg["ENVIRONMENT"]:
            return
    pub = pathlib.Path(cfg["SSH_PUBLIC_KEY"])
    if not pub.exists():
        sys.exit(f"Klucz SSH '{cfg['KEYPAIR_NAME']}' nie istnieje w Hyperstacku, a nie ma pliku {pub}. "
                 "Wygeneruj: ssh-keygen -t ed25519")
    say(f"importuję klucz SSH {pub.name} jako '{cfg['KEYPAIR_NAME']}'")
    hs.import_keypair(cfg["ENVIRONMENT"], cfg["KEYPAIR_NAME"], pub.read_text().strip())


def finish(hs, cfg, vm_id, action):
    if action == "keep":
        warn("VM zostaje WŁĄCZONA i nalicza pełną stawkę. Pamiętaj: python lab.py hibernate / down")
        return
    try:
        if action == "delete":
            hs.delete_vm(vm_id)
            say("VM usunięta - nic już nie płacisz.", "32")
        else:
            hs.hibernate(vm_id)
            say("VM hibernuje - płacisz tylko za dysk. Następne 'python lab.py up' ją wybudzi.", "32")
    except HSError as e:
        warn(f"Nie udało się ({e}). SPRAWDŹ PANEL HYPERSTACK, żeby VM nie naliczała opłat!")


def cmd_up(hs, cfg, keep=False, detach=False):
    if detach and not cfg["CLOUDFLARE_TUNNEL_TOKEN"]:
        sys.exit("'start' wymaga CLOUDFLARE_TUNNEL_TOKEN w lab.env (instrukcja w README, sekcja 6).")
    if cfg["CLOUDFLARE_TUNNEL_TOKEN"] and len(cfg["ACCESS_PASSWORD"]) < 10:
        sys.exit("Serwer będzie publiczny (CLOUDFLARE_TUNNEL_TOKEN): ustaw ACCESS_PASSWORD w lab.env (min. 10 znaków),\n"
                 "inaczej każdy, kto zna adres, użyje Twojego GPU.")
    env = find_env(hs, cfg)
    region = env.get("region")
    vm = find_vm(hs, cfg)
    if vm:
        st = status_of(vm)
        say(f"znaleziono VM '{cfg['VM_NAME']}' (id {vm['id']}) w stanie {st}")
        if st == "HIBERNATED":
            say("wybudzam z hibernacji…")
            hs.restore(vm["id"])
        elif st in ("SHUTOFF", "STOPPED"):
            say("uruchamiam…")
            hs.start(vm["id"])
        vm_id = vm["id"]
    else:
        ensure_keypair(hs, cfg)
        flavor, _ = pick_flavor(hs, cfg, region)
        image, _ = pick_image(hs, cfg, region)
        say(f"tworzę VM: {flavor.get('name')} ({flavor.get('gpu')} x{flavor.get('gpu_count')}, "
            f"{flavor.get('ram')} GB RAM, dysk {flavor.get('disk')} GB) · obraz: {image.get('name')}")
        payload = {
            "name": cfg["VM_NAME"], "environment_name": cfg["ENVIRONMENT"], "flavor_name": flavor["name"],
            "image_name": image["name"], "key_name": cfg["KEYPAIR_NAME"], "count": 1,
            "assign_floating_ip": True, "labels": ["mindlab"],
            "security_rules": [{"direction": "ingress", "ethertype": "IPv4", "protocol": "tcp",
                                "remote_ip_prefix": cfg["SSH_ALLOW_CIDR"], "port_range_min": 22, "port_range_max": 22}],
        }
        inst = hs.create_vm(payload)
        if not inst:
            sys.exit("API nie zwróciło utworzonej VM - sprawdź panel")
        vm_id = inst[0]["id"]
    t_start = time.time()
    tunnel = None
    try:
        vm = wait_active(hs, vm_id)
        ip = vm["floating_ip"]
        say(f"VM działa: {ip}. Łączę przez SSH…")
        ssh = ssh_connect(cfg, ip)
        lab = upload(ssh, cfg, vm_id)
        if cfg["WATCHDOG"] == "1" and cfg["END_ACTION"] != "keep":
            run(ssh, f"pkill -f mindlab/watchdog.py; nohup python3 {lab}/watchdog.py > {lab}/watchdog.log 2>&1 &", show=False)
            say(f"strażnik na VM: {cfg['END_ACTION']} po {cfg['IDLE_MINUTES']} min bezczynności lub po {cfg['MAX_HOURS']} h")
        say("instaluję środowisko i pobieram model (pierwszy raz: kilka-kilkanaście minut)…")
        code, _ = run(ssh, f"bash {lab}/setup_vm.sh 2>&1 | tee {lab}/setup.log; exit ${{PIPESTATUS[0]}}")
        if code != 0:
            raise RuntimeError("instalacja na VM nie powiodła się (szczegóły powyżej)")
        run(ssh, f"bash {lab}/start_server.sh")
        if not detach:
            port = free_port(int(cfg["LOCAL_PORT"]))
            tunnel = subprocess.Popen(ssh_base(cfg, ip)[:-1] + ["-N", "-L", f"{port}:127.0.0.1:8000", ssh_base(cfg, ip)[-1]])
            url = f"http://localhost:{port}"
        say("czekam, aż model się załaduje…")
        last = ""
        while True:
            code, out = run(ssh, "curl -s 127.0.0.1:8000/api/health || true", show=False)
            try:
                health = json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
            except (ValueError, IndexError):
                health = {}
            if health.get("status") == "ready":
                break
            if health.get("status") == "error":
                run(ssh, f"tail -n 40 {lab}/server.log")
                raise RuntimeError("serwer zgłosił błąd ładowania modelu (log powyżej; dashboard też go pokaże)")
            prog = health.get("progress")
            if prog and prog != last:
                last = prog
                say(f"  {last}")
            time.sleep(5)
        if detach:
            say("GOTOWE. Serwer działa publicznie przez tunel Cloudflare.", "32")
            if cfg["PUBLIC_URL"]:
                say(f"Dashboard: {cfg['PUBLIC_URL']}", "32")
                webbrowser.open(cfg["PUBLIC_URL"])
            say("Od teraz włączasz i wyłączasz VM w panelu Hyperstack: Hibernate / Restore (NIE Stop - "
                "zatrzymana VM nalicza pełną stawkę). Serwer i tunel wstają same po każdym starcie.", "32")
            warn(f"VM nalicza pełną stawkę, dopóki jej nie zahibernujesz (panel albo: python lab.py hibernate).\n"
                 f"   Strażnik i tak ją zahibernuje po {cfg['IDLE_MINUTES']} min bez aktywności lub po {cfg['MAX_HOURS']} h.")
            return
        say(f"GOTOWE → {url}   (Ctrl+C kończy sesję i wykonuje: {'keep' if keep else cfg['END_ACTION']})", "32")
        webbrowser.open(url)
        price = float(cfg["PRICE_PER_HOUR"] or 0)
        while True:
            time.sleep(300)
            h = (time.time() - t_start) / 3600
            say(f"sesja trwa {h:.1f} h" + (f" (~${h * price:.2f})" if price else ""))
            if tunnel.poll() is not None:
                warn("tunel SSH padł - otwieram ponownie")
                tunnel = subprocess.Popen(ssh_base(cfg, ip)[:-1] + ["-N", "-L", f"{port}:127.0.0.1:8000", ssh_base(cfg, ip)[-1]])
    except KeyboardInterrupt:
        print()
        say("kończę sesję…")
        if tunnel:
            tunnel.terminate()
        finish(hs, cfg, vm_id, "keep" if keep else cfg["END_ACTION"])
    except (Exception, SystemExit) as e:  # noqa: BLE001
        if tunnel:
            tunnel.terminate()
        warn(f"Problem: {e}")
        choice = "h"
        if sys.stdin.isatty():
            try:
                choice = input("Co zrobić z VM? [h]ibernuj / [d]usuń / [z]ostaw do debugowania (domyślnie h): ").strip().lower() or "h"
            except (EOFError, KeyboardInterrupt):
                choice = "h"
        finish(hs, cfg, vm_id, {"d": "delete", "z": "keep"}.get(choice[:1], "hibernate"))
        if choice.startswith("z"):
            say(f"Debugowanie: python lab.py ssh   (logi: ~/mindlab/setup.log, ~/mindlab/server.log)")
        sys.exit(1)


def cmd_check(hs, cfg):
    say("klucz API: OK" if hs.environments() is not None else "?")
    envs = hs.environments()
    print("\nEnvironments (ENVIRONMENT):")
    for e in envs:
        print(f"  {e.get('name'):<30} region={e.get('region')}")
    env = next((e for e in envs if e.get("name") == cfg["ENVIRONMENT"]), None)
    if not env:
        warn(f"ENVIRONMENT '{cfg['ENVIRONMENT']}' nie istnieje - utwórz go w panelu lub popraw lab.env")
        return
    region = env.get("region")
    print("\nKlucze SSH (KEYPAIR_NAME):")
    for k in hs.keypairs():
        print(f"  {k.get('name'):<30} env={(k.get('environment') or {}).get('name')}")
    print(f"\nGPU w regionie {region} (FLAVOR_NAME / GPU_MATCH):")
    m = norm(cfg["GPU_MATCH"])
    for f in sorted(hs.flavors(region), key=lambda f: (str(f.get("gpu")), f.get("gpu_count") or 0)):
        if not f.get("gpu"):
            continue
        mark = "←" if flavor_matches(f, cfg) else " "
        print(f" {mark} {f.get('name'):<32} {f.get('gpu')} x{f.get('gpu_count')}  RAM {f.get('ram')} GB  dysk {f.get('disk')} GB  "
              f"{'dostępny' if f.get('stock_available') else 'brak'}")
    print(f"\nObrazy Ubuntu w {region} (IMAGE_NAME):")
    for i in hs.images(region):
        if "ubuntu" in (i.get("name") or "").lower():
            print(f"   {i.get('name')}")
    try:
        f, _ = pick_flavor(hs, cfg, region)
        i, _ = pick_image(hs, cfg, region)
        say(f"wybór automatyczny: {f.get('name')} + {i.get('name')}", "32")
    except SystemExit as e:
        warn(str(e))
    vm = find_vm(hs, cfg)
    if vm:
        say(f"istniejąca VM '{cfg['VM_NAME']}': id {vm.get('id')}, stan {status_of(vm)}, IP {vm.get('floating_ip')}")


def main():
    ap = argparse.ArgumentParser(description="MindLab na Hyperstacku")
    ap.add_argument("command", choices=["check", "up", "start", "down", "hibernate", "status", "logs", "ssh"])
    ap.add_argument("--keep", action="store_true", help="po Ctrl+C zostaw VM włączoną")
    ap.add_argument("--config", default=str(ROOT / "lab.env"))
    a = ap.parse_args()
    cfg = load_cfg(pathlib.Path(a.config))
    hs = Hyperstack(cfg["HYPERSTACK_API_KEY"])
    try:
        if a.command == "check":
            cmd_check(hs, cfg)
        elif a.command == "up":
            cmd_up(hs, cfg, keep=a.keep)
        elif a.command == "start":
            cmd_up(hs, cfg, detach=True)
        else:
            vm = find_vm(hs, cfg)
            if not vm:
                say(f"Brak VM o nazwie '{cfg['VM_NAME']}' - nic nie nalicza opłat.", "32")
                return
            if a.command == "status":
                say(f"VM '{vm['name']}' id {vm['id']}: {status_of(vm)}  IP {vm.get('floating_ip') or '-'}  "
                    f"flavor {(vm.get('flavor') or {}).get('name')}  utworzona {vm.get('created_at')}")
                if status_of(vm) in ("ACTIVE", "SHUTOFF"):
                    warn("Ten stan nalicza PEŁNĄ stawkę (także SHUTOFF!).")
            elif a.command == "down":
                finish(hs, cfg, vm["id"], "delete")
            elif a.command == "hibernate":
                finish(hs, cfg, vm["id"], "hibernate")
            elif a.command == "logs":
                ssh = ssh_connect(cfg, vm["floating_ip"], timeout_s=30)
                run(ssh, "tail -n 60 ~/mindlab/server.log; echo ---; tail -n 20 ~/mindlab/watchdog.log 2>/dev/null; "
                         "echo ---; tail -n 15 ~/mindlab/tunnel.log 2>/dev/null")
            elif a.command == "ssh":
                print(" ".join(shlex.quote(x) for x in ssh_base(cfg, vm.get("floating_ip") or "IP")))
    except HSError as e:
        sys.exit(f"Błąd API Hyperstack: {e}")


if __name__ == "__main__":
    main()