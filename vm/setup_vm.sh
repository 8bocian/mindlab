#!/usr/bin/env bash
# Runs ON THE VM. Idempotent: safe to run again after hibernation/restore.
# Installs Python env + PyTorch matching the driver, downloads the model.
set -euo pipefail
LAB="$HOME/mindlab"
cd "$LAB"
source "$LAB/server.env"           # MODEL, HF_TOKEN, ... (written by lab.py)
log(){ echo -e "\033[1;34m[setup]\033[0m $*"; }

log "GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "Brak GPU / sterownika NVIDIA"; exit 1; }

# ---- uv (fast Python package manager) ----
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  log "Instaluję uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# ---- venv ----
if [ ! -x "$LAB/.venv/bin/python" ]; then
  log "Tworzę środowisko Python 3.12"
  uv venv --python 3.12 "$LAB/.venv"
fi
source "$LAB/.venv/bin/activate"

# ---- PyTorch: try newest CUDA build first, fall back to older ones ----
torch_ok(){ python - <<'EOF' 2>/dev/null
import torch, sys
ok = torch.cuda.is_available()
if ok:
    torch.zeros(1, device="cuda") * 2
    print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
sys.exit(0 if ok else 1)
EOF
}
if ! torch_ok; then
  for idx in "" cu130 cu128 cu126; do
    if [ -z "$idx" ]; then log "Instaluję PyTorch (PyPI)"; uv pip install -q --upgrade torch || continue
    else log "Próbuję PyTorch z indeksu $idx"; uv pip install -q --upgrade --reinstall torch --index-url "https://download.pytorch.org/whl/$idx" || continue; fi
    if torch_ok; then break; fi
  done
  torch_ok || { echo "PyTorch nie widzi GPU - sprawdź sterownik (nvidia-smi) i obraz systemu"; exit 1; }
fi

log "Instaluję zależności serwera"
uv pip install -q -r "$LAB/server/requirements.txt"
if [ -n "${EXTRA_PIP:-}" ]; then log "Dodatkowe pakiety: $EXTRA_PIP"; uv pip install -q $EXTRA_PIP; fi

# ---- where to keep model weights: root disk survives hibernation, ephemeral does not ----
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
if [ -z "${HF_HOME:-}" ]; then
  if [ "$FREE_GB" -lt 45 ] && [ -d /ephemeral ] && [ -w /ephemeral ]; then
    HF_HOME=/ephemeral/hf
    log "Mało miejsca na dysku systemowym (${FREE_GB} GB) - wagi idą na /ephemeral (po hibernacji trzeba je pobrać ponownie)"
  else
    HF_HOME="$HOME/.cache/huggingface"
  fi
fi
mkdir -p "$HF_HOME"
echo "export HF_HOME=$HF_HOME" > "$LAB/runtime.env"
export HF_HOME

# ---- cloudflared (only when a Cloudflare Tunnel token was configured) ----
if [ -f "$LAB/tunnel.env" ] && ! command -v cloudflared >/dev/null; then
  case "$(uname -m)" in aarch64|arm64) CF_ARCH=arm64 ;; *) CF_ARCH=amd64 ;; esac
  log "Instaluję cloudflared ($CF_ARCH)"
  mkdir -p "$HOME/.local/bin"
  curl -fsSL -o "$HOME/.local/bin/cloudflared" \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$CF_ARCH"
  chmod +x "$HOME/.local/bin/cloudflared"
fi

log "Pobieram model $MODEL do $HF_HOME (przy pierwszym razie może to chwilę potrwać)"
if [ -d "$MODEL" ]; then
  log "MODEL to lokalny katalog - pomijam pobieranie"
else
  HF_TOKEN="${HF_TOKEN:-}" hf download "$MODEL" --exclude "*.gguf" --exclude "original/*" >/dev/null
fi
log "Gotowe ✔"
