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
# CUDA build matching the driver's major version (newer driver = newer CUDA allowed)
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1 | tr -dc '0-9')
DRIVER_MAJOR=${DRIVER_MAJOR:-0}
if [ "$DRIVER_MAJOR" -ge 580 ]; then TORCH_IDX="cu130 cu128 cu126"
elif [ "$DRIVER_MAJOR" -ge 570 ]; then TORCH_IDX="cu128 cu126"
else TORCH_IDX="cu126"; fi
install_torch(){
  for idx in $TORCH_IDX; do
    log "Instaluję PyTorch z indeksu $idx (sterownik $DRIVER_MAJOR)"
    uv pip install -q --upgrade --reinstall torch --index-url "https://download.pytorch.org/whl/$idx" || continue
    if torch_ok; then return 0; fi
  done
  echo "PyTorch nie widzi GPU - sprawdź sterownik (nvidia-smi) i obraz systemu"; exit 1
}
torch_ok || install_torch

log "Instaluję zależności serwera"
uv pip install -q -r "$LAB/server/requirements.txt"
if [ -n "${EXTRA_PIP:-}" ]; then log "Dodatkowe pakiety: $EXTRA_PIP"; uv pip install -q $EXTRA_PIP; fi
# dependencies may have replaced PyTorch with a build that does not match the driver
if ! torch_ok; then log "Zależności podmieniły PyTorcha - instaluję ponownie"; install_torch; fi

uv cache clean -q || true
rm -rf "$HOME/.cache/pip"

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
# ---- fast kernels for models with linear/hybrid attention (Qwen3.5, Qwen3.6, Qwen3-Next…) ----
# without them transformers falls back to slow reference PyTorch code
NEEDS_FLA=$(python - <<'EOF' 2>/dev/null
import json, os, glob
from huggingface_hub import hf_hub_download
m = os.environ["MODEL"]
try:
    p = os.path.join(m, "config.json") if os.path.isdir(m) else hf_hub_download(m, "config.json")
    c = json.load(open(p))
    c = {**c, **c.get("text_config", {})}
    linear = "linear_attention" in json.dumps(c.get("layer_types", [])) or any(
        k in str(c.get("model_type", "")) for k in ("qwen3_5", "qwen3_6", "qwen3_next"))
    print(1 if linear else 0)
except Exception:
    print(0)
EOF
)
if [ "$NEEDS_FLA" = "1" ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
  python -c "import fla" 2>/dev/null || { log "Instaluję flash-linear-attention (szybka atencja liniowa)"; uv pip install -q flash-linear-attention || log "UWAGA: flash-linear-attention się nie zainstalował - model będzie wolniejszy"; }
  python -c "import causal_conv1d" 2>/dev/null || { log "Kompiluję causal-conv1d (jednorazowo, ok. 5-10 min)"; uv pip install -q causal-conv1d --no-build-isolation || log "UWAGA: causal-conv1d się nie zainstalował - model będzie wolniejszy"; }
fi
log "Gotowe ✔"
