#!/usr/bin/env bash
set -euo pipefail

# AutoDL one-click bootstrap for youtube-speaker-diarization
# Usage:
#   bash autodl_quickstart.sh
# Optional env:
#   REPO_DIR=/root/autodl-tmp/youtube-speaker-diarization
#   PYTHON_BIN=python3
#   HF_TOKEN=xxxx

REPO_DIR="${REPO_DIR:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/.cache}"

cd "$REPO_DIR"

echo "[1/7] System deps (ffmpeg, git)"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y || true
  sudo apt-get install -y ffmpeg git || true
fi

echo "[2/7] Create virtualenv"
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[3/7] Pip upgrade"
pip install -U pip setuptools wheel

echo "[4/7] Set cache env"
mkdir -p "$CACHE_DIR" "$CACHE_DIR/pip" "$CACHE_DIR/hf" "$CACHE_DIR/torch" "$REPO_DIR/output"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"

# persist env helper
cat > "$REPO_DIR/.env.autodl" <<EOF
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$TORCH_HOME"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
EOF

echo "[5/7] Install pinned torch stack"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

echo "[6/7] Install project requirements"
pip install -r requirements.txt

echo "[7/7] Smoke check"
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
try:
    import yt_dlp, faster_whisper
    print('yt_dlp:', yt_dlp.version.__version__)
    print('faster_whisper: OK')
except Exception as e:
    print('dependency check warning:', e)
PY

echo
echo "✅ AutoDL bootstrap done."
echo "Next:"
echo "  1) source $VENV_DIR/bin/activate"
echo "  2) source $REPO_DIR/.env.autodl"
echo "  3) export HF_TOKEN=你的token"
echo "  4) python main.py --youtube_url 'https://www.youtube.com/watch?v=xxx' --ref_audio /path/to/ref.wav --language zh --no_separation"

# python main.py --youtube_url 'https://www.youtube.com/watch?v=Zs8jUFaqtCI' --ref_audio biao.mp3 --language zh --no_separation