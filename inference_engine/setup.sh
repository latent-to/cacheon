#!/usr/bin/env bash
# inference_engine/setup.sh
#
# Paste this whole file as the "on-start script" in Lium/RunPod, OR run it
# manually after SSHing in. Safe to re-run — it pulls instead of re-cloning.
#
# Usage:
#   bash setup.sh                  # full setup + download models
#   bash setup.sh --no-model       # skip model download (deps only)
#
# Storage layout (Lium):
#   /mnt   — persistent S3-backed volume (slow, survives pod deletion)
#   /root  — local NVMe (fast, survives restart but wiped on pod delete)
#
# Strategy: download model weights to /mnt once so they survive across pods,
# then copy to /root on each pod startup for fast inference-time reads.

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/latent-to/cacheon.git"
LOCAL="/root"                                          # fast local NVMe (ephemeral on delete)
VOLUME="/mnt"                                          # persistent S3 volume (slow reads)
REPO_DIR="$LOCAL/cacheon"
VENV_DIR="$LOCAL/venv"
LOCAL_HF="$LOCAL/.cache/huggingface"
VOLUME_HF="$VOLUME/.cache/huggingface"
export HF_HOME="$LOCAL_HF"                             # inference always reads from local
SKIP_MODEL=false

for arg in "$@"; do
  [[ "$arg" == "--no-model" ]] && SKIP_MODEL=true
done

# ── system deps ───────────────────────────────────────────────────────────────
echo "=== System dependencies ==="
apt-get update
apt-get install -y --no-install-recommends git curl wget ca-certificates rsync tmux

# ── clone or pull repo ────────────────────────────────────────────────────────
echo ""
echo "=== Repo ==="
if [ -z "${GITHUB_PAT:-}" ]; then
  echo "ERROR: GITHUB_PAT environment variable is not set."
  exit 1
fi

if [ -d "$REPO_DIR/.git" ]; then
  echo "Repo exists — pulling latest..."
  git -C "$REPO_DIR" pull
else
  echo "Cloning $REPO_URL → $REPO_DIR"
  # Insert GITHUB_PAT into the clone URL if cloning from github.com
  if [[ "$REPO_URL" == https://github.com/* ]]; then
    CLONE_URL=$(echo "$REPO_URL" | sed -E "s#https://github.com/#https://$GITHUB_PAT@github.com/#")
    git clone "$CLONE_URL" "$REPO_DIR"
  else
    git clone "$REPO_URL" "$REPO_DIR"
  fi
fi

# ── python venv ───────────────────────────────────────────────────────────────
echo ""
echo "=== Python dependencies ==="
echo "Python: $(python3 --version)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not found')"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$REPO_DIR/inference_engine/requirements.txt"

# ── model weights ─────────────────────────────────────────────────────────────
if [[ "$SKIP_MODEL" == false ]]; then
  MODEL_HUB_DIR="hub/models--Qwen--Qwen2.5-7B-Instruct"

  if [ -d "$LOCAL_HF/$MODEL_HUB_DIR/snapshots" ]; then
    echo ""
    echo "=== Model weights already on local NVMe — skipping ==="
    echo "  $LOCAL_HF/$MODEL_HUB_DIR"

  elif [ -d "$VOLUME_HF/$MODEL_HUB_DIR/snapshots" ]; then
    echo ""
    echo "=== Copying model from volume → local NVMe ==="
    echo "  $VOLUME_HF → $LOCAL_HF"
    echo "  (s3fs → NVMe, may take 20–50 min for ~15 GB)"
    mkdir -p "$LOCAL_HF"
    rsync -ah --info=progress2 "$VOLUME_HF/" "$LOCAL_HF/"
    echo "  copy done."

  else
    echo ""
    echo "=== Downloading model weights ==="

    # Download to persistent volume first
    echo "  Downloading to volume ($VOLUME_HF) ..."
    mkdir -p "$VOLUME_HF"
    HF_HOME="$VOLUME_HF" python3 - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"  {model_name}...")
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
print(f"  saved to volume.")
EOF

    # Copy to local NVMe for fast reads
    echo "  Copying volume → local NVMe ..."
    mkdir -p "$LOCAL_HF"
    rsync -ah --info=progress2 "$VOLUME_HF/" "$LOCAL_HF/"
    echo "  copy done."
  fi

  echo "  HF_HOME=$HF_HOME (local NVMe)"
fi

# ── smoke test ────────────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test ==="
cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"
python scripts/smoke_test.py --device cuda --max-new-tokens 16

echo ""
echo "=== Setup complete ==="
