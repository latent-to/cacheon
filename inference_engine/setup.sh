#!/usr/bin/env bash
# inference_engine/setup.sh
#
# GPU inference pod only — provisions the machine that runs the harness (Qwen,
# monkey-patched attention, scoring).  The Phase 5 CPU validator (chain scan,
# sandbox precheck with firejail, set_weights) is a separate host and does NOT
# use this script.
#
# Paste this whole file as the "on-start script" in Targon/RunPod, OR run it
# manually after SSHing in. Safe to re-run — it pulls instead of re-cloning.
#
# Usage:
#   bash setup.sh                  # full setup + download models
#   bash setup.sh --no-model       # skip model download (deps only)
#
# Storage layout (Targon):
#   Mount your persistent volume at /workspace (recommended). Repo, venv, and
#   Hugging Face cache all live there — one tier, survives pod restarts as long
#   as the volume is attached.

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/latent-to/cacheon.git"
BASE="/workspace"
REPO_DIR="$BASE/cacheon"
VENV_DIR="$BASE/venv"
export HF_HOME="$BASE/.cache/huggingface"
SKIP_MODEL=false

for arg in "$@"; do
  [[ "$arg" == "--no-model" ]] && SKIP_MODEL=true
done

# ── system deps ───────────────────────────────────────────────────────────────
echo "=== System dependencies ==="
apt-get update
apt-get install -y --no-install-recommends git curl wget ca-certificates tmux python3-venv

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

# Recreate the venv if it's missing OR incomplete (e.g. an empty /workspace/venv
# pre-created by the volume mount or left over from a failed earlier run).
if [ ! -x "$VENV_DIR/bin/python" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Creating venv at $VENV_DIR..."
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$REPO_DIR/inference_engine/requirements.txt"

# ── model weights ─────────────────────────────────────────────────────────────
if [[ "$SKIP_MODEL" == false ]]; then
  MODEL_HUB_DIR="hub/models--Qwen--Qwen2.5-7B-Instruct"

  if [ -d "$HF_HOME/$MODEL_HUB_DIR/snapshots" ]; then
    echo ""
    echo "=== Model weights already present — skipping download ==="
    echo "  $HF_HOME/$MODEL_HUB_DIR"
  else
    echo ""
    echo "=== Downloading model weights ==="
    echo "  Target: $HF_HOME"
    mkdir -p "$HF_HOME"
    python3 - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"  {model_name}...")
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
print("  download complete.")
EOF
  fi

  echo "  HF_HOME=$HF_HOME"
fi

# ── smoke test ────────────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test ==="
cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"
python scripts/smoke_test.py --device cuda --max-new-tokens 16

echo ""
echo "=== Setup complete ==="
