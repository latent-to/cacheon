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
# Persistent volume is mounted at /workspace on Lium/RunPod.
# Model weights are cached there so they survive pod restarts.

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/latent-to/cacheon.git"   # TODO: replace with real URL
WORKSPACE="/root"
REPO_DIR="$WORKSPACE/cacheon"
export HF_HOME="$WORKSPACE/.cache/huggingface"    # weights persist across restarts
SKIP_MODEL=false

for arg in "$@"; do
  [[ "$arg" == "--no-model" ]] && SKIP_MODEL=true
done

# ── system deps ───────────────────────────────────────────────────────────────
echo "=== System dependencies ==="
apt-get update
apt-get install -y git curl wget ca-certificates

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

VENV_DIR="$WORKSPACE/venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$REPO_DIR/inference_engine/requirements.txt"

# ── model weights ─────────────────────────────────────────────────────────────
if [[ "$SKIP_MODEL" == false ]]; then
  echo ""
  echo "=== Model weights (HF_HOME=$HF_HOME) ==="
  mkdir -p "$HF_HOME"

  python3 - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"  {model_name}...")
AutoTokenizer.from_pretrained(model_name)
AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
print(f"  done.")

EOF
fi

# ── smoke test ────────────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test ==="
cd "$REPO_DIR"
source "$WORKSPACE/venv/bin/activate"
python scripts/smoke_test.py --device cuda --max-new-tokens 16

echo ""
echo "=== Setup complete ==="
