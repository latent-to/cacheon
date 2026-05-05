#!/usr/bin/env bash
# validator/setup.sh
#
# Single-machine validator host setup for Ubuntu 22.04 / 24.04 (GPU pod, e.g. Targon DinD / Lium).
# Installs Docker CE, NVIDIA Container Toolkit, Python venv, model weights, PG19 prefetch,
# and pulls the vLLM baseline image.
#
# Optional: export GITHUB_PAT for authenticated clone of private forks (never printed).
# Optional: export HF_TOKEN for huggingface-cli login (large model download).
#
# Usage:
#   bash validator/setup.sh          # full setup
#   bash validator/setup.sh --pull   # git pull + pip install only (no system installs, downloads, or Docker pulls)

set -euo pipefail

# -- config (same layout as legacy setup-cpu.sh) --
REPO_URL="https://github.com/latent-to/cacheon.git"
BASE="$HOME"
REPO_DIR="$BASE/cacheon"
VENV_DIR="$BASE/venv-cacheon"
MODEL_DIR="/workspace/models/Qwen2.5-72B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

PULL_ONLY=false
for arg in "$@"; do
  [[ "$arg" == "--pull" ]] && PULL_ONLY=true
done

if [[ "$PULL_ONLY" == true ]]; then
  echo "=== Pull-only (--pull): git pull + pip install ==="
  if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "ERROR: $REPO_DIR is not a git clone. Run full setup first (without --pull)."
    exit 1
  fi
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "ERROR: venv missing at $VENV_DIR. Run full setup first (without --pull)."
    exit 1
  fi
  git -C "$REPO_DIR" pull
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install -r "$REPO_DIR/validator/requirements-cpu.txt"
  echo "=== Pull-only run complete ==="
  exit 0
fi

# -- system deps --
echo ""
echo "=== System dependencies ==="
apt-get update -q
apt-get install -y --no-install-recommends \
  git curl ca-certificates tmux python3-venv jq gnupg

# -- Docker CE --
echo ""
echo "=== Docker CE ==="
if docker --version >/dev/null 2>&1; then
  echo "Docker already installed: $(docker --version)"
else
  echo "Installing Docker via get.docker.com..."
  curl -fsSL https://get.docker.com | sh
fi

# Ensure dockerd is running (systemctl may not work in containers)
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon not running, starting manually..."
  nohup dockerd > /var/log/dockerd.log 2>&1 &
  sleep 5
  if ! docker info >/dev/null 2>&1; then
    echo "ERROR: dockerd failed to start. Check /var/log/dockerd.log"
    exit 1
  fi
  echo "dockerd started (PID $(pgrep -x dockerd))"
fi

# -- NVIDIA Container Toolkit --
echo ""
echo "=== NVIDIA Container Toolkit ==="
if nvidia-ctk --version >/dev/null 2>&1; then
  echo "nvidia-ctk already present: $(nvidia-ctk --version | head -n 1)"
else
  echo "Installing nvidia-container-toolkit from NVIDIA repository..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  apt-get update -q
  apt-get install -y --no-install-recommends nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  # Restart dockerd to pick up the new runtime config
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active docker >/dev/null 2>&1; then
    systemctl restart docker
  else
    pkill -x dockerd && sleep 2
    nohup dockerd > /var/log/dockerd.log 2>&1 &
    sleep 5
    if ! docker info >/dev/null 2>&1; then
      echo "ERROR: dockerd failed to restart after nvidia-ctk configure. Check /var/log/dockerd.log"
      exit 1
    fi
    echo "dockerd restarted (PID $(pgrep -x dockerd))"
  fi
fi

# -- clone or update repo --
echo ""
echo "=== Repo ==="
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "Repo exists, pulling latest..."
  git -C "$REPO_DIR" pull
else
  echo "Cloning into $REPO_DIR"
  if [[ -n "${GITHUB_PAT:-}" ]] && [[ "$REPO_URL" == https://github.com/* ]]; then
    CLONE_URL="${REPO_URL/https:\/\/github.com\//https://${GITHUB_PAT}@github.com/}"
    git clone "$CLONE_URL" "$REPO_DIR"
  else
    git clone "$REPO_URL" "$REPO_DIR"
  fi
fi

# -- Python venv --
echo ""
echo "=== Python dependencies ==="
echo "Python: $(python3 --version)"
if [[ ! -x "$VENV_DIR/bin/python" ]] || [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Creating venv at $VENV_DIR..."
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
export PATH="$VENV_DIR/bin:$PATH"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$REPO_DIR/validator/requirements-cpu.txt"

# Hugging Face cache (all HF operations below use this)
export HF_HOME="/workspace/.cache/huggingface"
mkdir -p "$HF_HOME"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo ""
  echo "=== Hugging Face CLI (HF_TOKEN set) ==="
  "$VENV_DIR/bin/hf" auth login --token "$HF_TOKEN"
fi

# -- disk space (before large downloads) --
echo ""
echo "=== Disk space (/workspace) ==="
avail_kb="$(df -Pk /workspace 2>/dev/null | awk 'NR==2 {print $4}')"
if [[ -z "$avail_kb" ]] || ! [[ "$avail_kb" =~ ^[0-9]+$ ]]; then
  echo "WARNING: could not read free space on /workspace (df). Continuing."
else
  avail_gb=$((avail_kb / 1024 / 1024))
  if ((avail_gb < 300)); then
    echo "ERROR: need at least 300 GB free on the /workspace filesystem, found ${avail_gb} GB."
    exit 1
  fi
  if ((avail_gb < 500)); then
    echo "WARNING: only ${avail_gb} GB free on /workspace; recommend 500+ GB for model, dataset cache, and Docker layers."
  else
    echo "OK: ${avail_gb} GB free on /workspace."
  fi
fi

# -- model weights --
echo ""
echo "=== Model weights ($MODEL_NAME) ==="
"$VENV_DIR/bin/hf" download "$MODEL_NAME" --local-dir "$MODEL_DIR"

# -- PG19 (pinned revision, matches validator/prompts.py) --
echo ""
echo "=== PG19 dataset (pinned revision) ==="
"$VENV_DIR/bin/python" -c "from datasets import load_dataset; load_dataset('deepmind/pg19', split='train', revision='a0e01428956e39868fa36ccfa0ee236ff04e6a6b'); print('PG19 cached')"

# -- vLLM baseline image --
echo ""
echo "=== vLLM baseline Docker image ==="
docker pull vllm/vllm-openai:latest
REPO_DIGEST="$(docker image inspect vllm/vllm-openai:latest --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"
DIGEST=""
if [[ -n "$REPO_DIGEST" ]] && [[ "$REPO_DIGEST" == *"@"* ]]; then
  DIGEST="${REPO_DIGEST##*@}"
else
  echo "WARNING: could not read RepoDigests for vllm/vllm-openai:latest; set CACHEON_BASELINE_DIGEST manually."
fi
echo ""
echo "# Baseline pinning (add to .env alongside CACHEON_MODEL_VOLUME and CACHEON_GPUS)"
echo "CACHEON_BASELINE_IMAGE=vllm/vllm-openai:latest"
echo "CACHEON_BASELINE_DIGEST=${DIGEST:-sha256:REPLACE_WITH_ACTUAL_DIGEST}"

# -- verification --
echo ""
echo "=== Verification ==="
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
"$VENV_DIR/bin/python" -c "import bittensor, datasets; print('python deps ok')"
test -d "$MODEL_DIR"

echo ""
echo "=== Setup complete ==="
echo "Next: copy validator/.env.validator.example to your .env, set wallet vars, and paste CACHEON_BASELINE_DIGEST from above."
echo "Docs: https://cacheon.io/docs/validators/overview"
echo "Run the validator: https://cacheon.io/docs/architecture/validator"
