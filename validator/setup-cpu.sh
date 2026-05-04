#!/usr/bin/env bash
# validator/setup-cpu.sh
#
# Validator host setup. Provisions the machine that runs the chain scan
# loop, Docker-based miner evaluation, and set_weights.
#
# Prerequisites (see https://cacheon.io/docs/validators/overview):
#   export GITHUB_PAT=<your personal access token>
#
# Usage:
#   bash validator/setup-cpu.sh          # full setup
#   bash validator/setup-cpu.sh --pull   # re-run on an existing install (git pull only)

set -euo pipefail

# -- config --
REPO_URL="https://github.com/latent-to/cacheon.git"
BASE="$HOME"
REPO_DIR="$BASE/cacheon"
VENV_DIR="$BASE/venv-cacheon"
PULL_ONLY=false

for arg in "$@"; do
  [[ "$arg" == "--pull" ]] && PULL_ONLY=true
done

# -- system deps --
if [[ "$PULL_ONLY" == false ]]; then
  echo "=== System dependencies ==="
  apt-get update -q
  apt-get install -y --no-install-recommends \
    git curl ca-certificates tmux python3-venv docker.io
fi

# -- clone or pull repo --
echo ""
echo "=== Repo ==="
if [ -z "${GITHUB_PAT:-}" ]; then
  echo "ERROR: GITHUB_PAT environment variable is not set."
  echo "       export GITHUB_PAT=<your token> and re-run."
  exit 1
fi

if [ -d "$REPO_DIR/.git" ]; then
  echo "Repo exists, pulling latest..."
  git -C "$REPO_DIR" pull
else
  echo "Cloning $REPO_URL -> $REPO_DIR"
  if [[ "$REPO_URL" == https://github.com/* ]]; then
    CLONE_URL=$(echo "$REPO_URL" | sed -E "s#https://github.com/#https://$GITHUB_PAT@github.com/#")
    git clone "$CLONE_URL" "$REPO_DIR"
  else
    git clone "$REPO_URL" "$REPO_DIR"
  fi
fi

if [[ "$PULL_ONLY" == true ]]; then
  echo "=== Pull-only run complete ==="
  exit 0
fi

# -- python venv --
echo ""
echo "=== Python dependencies ==="
echo "Python: $(python3 --version)"

if [ ! -x "$VENV_DIR/bin/python" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "Creating venv at $VENV_DIR..."
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$REPO_DIR/validator/requirements-cpu.txt"

echo ""
echo "=== Setup complete ==="
echo "See https://cacheon.io/docs/validators/overview to run the validator."
