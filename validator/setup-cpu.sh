#!/usr/bin/env bash
# validator/setup-cpu.sh
#
# CPU validator setup: creates/activates the Python venv, installs deps,
# loads .env, tears down any running cpu-validator container, and starts
# docker compose.
#
# Usage:
#   cd cacheon/validator
#   bash setup-cpu.sh            # interactive: .env must exist
#   bash setup-cpu.sh --detach   # same, but run compose in background (-d)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/venv-cacheon}"
COMPOSE_FILE="$SCRIPT_DIR/cpu-compose.yml"
ENV_FILE="$SCRIPT_DIR/.env"

DETACH=false
for arg in "$@"; do
  [[ "$arg" == "--detach" || "$arg" == "-d" ]] && DETACH=true
done

# -- venv --
echo "=== Python venv ==="
if [[ -x "$VENV_DIR/bin/python" ]] && [[ -f "$VENV_DIR/bin/activate" ]]; then
  echo "Activating existing venv at $VENV_DIR"
else
  echo "Creating venv at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements-cpu.txt" -q
echo "Dependencies installed."

# -- .env --
echo ""
echo "=== Environment ==="
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found."
  echo "Copy .env.cpu.example to .env and fill in credentials before running this script."
  exit 1
fi
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a
echo ".env loaded from $ENV_FILE"

# -- docker prereqs --
echo ""
echo "=== Docker ==="
if ! docker --version >/dev/null 2>&1; then
  echo "ERROR: docker not found. Install Docker and Docker Compose first."
  exit 1
fi

# -- tear down existing cpu-validator if running --
if docker ps -q -f name=cacheon-cpu-validator 2>/dev/null | grep -q .; then
  echo "Stopping existing cacheon-cpu-validator container..."
  docker compose -f "$COMPOSE_FILE" down
  echo "Container stopped."
else
  echo "No running cacheon-cpu-validator found."
fi

# -- start --
echo ""
echo "=== Starting CPU validator ==="
if [[ "$DETACH" == true ]]; then
  docker compose -f "$COMPOSE_FILE" up --build -d
  echo ""
  echo "Running in background. Tail logs with:"
  echo "  docker compose -f $COMPOSE_FILE logs -f"
else
  docker compose -f "$COMPOSE_FILE" up --build
fi
