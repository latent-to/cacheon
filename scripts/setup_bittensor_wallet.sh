#!/usr/bin/env bash
# Restore Bittensor coldkey + hotkey under ~/.bittensor/wallets (or WALLET_PATH).
#
# Mnemonics via environment (recommended: prefix line with space so Bash may skip history):
#   COLDKEY_MNEMONIC='...' HOTKEY_MNEMONIC='...' ./scripts/setup_bittensor_wallet.sh
#
# Or interactively (hidden input, no argv):
#   ./scripts/setup_bittensor_wallet.sh --prompt
#
# Optional overrides: WALLET_NAME (default owner), WALLET_HOTKEY (default owner), WALLET_PATH.

set -euo pipefail

WALLET_NAME="${WALLET_NAME:-owner}"
WALLET_HOTKEY="${WALLET_HOTKEY:-owner}"
WALLET_PATH="${WALLET_PATH:-$HOME/.bittensor/wallets}"

usage() {
  cat <<'EOF'
Usage:
  COLDKEY_MNEMONIC='...' HOTKEY_MNEMONIC='...' ./scripts/setup_bittensor_wallet.sh
  ./scripts/setup_bittensor_wallet.sh --prompt

Optional env: WALLET_NAME WALLET_HOTKEY WALLET_PATH (defaults: owner, owner, ~/.bittensor/wallets)
EOF
}

if ! command -v btcli >/dev/null 2>&1; then
  echo "error: btcli not found in PATH (install bittensor-cli)" >&2
  exit 1
fi

PROMPT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt | -p)
      PROMPT=1
      shift
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$PROMPT" -eq 1 ]]; then
  read -r -s -p "Coldkey mnemonic: " COLDKEY_MNEMONIC
  echo "" >&2
  read -r -s -p "Hotkey mnemonic: " HOTKEY_MNEMONIC
  echo "" >&2
elif [[ -z "${COLDKEY_MNEMONIC:-}" || -z "${HOTKEY_MNEMONIC:-}" ]]; then
  echo "error: set COLDKEY_MNEMONIC and HOTKEY_MNEMONIC, or run with --prompt" >&2
  exit 1
fi

mkdir -p "$WALLET_PATH"

btcli w regen-coldkey \
  --wallet-name "$WALLET_NAME" \
  --mnemonic "$COLDKEY_MNEMONIC" \
  --wallet-path "$WALLET_PATH" \
  --no-use-password

btcli w regen-hotkey \
  --wallet-name "$WALLET_NAME" \
  --wallet-hotkey "$WALLET_HOTKEY" \
  --mnemonic "$HOTKEY_MNEMONIC" \
  --wallet-path "$WALLET_PATH" \
  --no-use-password

echo "done: wallets at $WALLET_PATH (cold + hotkey: $WALLET_NAME / $WALLET_HOTKEY)"
