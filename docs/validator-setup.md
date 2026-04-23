# Validator CPU Host — Setup Guide

This guide covers provisioning the **CPU validator** (chain scan, sandbox precheck, `set_weights`).
The GPU inference pod is a separate machine — see `inference_engine/setup.sh` for that.

## Prerequisites

- Ubuntu 22.04+ (or any Debian-based distro with `apt`)
- Python 3.10+
- A GitHub Personal Access Token (PAT) with **read access** to this repo

## Quick start

```bash
export GITHUB_PAT=<your token>
curl -fsSL https://raw.githubusercontent.com/latent-to/cacheon/main/validator/setup-cpu.sh | bash
```

Or clone first and run locally:

```bash
git clone https://$GITHUB_PAT@github.com/latent-to/cacheon.git ~/cacheon
bash ~/cacheon/validator/setup-cpu.sh
```

The script:

1. Installs system deps: `git`, `python3-venv`, **`firejail`**
2. Clones the repo to `~/cacheon` (or pulls if it already exists)
3. Creates a venv at `~/venv-cacheon` (skipped if already valid)
4. Installs `validator/requirements-cpu.txt` (`bittensor`, `bittensor-cli`) into the venv

### Re-running after a code update

```bash
export GITHUB_PAT=<your token>
bash ~/cacheon/validator/setup-cpu.sh --pull
```

`--pull` skips apt and venv steps — just pulls the latest code.

## Wallet setup

You'll need a Bittensor wallet with a registered hotkey on the Cacheon subnet (netuid 14 on mainnet, netuid 460 on testnet) before starting the validator. If you're new to Bittensor wallets and hotkeys, see the [official key management docs](https://docs.learnbittensor.org/keys/working-with-keys).

Register your hotkey on the subnet:

```bash
# Mainnet (netuid 14)
btcli subnet register --netuid 14 --wallet.name <name> --wallet.hotkey <hotkey-name>

# Testnet (netuid 460)
btcli subnet register --network test --netuid 460 --wallet.name <name> --wallet.hotkey <hotkey-name>
```

Verify registration:

```bash
btcli wallet overview --wallet.name <name>
```

For a full walkthrough on running a Bittensor validator, see the [validator docs](https://docs.learnbittensor.org/validators).

## Running the validator

```bash
source ~/venv-cacheon/bin/activate

# Mainnet (netuid 14)
python ~/cacheon/scripts/remote_validator.py \
  --network finney \
  --netuid 14 \
  --wallet-name <name> \
  --wallet-hotkey <hotkey-name>

# Testnet (netuid 460)
python ~/cacheon/scripts/remote_validator.py \
  --network test \
  --netuid 460 \
  --wallet-name <name> \
  --wallet-hotkey <hotkey-name>
```

Add `--dry-run` to scan the chain and exercise the full loop without writing weights (safe for
testing). Add `-v` for verbose output.

## Environment variable reference

All CLI flags can be set via env vars instead. Copy `validator/.env.validator.example` to `.env.validator`, fill in your values, and `source` it before running

| Variable                     | Default                    | Description                                            |
| ---------------------------- | -------------------------- | ------------------------------------------------------ |
| `CACHEON_NETWORK`            | `finney`                   | Bittensor network (`finney`, `test`, or `ws://…`)      |
| `CACHEON_NETUID`             | `14`                       | Subnet UID                                             |
| `CACHEON_WALLET_NAME`        | `default`                  | Wallet name                                            |
| `CACHEON_WALLET_HOTKEY`      | `default`                  | Hotkey name                                            |
| `CACHEON_POLL_INTERVAL_S`    | `360`                      | Seconds to sleep between idle scans                    |
| `CACHEON_STATE_DIR`          | `<repo>/state`             | Path where `state.json` is persisted                   |
| `CACHEON_DRY_RUN`            | `0`                        | Set to `1` to skip on-chain `set_weights` and GPU eval |
| `CACHEON_POLICY_CACHE_DIR`   | `<STATE_DIR>/policy-cache` | Where fetched `policy.py` files are cached             |
| `CACHEON_POLICY_MAX_BYTES`   | `1048576`                  | Hard size cap (bytes) on a single `policy.py` download |
| `CACHEON_HF_ETAG_TIMEOUT_S`  | `30.0`                     | Timeout (s) for HEAD/etag revalidation inside `hf_hub_download` (not blob download). Legacy: `CACHEON_HF_FETCH_TIMEOUT_S` still read if unset. |
| `CACHEON_HF_TOKEN`           | _(none)_                   | Optional HF token for private/gated repos              |

## Why firejail?

The Phase 3 sandbox precheck (`inference_engine/runner.py`) executes miner-submitted code
in a subprocess. When `firejail` is on `PATH`, the subprocess runs with:

- `--net=none` — no network access
- `--private=<workdir>` — isolated filesystem
- `--rlimit-as=8g` — 8 GB memory cap
- `--rlimit-nproc=64` — process limit

Without firejail the precheck falls back to a bare subprocess (warning logged) — this is
**not safe in production**: a malicious submission could reach the network or exhaust host
resources. Always run `apt install firejail` on the CPU validator box.
