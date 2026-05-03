# GPU Pod Setup

Small provisioning flows for the Cacheon eval harness on **Targon** or **Lium**. Pick a provider, set API credentials, run the matching script from the repo root.

## Customize before you run

Both scripts mark provider-specific values with `# TODO` comments at the top of each file. Replace those defaults with your own:

**`create_targon_pod.py`**

- `# TODO: replace this with your workload name` (`WORKLOAD_NAME`)
- `# TODO: replace this with your volume UID` (`VOLUME_UID`)
- `# TODO: replace with one of "H100", "H200", "B200"` (`TARGON_GPU`; Targon does not expose automatic fallback, so you choose one tier)

**`create_lium_pod.py`**

- `# TODO: replace this with your pod name` (`POD_NAME`)
- `# TODO: replace this with your volume name` (`VOLUME_NAME`)

SSH public keys live in `shared.py` (`SSH_KEYS`). Edit that dict so the pod accepts logins from your keys (and remove entries you do not need).

## GPU selection

**Lium** tries GPU types in this order and uses the first available (1x only):

1. H100 SXM (PCIe variants excluded)
2. H200
3. B200

**Targon** does not support that pattern in the API. Set `TARGON_GPU` in `create_targon_pod.py` to exactly one of `H100`, `H200`, or `B200` (mapped to Targon `resource_name` values in `TARGON_RESOURCE_BY_GPU`).

## Usage

```bash
# Targon
export TARGON_API_KEY=...
python scripts/gpu_setup/create_targon_pod.py

# Lium
export LIUM_API_KEY=...
python scripts/gpu_setup/create_lium_pod.py
```

Both scripts register SSH keys with the provider, attach a persistent volume at `/workspace`, inject environment variables from the repo-root `.env`, and poll until the pod is running. GPU picking differs by provider (see **GPU selection** above).

## Files

| File                   | Role                                           |
| ---------------------- | ---------------------------------------------- |
| `shared.py`            | SSH keys, `.env` loader, `GPU_PREFERENCE` list |
| `create_targon_pod.py` | Targon-specific provisioning via REST API      |
| `create_lium_pod.py`   | Lium-specific provisioning via `lium` SDK      |
