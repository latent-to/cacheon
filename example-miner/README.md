# Example miner

Minimal Cacheon submission using stock vLLM. Submitting this as-is scores zero (no improvement over baseline). It exists to verify the contract works.

Full docs: https://cacheon.ai/docs/miners/overview

## Build

```bash
cd example-miner
docker login
docker build -t docker.io/YOUR_USER/cacheon-miner:v1 .
```

## Test locally

See https://cacheon.ai/docs/miners/local-testing

## Push

```bash
docker push docker.io/YOUR_USER/cacheon-miner:v1
```

## Get the digest

```bash
docker inspect --format='{{index .RepoDigests 0}}' docker.io/YOUR_USER/cacheon-miner:v1
```

This prints something like `docker.io/YOUR_USER/cacheon-miner@sha256:abc123...`. Copy the `sha256:...` part.

## Commit on-chain

```bash
python miner/commit.py \
  --image docker.io/YOUR_USER/cacheon-miner:v1 \
  --digest sha256:abc123... \
  --wallet-name my-miner \
  --wallet-hotkey default \
  --network test --netuid 470
```

For mainnet use `--network finney --netuid 14`.

See https://cacheon.ai/docs/miners/registration for full details.
