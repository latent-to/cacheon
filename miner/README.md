# Mining on Cacheon

Full docs: https://cacheon.io/docs/miners/overview

## Quick start

```bash
python miner/commit.py \
  --image docker.io/myuser/cacheon-miner:v1 \
  --digest sha256:abc123... \
  --wallet-name my-miner \
  --wallet-hotkey default \
  --network test --netuid 470
```

This commits your Docker image reference on-chain. The validator picks it up within ~6 minutes, pulls the image, and evaluates it.

## What do miners build?

A Docker container that serves Qwen2.5-72B-Instruct via an OpenAI-compatible `/v1/chat/completions` endpoint. See the [miner overview](https://cacheon.io/docs/miners/overview) for the full spec.
