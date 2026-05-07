<div align="center">

# **Cacheon (SN14)**

[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Discord](https://discord.gg/bittensor) | [TAO.app](https://tao.app/subnets/14) | [Docs](https://cacheon-frontend.c-ad6.workers.dev/docs)

---

</div>

## What is Cacheon?

Cacheon is a Bittensor subnet (SN14) where miners compete to build the **fastest inference server** for a fixed open-source model.

**V1 arena:** serve `Qwen2.5-72B-Instruct` on 4x H200 faster than a pinned vLLM baseline, while preserving output correctness via a first-divergence logprob gate.

Miners submit Docker images that expose an OpenAI-compatible `/v1/chat/completions` endpoint. The validator pulls, launches, benchmarks, and scores each submission on TTFT and throughput improvement over baseline.

## Docs

All documentation lives at [cacheon.io/docs](https://cacheon.io/docs), built from the [cacheon-frontend](https://github.com/latent-to/cacheon-frontend) repo.

## Repository layout

```
validator/          Core validator logic (chain, loop, state, challengers)
scripts/            CLI entrypoints and GPU pod provisioning helpers
miner/              Miner commitment tool
tests/              Unit tests (pytest -m unit)
```

## Quick start

**Miners:** see [cacheon.io/docs/miners/overview](https://cacheon.io/docs/miners/overview)

**Validators:** see [cacheon.io/docs/validators/overview](https://cacheon.io/docs/validators/overview)

## License

MIT
