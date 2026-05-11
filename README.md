<div align="center">

# Cacheon (SN14)

**Inference optimization. Fastest server wins.**

[![Discord](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![Docs](https://img.shields.io/badge/docs-cacheon.ai-blue)](https://cacheon.ai/docs)
[![TAO.app](https://img.shields.io/badge/TAO.app-SN14-purple)](https://tao.app/subnets/14)
[![X](https://img.shields.io/badge/X-@cacheon__ai-000000?logo=x&logoColor=white)](https://x.com/cacheon_ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Website](https://cacheon.ai) | [Docs](https://cacheon.ai/docs) | [Discord](https://discord.gg/bittensor) | [TAO.app](https://tao.app/subnets/14)

---

</div>

Cacheon is a Bittensor subnet (SN14) that runs an open competition for **production-grade LLM inference optimization**. Miners submit containerized inference servers. Validators evaluate them against a vLLM baseline on the same hardware. The fastest correct server takes all emission.

**V1 arena:** `Qwen2.5-72B-Instruct` on 4x H200 or equivalent GPUs. Beat the pinned vLLM baseline on TTFT and throughput while passing a greedy-decoding correctness gate.

## How It Works

1. **Miners** build an inference server, package it as a Docker image, and then commit the image reference and image digest on-chain.
2. **Validators** scan the chain for new commitments, pull the image, and run it with model weights mounted at `/models`.
3. **Scoring** measures TTFT and throughput improvement over the vLLM baseline. Correctness is checked first -- fail it and the score is zero.
4. **The fastest correct server** becomes king and earns all subnet emission until someone beats it.
5. **Challengers** must exceed the king's score by a small decaying margin (~1% at crowning, decaying to 0 over ~7 days) to prevent noise-driven churn.

Score formula:

```python
if not correctness_pass:
    score = 0.0
else:
    ttft_imp = max(0, (baseline_ttft - miner_ttft) / baseline_ttft)
    tps_imp  = max(0, (miner_tps  - baseline_tps)  / baseline_tps)
    score = 0.5 * ttft_imp + 0.5 * tps_imp
```

## For Miners

Build an inference server that serves `Qwen2.5-72B-Instruct` via `/v1/chat/completions` with streaming and logprobs. Package it as a Docker image (maximum 20 GB; model weights are mounted at runtime, not baked into the image). Push it to a public registry and commit on-chain.

**Requirements:** public container registry, Bittensor wallet registered on SN14. GPU hardware is only needed for local testing.

```bash
# Push your image
docker tag my-server:latest docker.io/myuser/cacheon-miner:v1
docker push docker.io/myuser/cacheon-miner:v1

# Commit on-chain (one shot per hotkey -- test locally first)
python miner/commit.py \
  --wallet-name <wallet> \
  --wallet-hotkey <hotkey> \
  --image "docker.io/myuser/cacheon-miner:v1" \
  --digest "sha256:..." \
  --network finney \
  --netuid 14
```

Full guide: [cacheon.ai/docs/miners/overview](https://cacheon.ai/docs/miners/overview)

## For Validators

The validator has two components: an always-on CPU host (chain scanning, weight setting) and an ephemeral GPU pod (eval). The GPU pod is rented on-demand only when challengers are queued.

**GPU requirements:** NVLink/SXM interconnect, 4x H200 or equivalent, 400 GB storage, model weights at `/workspace/models/Qwen2.5-72B-Instruct`.

```bash
# CPU host (always-on)
git clone https://github.com/latent-to/cacheon
cd cacheon
cp .env.example .env   # add wallet and S3 config
docker compose up --build

# GPU pod (on-demand, run when challengers appear)
bash scripts/gpu_setup/setup.sh
docker compose -f docker-compose.gpu.yml up --build
```

Full guide: [cacheon.ai/docs/validators/overview](https://cacheon.ai/docs/validators/overview)

## Documentation

|                | Miners                                                      | Validators                                                          | Evaluation                                            |
| -------------- | ----------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------- |
| **Start here** | [Overview](https://cacheon.ai/docs/miners/overview)         | [Overview](https://cacheon.ai/docs/validators/overview)             | [Scoring](https://cacheon.ai/docs/evaluation/scoring) |
| **Reference**  | [API contract](https://cacheon.ai/docs/miners/api-contract) | [Architecture](https://cacheon.ai/docs/validators/architecture)     | [Harness](https://cacheon.ai/docs/evaluation/harness) |
| **Setup**      | [Quickstart](https://cacheon.ai/docs/miners/registration)   | [GPU pod setup](https://cacheon.ai/docs/validators/gpu-pod-setup)   | [Prompts](https://cacheon.ai/docs/evaluation/prompts) |
| **Rules**      | [Rules](https://cacheon.ai/docs/miners/rules)               | [CPU host setup](https://cacheon.ai/docs/validators/cpu-host-setup) | [Roadmap](https://cacheon.ai/docs/roadmap)            |

## License

MIT
