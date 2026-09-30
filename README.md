# Cacheon

Cacheon is an inference-speed competition built around SGLang and designed to
operate as a Bittensor subnet. Miners submit inspectable contributions at
registered boundaries of a validator-owned engine: modules of the served model,
or the scheduler's prefix cache. Validators admit those contributions through a
typed, isolated pipeline and reward only improvements measured end to end on the
registered agent workload under its speed and quality policy.

This repository—published as [`latent-to/cacheon`](https://github.com/latent-to/cacheon)—contains
the miner SDK and examples, validator and chain control plane, evaluation
runtime, and settlement and incentive machinery.

> [!IMPORTANT]
> Cacheon is pre-release software. Implemented paths, retained empirical evidence,
> and production readiness are separate claims. Start with the
> [validator first hour](docs/validator-guide/first-hour.md) for what a deployment must prove.

## Start here

| Goal | Documentation |
|---|---|
| Understand the system | [Concepts](docs/get-started/concepts.md) and [architecture overview](docs/architecture/overview.md) |
| Understand why miners participate and how rewards work | [How miners earn rewards](docs/miner-guide/incentives.md) |
| Build a miner contribution | [Miner guide](docs/miner-guide/overview.md) |
| Operate a validator | [Validator guide](docs/validator-guide/overview.md) |
| See what a crown does and does not authorize | [After a crown](docs/engine/integration.md) |
| Review trust boundaries | [Security model](docs/security/threat-model.md) |
| Contribute to the repository | [Contributing](CONTRIBUTING.md) |

The canonical documentation source lives under [`docs/`](docs/) in this
repository and is rendered at [cacheon.ai/docs](https://cacheon.ai/docs/).

## Local correctness loop

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cpu,dev]"

python -m cacheon.cli scan examples/miner_node_identity
python -m cacheon.cli verify examples/miner_node_identity
```

`scan` and `verify` are development checks; `check` runs the node audit and
graph checks in the published arena image. They do not establish serving
throughput, end-to-end quality, settlement eligibility, or a production
release. Those decisions belong to the validator-owned qualification path
described in the documentation.

## Design boundaries

- The validator owns the model, workload, references, timing, outputs, and
  reward policy. A miner contribution owns only its registered target.
- Candidate build and execution run in validator-owned, no-egress OCI workers;
  wallet and chain-signing authority remain outside candidate lifetimes.
- One complete audited qualification PASS makes a contribution eligible for
  settlement, which reopens its retained evidence before recording a crown.
- Evaluation acceptance and serving are different decisions. Nothing in the
  repository turns a crown into a release; that is a maintainer decision made
  outside it.

See the [product model](docs/architecture/product-model.md) and
[slot contract](docs/architecture/slot-contract.md) for the normative
invariants.

## Tests

```bash
python -m pytest -q tests
```

Documentation checks are described in
[Contributing](CONTRIBUTING.md#documentation).

## License

The repository is licensed under [Apache-2.0](LICENSE). Miner submissions are
governed separately by the
[draft submission terms](docs/legal/submission-terms.md); those terms require
legal review before production use.
