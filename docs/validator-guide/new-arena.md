# Commissioning a new arena

An arena is one sealed model, worker image, workload and lane width, served by
one or more commissioned pairs on a node. The repository carries the operator
that commissions a pair as tracked templates under `scripts/arena/`, with every
arena-specific value in one `operator.json` beside the arena's inputs. Bringing
up an arena is therefore: write its inputs, assemble its deployment-owned
packet, render the operator, review the render against the previous arena's,
and launch. The DeepSeek-V4.1-Flash and Muse-Spark-Open arenas render from the
same templates; their rendered operators differ only where their `operator.json`
files differ.

## 1. Arena inputs

`examples/arena_inputs/<arena>/` holds the development-check inputs and the
operator parameters:

| file | role |
|---|---|
| `Dockerfile` | the base engine image pinned by digest, with cacheon installed beside it |
| `engine-config.json` | the sealed flag set the stock arm boots with |
| `development-requests.json` | the public requests for `cacheon check` |
| `README.md` | the model, draft and mount contract, and the measured boot bounds |
| `operator.json` | the parameters below |

Run `cacheon check` on the published worker image first: it proves that the
node seal binds this model under CUDA-graph capture and replay (see
[Verification and diagnostics](running-evals.md)). Publish the worker image and
retain it by digest in a durable image proof, export the runtime source with its
metadata, and harvest a warm runtime seed from one stock boot of that exact
image; the launcher checks the proof and the source, and the commission seals
the seed in the READY receipt. The README of each arena records why the seed is
required: a cold boot from an empty runtime cache does not fit the per-engine
boot bound.

## 2. Deployment-owned inputs

The packet is deployment-owned and sealed outside this repository; the render
copies it by digest and never fabricates any part of it. One directory holds:

| path | content |
|---|---|
| `packet/qualification_capabilities.py` | the fixed capabilities of the commissioned worker; its replay load is rewritten to the arena's value |
| `packet/numeric_answer_judge.py` | the hidden judge |
| `packet/finalize-authority.py` | the authority finalize step |
| `packet/cacheon_b300_qualification_adapter.py` | the closed adapter the pod transport invokes |
| `packet/prompt-authority.current.json` | the sealed prompt authority built from the arena's prompt pools with its tokenizer |
| `model-receipts/model-provision-sha256-*.json` | the one provision receipt of the commissioned checkpoint |
| `pod-remote-worker-service.py` | the pod transport the registration binds |

## 3. The operator parameters

`operator.json` carries schema `cacheon-arena-operator-v1` and these fields:

| field | meaning |
|---|---|
| `arena_id` | the registered arena identity, as sealed in the deployment profile and the manifest |
| `slug` | the short name in unit names, credential ids, digest domains and rendered file names |
| `model_profile_key` | the model profile key the deployment profile seals |
| `tensor_parallel_size` | the lane width; a pair is two disjoint lanes of this width |
| `replay_load` | the sealed replay sessions per window |
| `tokenizer_asset` | the tokenizer directory name inside the replay inputs archive |
| `model_in_ram` | whether the checkpoint is served from a RAM mount or from its storage path |
| `owner` | the dispatcher owner name of the commissioned pair |
| `node.name` | the host class whose measured scheduler positions follow |
| `node.scheduler_starts` | one measured-fast vCPU start per GPU of the host, in GPU order |
| `node.client_cpus` | the validator client core set per lane, keyed by the lane's device list |

The node section is a measured fact of the host, not a tuning knob: scheduler
pins and the client set bind the sealed runtime digest.

## 4. Render and review

```bash
python scripts/arena/render_operator.py --arena musespark \
  --retained-inputs /path/to/retained --out /path/to/operator-musespark
```

The output is the task-directory layout the validator VM launches from:
`operator/` and `cpu-operator-<slug>.tar` (the scripts the pod and the VM run;
`bootstrap_worker.py` is copied from `scripts/`), `inputs/` and
`retained-bootstrap-inputs-<slug>.tar` (the commission inputs extracted into the
pod's shared root), the rendered `launch_stage.py`, a copy of `launch_node.py`
and `render-receipt.json`, which digests every template, retained input and
output and records the source commit. Every rendered file is parsed in the
language it runs as before it is written, and a placeholder left unfilled is an
error.

Review a new arena's render against the previous arena's with `diff -r` over
the two output directories: the only differences are the parameters. The same
command over the retained operator of a commissioned pair proves that a later
render still produces the bytes that commissioned it.

## 5. Launch

On the validator VM, from a task directory that holds the retained artifacts:

```bash
python3 -B scripts/arena/launch_node.py --operator /path/to/operator-musespark \
  --task-dir /path/to/task --base /path/to/base --image-proof durable-images.json \
  --source runtime-source.tar --source-metadata runtime-source.json \
  --source-revision <commit> --image-digest <sha256 hex> -- \
  --host <pod> --user root --known-hosts known_hosts --shared /data/<arena> \
  --tag <node tag> --lane 0,1,2,3 --baseline 4,5,6,7 \
  --model-input model-input.json --model-token /path/to/token \
  --runtime-seed /path/to/seed
```

The launcher checks the image proof, the source archive and the render receipt,
then runs the rendered stage detached with its output in the task directory.
The stage ships the archives to the pod, bootstraps the worker (environments,
model receipt re-hash, image pull by digest), binds the recipe to the pod's
paths, and runs the commission steps: READY receipt, deployment profile,
declaration, base authority, materialization, registration, calibration seal,
finalize, final materialization and registration, pod install and the CPU
handoff archive, followed by the runtime-UID filesystem preflight. READY is an
environment commission: no engine boots, and the first engine boot is the first
qualification. The stage ends with `COMMISSIONING_ONLY`.

## 6. Prove, then connect

Before the pair joins an intake, run the off-chain twin through the production
pod service and the rendered request driver (`req_<slug>.py`): the identity
bundle reads about 1.00 and fails the speed floor, and the registered broken and
faithful controls grade as documented in
[Arena service](arena-service.md#deployment-composition). Connecting the
handoff to the intake and starting the pair units
(`connect_worker.py --start`, wrapped by `await_commission.py`) is a separate,
deliberate step; nothing in the launch does it.
