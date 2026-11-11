# Commissioning a new arena

An arena is one sealed model, worker image, workload and lane width, served by
one or more commissioned pairs on a node; a pair is the operator's name for one
worker and the one lane it owns. The repository carries the operator
that commissions a pair as tracked templates under `scripts/arena/`, with every
arena-specific value in one `operator.json` beside the arena's inputs, and the
tools that produce the arena's own inputs: the prompt pools and authority, the
retained image proof and the warm runtime seed. Bringing up an arena is
therefore: write its inputs, build and retain, render the operator, review the
render against the previous arena's, launch, prove off-chain, and connect. The
DeepSeek-V4.1-Flash and GLM-5.3-Flash prefill/decode arenas render from the same
templates; their rendered operators differ only where their `operator.json` files differ.

## 1. Arena inputs

`examples/arena_inputs/<arena>/` holds the development-check inputs and the
operator parameters:

| file | role |
|---|---|
| `Dockerfile` | the base engine image pinned by digest, with cacheon installed beside it |
| `engine-config.json` | the sealed flag set the stock arm boots with |
| `development-requests.json` | the public requests for `cacheon check` |
| `README.md` | the model, draft and mount contract, and the measured boot bounds |
| `operator.json` | the parameters in section 4 |

Run `cacheon check` on the published worker image first: it proves that the
node seal binds this model under CUDA-graph capture and replay (see
[Verification and diagnostics](running-evals.md)). Publish the worker image
and export the runtime source with its metadata; the launcher checks both.

## 2. Build and retain

Four tools produce the inputs the commission binds. Each writes a receipt and
refuses to overwrite an output.

**Image proof.** On the validator VM, retain the published image by digest and
write the durable image proof every later step checks:

```bash
python3 scripts/arena/retain_image.py --published docker.io/<org>/<repo>@sha256:<index> \
  --repository <repo> --tag <revision prefix> --registry-root /path/to/registry \
  --sglang-image <base image>@sha256:<digest> --source-revision <commit> \
  --out durable-images.json --manifest-out manifest.json --log retain.log
```

**Prompt pools and authority.** Inside the worker image, with the model mounted
read-only and the previous arena's sealed authority and judge as the reference:

```bash
python3 scripts/arena/build_prompt_pools.py --model /cacheon/input/model \
  --source-jsonl /pools/gsm8k_train.jsonl --input-tokens 8192 --pool-size 2400 --out /pools/pool_8192.json
python3 scripts/arena/build_prompt_pools.py --model /cacheon/input/model \
  --source-jsonl /pools/gsm8k_train.jsonl --input-tokens 65536 --pool-size 120 --out /pools/pool_65536.json
python3 scripts/arena/build_prompt_authority.py --arena-inputs /arena --pools /pools \
  --judge-dir /packet --reference /packet/prompt-authority.previous.json \
  --model /cacheon/input/model --model-receipt /receipts/model-provision-sha256-<digest>.json \
  --out /packet/prompt-authority.current.json
```

The authority carries the reference's cell structure (two cells, seven
batches, 480 prompts), its selection policy digest and its corpus commitment;
the judge digest function must reproduce the reference's sealed values before
anything is written. The engine config is laid out as the pod's commission
rewrites it, so the digest the builder prints is the digest the pod seals.

**Runtime seed.** Every qualification lease mounts an empty runtime cache and
seeds it from a validator-provisioned tree, so a cold boot recompiles every
runtime artifact and does not fit the per-engine boot bound (the arena README
records the measured times). On the node, with the arena's lane free:

```bash
python3 scripts/arena/harvest_runtime_seed.py --arena-inputs examples/arena_inputs/<arena> \
  --image-proof durable-images.json --model /path/to/model --gpus 0,1,2,3 \
  --work /path/to/harvest --out /path/to/runtime-seed --verify
```

The tool boots the stock engine once in the exact image with the sealed flags
and the production cache environment, stops it, seals the cache's regular files
as the seed with the receipt the commissioner recomputes, and with `--verify`
boots once more from the seed and records both boot times. For a prefill/decode
pair it boots each role in turn, in its one-engine form (the shared options under
the role's own), into the same cache, since one seed serves every engine of a
lease; `--gpus` names the widest role's GPUs, and a narrower role uses the first of them.

**Deployment-owned packet.** The remaining packet files are sealed outside this
repository and copied by digest, never fabricated: the fixed qualification
capabilities (its replay load, per-load decode floors, scored windows and audit
controls are rewritten to the arena's values), the numeric
judge, the finalize step, the closed adapter, the model provision receipt and
the pod transport. One directory holds them:

| path | content |
|---|---|
| `packet/qualification_capabilities.py` | the fixed capabilities of the commissioned worker |
| `packet/numeric_answer_judge.py` | the hidden judge |
| `packet/finalize-authority.py` | the authority finalize step |
| `packet/cacheon_b300_qualification_adapter.py` | the closed adapter the pod transport invokes |
| `packet/prompt-authority.current.json` | the authority built above |
| `model-receipts/model-provision-sha256-*.json` | the one provision receipt of the commissioned checkpoint |
| `pod-remote-worker-service.py` | the pod transport the registration binds |

## 3. The operator parameters

`operator.json` carries schema `cacheon-arena-operator-v1` and these fields:

| field | meaning |
|---|---|
| `arena_id` | the registered arena identity, as sealed in the deployment profile and the manifest |
| `slug` | the short name in unit names, credential ids, digest domains and rendered file names |
| `model_profile_key` | the model profile key the deployment profile seals |
| `tensor_parallel_size` | the lane width; one engine at a time owns every GPU of the lane |
| `replay_load` | the sealed replay session concurrency: one count, or an ascending list (low, mid, high) a PD pair replays in every lifetime, highest first, and scores as a geometric mean |
| `decode_floors` | required with several loads, else optional: each load's per-user decode floor in tokens/s, keyed by the load as a string, as canonical decimals; written into the packet's goodput policy |
| `load_margin` | optional, with `decode_floors`: a canonical decimal in (0, 1); a load whose throughput-ratio upper bound is below 1 minus it fails the candidate; written into the packet's goodput policy |
| `replay_windows` | optional: the scored windows per load, replacing the packet's (one keeps a three-load lifetime within a one-load lifetime's replay time) |
| `tokens_per_prompt` | the largest recorded output budget of one turn in the sessions the replay serves; the commission refuses any other value |
| `tokenizer_asset` | the tokenizer directory name inside the replay inputs archive |
| `model_in_ram` | whether the checkpoint is served from a RAM mount or from its storage path |
| `owner` | the dispatcher owner name of the commissioned pair |
| `audit_control_delta_digests` | the registered controls' selected delta digests, sorted; a wrong control for this model's node address, so its audit runs after its speed failure |
| `audit_row_floor_ppm` | optional: the node audit's per-row floor in ppm, absent at the 20000 (2%) default and at most 400000; set from `cacheon check --audit-row-floor-ppm` runs in which an honest lossy control passes and the wrong control fails |
| `node.name` | the host class whose measured scheduler positions follow |
| `node.scheduler_starts` | one measured-fast vCPU start per GPU of the host, in GPU order |
| `node.client_cpus` | the validator client core set per lane, keyed by the lane's device list |
| `node.engine_cpus` | optional: per engine (each prefill server and the decode engine of a pair), keyed by its first GPU, the data-parallel controller's vCPU and the frontend's vCPU; an engine with DP 1 runs no controller |
| `node.lifetime_memory_gib` | optional: the host memory cap of one engine lifetime's container in GiB (default 1024), sealed into the runtime resource policy; a host KV tier (`hicache_size`, per rank) is pinned inside it |

The node section is a measured fact of the host, not a tuning knob: scheduler
pins, engine roles and the client set bind the sealed runtime digest. The
commission times the table rather than trusting `lscpu` or sysfs, which a guest
can misreport: Verda's 8x B300 VM listed 240 one-thread sockets while vCPU k and
k+120 shared a core, and 16 of its 120 cores ran at twice the speed of the rest.
`cacheon.eval.oci_cpuset.verify_cpu_plan` times every vCPU alone and every
relevant pair together. It refuses a scheduler, engine-role or client vCPU
slower than 0.75x the fastest, a scheduler or engine-role vCPU that shares a
core with any other named vCPU, and a pool vCPU that shares a core with the
client. A control's digest is
what intake computes for it: with the control bundle on disk,
`inspect_contribution(root, catalog=default_target_catalog()).selected_delta_digest`
from `cacheon.engine_tree` and `cacheon.target_catalog`; the render writes the
list into the sealed policy of the packet's fixed capabilities, replacing the
previous arena's controls as it replaces the replay load.

## 4. Render and review

```bash
python scripts/arena/render_operator.py --arena dsv41pd \
  --retained-inputs /path/to/retained --out /path/to/operator-dsv41pd
```

The output is the task-directory layout the validator VM launches from:
`operator/` and `cpu-operator-<slug>.tar` (the scripts the pod and the VM run,
with the off-chain twin; `bootstrap_worker.py` is copied from `scripts/`),
`inputs/` and `retained-bootstrap-inputs-<slug>.tar` (the commission inputs
extracted into the pod's shared root), the rendered `launch_stage.py`, a copy
of `launch_node.py` and `render-receipt.json`, which digests every template,
retained input and output and records the source commit. Every rendered file
is parsed in the language it runs as before it is written, and a placeholder
left unfilled is an error.

Review a new arena's render against the previous arena's with `diff -r` over
the two output directories: the only differences are the parameters. The same
command over the retained operator of a commissioned pair proves that a later
render still produces the bytes that commissioned it.

## 5. Launch

On the validator VM, from a task directory that holds the retained artifacts,
with the replay inputs archive (`replay-inputs-<slug>.tar`: the sealed workload
slice and the tokenizer) placed beside the rendered `launch_stage.py`:

```bash
python3 -B scripts/arena/launch_node.py --operator /path/to/operator-dsv41pd \
  --task-dir /path/to/task --base /path/to/base --image-proof durable-images.json \
  --source runtime-source.tar --source-metadata runtime-source.json \
  --source-revision <commit> --image-digest <sha256 hex> -- \
  --host <pod> --user root --known-hosts known_hosts --shared /data/<arena> \
  --tag <node tag> --lane 0,1,2,3 \
  --model-input model-input.json --model-token /path/to/token \
  --runtime-seed /path/to/runtime-seed
```

The launcher independently checks the retained image, exported controller source
and render receipt, then runs the stage detached with its output in the task directory.
The stage ships the archives to the pod, bootstraps the worker (environments,
model receipt re-hash, image pull by digest), binds the recipe to the pod's
paths, and runs the commission steps: READY receipt, deployment profile,
declaration, base authority, materialization, registration, calibration seal,
finalize, final materialization and registration, pod install and the CPU
handoff archive, followed by the runtime-UID filesystem preflight. READY is an
environment commission: no engine boots, and the first engine boot is the first
qualification. The stage ends with `COMMISSIONING_ONLY`.

## 6. Prove, then connect

Before the pair joins an intake, run the off-chain twin on the pod. It is
rendered into the pod's operator directory as `twin_<slug>.sh` and runs the
production pod service the way the pair units run it, on the lane's client
cores, plus the production request driver over a scratch intake:

```bash
sh /path/to/workers/<tag>/recovery-operator/twin_<slug>.sh /data/<arena> <tag> serve
sh /path/to/workers/<tag>/recovery-operator/twin_<slug>.sh /data/<arena> <tag> submit /path/to/control identity-aa-1
sh /path/to/workers/<tag>/recovery-operator/twin_<slug>.sh /data/<arena> <tag> status
```

A PD (policy 21) pair measures its stock reference here: the first two twin
requests each measure one stock pair and end `NO_DECISION`, and the third
reads its bundle against the completed reference. A connected pair would hold
the reservations those two results name, so a replacement reference is measured
the same way, with the worker stopped.

The identity bundle reads about 1.00 and fails the speed floor, and the
registered broken and faithful controls grade as documented in
[Arena service](arena-service.md#deployment-composition). Connecting the
handoff is a separate, deliberate step; nothing in the launch does it.
`connect_worker.py` without `--start` fetches the handoff and prepares the
pair's stage on the validator VM together with the arena's plane: its own
intake unit, intake database and fee, created by the arena's first pair and
found by later ones. Nothing is installed or started until `--start`, which
`await_commission.py` passes once the commission completes.
The start step installs and starts the intake unit first, waits up to 120 seconds
for its first finalized cursor, and checks queued identities before starting
the worker. A missing cursor stops the start with the intake log as the recovery
point; an existing plane keeps its database and cursor.
