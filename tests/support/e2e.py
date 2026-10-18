"""One private deployment root driven through the validator's production entrypoints.

The end-to-end modules share this root builder: miner archives sealed by
``chain-package``, the simulated chain, finalized intake, the commissioned arena,
the worker registration, the sealed dispatcher and standing configs, the
standing supervisor process, and the GPU pod played through the production
spool and response sealers. Which calls are subprocesses, which run in-process
and which values are test-supplied is stated in
``tests/test_validator_end_to_end.py``.
"""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.chain import fetch, validator_loop
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.chain.intake import IntakePolicy, IntakeScope
from cacheon.chain.payload import encode_payload
from cacheon.chain.qualification_request import require_commissioned_incumbent
from cacheon.chain.remote_evaluation_dispatcher import (
    RemoteEvaluationRequest, capture_remote_qualification_product, seal_remote_response,
)
from cacheon.chain.remote_worker_registration import (
    registration_credential, registration_transport_identity, verify_registration,
)
from cacheon.engine_tree import inspect_contribution, materialize_engine_tree
from cacheon.eval import b300_deployment as deployment
from cacheon.eval.b300_publication_intake import resolve_cohort_publications
from cacheon.eval.b300_registered_qualification_inputs import ATTRIBUTION_SCHEMA
from cacheon.eval.evidence_store import publish_evidence
from cacheon.eval.oci_backend import runtime_identity_from_preflight
from cacheon.eval.oci_session_protocol import SlotAuditPolicy
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest, QualificationIntakeBatch, QualificationIntakeOutcome,
    QualificationReservation,
)
from cacheon.eval.remote_run_forensics import append_event, journal_path
from cacheon.settlement import SettlementQualification
from cacheon.stack_identity import canonical_digest, canonical_json_bytes, sha256_hex
from cacheon.stack_manifest import ProposalContributionRef
from cacheon.stack_plan import plan_candidate_stack, plan_marginal_arm
from cacheon.target_catalog import default_target_catalog
from tests.support.b300 import gpu
from tests.support.chain import amend, reopen_state, write_state
from tests.support.preflight import preflight_receipt

REPO = Path(__file__).resolve().parents[2]
SIMULATED_SDK = REPO / "tests" / "support" / "simulated_chain"
NETUID, REVEAL_BLOCK, HEAD = 1, 90, 100
PROFILES = {
    "b300-tp4": SimpleNamespace(gpu="b300", host=8, lane=(0, 1, 2, 3), arch="sm103", sglang="0.5.18",
                                model="MiniMax-M3", kwargs={"page_size": 128, "quantization": "modelopt_fp4"}),
    "h100-tp1": SimpleNamespace(gpu="h100", host=2, lane=(0,), arch="sm90", sglang="0.5.19",
                                model="Qwen3.6-35B-A3B-BF16", kwargs={"kv_cache_dtype": "fp8_e4m3"}),
}


def private(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def sealed_json(path: Path, value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(raw)
    path.chmod(0o400)
    return sha256_hex(raw)


def run(*argv: str, env: dict[str, str] | None = None, code: int = 0) -> str:
    """Run one module entrypoint to its expected exit code; a nonzero exit also returns stderr."""

    done = subprocess.run([sys.executable, "-m", *argv], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == code, done.stdout + done.stderr
    return done.stdout if code == 0 else done.stdout + done.stderr


def chain_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join((str(SIMULATED_SDK), str(REPO)))}


def deployment_root(tmp_path: Path, name: str, monkeypatch):
    """A private root whose validator decryption key the intake process reads."""

    from nacl.public import PrivateKey

    root = private(tmp_path / name)
    key = PrivateKey.generate()
    key_file = root / "bundle-decryption.key"
    key_file.write_text(bytes(key).hex())
    key_file.chmod(0o600)
    monkeypatch.setenv("CACHEON_BUNDLE_DECRYPTION_KEY", str(key_file))
    private(root / "validator")
    return root, key


def miner_bundle(root: Path, hotkey: str, forward: str | None = None) -> Path:
    """The public node example, entered into this root's arena, optionally rewritten."""

    bundle = root / "miners" / hotkey
    shutil.copytree(REPO / "examples" / "miner_node_identity", bundle)
    manifest = bundle / "manifest.toml"
    manifest.write_text(manifest.read_text().replace('mode = "slot"', f'mode = "slot"\narena = "{root.name}"'))
    if forward is not None:
        (bundle / "kernels" / "forward.py").write_text(forward)
    return bundle


def package(bundle: Path, validator_key) -> tuple[str, Path]:
    """``chain-package`` seals the archive for the validator; returns its content hash."""

    archive = bundle.with_suffix(".sealed")
    out = run("cacheon.cli", "chain-package", str(bundle), "--out", str(archive),
              "--encrypt-for", bytes(validator_key.public_key).hex())
    return out.split("content_hash: ")[1].split()[0], archive


def reveal(hotkey: str, content_hash: str, archive: Path, origin: dict[str, Path], block: int = REVEAL_BLOCK):
    """One finalized commitment naming the archive at a canonical HTTPS URL."""

    url = f"https://{hotkey}.example/{content_hash}.sealed"
    origin[url] = archive
    return block, encode_payload(content_hash, url)


def chain(root: Path, reveals: dict[str, tuple], hotkeys: tuple[str, ...], owner: str = "") -> Path:
    return write_state(root / "chain.json", hotkeys=("validator", *hotkeys), revealed=reveals,
                       block=HEAD, validator="validator", owner=owner)


def intake(root: Path, state: Path, origin: dict[str, Path], monkeypatch, **options) -> list[str]:
    """One finalized intake pass; only the HTTPS transfer leg is the hermetic one."""

    fetched: list[str] = []

    def transfer(url, expected, private_root):
        fetched.append(expected)
        return fetch.fetch_bundle_from_local_file_for_testing(origin[url].as_uri(), expected, private_root)

    monkeypatch.setattr(validator_loop, "fetch_bundle", transfer)
    validator_loop.run_validator(
        reopen_state(state), NETUID, intake_db=root / "validator" / "intake.sqlite3",
        private_root=root / "validator" / "private", publication_root=root / "validator" / "worker",
        once=True, audit_log=root / "validator" / "chain-audit.jsonl", **options,
    )
    return fetched


def scope(root: Path) -> IntakeScope:
    return IntakeScope(reopen_state(root / "chain.json").get_block_hash(0), NETUID)


def weights_root(tmp_path: Path, monkeypatch, name: str, reveals: dict[str, int], *, profile=None,
                 peers: tuple[str, ...] = ()):
    """Distinct bundles revealed at the given blocks and fetched; a signed vector reveals three blocks later.

    UIDs run validator, subnet owner, burn sink, ``peers``, then ``reveals``. With a ``profile`` the
    arena is commissioned and the standing composition sealed.
    """

    root, key = deployment_root(tmp_path, name, monkeypatch)
    origin, revealed = {}, {}
    for hotkey, block in reveals.items():
        body = (f"def forward(module, *args, **kwargs):\n    label = {hotkey!r}\n"
                "    return module.forward(*args, **kwargs)\n")
        content_hash, archive = package(miner_bundle(root, hotkey, body), key)
        revealed[hotkey] = (reveal(hotkey, content_hash, archive, origin, block),)
    state = chain(root, revealed, ("subnet-owner", "burn-sink", *peers, *reveals), owner="subnet-owner-coldkey")
    amend(state, reveal_delay=3)
    assert len(intake(root, state, origin, monkeypatch)) == len(reveals)
    if profile is None:
        return root, state, None
    materialized, ready_path, ready = commission(root, profile)
    return root, state, seal_authority(root, materialized, ready_path, ready)


def reward_credit(record: tuple[str, int], earlier: tuple[str, int] | None, block: int,
                  decay_start: int | None) -> float:
    """``docs/reference/emissions-policy.md`` at half-life 100: ln(s) x stall bonus x decay from the confirmed start.

    A record is the PASS's speedup and reveal block; ``earlier`` is the best earlier rewarded record.
    """

    (c, b), (q, p) = record, earlier or ("1", record[1])
    s = math.floor(float(c) / float(q) * 1_000_000) / 1_000_000
    age = 0 if decay_start is None else block - decay_start
    return math.log(s) * (1 + math.sqrt((b - p) / 1800)) * 2 ** (-age / 100)


def reward_shares(**credits: float) -> dict[str, float]:
    total = sum(credits.values())
    return {hotkey.replace("_", "-"): pytest.approx(credit / total, abs=2e-6) for hotkey, credit in credits.items()}


def commission(root: Path, profile) -> tuple[dict, Path, dict]:
    """The pod commission's sealed inputs, materialized into the arena identity."""

    pod = private(root / "pod")
    model = private(pod / "model")
    (model / "config.json").write_text('{"architectures": ["E2EForCausalLM"]}\n')
    sealed = json.loads(run("cacheon.cli", "model-provision", str(model), str(private(pod / "receipts"))))
    image = sha256_hex(f"{profile.gpu}-image".encode())
    preflight = preflight_receipt(image=image, platform=sha256_hex(b"linux/amd64"),
                                  worker=sha256_hex(b"cacheon-harness"), sglang_version=profile.sglang,
                                  worker_file_count=100, worker_total_bytes=100_000)
    runtime = runtime_identity_from_preflight(preflight)
    lane = list(profile.lane)
    device = pod / "device-execution.json"
    prompt = pod / "prompt-authority.json"
    calibration = pod / "calibration-package.json"
    projection = pod / "calibration-projection-receipt.json"
    sealed_json(projection, {"schema": "cacheon-calibration-projection-v1"})
    authority = {
        "arena_id": root.name, "authority_role": "primary",
        "calibration": {"evidence_root": str(pod / "calibration-evidence"), "package": str(calibration),
                        "package_sha256": sealed_json(calibration, {"schema": "cacheon-calibration-v1"})},
        "device_execution": {"path": str(device), "sha256": sealed_json(device, {
            "schema": "cacheon-private-device-control-v1",
            "worker": {"preflight": preflight.canonical_payload()}})},
        "fmha_cache_seed": {"directory_count": 2, "file_count": 3, "plan_sha256": sha256_hex(b"fmha-plan"),
                            "root": str(pod / "fmha-cache"), "schema": "cacheon-fmha-cache-v1",
                            "sealed_root_owned_read_only": True, "total_bytes": 4096,
                            "tree_sha256": sha256_hex(b"fmha-tree")},
        "model": {"content_digest": sealed["content_digest"], "manifest_digest": sealed["receipt_digest"],
                  "revision_digest": sha256_hex(profile.model.encode()), "root": str(model)},
        "prompt": {"path": str(prompt), "sha256": sealed_json(prompt, {
            "accepted_token_subsequences": [], "registered_targets": ["forward_pass"],
            "hidden_corpus_commitment": sha256_hex(b"hidden-corpus"),
            "hidden_judge_digest": sha256_hex(b"hidden-judge"),
            "hidden_task_policy_digest": sha256_hex(b"hidden-task-policy"),
            "engine_config": {
                "attention_backend": None, "deterministic": False, "disable_custom_all_reduce": True,
                "dtype": "bfloat16", "engine_env": {}, "log_level": "error", "mem_fraction_static": 0.8,
                "moe_runner_backend": "triton", "tp_size": len(lane),
                "engine_kwargs": {"disable_radix_cache": True, **profile.kwargs}},
            "model_profile_key": profile.model, "prompt_batches": [["one"], ["two"], ["three"]],
            "prompt_seed_scheme": "sealed-prompts-v1", "schema": "cacheon-private-prompt-authority-v1",
            "selection_policy_digest": sha256_hex(b"selection-policy"),
            "tokenizer_digest": sha256_hex(b"tokenizer"),
            "workload_cell": {"cell_id": "s8", "concurrency": 1, "input_tokens": 8192,
                              "output_tokens": 1024, "timed_reads": 2}})},
        "qualification_builder_digest": sha256_hex(b"qualification-builder"),
        "schema": "cacheon-private-b300-authority-v1",
        "source": {"controller_digest": sha256_hex(b"controller")},
        "topology": {"architecture": profile.arch, "gpu_count": len(lane), "lane": list(map(str, lane)),
                     "lane_digest": sha256_hex(repr(lane).encode()), "tensor_parallel_size": len(lane),
                     "topology_class": "nvlink-sxm"},
        "worker": {"base_engine_digest": runtime.base_engine_digest, "image": preflight.requested_image,
                   "local_image_id": preflight.local_image_id, "runtime_digest": runtime.runtime_digest,
                   "validator_overlay_digest": runtime.validator_overlay_digest,
                   "worker_distribution_digest": preflight.worker_distribution_digest},
    }
    authority_path, measurement = pod / "authority-config.json", pod / "measurement-config.json"
    sealed_json(authority_path, authority)
    sealed_json(measurement, authority)
    gpus = tuple(gpu(index, profile.gpu) for index in range(profile.host))
    python = Path(sys.executable)
    receipt = Path(sealed["receipt_path"])
    ready = {
        "created_at_unix": int(time.time()),
        "gpu": {"count": profile.host, "inventory": [
            {"index": g.physical_id, "memory_mib": g.memory_total_mib, "name": g.name,
             "pci_bus_id": g.pci_bus_id, "uuid": g.uuid} for g in gpus],
            "inventory_sha256": sha256_hex(canonical_json_bytes([g.uuid for g in gpus])),
            "topology_sha256": sha256_hex(b"nvlink-sxm")},
        "lane": {"devices": lane, "lane_digest": sha256_hex(repr(lane).encode()), "tensor_parallel_size": len(lane)},
        "model": {"content_digest": sealed["content_digest"], "path": str(model), "readonly_inventory_verified": True,
                  "receipt_digest": sealed["receipt_digest"], "receipt_file_sha256": spool.file_sha256(receipt),
                  "receipt_path": str(receipt)},
        "provider": {"hostname": "pod", "machine_id_sha256": sha256_hex(b"machine"), "pod_endpoint": "pod.example:22"},
        "python": {"path": str(python), "resolved_path": str(python.resolve()),
                   "executable_sha256": spool.file_sha256(python.resolve()), "version": "Python 3.12.0"},
        "runtime": {"path": str(pod / "runtime"), "tree_digest": sha256_hex(b"runtime-tree")},
        "schema": "cacheon-current-pod-commission-v1",
        "source": {"path": str(REPO), "tree_digest": sha256_hex(b"source-tree"), "revision": "0" * 40},
        "state": "READY_FOR_REGISTRATION", "worker_epoch": sha256_hex(root.name.encode())[:32],
        "worker_image": preflight.requested_image,
    }
    ready_path = _seal_ready(pod / "ready-receipt.json", ready)
    return _materialize(root, profile, ready_path, pod / "commissioned"), ready_path, ready


def _seal_ready(path: Path, ready: dict) -> Path:
    ready["receipt_digest"] = spool.spool_digest("cacheon.current-pod-commission.v1", ready)
    sealed_json(path, ready)
    return path


def _materialize(root: Path, profile, ready_path: Path, output: Path) -> dict:
    pod = root / "pod"
    gpus = tuple(gpu(index, profile.gpu) for index in range(profile.host))
    return deployment.materialize_b300_identities(
        ready_receipt=ready_path, authority_config=pod / "authority-config.json",
        measurement_config=pod / "measurement-config.json", calibration_package=pod / "calibration-package.json",
        calibration_projection_receipt=pod / "calibration-projection-receipt.json",
        prompt_authority=pod / "prompt-authority.json", output_root=private(output),
        gpu_provisioner=lambda selected, *, deadline: tuple(gpus[index] for index in selected),
    )


def _register(root: Path, materialized: dict, ready_path: Path, output: Path) -> dict:
    """``make-registration`` binds the READY receipt, readiness and credential for the CPU side."""

    manifest = json.loads(Path(materialized["arena_service_manifest"]).read_text())
    run("cacheon.chain.remote_worker_service", "make-registration", "--ready-receipt", str(ready_path),
        "--worker-readiness", materialized["worker_readiness"], "--known-hosts", str(root / "known_hosts"),
        "--pod-host", "pod.example", "--pod-port", "22",
        "--service-identity", deployment._manifest_from_dict(manifest).service_id,
        "--remote-service", str(REPO / "cacheon/chain/remote_worker_service.py"),
        "--adapter", str(REPO / "cacheon/eval/b300_remote_worker_adapter.py"),
        "--credential", str(root / "credential.secret"), "--credential-id", "e2e-qualification-v1",
        "--output", str(output))
    return verify_registration(spool.load_json(output))


def replace_worker(root: Path, profile, ready: dict, tag: str) -> dict:
    """The operator replaces the pod: a fresh READY epoch, its registration, the dispatcher re-pointed."""

    fresh = {**ready, "created_at_unix": int(time.time()), "worker_epoch": sha256_hex(tag.encode())[:32]}
    del fresh["receipt_digest"]
    ready_path = _seal_ready(root / "pod" / f"ready-receipt-{tag}.json", fresh)
    materialized = _materialize(root, profile, ready_path, root / "pod" / f"commissioned-{tag}")
    path = root / f"registration-{tag}.json"
    registration = _register(root, materialized, ready_path, path)
    dispatcher = root / "validator" / "dispatcher.json"
    config = json.loads(dispatcher.read_text())
    config.update(registration_digest=registration["registration_digest"], registration_path=str(path),
                  transport_identity_digest=registration["transport_identity_digest"],
                  worker_readiness=json.loads(Path(materialized["worker_readiness"]).read_text()))
    dispatcher.unlink()
    sealed_json(dispatcher, config)
    return registration


def seal_authority(root: Path, materialized: dict, ready_path: Path, ready: dict):
    """The operator registers the worker and seals the standing composition."""

    secret = root / "credential.secret"
    secret.write_bytes(os.urandom(32))
    secret.chmod(0o400)
    known_hosts = root / "known_hosts"
    known_hosts.write_text("pod.example ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE2E\n")
    known_hosts.chmod(0o600)
    manifest = json.loads(Path(materialized["arena_service_manifest"]).read_text())
    registration_path = root / "registration.json"
    registration = _register(root, materialized, ready_path, registration_path)
    inputs, composition, _ = deployment.replay_commissioned_composition(
        registration, ready, commissioned_root=Path(materialized["deployment"]).parent)
    catalog = default_target_catalog()
    _, context, incumbent, tree = deployment._commissioned_stock_authority(
        inputs, composition.manifest, catalog, catalog.snapshot(),
        error=deployment.B300DeploymentError, label="end-to-end genesis")
    validator = root / "validator"
    sealed_json(validator / "incumbent-stack.json", incumbent.to_dict())
    policy = IntakePolicy()
    sealed_json(validator / "dispatcher.json", {
        "accept_legacy_bundles": False, "arena_service_manifest": manifest,
        "credential_digest": registration["credential_digest"], "credential_path": str(secret),
        "heartbeat_interval_ms": 1000, "heartbeat_join_timeout_ms": 2000, "idle_poll_ms": 100,
        "intake_db": str(validator / "intake.sqlite3"),
        "intake_policy": {name: getattr(policy, name) for name in policy.__dataclass_fields__},
        "intake_scope": scope(root).to_dict(), "lease_blocks": 20, "lock_attempts": 20, "lock_retry_delay_ms": 50,
        "owner": f"e2e-{root.name}", "registration_digest": registration["registration_digest"],
        "registration_path": str(registration_path), "response_timeout_seconds": 1,
        "restart_initial_backoff_ms": 100, "restart_max_backoff_ms": 400,
        "schema": "cacheon-mainnet-screen-dispatcher-config-v1", "spool_root": str(private(root / "spool")),
        "transport_identity_digest": registration["transport_identity_digest"], "transport_poll_seconds": 1,
        "worker_readiness": json.loads(Path(materialized["worker_readiness"]).read_text()),
    })
    sealed_json(validator / "standing.json", {
        "enable_qualification": True, "enable_settlement": True, "enable_weights": False,
        "idle_poll_ms": 100, "pruning_review_root": "",
        "qualification_evidence_root": str(private(validator / "evidence")),
        "qualification_incumbent_stack_path": str(validator / "incumbent-stack.json"),
        "qualification_incumbent_tree_digest": tree.tree_digest,
        "restart_initial_backoff_ms": 100, "restart_max_backoff_ms": 400,
        "schema": "cacheon-standing-supervisor-config-v1",
        "screen_dispatcher_config": str(validator / "dispatcher.json"),
        "settlement_network": str(root / "chain.json"), "stall_timeout_ms": 600_000,
        "weights_stage_config": "",
    })
    return SimpleNamespace(registration=registration, catalog=catalog, context=context, manifest=manifest,
                           incumbent=incumbent, incumbent_tree=tree, scope=scope(root), policy=policy)


class Supervisor:
    """The standing CPU supervisor process, observed through its status lines."""

    def __init__(self, root: Path, config: Path | None = None) -> None:
        self.log = (root / "supervisor.stderr").open("a")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "cacheon.chain.standing_cpu_supervisor",
             "--config", str(config or root / "validator" / "standing.json")],
            cwd=REPO, env=chain_env(), stdout=subprocess.PIPE, stderr=self.log, text=True)
        self.lines: queue.Queue[dict] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.process.stdout:
            self.lines.put(json.loads(line))

    def until(self, predicate, timeout: float = 90.0, *, allow=frozenset()) -> dict:
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                status = self.lines.get(timeout=remaining)
            except queue.Empty:
                break
            assert status["phase"] not in {"failed", "hold"} - allow, status
            if predicate(status):
                return status
        raise AssertionError(f"supervisor never reached the expected status: {self.process.poll()}")

    def stop(self, signum: int) -> int:
        self.process.send_signal(signum)
        code = self.process.wait(timeout=60)
        self.log.close()
        return code


def supervisor_processes():
    """Fixture body: start supervisors; reap any a failed assertion left running."""

    started: list[Supervisor] = []
    yield lambda root, config=None: started.append(Supervisor(root, config)) or started[-1]
    for supervisor in started:
        if supervisor.process.poll() is None:
            supervisor.process.kill()
            supervisor.process.wait(timeout=30)


def waiting(status: dict) -> bool:
    return status["last_disposition"] == "waiting" and status["request_id"] is not None


def carriers(root: Path, spool_name: str = "spool") -> list[Path]:
    outbox = root / spool_name / "outbox"
    return sorted(path for path in outbox.iterdir() if path.is_dir()) if outbox.is_dir() else []


def pod_request(root: Path, sealed, carrier: Path):
    """The pod side of one carrier: the authenticated request and its publication."""

    registration = sealed.registration
    identity = registration_transport_identity(registration)
    credential = registration_credential(registration, Path(registration["credential_path"]))
    request = spool.verify_request(spool.load_json(carrier / "request.json"), carrier, registration,
                                   identity=identity, credential=credential)
    wire = RemoteEvaluationRequest.from_dict(spool.authenticated_wire_request(
        request, carrier, identity=identity, credential=credential))
    require_commissioned_incumbent(wire.body, SimpleNamespace(
        incumbent_stack=sealed.incumbent, incumbent_tree_digest=sealed.incumbent_tree.tree_digest))
    (publication,) = resolve_cohort_publications(
        request, carrier, wire.body["candidates"], private(root / "pod" / "publications"))
    return SimpleNamespace(carrier=carrier, request=request, wire=wire, publication=publication,
                           identity=identity, credential=credential)


def pod_answer(root: Path, sealed, carrier: Path, decision: QualificationDecision, *, speedup: str = "1.06",
               incumbent_tree_digest: str | None = None, tamper=None) -> str:
    """Play the GPU pod: verify the authenticated request, then seal its verdict.

    ``speedup`` is the PASS's measured speedup over the commissioned baseline;
    ``incumbent_tree_digest`` plays a pod that measured against another incumbent tree;
    ``tamper`` rewrites the finalized result before RESULT_READY, as a hostile pod or shuttle would.
    """

    pod = pod_request(root, sealed, carrier)
    request, wire, publication = pod.request, pod.wire, pod.publication
    (reservation,) = (QualificationReservation.from_dict(row["reservation"]) for row in wire.body["candidates"])

    def pod_output(name: str) -> str:
        return canonical_digest("e2e.pod-output", {"field": name, "request": request["request_id"]})

    evidence = private(root / "pod" / "evidence")
    attempt = publish_evidence(evidence, canonical_json_bytes({
        "decision": decision.value, "request_id": request["request_id"],
        "speed_witness": {"resident_policy": {"version": 17}}}),  # the live paired-replay policy
        domain="qualification.cohort-attempt", media_type="application/json",
        schema="cacheon.qualification.cohort-attempt.v1")
    authority = QualificationAuthorityManifest(
        "registered", pod_output("authority"), pod_output("source"), pod_output("commitment"),
        pod_output("secret-reference"), (reservation.selected_delta_digest,), (reservation,))
    settlement = None
    if decision is QualificationDecision.PASS:
        catalog, context, incumbent = sealed.catalog, sealed.context, sealed.incumbent
        inspected = inspect_contribution(publication.root, catalog=catalog)
        contribution = ProposalContributionRef(
            reservation.target_id, inspected.target_spec_digest, publication.content_hash,
            inspected.selected_payload_digest, canonical_digest(ATTRIBUTION_SCHEMA, {
                "finalized_block": reservation.finalized_block,
                "finalized_event_index": reservation.finalized_event_index,
                "finalized_event_subindex": reservation.finalized_event_subindex,
                "hotkey": reservation.hotkey, "reservation_digest": reservation.reservation_digest,
                "submission_digest": reservation.submission_digest}))
        candidate = plan_candidate_stack(incumbent, contribution, catalog=catalog, expected_context=context)
        tree = materialize_engine_tree(
            candidate, context=context, catalog=catalog,
            resolver={("proposal", contribution.artifact_digest): publication.root},
            destination=root / "pod" / "trees" / candidate.digest)
        arm = plan_marginal_arm(incumbent, contribution, catalog=catalog,
                                incumbent_tree_digest=sealed.incumbent_tree.tree_digest,
                                candidate_tree_digest=tree.tree_digest, expected_context=context)
        audit = SlotAuditPolicy(pod_output("audit-seed")[:32], 100_000, 32, reservation.target_members, 1)
        settlement = SettlementQualification(
            lane="registered", arena_digest=incumbent.arena_digest,
            reservation_digest=reservation.reservation_digest, finalized_block=reservation.finalized_block,
            event_index=reservation.finalized_event_index, event_subindex=reservation.finalized_event_subindex,
            hotkey=reservation.hotkey, target_id=reservation.target_id, members=reservation.target_members,
            selected_delta_digest=reservation.selected_delta_digest,
            qualification_authority_digest=authority.digest, qualification_plan_digest=pod_output("plan"),
            qualification_attempt_digest=attempt.sha256, qualification_report_digest=pod_output("report"),
            selection_commitment_digest=authority.commitment_digest,
            selection_secret_commitment_digest=pod_output("secret-commitment"),
            selection_evidence_digest=pod_output("selection-evidence"), arm_digest=arm.digest,
            incumbent_stack_digest=arm.baseline_before.stack_digest,
            incumbent_tree_digest=arm.baseline_before.tree_digest,
            candidate_stack_digest=arm.challenger.stack_digest, candidate_tree_digest=arm.challenger.tree_digest,
            speedup=speedup, incumbent_manifest=incumbent, candidate_manifest=arm.candidate,
            audit_control_digest=audit.control.digest, audit_policy=audit,
            audit_evidence_digest=pod_output("audit-evidence"))
    outcome = QualificationIntakeOutcome(
        reservation.reservation_digest, reservation.selected_delta_digest, authority.digest, decision,
        "qualified" if settlement else "speed_regression", False, attempt_artifact_sha256=attempt.sha256,
        report_digest=pod_output("report"), settlement_qualification=settlement)
    product = capture_remote_qualification_product(
        batch=QualificationIntakeBatch(authority.digest, (outcome,), attempt), authority_manifest=authority,
        incumbent_stack=sealed.incumbent,
        incumbent_tree_digest=incumbent_tree_digest or sealed.incumbent_tree.tree_digest,
        screen_lane=wire.body["screen_lane"], service_digest=wire.body["service_digest"],
        readiness=WorkerReadiness(**sealed.registration["worker_readiness"]), evidence_root=evidence,
        evidence_references=(attempt,))
    write_response(pod, seal_remote_response(wire, product, pod.identity, pod.credential), tamper)
    return reservation.reservation_digest


def write_response(pod, response, tamper=None) -> None:
    """The adapter's terminal write: the sealed response, its journal, then RESULT_READY."""

    request_id = pod.request["request_id"]
    result = private(pod.carrier.parent.parent / "results" / request_id)  # the carrier's own spool
    (result / "response.json").write_bytes(spool.spool_canonical_json(response.to_dict()) + b"\n")
    append_event(journal_path(result), request_id, "adapter.terminal", "completed")
    spool.finalize_adapter_response(pod.request, pod.carrier, result, identity=pod.identity,
                                    credential=pod.credential)
    if tamper is not None:
        tamper(result)
    spool.atomic_bytes(result / "RESULT_READY", (request_id + "\n").encode(), mode=0o400)
