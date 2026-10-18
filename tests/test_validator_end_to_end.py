"""The validator, end to end on CPU, through its production entrypoints.

One private state root per arena profile holds what a deployment holds: sealed
miner archives, the chain, the intake SQLite authority, immutable publications,
the commissioned arena, the worker registration, the sealed dispatcher and
standing configs, the durable spool, and the evidence roots. Three miners
reveal in one finalized block: a verbatim copy of the public example, a bundle
the pod fails, and a bundle the pod passes.

Production processes, run as subprocesses:
``python -m cacheon.cli chain-package``, ``model-provision`` and
``set-weights --dry-run``; ``python -m cacheon.chain.remote_worker_service
make-registration``; ``python -m cacheon.chain.standing_cpu_supervisor
--config`` (killed mid-wait and restarted, then stopped with SIGTERM). The
contributor commands ``scan`` and ``verify`` run over ``examples/`` in their own test.

Production calls made in-process because the CPU host has no seam (owner
decision 2026-10-19): intake runs ``validator_loop.run_validator`` with its
HTTPS transfer leg replaced by the repository's hermetic
``fetch_bundle_from_local_file_for_testing``; the commission calls
``materialize_b300_identities`` through its own ``gpu_provisioner`` parameter,
because the CLI reads the fixed ``/usr/bin/nvidia-smi``.

Simulated boundaries: the chain (``tests/support/chain.py``, reached by
subprocesses through ``tests/support/simulated_chain``); the GPU pod and its SSH
shuttle, which this module plays with the shuttle's own liveness writer and by
writing the sealed response into the spool with ``seal_remote_response`` and
``finalize_adapter_response``; and the pod commissioning inputs the repository
does not produce (runtime preflight, READY receipt, GPU inventory, the private
authority and prompt configs). The model digests come from ``model-provision``.
Every value the CPU checks against the commissioned artifacts is read from
them: the genesis incumbent and its tree come from the commission's own
``_commissioned_stock_authority`` over the replayed composition, and the PASS
arm from ``inspect_contribution``, ``plan_marginal_arm`` and
``materialize_engine_tree`` over the carrier's publication. Pod outputs,
test-supplied and measurement-only: the verdict, speedup, audit policy and
evidence, the selection commitment and secret reference sealed by the pod
secret, the authority, source and plan digests, and the attempt and report bytes.

Known gap, left unasserted: the intake pass that demotes the verbatim example
writes no ``chain-audit.jsonl`` record. ``run_pass`` reports that copy as
``"validator_reference:<name>"``, ``pass_audit_record`` rejects the non-digest
value ("chain audit copies digest is malformed"), and ``run_validator`` logs the
append failure and continues. Reproduce by asserting one record after
``_intake`` below.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from nacl.public import PrivateKey

from cacheon.chain import fetch, validator_loop
from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.intake import FinalizedIntakeStore, IntakePolicy, IntakeScope
from cacheon.chain.payload import encode_payload
from cacheon.chain.qualification_request import require_commissioned_incumbent
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.remote_evaluation_dispatcher import (
    RemoteEvaluationRequest, capture_remote_qualification_product, seal_remote_response,
)
from cacheon.chain.remote_worker_registration import (
    registration_credential, registration_transport_identity, verify_registration,
)
from cacheon.chain.ssh_worker_transport import _dispatcher_heartbeat
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
from tests.support.chain import reopen_state, write_state
from tests.support.preflight import preflight_receipt

REPO = Path(__file__).resolve().parents[1]
SIMULATED_SDK = REPO / "tests" / "support" / "simulated_chain"
NETUID, REVEAL_BLOCK, HEAD = 1, 90, 100
MINERS = ("miner-copy", "miner-fail", "miner-pass")
BODIES = {  # Not copies of each other or of the public example: each must reach the pod.
    "miner-fail": "def forward(module, hidden_states, *args, **kwargs):\n"
                  "    return module.forward(hidden_states.contiguous(), *args, **kwargs)\n",
    "miner-pass": "import torch\n\n\ndef forward(module, *args, **kwargs):\n"
                  "    with torch.inference_mode():\n        staged = module.forward(*args, **kwargs)\n"
                  "    return staged\n",
}
PROFILES = {
    "b300-tp4": SimpleNamespace(gpu="b300", host=8, lane=(0, 1, 2, 3), arch="sm103", sglang="0.5.18",
                                model="MiniMax-M3", kwargs={"page_size": 128, "quantization": "modelopt_fp4"}),
    "h100-tp1": SimpleNamespace(gpu="h100", host=2, lane=(0,), arch="sm90", sglang="0.5.19",
                                model="Qwen3.6-35B-A3B-BF16", kwargs={"kv_cache_dtype": "fp8_e4m3"}),
}


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _sealed(path: Path, value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(raw)
    path.chmod(0o400)
    return sha256_hex(raw)


def _run(*argv: str, env: dict[str, str] | None = None) -> str:
    done = subprocess.run([sys.executable, "-m", *argv], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def _chain_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join((str(SIMULATED_SDK), str(REPO)))}


def _submit(root: Path, validator_key: PrivateKey) -> tuple[Path, dict[str, Path]]:
    """Miners package sealed archives and reveal them in one finalized block."""

    reveals, origin = {}, {}
    for hotkey in MINERS:
        bundle = root / "miners" / hotkey
        shutil.copytree(REPO / "examples" / "miner_node_identity", bundle)
        manifest = bundle / "manifest.toml"
        manifest.write_text(manifest.read_text().replace(
            'mode = "slot"', f'mode = "slot"\narena = "{root.name}"'))
        if hotkey in BODIES:
            (bundle / "kernels" / "forward.py").write_text(BODIES[hotkey])
        archive = bundle.with_suffix(".sealed")
        out = _run("cacheon.cli", "chain-package", str(bundle), "--out", str(archive),
                   "--encrypt-for", bytes(validator_key.public_key).hex())
        content_hash = out.split("content_hash: ")[1].split()[0]
        url = f"https://{hotkey}.example/{content_hash}.sealed"
        reveals[hotkey] = ((REVEAL_BLOCK, encode_payload(content_hash, url)),)
        origin[url] = archive
    state = write_state(root / "chain.json", hotkeys=("validator", *MINERS),
                        revealed=reveals, block=HEAD, validator="validator")
    return state, origin


def _intake(root: Path, state: Path, origin: dict[str, Path], monkeypatch) -> list[str]:
    """One finalized intake pass; only the HTTPS transfer leg is the hermetic one."""

    fetched: list[str] = []

    def transfer(url, expected, private_root):
        fetched.append(expected)
        return fetch.fetch_bundle_from_local_file_for_testing(origin[url].as_uri(), expected, private_root)

    monkeypatch.setattr(validator_loop, "fetch_bundle", transfer)
    validator_loop.run_validator(
        reopen_state(state), NETUID, intake_db=root / "validator" / "intake.sqlite3",
        private_root=root / "validator" / "private", publication_root=root / "validator" / "worker",
        once=True, audit_log=root / "validator" / "chain-audit.jsonl",
    )
    return fetched


def _commission(root: Path, profile) -> tuple[dict, Path, dict]:
    """The pod commission's sealed inputs, materialized into the arena identity."""

    pod = _private(root / "pod")
    model = _private(pod / "model")
    (model / "config.json").write_text('{"architectures": ["E2EForCausalLM"]}\n')
    sealed = json.loads(_run("cacheon.cli", "model-provision", str(model), str(_private(pod / "receipts"))))
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
    _sealed(projection, {"schema": "cacheon-calibration-projection-v1"})
    authority = {
        "arena_id": root.name, "authority_role": "primary",
        "calibration": {"evidence_root": str(pod / "calibration-evidence"), "package": str(calibration),
                        "package_sha256": _sealed(calibration, {"schema": "cacheon-calibration-v1"})},
        "device_execution": {"path": str(device), "sha256": _sealed(device, {
            "schema": "cacheon-private-device-control-v1",
            "worker": {"preflight": preflight.canonical_payload()}})},
        "fmha_cache_seed": {"directory_count": 2, "file_count": 3, "plan_sha256": sha256_hex(b"fmha-plan"),
                            "root": str(pod / "fmha-cache"), "schema": "cacheon-fmha-cache-v1",
                            "sealed_root_owned_read_only": True, "total_bytes": 4096,
                            "tree_sha256": sha256_hex(b"fmha-tree")},
        "model": {"content_digest": sealed["content_digest"], "manifest_digest": sealed["receipt_digest"],
                  "revision_digest": sha256_hex(profile.model.encode()), "root": str(model)},
        "prompt": {"path": str(prompt), "sha256": _sealed(prompt, {
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
    _sealed(authority_path, authority)
    _sealed(measurement, authority)
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
    ready["receipt_digest"] = spool.spool_digest("cacheon.current-pod-commission.v1", ready)
    ready_path = pod / "ready-receipt.json"
    _sealed(ready_path, ready)
    materialized = deployment.materialize_b300_identities(
        ready_receipt=ready_path, authority_config=authority_path, measurement_config=measurement,
        calibration_package=calibration, calibration_projection_receipt=projection,
        prompt_authority=prompt, output_root=_private(pod / "commissioned"),
        gpu_provisioner=lambda selected, *, deadline: tuple(gpus[index] for index in selected),
    )
    return materialized, ready_path, ready


def _seal_authority(root: Path, materialized: dict, ready_path: Path, ready: dict):
    """The operator registers the worker and seals the standing composition."""

    secret = root / "credential.secret"
    secret.write_bytes(os.urandom(32))
    secret.chmod(0o400)
    known_hosts = root / "known_hosts"
    known_hosts.write_text("pod.example ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE2E\n")
    known_hosts.chmod(0o600)
    manifest = json.loads(Path(materialized["arena_service_manifest"]).read_text())
    service_id = deployment._manifest_from_dict(manifest).service_id
    registration_path = root / "registration.json"
    _run("cacheon.chain.remote_worker_service", "make-registration", "--ready-receipt", str(ready_path),
         "--worker-readiness", materialized["worker_readiness"], "--known-hosts", str(known_hosts),
         "--pod-host", "pod.example", "--pod-port", "22", "--service-identity", service_id,
         "--remote-service", str(REPO / "cacheon/chain/remote_worker_service.py"),
         "--adapter", str(REPO / "cacheon/eval/b300_remote_worker_adapter.py"),
         "--credential", str(secret), "--credential-id", "e2e-qualification-v1",
         "--output", str(registration_path))
    registration = verify_registration(spool.load_json(registration_path))
    inputs, composition, _ = deployment.replay_commissioned_composition(
        registration, ready, commissioned_root=Path(materialized["deployment"]).parent)
    catalog = default_target_catalog()
    _, context, incumbent, tree = deployment._commissioned_stock_authority(
        inputs, composition.manifest, catalog, catalog.snapshot(),
        error=deployment.B300DeploymentError, label="end-to-end genesis")
    validator = root / "validator"
    _sealed(validator / "incumbent-stack.json", incumbent.to_dict())
    scope = IntakeScope(reopen_state(root / "chain.json").get_block_hash(0), NETUID)
    policy = IntakePolicy()
    _sealed(validator / "dispatcher.json", {
        "accept_legacy_bundles": False, "arena_service_manifest": manifest,
        "credential_digest": registration["credential_digest"], "credential_path": str(secret),
        "heartbeat_interval_ms": 1000, "heartbeat_join_timeout_ms": 2000, "idle_poll_ms": 100,
        "intake_db": str(validator / "intake.sqlite3"),
        "intake_policy": {name: getattr(policy, name) for name in policy.__dataclass_fields__},
        "intake_scope": scope.to_dict(), "lease_blocks": 20, "lock_attempts": 20, "lock_retry_delay_ms": 50,
        "owner": f"e2e-{root.name}", "registration_digest": registration["registration_digest"],
        "registration_path": str(registration_path), "response_timeout_seconds": 1,
        "restart_initial_backoff_ms": 100, "restart_max_backoff_ms": 400,
        "schema": "cacheon-mainnet-screen-dispatcher-config-v1", "spool_root": str(_private(root / "spool")),
        "transport_identity_digest": registration["transport_identity_digest"], "transport_poll_seconds": 1,
        "worker_readiness": json.loads(Path(materialized["worker_readiness"]).read_text()),
    })
    _sealed(validator / "standing.json", {
        "enable_qualification": True, "enable_settlement": True, "enable_weights": False,
        "idle_poll_ms": 100, "pruning_review_root": "",
        "qualification_evidence_root": str(_private(validator / "evidence")),
        "qualification_incumbent_stack_path": str(validator / "incumbent-stack.json"),
        "qualification_incumbent_tree_digest": tree.tree_digest,
        "restart_initial_backoff_ms": 100, "restart_max_backoff_ms": 400,
        "schema": "cacheon-standing-supervisor-config-v1",
        "screen_dispatcher_config": str(validator / "dispatcher.json"),
        "settlement_network": str(root / "chain.json"), "stall_timeout_ms": 600_000,
        "weights_stage_config": "",
    })
    return SimpleNamespace(registration=registration, catalog=catalog, context=context,
                           incumbent=incumbent, incumbent_tree=tree, scope=scope, policy=policy)


class _Supervisor:
    """The standing CPU supervisor process, observed through its status lines."""

    def __init__(self, root: Path) -> None:
        self.log = (root / "supervisor.stderr").open("a")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "cacheon.chain.standing_cpu_supervisor",
             "--config", str(root / "validator" / "standing.json")],
            cwd=REPO, env=_chain_env(), stdout=subprocess.PIPE, stderr=self.log, text=True)
        self.lines: queue.Queue[dict] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.process.stdout:
            self.lines.put(json.loads(line))

    def until(self, predicate, timeout: float = 90.0) -> dict:
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            try:
                status = self.lines.get(timeout=remaining)
            except queue.Empty:
                break
            assert status["phase"] not in {"failed", "hold"}, status
            if predicate(status):
                return status
        raise AssertionError(f"supervisor never reached the expected status: {self.process.poll()}")

    def stop(self, signum: int) -> int:
        self.process.send_signal(signum)
        code = self.process.wait(timeout=60)
        self.log.close()
        return code


@pytest.fixture
def supervisors():
    """Start supervisors; reap any a failed assertion left running."""

    started: list[_Supervisor] = []
    yield lambda root: started.append(_Supervisor(root)) or started[-1]
    for supervisor in started:
        if supervisor.process.poll() is None:
            supervisor.process.kill()
            supervisor.process.wait(timeout=30)


def _waiting(status: dict) -> bool:
    return status["last_disposition"] == "waiting" and status["request_id"] is not None


def _carriers(root: Path) -> list[Path]:
    return sorted(path for path in (root / "spool" / "outbox").iterdir() if path.is_dir())


def _pod_answer(root: Path, sealed, carrier: Path, decision: QualificationDecision) -> str:
    """Play the GPU pod: verify the authenticated request, then seal its verdict."""

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
        request, carrier, wire.body["candidates"], _private(root / "pod" / "publications"))
    (reservation,) = (QualificationReservation.from_dict(row["reservation"]) for row in wire.body["candidates"])

    def pod_output(name: str) -> str:
        return canonical_digest("e2e.pod-output", {"field": name, "request": request["request_id"]})

    evidence = _private(root / "pod" / "evidence")
    attempt = publish_evidence(evidence, canonical_json_bytes({
        "decision": decision.value, "request_id": request["request_id"],
        "speed_witness": {"resident_policy": {"min_margin": "0.01"}}}),
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
            speedup="1.06", incumbent_manifest=incumbent, candidate_manifest=arm.candidate,
            audit_control_digest=audit.control.digest, audit_policy=audit,
            audit_evidence_digest=pod_output("audit-evidence"))
    outcome = QualificationIntakeOutcome(
        reservation.reservation_digest, reservation.selected_delta_digest, authority.digest, decision,
        "qualified" if settlement else "speed_regression", False, attempt_artifact_sha256=attempt.sha256,
        report_digest=pod_output("report"), settlement_qualification=settlement)
    product = capture_remote_qualification_product(
        batch=QualificationIntakeBatch(authority.digest, (outcome,), attempt), authority_manifest=authority,
        incumbent_stack=sealed.incumbent, incumbent_tree_digest=sealed.incumbent_tree.tree_digest,
        screen_lane=wire.body["screen_lane"], service_digest=wire.body["service_digest"],
        readiness=_readiness(registration), evidence_root=evidence, evidence_references=(attempt,))
    result = _private(root / "spool" / "results" / request["request_id"])
    response = seal_remote_response(wire, product, identity, credential)
    (result / "response.json").write_bytes(spool.spool_canonical_json(response.to_dict()) + b"\n")
    append_event(journal_path(result), request["request_id"], "adapter.terminal", "completed")
    spool.finalize_adapter_response(request, carrier, result, identity=identity, credential=credential)
    spool.atomic_bytes(result / "RESULT_READY", (request["request_id"] + "\n").encode(), mode=0o400)
    return reservation.reservation_digest


def _readiness(registration: dict):
    from cacheon.chain.evaluation_coordinator import WorkerReadiness

    return WorkerReadiness(**registration["worker_readiness"])


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_finalized_submissions_qualify_settle_and_project_weights(tmp_path, monkeypatch, supervisors, profile):
    started = time.monotonic()
    root = _private(tmp_path / f"e2e-{profile}")
    validator_key = PrivateKey.generate()
    key_file = root / "bundle-decryption.key"
    key_file.write_text(bytes(validator_key).hex())
    key_file.chmod(0o600)
    monkeypatch.setenv("CACHEON_BUNDLE_DECRYPTION_KEY", str(key_file))
    _private(root / "validator")

    # Intake: finalized order, private fetch, immutable publication, copy demotion.
    state, origin = _submit(root, validator_key)
    fetched = _intake(root, state, origin, monkeypatch)
    assert len(fetched) == 3
    scope = IntakeScope(reopen_state(state).get_block_hash(0), NETUID)
    with FinalizedIntakeStore(root / "validator" / "intake.sqlite3", scope=scope) as store:
        rows = {row.arrival.hotkey: row for row in store.all()}
    assert sorted(rows) == sorted(MINERS)
    assert rows["miner-copy"].status == "failed" and "validator_reference" in rows["miner-copy"].reason
    assert {rows[name].status for name in BODIES} == {"published"}
    assert [rows[name].arrival.event_index for name in MINERS] == [0, 1, 2]
    assert _intake(root, state, origin, monkeypatch) == []  # restart: nothing refetched or re-reserved

    # Commission and registration from the produced artifacts only.
    materialized, ready_path, ready = _commission(root, PROFILES[profile])
    sealed = _seal_authority(root, materialized, ready_path, ready)
    assert sealed.registration["service_identity"] == deployment._manifest_from_dict(
        json.loads(Path(materialized["arena_service_manifest"]).read_text())).service_id
    assert sealed.registration["worker_readiness_digest"] == materialized["worker_readiness_digest"]

    # Qualification: FIFO head claimed, one carrier; a crash mid-wait reopens the same request.
    # The shuttle's own liveness writer stands in for the SSH relay process.
    heartbeat = _private(root / "spool" / "state") / "heartbeat.json"
    with _dispatcher_heartbeat(sealed.registration, heartbeat, lambda: None):
        supervisor = supervisors(root)
        first = supervisor.until(_waiting)
        assert len(_carriers(root)) == 1
        assert supervisor.stop(signal.SIGKILL) == -signal.SIGKILL
        supervisor = supervisors(root)
        assert supervisor.until(_waiting)["request_id"] == first["request_id"]
        assert len(_carriers(root)) == 1
        failed = _pod_answer(root, sealed, _carriers(root)[0], QualificationDecision.FAIL)
        second = supervisor.until(lambda status: _waiting(status) and status["request_id"] != first["request_id"])
        carriers = _carriers(root)
        assert [path.name.split("-", 1)[1] for path in carriers] == [first["request_id"], second["request_id"]]
        passed = _pod_answer(root, sealed, carriers[-1], QualificationDecision.PASS)
        supervisor.until(lambda status: status["last_stage"] == "settlement"
                         and status["last_disposition"] == "committed")
        supervisor.until(lambda status: status["phase"] == "idle")
        assert supervisor.stop(signal.SIGTERM) == 0

    with RecoverableFinalizedIntakeStore(
        root / "validator" / "intake.sqlite3", sealed.policy, scope=scope,
    ) as store:
        assert rows["miner-fail"].reservation_id == failed and store.get(failed).decision == "FAIL"
        assert store.get(failed).status == "failed"
        assert rows["miner-pass"].reservation_id == passed and store.get(passed).decision == "PASS"
        assert store.get(passed).status == "qualified"
        assert store.qualification_attempts(failed) == store.qualification_attempts(passed) == 1
        stack = store.evaluation_stack(sealed.incumbent.arena_digest)
        assert stack.manifest.digest != sealed.incumbent.digest
        assert stack.manifest.entries["forward_pass"].artifact_digest == rows["miner-pass"].arrival.content_hash
        assert store.pending_qualification_recovery() is None

    # Weights: the signer projects the crowned hotkey over the same durable store.
    out = _run("cacheon.cli", "set-weights", "--dry-run", "--intake-db", str(root / "validator" / "intake.sqlite3"),
               "--netuid", str(NETUID), "--network", str(state), "--wallet-path", str(state),
               "--half-life-blocks", "100", "--discovery-lifetime-blocks", "20",
               "--discovery-pool-ppm", "100000", "--refresh-blocks", "10", env=_chain_env())
    uid = ("validator", *MINERS).index("miner-pass")
    assert f"rows=[{{'uid': {uid}, 'weight': 1.0, 'hotkey': 'miner-pass'}}]" in out, out
    assert "status=dry_run" in out and "submitted=False" in out, out
    with FinalizedIntakeStore(root / "validator" / "intake.sqlite3", scope=scope) as store:
        assert store._db.execute("SELECT COUNT(*) FROM weight_publications").fetchone()[0] == 0
    assert time.monotonic() - started < 60


def test_contributor_scan_and_verify_give_the_documented_verdicts(tmp_path):
    """The miner guide's commands: exit 0 with clean and interface verdicts, exit 2 on a FAIL."""
    assert "[clean] model.layers.*.mlp" in _run("cacheon.cli", "scan", "examples/miner_node_identity")
    assert "[INTERFACE OK] model.layers.*.mlp" in _run("cacheon.cli", "verify", "examples/miner_node_identity")
    broken = tmp_path / "broken"
    shutil.copytree(REPO / "examples" / "miner_node_identity", broken)
    (broken / "kernels" / "forward.py").write_text("def forward():\n    return None\n")
    done = subprocess.run([sys.executable, "-m", "cacheon.cli", "verify", str(broken)], cwd=REPO,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 2 and "[FAIL]" in done.stderr, done.stdout + done.stderr
