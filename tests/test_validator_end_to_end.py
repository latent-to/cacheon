"""The validator, end to end on CPU, through its production entrypoints.

One private state root per arena profile holds what a deployment holds: sealed
miner archives, the chain, the intake SQLite authority, immutable publications,
the commissioned arena, the worker registration, the sealed dispatcher and
standing configs, the durable spool, and the evidence roots. Three miners
reveal in one finalized block: a verbatim copy of the public example, a bundle
the pod fails, and a bundle the pod passes. ``tests/support/e2e.py`` builds the
root; the hostile-intake and qualification-fault modules reuse it.

Production processes, run as subprocesses:
``python -m cacheon.cli chain-package``, ``model-provision`` and
``set-weights --dry-run``; ``python -m cacheon.chain.remote_worker_service
make-registration``; ``python -m cacheon.chain.standing_cpu_supervisor
--config`` (killed mid-wait, restarted on the store downgraded to lease and
recovery schema 2, which it migrates, then stopped with SIGTERM); the
first intake pass, ``chain-validate --intake-only --once`` entered through
``cacheon.cli.main`` in its own process. The contributor commands ``scan`` and
``verify`` run over ``examples/`` in their own test, and so do the commission's
refusals of drifted READY, device, prompt and registration inputs.

Production calls made in-process because the CPU host has no seam (owner
decision 2026-10-19): intake's HTTPS transfer leg is the repository's hermetic
``fetch_bundle_from_local_file_for_testing`` (swapped inside the intake process),
and the restart pass runs ``validator_loop.run_validator`` here; the commission
calls ``materialize_b300_identities`` through its own ``gpu_provisioner``
parameter, because the CLI reads the fixed ``/usr/bin/nvidia-smi``.

The trusted controller never runs candidate code: the intake process and the
supervisor run with ``-X importtime`` and with the passing bundle armed in
their environment (``CACHEON_BUNDLE_PATH``, ``CACHEON_ACTIVE``). Both load the
launch, qualification and stack identity modules, and neither imports torch,
SGLang, Triton or FlashInfer. The passing bundle's top-level ``import torch``
is the witness: however a controller loaded that bundle, the import would be
logged.

Simulated boundaries: the chain (``tests/support/chain.py``, reached by
subprocesses through ``tests/support/simulated_chain``); the GPU pod and its SSH
shuttle, which these modules play with the shuttle's own liveness writer and by
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
append failure and continues. Reproduce by asserting one record after the
first intake pass below.
"""

from __future__ import annotations

import json
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.chain.intake import FinalizedIntakeStore
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.ssh_worker_transport import _dispatcher_heartbeat
from cacheon.eval import b300_deployment as deployment
from cacheon.eval.qualification import QualificationDecision
from cacheon.stack_identity import sha256_hex
from tests.support import e2e
from tests.support.b300 import gpu
from tests.support.pod import pod_answer

supervisors = pytest.fixture(e2e.supervisor_processes, name="supervisors")
MINERS = ("miner-copy", "miner-fail", "miner-pass")
BODIES = {  # Not copies of each other or of the public example: each must reach the pod.
    "miner-fail": "def forward(module, hidden_states, *args, **kwargs):\n"
                  "    return module.forward(hidden_states.contiguous(), *args, **kwargs)\n",
    "miner-pass": "import torch\n\n\ndef forward(module, *args, **kwargs):\n"
                  "    with torch.inference_mode():\n        staged = module.forward(*args, **kwargs)\n"
                  "    return staged\n",
}
LOADED = {"cacheon.eval.engine_launch", "cacheon.eval.qualification", "cacheon.stack_manifest"}
# importtime logs import statements, not importlib loads: a candidate run in the controller shows as miner-pass's torch.
GPU_STACKS = {"torch", "sglang", "triton", "flashinfer"}
INTAKE = ("import json, sys\nfrom cacheon.chain import fetch, validator_loop\nfrom cacheon.cli import main\n"
          "origin = json.loads(sys.argv[1])\nvalidator_loop.fetch_bundle = lambda url, expected, root: "
          "fetch.fetch_bundle_from_local_file_for_testing(origin[url], expected, root)\nsys.exit(main(sys.argv[2:]))\n")
STAMPS = ("evaluation_lease_schema", "evaluation_recovery_schema")
SINGLETONS = ("evaluation_leases_one_active_qualification", "evaluation_recoveries_one_unresolved")
# The store a pre-worker-scope controller (lease and recovery schema 2) left: one arena-wide active lease and one
# unresolved recovery, where version 3 keys them per worker and per lease.
SCHEMA_2 = (f"DROP INDEX {SINGLETONS[0]}; DROP INDEX {SINGLETONS[1]};"
            f"CREATE UNIQUE INDEX {SINGLETONS[0]} ON evaluation_leases(stage) WHERE state='active' AND stage='qualification';"
            f"CREATE UNIQUE INDEX {SINGLETONS[1]} ON evaluation_recoveries(resolution) WHERE resolution='';"
            f"UPDATE metadata SET value='2' WHERE key IN {STAMPS}")
LANE = "provisioned GPU set differs from the commissioned lane"
COMMISSION_REFUSALS = (  # (label, READY edits as (*keys, value), provisioned devices, reason) on the 8-device B300 host
    ("short-lane", (), lambda gpus, selected: gpus[:3], LANE),
    ("device-id-drift", (), lambda gpus, selected: gpus[:3] + (gpu(9),), LANE),
    ("device-model-drift", (), lambda gpus, selected: gpus[:3] + (replace(gpus[3], name="NVIDIA H100 SXM5"),),
     "provisioned GPU configuration differs from READY inventory"),
    ("lane-outside-inventory", (("lane", "devices", [8, 9, 10, 11]),), None,
     "READY lane lies outside its inventory"),
    ("ready-count", (("gpu", "count", 7),), None, "READY GPU count differs from its inventory"),
    ("ready-duplicate-index", (("gpu", "inventory", 7, "index", 6),), None, "ordered unique GPU indices"),
    ("ready-unordered-indices", (("gpu", "inventory", 0, "index", 1), ("gpu", "inventory", 1, "index", 0)), None,
     "ordered unique GPU indices"),
)


def _imported(stderr: str) -> set[str]:
    """The modules a ``-X importtime`` controller process imported, after checking it loaded no GPU stack."""

    names = {line.rsplit("|", 1)[1].strip() for line in stderr.splitlines() if line.startswith("import time:")}
    assert sorted(name for name in names if name.split(".")[0] in GPU_STACKS) == []
    return names


def _submit(root: Path, validator_key, native_prebuild: bool) -> tuple[Path, dict[str, Path]]:
    """Miners package sealed archives and reveal them in one finalized block."""

    reveals, origin = {}, {}
    for hotkey in MINERS:
        bundle = e2e.miner_bundle(root, hotkey, BODIES.get(hotkey))
        if hotkey == "miner-pass":
            (bundle / "kernels/kernel.cuh").write_text("#define JIT_VALUE 1\n")
            manifest = bundle / "manifest.toml"
            sources = ["kernels/kernel.cuh"]
            if native_prebuild:
                sources.append("kernels/native.cu")
                (bundle / sources[-1]).write_text('#include "kernel.cuh"\n')
                (bundle / "rebuild.json").write_text(
                    '{"steps":[{"type":"repo_python","path":"build_cuda_ext.py"}]}\n')
            manifest.write_text(manifest.read_text() + f"cuda_sources = {json.dumps(sources)}\n")
        content_hash, archive = e2e.package(bundle, validator_key)
        reveals[hotkey] = (e2e.reveal(hotkey, content_hash, archive, origin),)
    return e2e.chain(root, reveals, MINERS), origin


def _store_schema(root: Path, script: str = "") -> tuple[str, ...]:
    """Run ``script`` on the root's intake store in place; return its lease and recovery stamps and singleton indexes."""

    with closing(sqlite3.connect(root / "validator" / "intake.sqlite3")) as db:
        db.executescript(script)
        stamps = db.execute("SELECT value FROM metadata WHERE key IN (?, ?) ORDER BY key", STAMPS).fetchall()
        indexes = db.execute("SELECT sql FROM sqlite_master WHERE name IN (?, ?) ORDER BY name", SINGLETONS).fetchall()
        return (*(row[0] for row in stamps), *(" ".join(row[0].split()).split(" ON ")[1] for row in indexes))


@pytest.mark.parametrize("profile", sorted(e2e.PROFILES))
@pytest.mark.parametrize("native_prebuild", [False, True])
def test_finalized_submissions_qualify_settle_and_project_weights(
    tmp_path, monkeypatch, supervisors, profile, native_prebuild,
):
    started = time.monotonic()
    root, validator_key = e2e.deployment_root(tmp_path, f"e2e-{profile}", monkeypatch)

    # Intake: finalized order, private fetch, immutable publication, copy demotion, in the CLI's own process.
    state, origin = _submit(root, validator_key, native_prebuild)
    armed = {"CACHEON_BUNDLE_PATH": str(root / "miners" / "miner-pass"), "CACHEON_ACTIVE": "1"}
    validator = root / "validator"
    intake = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", INTAKE, json.dumps({u: p.as_uri() for u, p in origin.items()}),
         "chain-validate", "--intake-only", "--once", "--netuid", str(e2e.NETUID), "--network", str(state),
         "--intake-db", str(validator / "intake.sqlite3"), "--private-root", str(validator / "private"),
         "--publication-root", str(validator / "worker"), "--audit-log", str(validator / "chain-audit.jsonl")],
        cwd=e2e.REPO, env={**e2e.chain_env(), **armed}, capture_output=True, text=True, timeout=300)
    assert intake.returncode == 0 and "seen=3 reserved=3 published=2 copies=1 rejected=0" in intake.stdout, intake
    assert LOADED <= _imported(intake.stderr)
    scope = e2e.scope(root)
    with FinalizedIntakeStore(root / "validator" / "intake.sqlite3", scope=scope) as store:
        rows = {row.arrival.hotkey: row for row in store.all()}
    assert sorted(rows) == sorted(MINERS)
    assert rows["miner-copy"].status == "failed" and "validator_reference" in rows["miner-copy"].reason
    assert {rows[name].status for name in BODIES} == {"published"}
    assert [rows[name].arrival.event_index for name in MINERS] == [0, 1, 2]
    assert e2e.intake(root, state, origin, monkeypatch) == []  # restart: nothing refetched or re-reserved

    # Commission and registration from the produced artifacts only.
    materialized, ready_path, ready = e2e.commission(root, e2e.PROFILES[profile])
    sealed = e2e.seal_authority(root, materialized, ready_path, ready)
    assert sealed.registration["service_identity"] == deployment._manifest_from_dict(
        json.loads(Path(materialized["arena_service_manifest"]).read_text())).service_id
    assert sealed.registration["worker_readiness_digest"] == materialized["worker_readiness_digest"]

    # Qualification: FIFO head claimed, one carrier; a crash mid-wait reopens the same request.
    # The shuttle's own liveness writer stands in for the SSH relay process.
    heartbeat = e2e.private(root / "spool" / "state") / "heartbeat.json"
    for name, value in {**armed, "PYTHONPROFILEIMPORTTIME": "1"}.items():
        monkeypatch.setenv(name, value)
    with _dispatcher_heartbeat(sealed.registration, heartbeat, lambda: None):
        supervisor = supervisors(root)
        first = supervisor.until(e2e.waiting)
        assert len(e2e.carriers(root)) == 1
        assert supervisor.stop(signal.SIGKILL) == -signal.SIGKILL
        # The restart reopens a schema-2 store in place, migrates it to version 3 and resumes the same request.
        assert _store_schema(root, SCHEMA_2) == ("2", "2", "evaluation_leases(stage) WHERE state='active' AND "
                                                 "stage='qualification'", "evaluation_recoveries(resolution) WHERE resolution=''")
        supervisor = supervisors(root)
        assert supervisor.until(e2e.waiting)["request_id"] == first["request_id"]
        assert _store_schema(root) == ("3", "3", "evaluation_leases(competition_arena,owner) WHERE state='active' AND "
                                       "stage='qualification'", "evaluation_recoveries(lease_id) WHERE resolution=''")
        assert len(e2e.carriers(root)) == 1
        failed = pod_answer(root, sealed, e2e.carriers(root)[0], QualificationDecision.FAIL)
        second = supervisor.until(lambda status: e2e.waiting(status) and status["request_id"] != first["request_id"])
        carriers = e2e.carriers(root)
        assert [path.name.split("-", 1)[1] for path in carriers] == [first["request_id"], second["request_id"]]
        passed = pod_answer(root, sealed, carriers[-1], QualificationDecision.PASS)
        supervisor.until(lambda status: status["last_stage"] == "settlement"
                         and status["last_disposition"] == "committed")
        supervisor.until(lambda status: status["phase"] == "idle")
        assert supervisor.stop(signal.SIGTERM) == 0
    assert LOADED <= _imported((root / "supervisor.stderr").read_text())

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

    # A library JIT finds its declared header relative to the materialized Python module's __file__.
    (header,) = (root / "pod" / "trees" / stack.manifest.digest).glob("cacheon_c_*/kernels/kernel.cuh")
    assert header.read_text() == "#define JIT_VALUE 1\n" and header.with_name("forward.py").is_file()
    from cacheon.manifest import all_declared_cuda_sources, load_manifest
    from cacheon.sandbox import scan_tree

    tree = header.parent.parent.parent
    assert scan_tree(tree, declared_cuda_sources=all_declared_cuda_sources(tree, load_manifest(tree))).ok
    if native_prebuild:
        from cacheon.patchers.build_cuda_ext import _declared_sources

        # Muse prefill capture (2026-10-24): a prebuild must not remove a library JIT's relative inputs.
        units, inventory, selected = _declared_sources(tree, e2e.PROFILES[profile].arch)
        assert len(units) == 1 and len(inventory) == len(selected) == 4
        source = units[0][0]
        assert source.relative_to(tree).parts[0] == "cuda"
        assert source.with_name("kernel.cuh").read_bytes() == header.read_bytes()
        assert header.with_name("native.cu").read_bytes() == source.read_bytes()
    else:
        _check_jit_only_prebuild(root, header, stack.manifest.digest, profile)

    # Weights: the signer projects the crowned hotkey over the same durable store.
    sign = ("cacheon.cli", "set-weights", "--dry-run", "--intake-db", str(root / "validator" / "intake.sqlite3"),
            "--netuid", str(e2e.NETUID), "--network", str(state), "--wallet-path", str(state),
            "--half-life-blocks", "100", "--discovery-lifetime-blocks", "20",
            "--discovery-pool-ppm", "100000", "--refresh-blocks", "10")
    out = e2e.run(*sign, env=e2e.chain_env())
    uid = ("validator", *MINERS).index("miner-pass")
    assert f"rows=[{{'uid': {uid}, 'weight': 1.0, 'hotkey': 'miner-pass'}}]" in out, out
    assert "status=dry_run" in out and "submitted=False" in out, out
    with FinalizedIntakeStore(root / "validator" / "intake.sqlite3", scope=scope) as store:
        assert store._db.execute("SELECT COUNT(*) FROM weight_publications").fetchone()[0] == 0
    # A stamp no controller wrote refuses the same reader, with its reason.
    _store_schema(root, "UPDATE metadata SET value='999' WHERE key='evaluation_lease_schema'")
    assert "evaluation lease schema is unsupported" in e2e.run(*sign, env=e2e.chain_env(), code=1)
    assert time.monotonic() - started < 60


def _check_jit_only_prebuild(root, header, stack_digest, profile):
    """Exercise the real prebuild worker without native compilation or candidate imports."""
    # Muse k10 (2026-10-21): declared JIT headers must pass prebuild without a separate rebuild plan.
    tree, native = header.parent.parent.parent, e2e.private(root / "jit-prebuild")
    program = ("import json,os,sys; from cacheon.eval import oci_prebuild as p; "
               "from cacheon.engine_tree import reopen_materialized_engine_tree; "
               "p.CONTAINER_TREE,p.CONTAINER_STAGE=sys.argv[1:]; "
               "os.environ['CACHEON_ENGINE_TREE_DIGEST']=reopen_materialized_engine_tree(p.CONTAINER_TREE).tree_digest; "
               "print(p.container_build().read_text())")
    checked = subprocess.run([sys.executable, "-B", "-c", program, str(tree), str(native)],
        capture_output=True, text=True, timeout=30, env={**e2e.chain_env(),
            "CACHEON_NATIVE_BUILD_SPEC_DIGEST": stack_digest,
            "CACHEON_TARGET_GPU_ARCH": e2e.PROFILES[profile].arch,
            "CACHEON_NATIVE_ARTIFACT_STAGE": str(native), "CACHEON_PINNED_BUILD_ROOTS": "/usr",
            "CACHEON_BUILD_PATH": "/usr/bin", "CACHEON_BUILD_TMPDIR": "/tmp",
            "CACHEON_NATIVE_COMPILE_TIMEOUT_S": "10", "CACHEON_REBUILD_CONTAINER": "1"})
    assert checked.returncode == 0, checked.stdout + checked.stderr
    built = json.loads(checked.stdout)
    assert built["rebuild_applied"] is False and built["stage_entries"] == ["prebuild.json"]

def test_commission_refuses_each_drift_from_its_sealed_inputs(tmp_path, monkeypatch):
    """The pod commission (``materialize_b300_identities``, then the registration replay) refuses each drift by name."""

    profile = e2e.PROFILES["b300-tp4"]
    root, _ = e2e.deployment_root(tmp_path, "commission", monkeypatch)
    e2e.chain(root, {}, ())  # the registration's scope
    materialized, ready_path, ready = e2e.commission(root, profile)
    pod, deployed = root / "pod", lambda result: json.loads(Path(result["deployment"]).read_text())
    lane = deployed(materialized)["declared_qualification"]["lane"]
    assert lane["physical_gpu_ids"] == [0, 1, 2, 3]
    for label, edits, devices, reason in COMMISSION_REFUSALS:
        drifted = json.loads(json.dumps({key: value for key, value in ready.items() if key != "receipt_digest"}))
        for *keys, last, value in edits:
            node = drifted
            for key in keys:
                node = node[key]
            node[last] = value
        path = e2e._seal_ready(pod / f"ready-{label}.json", drifted) if edits else ready_path
        with pytest.raises(deployment.B300DeploymentError, match=reason):
            e2e._materialize(root, profile, path, pod / f"commissioned-{label}", devices)

    # A registration naming another service identity does not replay.
    sealed = e2e.seal_authority(root, materialized, ready_path, ready)
    with pytest.raises(deployment.B300DeploymentError, match="registration differs"):
        deployment.replay_commissioned_composition({**sealed.registration, "service_identity": materialized[
            "service_digest"]}, ready, commissioned_root=Path(materialized["deployment"]).parent)
    # Calibration names the arena manifest, never the service identity: a new package keeps the service digest.
    package = e2e.reseal(pod / "calibration-package.json", pod / "calibration-package.json", generation=2)
    for config in (pod / "authority-config.json", pod / "measurement-config.json"):
        calibration = json.loads(config.read_text())["calibration"]
        e2e.reseal(config, config, calibration={**calibration, "package_sha256": sha256_hex(package.read_bytes())})
    recalibrated = e2e._materialize(root, profile, ready_path, pod / "commissioned-recalibrated")
    assert recalibrated["service_digest"] == materialized["service_digest"]
    assert deployed(recalibrated)["authorities"]["calibration_package"]["sha256"] != deployed(materialized)[
        "authorities"]["calibration_package"]["sha256"]
    # A sealed prompt changed after its digest was taken, and a prompt naming a target the catalog lacks.
    (pod / "prompt-authority.json").chmod(0o600)
    with (pod / "prompt-authority.json").open("a") as prompt:
        prompt.write(" ")
    with pytest.raises(deployment.B300DeploymentError, match="explicit sealed authority paths or SHA-256"):
        e2e._materialize(root, profile, ready_path, pod / "commissioned-prompt")
    with pytest.raises(deployment.B300DeploymentError, match="not registered"):
        e2e.commission(e2e.deployment_root(tmp_path, "unregistered", monkeypatch)[0], profile, ("made.up_target",))


def test_contributor_scan_and_verify_give_the_documented_verdicts(tmp_path):
    """The miner guide's commands: exit 0 with clean and interface verdicts, exit 2 on a FAIL."""
    assert "[clean] model.layers.*.mlp" in e2e.run("cacheon.cli", "scan", "examples/miner_node_identity")
    assert "[INTERFACE OK] model.layers.*.mlp" in e2e.run("cacheon.cli", "verify", "examples/miner_node_identity")
    broken = tmp_path / "broken"
    shutil.copytree(e2e.REPO / "examples" / "miner_node_identity", broken)
    (broken / "kernels" / "forward.py").write_text("def forward():\n    return None\n")
    done = subprocess.run([sys.executable, "-m", "cacheon.cli", "verify", str(broken)], cwd=e2e.REPO,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 2 and "[FAIL]" in done.stderr, done.stdout + done.stderr
