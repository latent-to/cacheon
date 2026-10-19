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
--config`` (killed mid-wait and restarted, then stopped with SIGTERM); the
first intake pass, ``chain-validate --intake-only --once`` entered through
``cacheon.cli.main`` in its own process. The contributor commands ``scan`` and
``verify`` run over ``examples/`` in their own test.

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
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cacheon.chain.intake import FinalizedIntakeStore
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.ssh_worker_transport import _dispatcher_heartbeat
from cacheon.eval import b300_deployment as deployment
from cacheon.eval.qualification import QualificationDecision
from tests.support import e2e
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


def _imported(stderr: str) -> set[str]:
    """The modules a ``-X importtime`` controller process imported, after checking it loaded no GPU stack."""

    names = {line.rsplit("|", 1)[1].strip() for line in stderr.splitlines() if line.startswith("import time:")}
    assert sorted(name for name in names if name.split(".")[0] in GPU_STACKS) == []
    return names


def _submit(root: Path, validator_key) -> tuple[Path, dict[str, Path]]:
    """Miners package sealed archives and reveal them in one finalized block."""

    reveals, origin = {}, {}
    for hotkey in MINERS:
        content_hash, archive = e2e.package(e2e.miner_bundle(root, hotkey, BODIES.get(hotkey)), validator_key)
        reveals[hotkey] = (e2e.reveal(hotkey, content_hash, archive, origin),)
    return e2e.chain(root, reveals, MINERS), origin


@pytest.mark.parametrize("profile", sorted(e2e.PROFILES))
def test_finalized_submissions_qualify_settle_and_project_weights(tmp_path, monkeypatch, supervisors, profile):
    started = time.monotonic()
    root, validator_key = e2e.deployment_root(tmp_path, f"e2e-{profile}", monkeypatch)

    # Intake: finalized order, private fetch, immutable publication, copy demotion, in the CLI's own process.
    state, origin = _submit(root, validator_key)
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
        supervisor = supervisors(root)
        assert supervisor.until(e2e.waiting)["request_id"] == first["request_id"]
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

    # Weights: the signer projects the crowned hotkey over the same durable store.
    out = e2e.run("cacheon.cli", "set-weights", "--dry-run", "--intake-db", str(root / "validator" / "intake.sqlite3"),
                  "--netuid", str(e2e.NETUID), "--network", str(state), "--wallet-path", str(state),
                  "--half-life-blocks", "100", "--discovery-lifetime-blocks", "20",
                  "--discovery-pool-ppm", "100000", "--refresh-blocks", "10", env=e2e.chain_env())
    uid = ("validator", *MINERS).index("miner-pass")
    assert f"rows=[{{'uid': {uid}, 'weight': 1.0, 'hotkey': 'miner-pass'}}]" in out, out
    assert "status=dry_run" in out and "submitted=False" in out, out
    with FinalizedIntakeStore(root / "validator" / "intake.sqlite3", scope=scope) as store:
        assert store._db.execute("SELECT COUNT(*) FROM weight_publications").fetchone()[0] == 0
    assert time.monotonic() - started < 60


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
