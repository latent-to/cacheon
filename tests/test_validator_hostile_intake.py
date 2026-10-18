"""Hostile submissions, end to end: each refusal lands where production makes it.

A miner proposal is hostile input. These tests drive the same deployment root
as ``tests/test_validator_end_to_end.py`` (built by ``tests/support/e2e.py``)
and assert the durable disposition of every hostile shape at the production
step that refuses it:

- finalized intake (``validator_loop.run_validator``) refuses malformed reveal
  payloads, archives whose transport, encryption, members or content hash are
  wrong, manifests the component parser rejects (path escape, foreign ABI, a
  node claimed twice or outside the model, dynamic imports and rebound DSL
  compile receivers), and demotes copies of an earlier miner's bundle or of
  the validator's own library. A ``failed`` row is terminal: the claim query
  selects only ``published`` and ``reproduction_pending`` rows;
- the static scanner runs in ``cacheon scan`` and ``verify`` and in the pod's
  seam load, never at intake, so its refusals are proven through the CLI;
- a paid arena admits only a reveal whose ``chain-submit --pay`` transfer
  covers the fee, read back from the simulated chain the miner paid on;
- the standing supervisor's claim parks a bundle for a target the arena
  closed and replays a terminal FAIL onto the same miner's byte-identical
  resubmission without a second carrier.
"""

from __future__ import annotations

import io
import json
import signal
import subprocess
import sys
import tarfile

import pytest
from nacl.public import PrivateKey

from cacheon.chain.bundle_privacy import encrypt_archive
from cacheon.chain.eval_cost import EvalCostPolicy
from cacheon.chain.fetch import MAX_ARCHIVE_BYTES
from cacheon.chain.intake import FinalizedIntakeStore
from cacheon.chain.recoverable_intake import RecoverableFinalizedIntakeStore
from cacheon.chain.ssh_worker_transport import _dispatcher_heartbeat
from cacheon.eval.qualification import QualificationDecision
from tests.support import e2e
from tests.support.chain import advance

supervisors = pytest.fixture(e2e.supervisor_processes, name="supervisors")


def _identity(tag: str) -> str:
    """An identity-preserving body distinct from the example and every other tag."""
    return f"def forward(module, *args, **kwargs):\n    label = {tag!r}\n    return module.forward(*args, **kwargs)\n"


def _tar(members) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as archive:
        for name, kind, data in members:
            info = tarfile.TarInfo(name)
            if kind == tarfile.REGTYPE:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            else:
                info.type, info.linkname = kind, data
                archive.addfile(info)
    return raw.getvalue()


def _drain(root, state, origin, monkeypatch, **options) -> int:
    """Intake fetches one cohort per pass; run passes until nothing is left to fetch."""

    passes = 0
    while e2e.intake(root, state, origin, monkeypatch, **options):
        passes += 1
    return passes


def _rows(root):
    with FinalizedIntakeStore(root / "validator" / "intake.sqlite3", scope=e2e.scope(root)) as store:
        return {row.arrival.hotkey: row for row in store.all()}


def test_hostile_reveals_archives_and_manifests_fail_at_intake(tmp_path, monkeypatch):
    root, key = e2e.deployment_root(tmp_path, "hostile-intake", monkeypatch)
    origin, reveals, refused = {}, {}, {}
    sealed = lambda data: encrypt_archive(data, bytes(key.public_key).hex())  # noqa: E731

    def stage(hotkey, data, content_hash, reason):
        path = e2e.private(root / "archives") / f"{hotkey}.sealed"
        path.write_bytes(data)
        reveals[hotkey], refused[hotkey] = (e2e.reveal(hotkey, content_hash, path, origin),), reason

    def submit(hotkey, forward, edit=None, recipient=key):
        bundle = e2e.miner_bundle(root, hotkey, forward)
        if edit is not None:
            (bundle / "manifest.toml").write_text(edit((bundle / "manifest.toml").read_text()))
        content_hash, archive = e2e.package(bundle, recipient)
        reveals[hotkey] = (e2e.reveal(hotkey, content_hash, archive, origin),)
        return content_hash, archive

    ok_hash, ok = submit("miner-ok", _identity("ok"))
    submit("dsl-compile", "import cutlass.cute as cute\n\n\ndef forward(module, *args, **kwargs):\n"
                          "    cute.compile(module.forward)\n    return module.forward(*args, **kwargs)\n")
    example = e2e.REPO / "examples" / "miner_node_identity"
    base = [("manifest.toml", tarfile.REGTYPE, (example / "manifest.toml").read_bytes()),
            ("kernels/forward.py", tarfile.REGTYPE, (example / "kernels" / "forward.py").read_bytes())]
    unknown = "ab" * 32
    for hotkey, extra, reason in (
        ("escape", ("../evil.py", tarfile.REGTYPE, b"x"), "fetch:archive member escapes destination"),
        ("symlink", ("kernels/link.py", tarfile.SYMTYPE, "/etc/passwd"), "fetch:archive member is not a regular file"),
        ("hardlink", ("kernels/hard.py", tarfile.LNKTYPE, "manifest.toml"), "fetch:archive member is not a regular file"),
        ("duplicate", base[0], "fetch:archive contains duplicate member path"),
        ("excluded", ("kernels/__pycache__/forward.cpython-312.pyc", tarfile.REGTYPE, b"x"),
         "fetch:archive member is excluded from bundle identity"),
    ):
        stage(hotkey, sealed(_tar([*base, extra])), unknown, reason)
    stage("not-gzip", sealed(b"plain bytes"), unknown, "fetch:bundle archive must be gzip-compressed tar")
    stage("oversize", b"", unknown, "fetch:archive exceeds")
    with (root / "archives" / "oversize.sealed").open("r+b") as handle:
        handle.truncate(MAX_ARCHIVE_BYTES + 1)
    stage("hash-mismatch", ok.read_bytes(), unknown, "fetch:content hash mismatch")
    submit("wrong-recipient", _identity("other key"), recipient=PrivateKey.generate())
    refused["wrong-recipient"] = "fetch:encrypted bundle names another validator key"
    tampered_hash, tampered = submit("tampered", _identity("tampered"))
    stage("tampered", tampered.read_bytes()[:-1] + bytes([tampered.read_bytes()[-1] ^ 1]), tampered_hash,
          "fetch:encrypted bundle authentication failed")
    component = "manifest:submission is not a registered component: "
    for hotkey, forward, edit, reason in (
        ("manifest-escape", _identity("escape"), lambda m: m.replace('"kernels/forward.py"', '"../forward.py"'),
         "source path escapes bundle root"),
        ("foreign-abi", _identity("abi"), lambda m: m.replace("cacheon-op-abi-v0", "foreign-abi-v9"),
         "unsupported abi_version"),
        ("node-twice", _identity("twice"), lambda m: m + '\n[[ops]]\nslot = "model.layers.*.mlp"\n'
         'source = "kernels/forward.py"\nentry = "forward"\nvariant = "again"\n', "duplicate slot"),
        ("outside-roots", _identity("roots"), lambda m: m.replace('"model.layers.*.mlp"', '"lm_head"'),
         "node addresses must sit under"),
        ("dynamic-import", "import importlib\n\n\ndef forward(module, *args, **kwargs):\n"
         "    importlib.import_module('kernels.helper')\n    return module.forward(*args, **kwargs)\n", None,
         "dynamic import is unsupported"),
        ("dsl-rebound", "import cutlass.cute as cute\n\n\ndef forward(module, *args, **kwargs):\n"
         "    cute = module\n    cute.compile(module.forward)\n    return module.forward(*args, **kwargs)\n", None,
         "dynamic import is unsupported"),
    ):
        submit(hotkey, forward, edit)
        refused[hotkey] = component + reason
    library = (e2e.REPO / "cacheon_kernels" / "codec" / "nvfp4.py").read_text() + "\n\n" + _identity("library")
    submit("library-copy", library)
    refused["library-copy"] = "copy_of:validator_reference:library-codec-nvfp4.py"
    for hotkey, url in (("garbage", None), ("file-url", "file:///etc/passwd"), ("plain-http", "http://x.example/a")):
        data = "not a payload" if url is None else json.dumps({"v": 1, "h": ok_hash, "u": url}, separators=(",", ":"))
        reveals[hotkey], refused[hotkey] = ((e2e.REVEAL_BLOCK, data),), "invalid_payload"
    later = submit("reformatted", '"""Reworded."""\n\n\ndef forward(module, *args, **kwargs):\n'
                                  "    # A comment.\n    label = 'ok'\n\n    return module.forward(*args, **kwargs)\n")
    reveals["reformatted"] = (e2e.reveal("reformatted", *later, origin, block=e2e.REVEAL_BLOCK + 1),)

    state = e2e.chain(root, reveals, tuple(reveals))
    assert _drain(root, state, origin, monkeypatch) == 3  # one cohort of eight per pass
    rows = _rows(root)
    refused["reformatted"] = f"copy_of:{rows['miner-ok'].reservation_id}"
    assert {hotkey for hotkey, row in rows.items() if row.status == "published"} == {"miner-ok", "dsl-compile"}
    for hotkey, reason in refused.items():
        assert (rows[hotkey].status, rows[hotkey].reason[:len(reason)]) == ("failed", reason), hotkey
    assert e2e.intake(root, state, origin, monkeypatch) == []  # terminal rows are never fetched again


def test_scan_refuses_egress_exec_escape_and_undeclared_artifacts(tmp_path):
    root = e2e.private(tmp_path / "scan")
    for name, forward, extra, violation in (
        ("egress", "import socket\n\n\ndef forward(module, *args, **kwargs):\n"
         "    socket.create_connection(('x.example', 80))\n    return module.forward(*args, **kwargs)\n",
         None, "banned import 'socket'"),
        ("exec", "import os\n\n\ndef forward(module, *args, **kwargs):\n    os.system('id')\n"
         "    return module.forward(*args, **kwargs)\n", None, "banned call .system()"),
        ("escape", "def forward(module, *args, **kwargs):\n    getattr(__builtins__, 'ev' + 'al')('1')\n"
         "    return module.forward(*args, **kwargs)\n", None, "dynamic getattr() with a non-literal attribute"),
        ("cuda", None, ("extra.cu", b"__global__ void k() {}\n"), "not declared in manifest cuda_sources"),
        ("binary", None, ("fast.so", b"\x7fELF"), "binary/artifact file not allowed in a bundle (.so)"),
    ):
        bundle = e2e.miner_bundle(root, name, forward)
        if extra is not None:
            (bundle / "kernels" / extra[0]).write_bytes(extra[1])
        done = subprocess.run([sys.executable, "-m", "cacheon.cli", "scan", str(bundle)], cwd=e2e.REPO,
                              capture_output=True, text=True, timeout=300)
        assert done.returncode == 2 and "[VIOLATIONS]" in done.stdout and violation in done.stdout, done.stdout
    declared = e2e.miner_bundle(root, "declared-cuda")
    (declared / "kernels" / "extra.cu").write_bytes(b"__global__ void k() {}\n")
    manifest = declared / "manifest.toml"
    manifest.write_text(manifest.read_text() + 'cuda_sources = ["kernels/extra.cu"]\n')
    controls = [path.parent for path in sorted((e2e.REPO / "examples").glob("*/manifest.toml"))]
    for bundle in (declared, *controls):  # the sanctioned CUDA tier and every public control scan clean
        assert "[VIOLATIONS]" not in e2e.run("cacheon.cli", "scan", str(bundle))


def test_paid_arena_admits_only_a_covering_payment(tmp_path, monkeypatch):
    root, key = e2e.deployment_root(tmp_path, "paid-intake", monkeypatch)
    miners = ("miner-paid", "miner-unpaid", "miner-short")
    state = e2e.chain(root, {}, miners, owner="subnet-owner-coldkey")
    origin, policy = {}, EvalCostPolicy(amount_rao=500_000_000)

    def submit(bundle, url, *flags):
        return subprocess.run([sys.executable, "-m", "cacheon.cli", "chain-submit", str(bundle), "--url", url,
                               "--netuid", str(e2e.NETUID), "--network", str(state), "--wallet", "miner",
                               "--hotkey", bundle.name, *flags],
                              cwd=e2e.REPO, env=e2e.chain_env(), capture_output=True, text=True, timeout=300)

    for hotkey, flags in (("miner-paid", ("--pay",)), ("miner-unpaid", ()),
                          ("miner-short", ("--pay", "--eval-cost-tao-rao", "1"))):
        bundle = e2e.miner_bundle(root, hotkey, _identity(hotkey))
        content_hash, archive = e2e.package(bundle, key)
        url = f"https://{hotkey}.example/{content_hash}.sealed"
        origin[url] = archive
        done = submit(bundle, url, *flags)
        assert done.returncode == 0 and "set_reveal_commitment submitted=True" in done.stdout, done.stdout + done.stderr
    for flags, refusal in (
        (("--dry-run",), "url must be canonical and use one of ('https',)"),
        (("--dry-run", "--pay", "--eval-cost-payment-block", "100"), "cannot pay and reuse"),
    ):
        done = submit(bundle, url.replace("https://", "http://") if refusal.startswith("url") else url, *flags)
        assert done.returncode == 2 and "REFUSED before signing" in done.stdout and refusal in done.stdout, done.stdout
    assert submit(bundle, url + "/" + "x" * 1100, "--dry-run").returncode == 2  # over the chain's payload cap

    assert _drain(root, state, origin, monkeypatch, eval_cost_policy=policy) == 0  # timelocks still closed
    advance(state, e2e.HEAD + 10)
    assert _drain(root, state, origin, monkeypatch, eval_cost_policy=policy) == 1
    rows = _rows(root)
    assert rows["miner-paid"].status == "published" and rows["miner-paid"].arrival.block == e2e.HEAD + 10
    assert (rows["miner-unpaid"].status, rows["miner-unpaid"].reason) == ("failed", "missing_eval_cost_payment")
    assert (rows["miner-short"].status, rows["miner-short"].reason) == ("failed", "eval_cost_payment_invalid")
    with FinalizedIntakeStore(root / "validator" / "intake.sqlite3", scope=e2e.scope(root)) as store:
        payments = store._db.execute("SELECT payment_block,payment_extrinsic_index,reservation_id,"
                                     "amount_tao_rao FROM eval_cost_payments").fetchall()
    assert [tuple(row) for row in payments] == [(e2e.HEAD, 0, rows["miner-paid"].reservation_id, policy.amount_rao)]


def test_claim_parks_a_closed_target_and_replays_a_duplicate_fail(tmp_path, monkeypatch, supervisors):
    root, key = e2e.deployment_root(tmp_path, "hostile-claim", monkeypatch)
    origin = {}
    fail_hash, fail = e2e.package(e2e.miner_bundle(root, "miner-fail", _identity("fail")), key)
    cache = e2e.miner_bundle(root, "miner-cache")
    (cache / "kernels" / "forward.py").unlink()
    e2e.private(cache / "cache")
    (cache / "cache" / "policy.py").write_text("def build(cache):\n    return type(cache)\n")
    (cache / "manifest.toml").write_text(
        'bundle_id = "cache-identity"\nabi_version = "cacheon-op-abi-v0"\n\n[competition]\n'
        f'target = "prefix_cache"\nmode = "slot"\narena = "{root.name}"\n\n'
        '[[ops]]\nslot = "tree_cache"\nsource = "cache/policy.py"\nentry = "build"\n')
    cache_hash, cache_archive = e2e.package(cache, key)
    state = e2e.chain(root, {
        "miner-fail": (e2e.reveal("miner-fail", fail_hash, fail, origin),
                       e2e.reveal("miner-fail", fail_hash, fail, origin, block=e2e.REVEAL_BLOCK + 1)),
        "miner-cache": (e2e.reveal("miner-cache", cache_hash, cache_archive, origin),),
    }, ("miner-fail", "miner-cache"))
    assert _drain(root, state, origin, monkeypatch) == 1
    assert {row.status for row in _rows(root).values()} == {"published"}

    materialized, ready_path, ready = e2e.commission(root, e2e.PROFILES["h100-tp1"])
    sealed = e2e.seal_authority(root, materialized, ready_path, ready)
    assert sealed.manifest["closed_targets"] == ["prefix_cache"]  # the prompt registers forward_pass only
    heartbeat = e2e.private(root / "spool" / "state") / "heartbeat.json"
    with _dispatcher_heartbeat(sealed.registration, heartbeat, lambda: None):
        supervisor = supervisors(root)
        supervisor.until(e2e.waiting)
        (carrier,) = e2e.carriers(root)
        first = e2e.pod_answer(root, sealed, carrier, QualificationDecision.FAIL)
        supervisor.until(lambda status: status["phase"] == "idle")
        assert supervisor.stop(signal.SIGTERM) == 0
    assert e2e.carriers(root) == [carrier]  # the resubmission never reached the pod

    with RecoverableFinalizedIntakeStore(root / "validator" / "intake.sqlite3", sealed.policy,
                                         scope=sealed.scope) as store:
        rows = {(row.arrival.hotkey, row.arrival.block): row for row in store.all()}
        attempts = {name: store.qualification_attempts(row.reservation_id) for name, row in rows.items()}
    original, again = rows["miner-fail", e2e.REVEAL_BLOCK], rows["miner-fail", e2e.REVEAL_BLOCK + 1]
    parked = rows["miner-cache", e2e.REVEAL_BLOCK]
    assert original.reservation_id == first and (original.status, original.decision) == ("failed", "FAIL")
    assert (again.status, again.decision) == ("failed", "FAIL")
    assert again.reason == f"duplicate_of:{first[:16]}:{original.reason}"
    assert (parked.status, parked.decision, parked.reason) == ("expired", "NO_DECISION",
                                                               "target_unavailable:prefix_cache")
    assert attempts == {("miner-fail", e2e.REVEAL_BLOCK): 1, ("miner-fail", e2e.REVEAL_BLOCK + 1): 0,
                        ("miner-cache", e2e.REVEAL_BLOCK): 0}
