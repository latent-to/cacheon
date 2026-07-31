"""Private validator snapshot, restore, and redacted audit-log contracts."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

import cacheon.chain.validator_loop as loop
from cacheon.bundle_hash import content_hash
from cacheon.chain import FinalizedRevealSnapshot, RevealedCommitment
from cacheon.chain.archive import (
    ValidatorArchiveError,
    ValidatorArchiveManifest,
    create_validator_archive,
    parse_named_roots,
    restore_validator_archive,
    verify_validator_archive,
)
from cacheon.chain.audit_log import (
    ChainAuditLogError,
    append_chain_audit,
    fault_audit_record,
    pass_audit_record,
)
from cacheon.chain.intake import FinalizedIntakeStore, IntakeScope
from cacheon.chain.payload import encode_payload
from cacheon.eval.evidence_store import publish_evidence, reopen_evidence
from cacheon.object_store import MemoryObjectStore


BLOCK = 90
BLOCK_HASH = "0x" + "9" * 64
SCOPE = IntakeScope("0x" + "0" * 64, 307)


class _Subtensor:
    def get_block_hash(self, block):
        assert block == 0
        return SCOPE.genesis_hash


def _bundle(root: Path) -> Path:
    (root / "kernels").mkdir(parents=True)
    (root / "manifest.toml").write_text(
        'bundle_id = "archive-test"\n'
        'abi_version = "optima-op-abi-v0"\n\n'
        "[[ops]]\n"
        'slot = "activation.silu_and_mul"\n'
        'source = "kernels/k.py"\n'
        'entry = "silu_and_mul"\n'
        'dtypes = ["float32"]\n'
    )
    (root / "kernels/k.py").write_text(
        "def silu_and_mul(x, out):\n    out.copy_(x)\n"
    )
    for directory in (root, root / "kernels"):
        directory.chmod(0o700)
    for file in (root / "manifest.toml", root / "kernels/k.py"):
        file.chmod(0o600)
    return root


def _published_database(tmp_path: Path, monkeypatch):
    source = _bundle(tmp_path / "source")
    digest = content_hash(source)
    payload = encode_payload(digest, "https://objects.example/bundle.tar.gz")
    snapshot = FinalizedRevealSnapshot(
        BLOCK,
        BLOCK_HASH,
        (RevealedCommitment("miner", payload, BLOCK, BLOCK_HASH, 0),),
    )
    monkeypatch.setattr(
        loop.chain,
        "read_finalized_reveal_history",
        lambda *_, **__: snapshot,
    )
    monkeypatch.setattr(
        loop.chain,
        "read_finalized_head",
        lambda *_: (BLOCK, BLOCK_HASH),
    )
    monkeypatch.setattr(loop, "fetch_bundle", lambda *_: source)
    database = tmp_path / "state" / "intake.sqlite3"
    result = loop.run_pass(
        _Subtensor(),
        307,
        intake_db=database,
        private_root=tmp_path / "private",
        publication_root=tmp_path / "worker",
        intake_only=True,
    )
    assert len(result.published) == 1
    return database, result, digest


def test_redacted_chain_audit_is_append_only_and_excludes_messages(tmp_path):
    path = tmp_path / "audit" / "chain.jsonl"
    result = loop.PassResult(BLOCK, BLOCK_HASH)
    result.seen = 1
    result.reserved.append("a" * 64)
    result.rejected["b" * 64] = (
        "fetch:https://secret.example/path?token=do-not-retain"
    )
    append_chain_audit(path, pass_audit_record(result, timestamp_ns=1))
    append_chain_audit(
        path,
        fault_audit_record(
            RuntimeError("wallet secret must not be retained"),
            consecutive_failures=1,
            timestamp_ns=2,
        ),
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["pass", "validator_fault"]
    assert rows[0]["rejected"] == {"b" * 64: "fetch"}
    assert rows[1]["error_type"] == "RuntimeError"
    retained = path.read_text()
    assert "secret.example" not in retained
    assert "wallet secret" not in retained
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ChainAuditLogError, match="not closed"):
        append_chain_audit(
            path,
            {
                "event": "pass",
                "schema": "optima.chain-audit.v1",
                "url": "https://must-not-be-retained.example",
            },
        )


def test_chain_audit_heals_wrong_mode_on_owned_parent_directory(tmp_path):
    parent = tmp_path / "chain_intake"
    parent.mkdir(mode=0o755)
    os.chmod(parent, 0o755)
    path = parent / "chain_audit.jsonl"
    result = loop.PassResult(BLOCK, BLOCK_HASH)
    result.seen = 1

    append_chain_audit(path, pass_audit_record(result, timestamp_ns=1))

    assert parent.stat().st_mode & 0o777 == 0o700
    assert path.stat().st_mode & 0o777 == 0o600
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["pass"]


def test_chain_audit_still_refuses_symlinked_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    result = loop.PassResult(BLOCK, BLOCK_HASH)
    result.seen = 1

    with pytest.raises(ChainAuditLogError, match="canonical owner-only"):
        append_chain_audit(
            link / "chain_audit.jsonl", pass_audit_record(result, timestamp_ns=1)
        )
    assert real.stat().st_mode & 0o777 == 0o700


def test_validator_loop_writes_success_and_redacted_fault_audits(
    tmp_path,
    monkeypatch,
):
    audit_log = tmp_path / "state" / "chain-audit.jsonl"
    result = loop.PassResult(BLOCK, BLOCK_HASH, seen=1)
    monkeypatch.setattr(loop, "run_pass", lambda *_, **__: result)

    returned = loop.run_validator(
        object(),
        307,
        intake_db=tmp_path / "unused.sqlite3",
        private_root=tmp_path / "private",
        publication_root=tmp_path / "worker",
        intake_only=True,
        once=True,
        audit_log=audit_log,
    )
    assert returned is result

    def fail(*_, **__):
        raise RuntimeError("https://secret.example/?token=never-log")

    monkeypatch.setattr(loop, "run_pass", fail)
    with pytest.raises(RuntimeError, match="secret.example"):
        loop.run_validator(
            object(),
            307,
            intake_db=tmp_path / "unused.sqlite3",
            private_root=tmp_path / "private",
            publication_root=tmp_path / "worker",
            intake_only=True,
            once=True,
            audit_log=audit_log,
        )

    retained = audit_log.read_text()
    rows = [json.loads(line) for line in retained.splitlines()]
    assert [row["event"] for row in rows] == ["pass", "validator_fault"]
    assert "secret.example" not in retained


def test_snapshot_roundtrip_archives_database_bundle_evidence_audit_and_sealed(
    tmp_path,
    monkeypatch,
):
    database, result, digest = _published_database(tmp_path, monkeypatch)
    reservation_id = result.reserved[0]

    evidence_root = tmp_path / "evidence"
    reference = publish_evidence(
        evidence_root,
        b'{"attempt":"retained"}',
        domain="qualification",
        media_type="application/json",
        schema="attempt.v1",
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "INSERT INTO settlement_qualifications("
            "reservation_id,reproduction_index,qualification_digest,"
            "qualification_json,attempt_ref_json,evidence_root,retained_block"
            ") VALUES(?,?,?,?,?,?,?)",
            (
                reservation_id,
                0,
                "a" * 64,
                "{}",
                json.dumps(reference.to_dict(), separators=(",", ":"), sort_keys=True),
                str(evidence_root),
                BLOCK,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    audit_log = tmp_path / "state" / "chain-audit.jsonl"
    append_chain_audit(
        audit_log,
        pass_audit_record(result, timestamp_ns=123),
    )
    sealed = tmp_path / "sealed"
    sealed.mkdir(mode=0o700)
    (sealed / "policy.json").write_text('{"policy":"frozen"}\n')
    (sealed / "policy.json").chmod(0o400)

    object_store = MemoryObjectStore()
    publication = create_validator_archive(
        database,
        object_store,
        audit_log=audit_log,
        sealed_inputs={"qualification-inputs": sealed},
        created_at_ns=456,
    )

    manifest = publication.manifest
    assert manifest.finalized_cursor == (BLOCK, BLOCK_HASH)
    assert len(manifest.admitted_bundles) == 1
    assert manifest.admitted_bundles[0].content_hash == digest
    assert len(manifest.qualification_evidence) == 1
    assert len(manifest.sealed_inputs) == 1
    assert manifest.audit_log is not None
    assert ValidatorArchiveManifest.from_bytes(
        object_store.get_bytes(publication.manifest_key)
    ) == manifest

    restored = restore_validator_archive(
        object_store,
        publication.manifest_key,
        tmp_path / "restore",
    )
    restored_db = sqlite3.connect(restored.database_path)
    try:
        status = restored_db.execute(
            "SELECT status FROM reservations WHERE reservation_id=?",
            (reservation_id,),
        ).fetchone()[0]
    finally:
        restored_db.close()
    assert status == "published"
    restored_evidence_root = next((tmp_path / "restore" / "evidence").iterdir())
    assert reopen_evidence(restored_evidence_root, reference) == b'{"attempt":"retained"}'
    assert (
        tmp_path
        / "restore"
        / "sealed"
        / "qualification-inputs"
        / "policy.json"
    ).read_text() == '{"policy":"frozen"}\n'
    assert verify_validator_archive(object_store, publication.manifest_key) == manifest


def test_snapshot_uses_online_sqlite_backup_while_store_is_open(tmp_path):
    database = tmp_path / "state" / "intake.sqlite3"
    object_store = MemoryObjectStore()

    with FinalizedIntakeStore(database, scope=SCOPE):
        publication = create_validator_archive(
            database,
            object_store,
            created_at_ns=7,
        )

    assert publication.manifest.scope == SCOPE
    assert publication.manifest.finalized_cursor is None
    assert dict(publication.manifest.table_counts)["reservations"] == 0
    verify_validator_archive(object_store, publication.manifest_key)


def test_snapshot_detects_content_address_conflict(tmp_path):
    database = tmp_path / "state" / "intake.sqlite3"
    object_store = MemoryObjectStore()
    with FinalizedIntakeStore(database, scope=SCOPE):
        first = create_validator_archive(database, object_store, created_at_ns=8)

    assert object_store._objects is not None
    database_key = first.manifest.database.key
    media_type = object_store._objects[database_key][1]
    object_store._objects[database_key] = (b"conflicting bytes", media_type)

    with pytest.raises(ValidatorArchiveError, match="conflicts"):
        create_validator_archive(database, object_store, created_at_ns=9)


def test_manifest_parser_closes_nested_type_failures() -> None:
    malformed = json.dumps(
        {
            "manifest": {
                "admitted_bundles": [],
                "audit_log": None,
                "created_at_ns": 1,
                "database": {
                    "key": "blobs/sha256/" + "a" * 64,
                    "media_type": "application/vnd.sqlite3",
                    "sha256": "a" * 64,
                    "size": 1,
                },
                "database_schema": 1,
                "finalized_cursor": None,
                "qualification_evidence": [],
                "schema": "optima.validator-archive.v1",
                "scope": {"genesis_hash": [], "netuid": 307},
                "scope_digest": "a" * 64,
                "sealed_inputs": [],
                "table_counts": [],
            },
            "manifest_digest": "a" * 64,
        }
    ).encode()
    with pytest.raises(ValidatorArchiveError, match="fields are malformed"):
        ValidatorArchiveManifest.from_bytes(malformed)


def test_restore_requires_the_canonical_manifest_address(
    tmp_path,
) -> None:
    database = tmp_path / "state" / "intake.sqlite3"
    object_store = MemoryObjectStore()
    with FinalizedIntakeStore(database, scope=SCOPE):
        publication = create_validator_archive(
            database,
            object_store,
            created_at_ns=11,
        )
    payload = object_store.get_bytes(publication.manifest_key)
    alias = f"aliases/archive-{publication.manifest.digest}.json"
    object_store.put_bytes(alias, payload, content_type="application/json")
    with pytest.raises(ValidatorArchiveError, match="does not bind"):
        restore_validator_archive(object_store, alias, tmp_path / "restore-alias")


def test_sealed_input_rejects_symlinks_and_named_root_parser_is_closed(
    tmp_path,
):
    database = tmp_path / "state" / "intake.sqlite3"
    with FinalizedIntakeStore(database, scope=SCOPE):
        pass
    sealed = tmp_path / "sealed"
    sealed.mkdir(mode=0o700)
    target = sealed / "target"
    target.write_text("secret")
    target.chmod(0o400)
    (sealed / "link").symlink_to(target)

    with pytest.raises(ValidatorArchiveError, match="unsafe entry"):
        create_validator_archive(
            database,
            MemoryObjectStore(),
            sealed_inputs={"inputs": sealed},
            created_at_ns=10,
        )
    assert parse_named_roots(["inputs=/private/inputs"]) == {
        "inputs": Path("/private/inputs")
    }
    with pytest.raises(ValidatorArchiveError, match="duplicated"):
        parse_named_roots(["inputs=/a", "inputs=/b"])
