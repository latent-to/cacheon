"""Rename compatibility for the immutable validator archive v1 contract."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from cacheon.chain.archive import (
    ARCHIVE_SCHEMA,
    DEFAULT_VALIDATOR_ARCHIVE_PREFIX,
    ValidatorArchiveManifest,
    create_validator_archive,
    restore_validator_archive,
)
from cacheon.chain.intake import FinalizedIntakeStore, IntakeScope
from cacheon.object_store import MemoryObjectStore
from cacheon.stack_identity import canonical_digest, canonical_json_bytes, sha256_hex


LEGACY_ARCHIVE_SCHEMA = "optima.validator-archive.v1"
LEGACY_ARCHIVE_PREFIX = "optima/validator-archive/v1"
LEGACY_SCOPE_DOMAIN = "optima.chain.intake-scope"
LEGACY_TREE_DOMAIN = "optima.validator.archived-tree"
LEGACY_MANIFEST_DOMAIN = "optima.validator.archive-manifest"
SCOPE = IntakeScope("0x" + "0" * 64, 307)


def _blob(payload: bytes, media_type: str) -> dict[str, object]:
    digest = sha256_hex(payload)
    return {
        "key": f"blobs/sha256/{digest}",
        "media_type": media_type,
        "sha256": digest,
        "size": len(payload),
    }


def _put_blob(
    store: MemoryObjectStore,
    payload: bytes,
    media_type: str,
) -> dict[str, object]:
    reference = _blob(payload, media_type)
    store.put_bytes(
        reference["key"],  # type: ignore[arg-type]
        payload,
        content_type=media_type,
    )
    return reference


def _legacy_manifest(
    store: MemoryObjectStore,
    database_payload: bytes,
) -> tuple[bytes, str, str]:
    """Build bytes exactly as the pre-rename archive writer did."""

    database = _put_blob(store, database_payload, "application/vnd.sqlite3")
    sealed_payload = b'{"policy":"frozen"}\n'
    sealed_blob = _put_blob(store, sealed_payload, "application/octet-stream")
    tree: dict[str, object] = {
        "content_hash": "",
        "directories": [{"mode": 0o700, "path": "."}],
        "files": [{"blob": sealed_blob, "mode": 0o400, "path": "policy.json"}],
        "kind": "sealed_input",
        "name": "qualification-inputs",
        "publication_digest": "",
        "source_root": "/srv/optima/sealed-inputs",
    }
    tree_digest = canonical_digest(LEGACY_TREE_DOMAIN, tree)
    tree["tree_digest"] = tree_digest

    scope_digest = canonical_digest(LEGACY_SCOPE_DOMAIN, SCOPE.to_dict())
    manifest: dict[str, object] = {
        "admitted_bundles": [],
        "audit_log": None,
        "created_at_ns": 456,
        "database": database,
        # The schema number is an authenticated manifest fact. Restore performs
        # SQLite integrity checks without rewriting that historical value.
        "database_schema": 1,
        "finalized_cursor": None,
        "qualification_evidence": [],
        "schema": LEGACY_ARCHIVE_SCHEMA,
        "scope": SCOPE.to_dict(),
        "scope_digest": scope_digest,
        "sealed_inputs": [tree],
        "table_counts": [],
    }
    manifest_digest = canonical_digest(LEGACY_MANIFEST_DOMAIN, manifest)
    payload = canonical_json_bytes(
        {"manifest": manifest, "manifest_digest": manifest_digest}
    )
    manifest_key = f"snapshots/{scope_digest}/456-{manifest_digest}.json"
    store.put_bytes(manifest_key, payload, content_type="application/json")
    return payload, manifest_key, tree_digest


def test_pre_rename_archive_bytes_and_tree_digests_restore(tmp_path: Path) -> None:
    database_path = tmp_path / "source" / "intake.sqlite3"
    with FinalizedIntakeStore(database_path, scope=SCOPE):
        pass
    database_payload = database_path.read_bytes()

    store = MemoryObjectStore()
    payload, manifest_key, tree_digest = _legacy_manifest(store, database_payload)

    manifest = ValidatorArchiveManifest.from_bytes(payload)
    assert manifest.to_bytes() == payload
    assert manifest.sealed_inputs[0].digest == tree_digest

    restored = restore_validator_archive(store, manifest_key, tmp_path / "restore")
    assert restored.manifest == manifest
    assert (
        restored.restore_root
        / "sealed"
        / "qualification-inputs"
        / "policy.json"
    ).read_bytes() == b'{"policy":"frozen"}\n'
    assert json.loads(restored.restore_map_path.read_bytes())["schema"] == (
        "optima.validator-archive-restore-map.v1"
    )
    with sqlite3.connect(restored.database_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_new_writer_keeps_v1_schema_digests_and_discovery_prefix(
    tmp_path: Path,
) -> None:
    assert ARCHIVE_SCHEMA == LEGACY_ARCHIVE_SCHEMA
    assert DEFAULT_VALIDATOR_ARCHIVE_PREFIX == LEGACY_ARCHIVE_PREFIX

    database_path = tmp_path / "source" / "intake.sqlite3"
    with FinalizedIntakeStore(database_path, scope=SCOPE):
        pass
    store = MemoryObjectStore()
    publication = create_validator_archive(
        database_path,
        store,
        created_at_ns=789,
    )

    encoded = json.loads(store.get_bytes(publication.manifest_key))
    assert encoded["manifest"]["schema"] == LEGACY_ARCHIVE_SCHEMA
    assert encoded["manifest"]["scope_digest"] == canonical_digest(
        LEGACY_SCOPE_DOMAIN,
        SCOPE.to_dict(),
    )
    assert encoded["manifest_digest"] == canonical_digest(
        LEGACY_MANIFEST_DOMAIN,
        encoded["manifest"],
    )
    assert ValidatorArchiveManifest.from_bytes(
        store.get_bytes(publication.manifest_key)
    ) == publication.manifest
