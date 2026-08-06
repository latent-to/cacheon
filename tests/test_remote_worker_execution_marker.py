from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path

import pytest

from cacheon.chain.remote_worker_execution_marker import (
    RESIDENT_ENTRY_MARKER,
    RemoteWorkerExecutionMarkerError,
    marker_for_request,
    publish_resident_entry,
    reopen_resident_entry,
)
from cacheon.chain.remote_worker_spool import load_json, spool_canonical_json
from tests.test_remote_worker_spool import _screen_authority


def _request(tmp_path: Path) -> dict[str, object]:
    authority = _screen_authority(tmp_path)
    return load_json(authority[-1] / "request.json")


def _result_root(tmp_path: Path) -> Path:
    result = tmp_path / "results" / ("." + "f" * 64 + ".123")
    result.mkdir(parents=True, mode=0o700)
    return result


def test_marker_publishes_canonically_and_reopens_exact_request(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    result = _result_root(tmp_path)

    published = publish_resident_entry(result, request)
    path = result / RESIDENT_ENTRY_MARKER

    assert reopen_resident_entry(result, request) == published
    assert path.read_bytes() == spool_canonical_json(published) + b"\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    payload = next(
        row["sha256"]
        for row in request["artifacts"]
        if row["role"] == "screen_payload"
    )
    assert published["remote_request_sha256"] == payload
    assert published["stage"] == "screen"

    with pytest.raises(RemoteWorkerExecutionMarkerError, match="cannot publish"):
        publish_resident_entry(result, request)


def test_marker_derivation_supports_qualification_without_target_constants(
    tmp_path: Path,
) -> None:
    request = copy.deepcopy(_request(tmp_path))
    request["lease"]["stage"] = "qualification"
    request["lease"]["members"][0]["prior_status"] = "promoted"
    for artifact in request["artifacts"]:
        if artifact["role"] == "screen_payload":
            artifact["role"] = "qualification_payload"
    marker = marker_for_request(request)
    assert marker["stage"] == "qualification"
    assert marker["lease_id"] == request["lease"]["lease_id"]


def test_marker_rejects_request_or_file_identity_drift(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = _result_root(tmp_path)
    publish_resident_entry(result, request)

    changed = copy.deepcopy(request)
    changed["request_id"] = "0" * 64
    with pytest.raises(RemoteWorkerExecutionMarkerError, match="differs"):
        reopen_resident_entry(result, changed)

    path = result / RESIDENT_ENTRY_MARKER
    path.chmod(0o600)
    with pytest.raises(RemoteWorkerExecutionMarkerError, match="identity is unsafe"):
        reopen_resident_entry(result, request)


def test_marker_rejects_symlink_noncanonical_and_truncated_files(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    symlink_root = _result_root(tmp_path / "symlink")
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (symlink_root / RESIDENT_ENTRY_MARKER).symlink_to(outside)
    with pytest.raises(RemoteWorkerExecutionMarkerError, match="cannot reopen"):
        reopen_resident_entry(symlink_root, request)

    noncanonical_root = _result_root(tmp_path / "noncanonical")
    expected = marker_for_request(request)
    noncanonical = noncanonical_root / RESIDENT_ENTRY_MARKER
    noncanonical.write_text(json.dumps(expected) + "\n", encoding="utf-8")
    noncanonical.chmod(0o400)
    with pytest.raises(RemoteWorkerExecutionMarkerError, match="differs"):
        reopen_resident_entry(noncanonical_root, request)

    truncated_root = _result_root(tmp_path / "truncated")
    truncated = truncated_root / RESIDENT_ENTRY_MARKER
    truncated.write_bytes(b'{"schema":')
    truncated.chmod(0o400)
    with pytest.raises(RemoteWorkerExecutionMarkerError, match="invalid"):
        reopen_resident_entry(truncated_root, request)


def test_marker_rejects_unsafe_or_symlinked_result_root(tmp_path: Path) -> None:
    request = _request(tmp_path)
    unsafe = _result_root(tmp_path / "unsafe")
    unsafe.chmod(0o755)
    with pytest.raises(RemoteWorkerExecutionMarkerError, match="mode 0700"):
        publish_resident_entry(unsafe, request)

    target = _result_root(tmp_path / "target")
    link = tmp_path / "result-link"
    os.symlink(target, link)
    with pytest.raises(RemoteWorkerExecutionMarkerError, match="mode 0700"):
        publish_resident_entry(link, request)
