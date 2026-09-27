"""Sealed slice manifests (cacheon/eval/agent_slice.py).

A manifest reopens only over the exact directory it was sealed on: the named files in order, their bytes, and the
session each holds. Expected work per load is the per-root turn accounting the scorer checks completed work against.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cacheon.eval.agent_slice import SliceManifestError, load_slice_manifest


def _session(i: int, *, k: int, inner: int) -> dict:
    return {
        "id": f"{i:02d}" + "ab" * 17, "n_main_total": 40, "start": 10, "k": k, "groups": 1 if inner else 0,
        "inner": inner, "cold_prefix": 90_000, "peak": 150_000, "new_input": 200_000, "reused_input": 900_000,
        "output": 12_000, "inner_input": 10_000 * inner, "inner_output": 200 * inner, "recorded_span_s": 300.0,
    }


def _write_slice(root: Path, sessions: list[dict], *, digest_override: str | None = None) -> Path:
    directory = root / "slice_test"
    directory.mkdir()
    digest = hashlib.sha256()
    for i, s in enumerate(sessions):
        name = f"{i:03d}_{s['id']}.json"
        body = json.dumps({"id": s["id"], "models": ["m"], "block_size": 64, "hash_id_scope": "local",
                           "requests": []}, separators=(",", ":")).encode()
        (directory / name).write_bytes(body)
        digest.update(name.encode())
        digest.update(body)
    manifest = {
        "dataset": "semianalysisai/cc-traces-weka-062126-256k", "revision": "8fecd2fc", "loader": "weka_trace",
        "rules": {"k_main_turns": 20, "seed": 1}, "rejected": {}, "dir": "slice_test", "files": len(sessions),
        "order": "sorted filename = sealed order",
        "sha256_of_named_files": digest_override or digest.hexdigest(), "sessions": sessions,
    }
    path = root / "manifest_test.json"
    path.write_text(json.dumps(manifest))
    return path


def test_manifest_reopens_and_accounts_work_per_load(tmp_path):
    sessions = [_session(0, k=20, inner=0), _session(1, k=12, inner=5), _session(2, k=20, inner=0)]
    manifest = load_slice_manifest(_write_slice(tmp_path, sessions))
    assert manifest.max_load == 3
    assert manifest.loader == "weka_trace" and manifest.directory == tmp_path / "slice_test"
    assert manifest.expected_work(2) == {sessions[0]["id"]: (20, 0), sessions[1]["id"]: (12, 5)}
    assert manifest.turns(3) == 20 + 12 + 5 + 20
    assert manifest.sessions[1].inner_requests == 5 and manifest.sessions[1].cold_prefix_tokens == 90_000
    with pytest.raises(SliceManifestError):
        manifest.expected_work(4)
    with pytest.raises(SliceManifestError):
        manifest.expected_work(0)


def test_manifest_refuses_a_directory_that_differs(tmp_path):
    sessions = [_session(0, k=20, inner=0), _session(1, k=20, inner=0)]
    path = _write_slice(tmp_path, sessions)
    target = tmp_path / "slice_test" / f"001_{sessions[1]['id']}.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(SliceManifestError, match="sealed digest"):
        load_slice_manifest(path)


def test_manifest_refuses_reordered_or_missing_files(tmp_path):
    sessions = [_session(0, k=20, inner=0), _session(1, k=20, inner=0)]
    path = _write_slice(tmp_path, sessions)
    first = tmp_path / "slice_test" / f"000_{sessions[0]['id']}.json"
    first.rename(tmp_path / "slice_test" / f"002_{sessions[0]['id']}.json")
    with pytest.raises(SliceManifestError, match="manifest order"):
        load_slice_manifest(path)


def test_manifest_refuses_a_wrong_seal_or_open_fields(tmp_path):
    sessions = [_session(0, k=20, inner=0)]
    path = _write_slice(tmp_path, sessions, digest_override="0" * 64)
    with pytest.raises(SliceManifestError, match="sealed digest"):
        load_slice_manifest(path)
    raw = json.loads(path.read_text())
    raw["extra"] = 1
    path.write_text(json.dumps(raw))
    with pytest.raises(SliceManifestError, match="not closed"):
        load_slice_manifest(path)
    raw.pop("extra")
    raw["sessions"][0]["k"] = 0
    path.write_text(json.dumps(raw))
    with pytest.raises(SliceManifestError, match="count field"):
        load_slice_manifest(path)
