"""The pointer from a finished launch to its retained log.

The behaviour under test is not "a file appears". It is that the pointer exists
for the quiet outcomes — a PASS, a speed FAIL, a HOLD — because those are the
runs whose logs used to sit on disk with nothing naming them, and that a broken
pointer can never take a run down with it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cacheon.eval import run_log_index


class _Artifact:
    def __init__(self, path: Path) -> None:
        self.artifact_path = path
        self.artifact_sha256 = "a" * 64
        self.artifact_bytes = 4096
        self.truncated = False


class _Diagnostic:
    def __init__(self, artifact: _Artifact | None, *, complete: bool = True) -> None:
        self.artifact = artifact
        self.capture_complete = complete
        self.capture_error = None
        self.client_returncode = 0
        self.stream_bytes = 4096
        self.stream_sha256 = "b" * 64


def _written(root: Path, launch_id: str) -> dict:
    return json.loads((root / (launch_id + run_log_index.SUFFIX)).read_text())


def test_a_clean_run_still_names_its_log(tmp_path: Path) -> None:
    log = tmp_path / "runtime-1.token.stderr"
    log.write_text("engine output")
    run_log_index.record(
        tmp_path,
        launch_id="runtime-1",
        launch_digest="d" * 64,
        session_protocol="ordinary",
        diagnostic=_Diagnostic(_Artifact(log)),
    )
    payload = _written(tmp_path, "runtime-1")
    assert payload["outcome"] == "ok"
    assert payload["error"] is None
    assert payload["launch_digest"] == "d" * 64
    assert payload["log"]["path"] == str(log)
    assert payload["log"]["bytes"] == 4096


def test_the_launch_id_prefix_finds_the_log_beside_the_pointer(tmp_path: Path) -> None:
    """The lookup contract: one prefix reaches both files."""

    log = tmp_path / "runtime-7.abc.stderr"
    log.write_text("engine output")
    run_log_index.record(
        tmp_path,
        launch_id="runtime-7",
        launch_digest="d" * 64,
        session_protocol="ordinary",
        diagnostic=_Diagnostic(_Artifact(log)),
    )
    found = sorted(p.name for p in tmp_path.glob("runtime-7*"))
    assert found == ["runtime-7.abc.stderr", "runtime-7.run.json"]


def test_a_failed_run_records_the_failure_text(tmp_path: Path) -> None:
    run_log_index.record(
        tmp_path,
        launch_id="runtime-2",
        launch_digest="d" * 64,
        session_protocol="ordinary",
        diagnostic=_Diagnostic(_Artifact(tmp_path / "runtime-2.token.stderr")),
        error=RuntimeError("the engine died during capture"),
    )
    payload = _written(tmp_path, "runtime-2")
    assert payload["outcome"] == "error"
    assert payload["error"] == "RuntimeError: the engine died during capture"


def test_a_hostile_failure_string_cannot_grow_the_pointer(tmp_path: Path) -> None:
    run_log_index.record(
        tmp_path,
        launch_id="runtime-3",
        launch_digest="d" * 64,
        session_protocol="ordinary",
        error=RuntimeError("x" * 100_000),
    )
    assert len(_written(tmp_path, "runtime-3")["error"]) <= 512


def test_an_unpublished_capture_says_so_instead_of_going_quiet(tmp_path: Path) -> None:
    """"No log" and "a log nobody indexed" are opposite diagnoses."""

    run_log_index.record(
        tmp_path,
        launch_id="runtime-4",
        launch_digest="d" * 64,
        session_protocol="ordinary",
        diagnostic=_Diagnostic(None, complete=False),
    )
    payload = _written(tmp_path, "runtime-4")
    assert payload["log"] is None
    assert "did not finish" in payload["log_absent_reason"]


def test_a_session_that_never_attached_says_that(tmp_path: Path) -> None:
    run_log_index.record(
        tmp_path,
        launch_id="runtime-5",
        launch_digest="d" * 64,
        session_protocol="reference",
        diagnostic=None,
    )
    payload = _written(tmp_path, "runtime-5")
    assert payload["log"] is None
    assert "never attached" in payload["log_absent_reason"]


def test_a_retry_of_one_launch_leaves_one_pointer(tmp_path: Path) -> None:
    for attempt in ("first", "second"):
        run_log_index.record(
            tmp_path,
            launch_id="runtime-6",
            launch_digest="d" * 64,
            session_protocol="ordinary",
            error=RuntimeError(attempt),
        )
    assert len(list(tmp_path.glob("runtime-6*"))) == 1
    assert _written(tmp_path, "runtime-6")["error"].endswith("second")


def test_an_unwritable_root_never_fails_the_run(tmp_path: Path) -> None:
    """A run that could not write its pointer is still a valid run."""

    sealed = tmp_path / "sealed"
    sealed.mkdir(mode=0o500)
    try:
        run_log_index.record(
            sealed,
            launch_id="runtime-8",
            launch_digest="d" * 64,
            session_protocol="ordinary",
        )
        assert list(sealed.glob("*")) == []
    finally:
        os.chmod(sealed, 0o700)


def test_a_launch_id_cannot_escape_the_diagnostics_root(tmp_path: Path) -> None:
    run_log_index.record(
        tmp_path,
        launch_id="../escaped",
        launch_digest="d" * 64,
        session_protocol="ordinary",
    )
    assert not (tmp_path.parent / ("escaped" + run_log_index.SUFFIX)).exists()
    assert list(tmp_path.glob("*")) == []
