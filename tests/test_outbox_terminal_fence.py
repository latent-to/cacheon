from __future__ import annotations

import json
from pathlib import Path

import pytest

from cacheon.chain.outbox_terminal_fence import (
    FENCE_MARKER,
    OutboxTerminalFenceError,
    assert_no_active_recovery_overlap,
    classify_outbox,
    fence_terminal_carriers,
    terminal_result_received,
)


def _carrier(root: Path, name: str, *, state: str, request_id: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "dispatch-state.json").write_text(
        json.dumps(
            {
                "archive_sha256": "a" * 64,
                "request_id": request_id,
                "schema": "cacheon-remote-dispatch-state-v1",
                "state": state,
                "updated_at_unix": 1,
                "worker_epoch": "b" * 32,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_classify_and_fence_terminal_result_received(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    req = "c" * 64
    _carrier(outbox, f"1-{req}", state="result_received", request_id=req)
    _carrier(outbox, "2-pending", state="transferred", request_id="d" * 64)

    carriers = classify_outbox(outbox)
    assert len(carriers) == 2
    terminal = terminal_result_received(carriers)
    assert len(terminal) == 1
    assert terminal[0].request_id == req

    assert_no_active_recovery_overlap(terminal, active_request_ids=())
    with pytest.raises(OutboxTerminalFenceError, match="overlaps an active"):
        assert_no_active_recovery_overlap(terminal, active_request_ids=(req,))

    fence = tmp_path / "terminal-fence"
    moved = fence_terminal_carriers(outbox, fence, terminal)
    assert len(moved) == 1
    assert not (outbox / f"1-{req}").exists()
    assert (moved[0] / FENCE_MARKER).is_file()
    # Non-terminal remains in the live outbox.
    assert (outbox / "2-pending").is_dir()
    assert classify_outbox(outbox)[0].state == "transferred"


def test_fence_refuses_delete_semantics_and_non_terminal(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    live = _carrier(outbox, "x", state="transferred", request_id="e" * 64)
    carriers = classify_outbox(outbox)
    with pytest.raises(OutboxTerminalFenceError, match="non-terminal"):
        fence_terminal_carriers(tmp_path / "fence", tmp_path / "fence2", carriers)
    assert live.is_dir()
