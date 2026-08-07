"""Classify and non-destructively fence terminal outbox carriers (handoff §8).

The CPU outbox may retain ``result_received`` carriers that are not pending
work.  Before a standing supervisor starts, classify them through the durable
dispatch-state files, prove none names an active protected recovery lease, and
move them into an explicitly fenced terminal area.  Never delete blindly.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DISPATCH_SCHEMA = "cacheon-remote-dispatch-state-v1"
FENCE_MARKER = "TERMINAL_FENCED"
_TERMINAL_STATES = frozenset({"result_received"})


class OutboxTerminalFenceError(RuntimeError):
    """Outbox classification or fence operation failed closed."""


@dataclass(frozen=True)
class OutboxCarrier:
    directory: Path
    request_id: str
    state: str
    worker_epoch: str | None
    archive_sha256: str | None
    updated_at_unix: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_sha256": self.archive_sha256,
            "dir": self.directory.name,
            "request_id": self.request_id,
            "state": self.state,
            "updated_at_unix": self.updated_at_unix,
            "worker_epoch": self.worker_epoch,
        }


def _load_dispatch_state(carrier: Path) -> dict[str, object]:
    path = carrier / "dispatch-state.json"
    if not carrier.is_dir() or carrier.is_symlink() or not path.is_file():
        raise OutboxTerminalFenceError(f"carrier is not a regular directory: {carrier}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise OutboxTerminalFenceError(f"dispatch-state unreadable: {exc}") from None
    if type(payload) is not dict or payload.get("schema") != DISPATCH_SCHEMA:
        raise OutboxTerminalFenceError("dispatch-state schema is not closed")
    request_id = payload.get("request_id")
    state = payload.get("state")
    if type(request_id) is not str or not request_id or type(state) is not str:
        raise OutboxTerminalFenceError("dispatch-state identity is malformed")
    return payload


def classify_outbox(outbox: Path) -> tuple[OutboxCarrier, ...]:
    """Return every carrier under ``outbox`` with a closed dispatch-state."""

    if not isinstance(outbox, Path) or not outbox.is_absolute():
        raise OutboxTerminalFenceError("outbox must be one absolute path")
    if not outbox.exists():
        return ()
    rows: list[OutboxCarrier] = []
    for path in sorted(outbox.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or path.is_symlink() or path.name.startswith("."):
            continue
        if not (path / "dispatch-state.json").is_file():
            continue
        payload = _load_dispatch_state(path)
        rows.append(
            OutboxCarrier(
                directory=path,
                request_id=str(payload["request_id"]),
                state=str(payload["state"]),
                worker_epoch=(
                    str(payload["worker_epoch"])
                    if isinstance(payload.get("worker_epoch"), str)
                    else None
                ),
                archive_sha256=(
                    str(payload["archive_sha256"])
                    if isinstance(payload.get("archive_sha256"), str)
                    else None
                ),
                updated_at_unix=(
                    int(payload["updated_at_unix"])
                    if type(payload.get("updated_at_unix")) is int
                    else None
                ),
            )
        )
    return tuple(rows)


def terminal_result_received(
    carriers: Iterable[OutboxCarrier],
) -> tuple[OutboxCarrier, ...]:
    return tuple(row for row in carriers if row.state in _TERMINAL_STATES)


def assert_no_active_recovery_overlap(
    carriers: Iterable[OutboxCarrier],
    active_request_ids: Iterable[str],
) -> None:
    """HOLD the fence if any terminal carrier shares an active recovery request."""

    active = set(active_request_ids)
    overlap = sorted(
        {row.request_id for row in carriers if row.request_id in active}
    )
    if overlap:
        raise OutboxTerminalFenceError(
            "terminal outbox carrier overlaps an active protected request: "
            + ",".join(overlap)
        )


def fence_terminal_carriers(
    outbox: Path,
    fence_root: Path,
    carriers: Iterable[OutboxCarrier],
) -> tuple[Path, ...]:
    """Move terminal carriers into ``fence_root``; leave a marker; never delete."""

    if not isinstance(outbox, Path) or not isinstance(fence_root, Path):
        raise OutboxTerminalFenceError("fence paths must be Path objects")
    if not outbox.is_absolute() or not fence_root.is_absolute():
        raise OutboxTerminalFenceError("fence paths must be absolute")
    if fence_root == outbox or outbox in fence_root.parents:
        raise OutboxTerminalFenceError("fence root must be outside the live outbox")
    fence_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    fence_root.chmod(0o700)
    moved: list[Path] = []
    for row in carriers:
        if row.state not in _TERMINAL_STATES:
            raise OutboxTerminalFenceError(
                f"refusing to fence non-terminal state {row.state!r}"
            )
        if row.directory.parent.resolve() != outbox.resolve():
            raise OutboxTerminalFenceError("carrier is not under the live outbox")
        destination = fence_root / row.directory.name
        if destination.exists():
            raise OutboxTerminalFenceError(
                f"fence destination already exists: {destination}"
            )
        shutil.move(str(row.directory), str(destination))
        marker = destination / FENCE_MARKER
        marker.write_text(
            json.dumps(
                {
                    "request_id": row.request_id,
                    "schema": "cacheon-outbox-terminal-fence-v1",
                    "source_outbox": str(outbox),
                    "state": row.state,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.chmod(marker, 0o400)
        moved.append(destination)
    directory_fd = os.open(fence_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return tuple(moved)


__all__ = [
    "DISPATCH_SCHEMA",
    "FENCE_MARKER",
    "OutboxCarrier",
    "OutboxTerminalFenceError",
    "assert_no_active_recovery_overlap",
    "classify_outbox",
    "fence_terminal_carriers",
    "terminal_result_received",
]
