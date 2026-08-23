"""One durable pointer per runtime launch, from the launch to its retained log.

Every attached OCI session already streams its container stderr to a private
host file and keeps that file on both clean and failed teardown. Nothing
recorded which run the file belonged to. The name carries the launch id, and on
the success path the host dropped the diagnostic entirely, so a PASS, a speed
FAIL and a HOLD each left a complete log on disk with nothing pointing at it —
the three outcomes that most need one. Two qualification lanes run at once, so
the file mtime does not disambiguate them either.

This writes ``<launch_id>.run.json`` beside the artifact. It carries the launch
digest, which is the value qualification evidence already records as
``candidate_launch_digest``, so a reservation reaches its own log by a lookup
instead of a guess, and the log itself is found by launch-id prefix.

Nothing here is evidence. It is written outside the content-addressed store, it
is never hashed into a verdict, and it never raises: a run that could not write
its diagnostic pointer is still a valid run, and an instrument that can fail a
qualification is worse than no instrument.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = "cacheon.run-log-index.v1"

#: Suffix of the per-launch pointer. The launch id prefix is shared with the
#: stderr artifact, so ``<launch_id>.*`` is the whole lookup.
SUFFIX = ".run.json"

#: A failure string comes from an exception whose text may be attacker-shaped.
#: The pointer is a diagnostic, not an archive; keep it small and typed.
_MAX_ERROR = 512


def _artifact_fields(diagnostic: object) -> dict[str, Any]:
    """Describe the retained log, or say why there is none."""

    if diagnostic is None:
        return {"log": None, "log_absent_reason": "the session never attached a client"}
    fields: dict[str, Any] = {
        "stream_bytes": getattr(diagnostic, "stream_bytes", None),
        "stream_sha256": getattr(diagnostic, "stream_sha256", None),
        "capture_complete": bool(getattr(diagnostic, "capture_complete", False)),
        "capture_error": getattr(diagnostic, "capture_error", None),
        "client_returncode": getattr(diagnostic, "client_returncode", None),
    }
    artifact = getattr(diagnostic, "artifact", None)
    if artifact is None:
        # The drain reached no clean EOF, so the writer aborted and unlinked its
        # partial file. Saying so is the difference between "no log" and "a log
        # nobody indexed", which are opposite diagnoses.
        fields["log"] = None
        fields["log_absent_reason"] = (
            "the stderr capture did not finish, so no artifact was published"
        )
        return fields
    fields["log"] = {
        "path": str(getattr(artifact, "artifact_path", "")),
        "sha256": getattr(artifact, "artifact_sha256", ""),
        "bytes": getattr(artifact, "artifact_bytes", 0),
        "truncated": bool(getattr(artifact, "truncated", False)),
    }
    return fields


def record(
    diagnostics_root: object,
    *,
    launch_id: str,
    launch_digest: str,
    session_protocol: str,
    diagnostic: object = None,
    error: BaseException | None = None,
) -> None:
    """Write this launch's pointer. Best effort by contract; never raises."""

    try:
        root = Path(str(diagnostics_root))
        if not launch_id or os.sep in launch_id or launch_id.startswith("."):
            raise ValueError(f"unusable launch id {launch_id!r}")
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "launch_id": launch_id,
            "launch_digest": launch_digest,
            "session_protocol": session_protocol,
            "outcome": "error" if error is not None else "ok",
            "error": (
                None
                if error is None
                else f"{type(error).__name__}: {str(error)[:_MAX_ERROR]}"[:_MAX_ERROR]
            ),
            # Wall clock, so several runs of one launch digest stay orderable.
            # Nothing downstream compares it against a signed time.
            "recorded_unix_seconds": time.time(),
            **_artifact_fields(diagnostic),
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode()
        destination = root / (launch_id + SUFFIX)
        temporary = root / f".{launch_id}.{secrets.token_hex(8)}.run.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temporary, flags, 0o600)
        try:
            os.write(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        # Replace rather than link: a retry of the same launch id should end with
        # one pointer, and the pointer is not an authority anyone has read yet.
        os.replace(temporary, destination)
    except Exception:  # noqa: BLE001 - a pointer must never fail its own run
        logger.exception("cacheon: run log pointer not written for %s", launch_id)
        try:
            temporary.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:  # noqa: BLE001
            pass
