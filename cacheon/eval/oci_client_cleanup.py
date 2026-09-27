"""Close the foreground client before the final exact-lease absence proof."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cacheon.eval.oci_process import OCIAttachedClient, OCILease, OCIProcessManager


def cleanup_client(
    manager: OCIProcessManager,
    lease: OCILease,
    process: subprocess.Popen[bytes],
    *,
    client: OCIAttachedClient | None = None,
) -> None:
    """Retain cleanup errors unless only the first removal was still pending.

    A bounded Docker removal can precede destruction of a GPU container. Only the
    final absence proof resolves this; identity, process and stream errors remain fatal.
    """
    from cacheon.eval.oci_process import OCIProcessError, _ContainerRemovalPending

    failures: list[BaseException] = []
    try:
        manager.force_remove_container(lease)
    except _ContainerRemovalPending:
        pass
    except BaseException as exc:
        failures.append(exc)
    try:
        manager._terminate_client(process)
    except BaseException as exc:
        failures.append(exc)
    if client is not None:
        client._finish_stderr_capture()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is None or stream.closed:
                continue
            try:
                stream.close()
            except BaseException as exc:
                failures.append(exc)
        client._finish_stderr_capture()
    # A foreground Docker client can create the named container after the first
    # absence proof. Only this proof after process-group death closes that race.
    try:
        manager.force_remove_container(lease)
    except BaseException as exc:
        failures.append(exc)
    if failures:
        role = "attached OCI" if client is not None else "OCI failure"
        raise OCIProcessError(
            f"{role} cleanup could not prove both client and container removal"
        ) from failures[0]
