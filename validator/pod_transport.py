"""Thin SSH/SFTP wrapper for communicating with the GPU eval pod.

Uses paramiko under the hood.  The validator CPU server connects to the
GPU pod, uploads job files + policy sources, executes ``pod_eval.py``
over SSH, and downloads the results JSON back.

No SSH key path is needed — paramiko auto-discovers
``~/.ssh/id_ed25519``, ``~/.ssh/id_rsa``, etc. and uses the SSH agent.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import paramiko

logger = logging.getLogger(__name__)

_KEEPALIVE_S: int = 30


class PodTransport:
    """Connect to a remote GPU pod via SSH and transfer files via SFTP."""

    def __init__(self, host: str, user: str, port: int = 22) -> None:
        self.host = host
        self.user = user
        self.port = port
        self._client: paramiko.SSHClient | None = None

    # -- lifecycle ------------------------------------------------------------

    def connect(self) -> None:
        self.close()  # drop any prior connection before opening a new one
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.WarningPolicy())
        client.connect(
            hostname=self.host,
            port=self.port,
            username=self.user,
            look_for_keys=True,
            allow_agent=True,
        )
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(_KEEPALIVE_S)
        self._client = client
        logger.info(
            "SSH connected to %s@%s:%d", self.user, self.host, self.port,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> PodTransport:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- commands -------------------------------------------------------------

    @property
    def _ssh(self) -> paramiko.SSHClient:
        if self._client is None:
            raise RuntimeError("PodTransport is not connected; call connect() first")
        return self._client

    def exec(
        self,
        command: str,
        *,
        timeout: float | None = None,
    ) -> tuple[str, str, int]:
        """Run *command* on the pod and block until it exits.

        Returns ``(stdout, stderr, exit_code)``.
        """
        logger.debug("SSH exec: %s", command)
        _stdin, stdout, stderr = self._ssh.exec_command(
            command, timeout=timeout,
        )
        # Drain stdout and stderr concurrently — sequential reads can deadlock
        # when one stream fills the shared channel window (paramiko#1778).
        err_chunks: list[bytes] = []

        def _drain_stderr():
            err_chunks.append(stderr.read())

        t = threading.Thread(target=_drain_stderr, daemon=True)
        t.start()
        out = stdout.read().decode("utf-8", errors="replace")
        t.join()
        err = b"".join(err_chunks).decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        return out, err, exit_code

    # -- file transfer --------------------------------------------------------

    def upload(self, local_path: str | os.PathLike, remote_path: str) -> None:
        """SFTP-put a local file to *remote_path* on the pod."""
        sftp = self._ssh.open_sftp()
        try:
            sftp.put(str(local_path), remote_path)
            logger.debug("SFTP put %s → %s", local_path, remote_path)
        finally:
            sftp.close()

    def download(self, remote_path: str, local_path: str | os.PathLike) -> None:
        """SFTP-get a file from the pod to *local_path*."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        sftp = self._ssh.open_sftp()
        try:
            sftp.get(remote_path, str(local_path))
            logger.debug("SFTP get %s → %s", remote_path, local_path)
        finally:
            sftp.close()
