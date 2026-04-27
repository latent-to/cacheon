"""Unit tests for validator.pod_transport — mocks paramiko internals so
we can verify the SSH/SFTP lifecycle without a real SSH connection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from validator.pod_transport import PodTransport, _KEEPALIVE_S

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _mock_ssh_client():
    """Return a mock ``paramiko.SSHClient`` with sensible defaults."""
    client = MagicMock()
    transport = MagicMock()
    client.get_transport.return_value = transport
    return client, transport


# --------------------------------------------------------------------------- #
# Connection lifecycle
# --------------------------------------------------------------------------- #


class TestConnect:
    @patch("validator.pod_transport.paramiko.SSHClient")
    def test_connect_calls_paramiko_with_correct_args(self, MockSSHClient):
        client, transport = _mock_ssh_client()
        MockSSHClient.return_value = client

        pt = PodTransport(host="gpu.example.com", user="worker", port=2222)
        pt.connect()

        client.set_missing_host_key_policy.assert_called_once()
        client.connect.assert_called_once_with(
            hostname="gpu.example.com",
            port=2222,
            username="worker",
            look_for_keys=True,
            allow_agent=True,
        )

    @patch("validator.pod_transport.paramiko.SSHClient")
    def test_keepalive_is_set(self, MockSSHClient):
        client, transport = _mock_ssh_client()
        MockSSHClient.return_value = client

        pt = PodTransport(host="gpu.example.com", user="worker")
        pt.connect()

        transport.set_keepalive.assert_called_once_with(_KEEPALIVE_S)

    @patch("validator.pod_transport.paramiko.SSHClient")
    def test_second_connect_closes_prior_client(self, MockSSHClient):
        c1, _t1 = _mock_ssh_client()
        c2, _t2 = _mock_ssh_client()
        MockSSHClient.side_effect = [c1, c2]

        pt = PodTransport(host="h", user="u")
        pt.connect()
        pt.connect()

        c1.close.assert_called_once()
        assert pt._client is c2


class TestClose:
    @patch("validator.pod_transport.paramiko.SSHClient")
    def test_close_calls_client_close(self, MockSSHClient):
        client, transport = _mock_ssh_client()
        MockSSHClient.return_value = client

        pt = PodTransport(host="h", user="u")
        pt.connect()
        pt.close()

        client.close.assert_called_once()

    @patch("validator.pod_transport.paramiko.SSHClient")
    def test_close_without_connect_is_safe(self, MockSSHClient):
        pt = PodTransport(host="h", user="u")
        pt.close()


class TestContextManager:
    @patch("validator.pod_transport.paramiko.SSHClient")
    def test_context_manager_connects_and_closes(self, MockSSHClient):
        client, transport = _mock_ssh_client()
        MockSSHClient.return_value = client

        with PodTransport(host="h", user="u") as pt:
            assert pt._client is not None

        client.close.assert_called_once()


# --------------------------------------------------------------------------- #
# Exec
# --------------------------------------------------------------------------- #


class TestExec:
    @patch("validator.pod_transport.paramiko.SSHClient")
    def test_exec_returns_stdout_stderr_exit_code(self, MockSSHClient):
        client, transport = _mock_ssh_client()
        MockSSHClient.return_value = client

        stdout_mock = MagicMock()
        stderr_mock = MagicMock()
        stdout_mock.read.return_value = b"hello\n"
        stderr_mock.read.return_value = b"warn\n"
        stdout_mock.channel.recv_exit_status.return_value = 0
        client.exec_command.return_value = (MagicMock(), stdout_mock, stderr_mock)

        pt = PodTransport(host="h", user="u")
        pt.connect()
        out, err, rc = pt.exec("echo hello")

        assert out == "hello\n"
        assert err == "warn\n"
        assert rc == 0
        client.exec_command.assert_called_once_with("echo hello", timeout=None)

    @patch("validator.pod_transport.paramiko.SSHClient")
    def test_exec_passes_timeout(self, MockSSHClient):
        client, _ = _mock_ssh_client()
        MockSSHClient.return_value = client

        stdout_mock = MagicMock()
        stderr_mock = MagicMock()
        stdout_mock.read.return_value = b""
        stderr_mock.read.return_value = b""
        stdout_mock.channel.recv_exit_status.return_value = 0
        client.exec_command.return_value = (MagicMock(), stdout_mock, stderr_mock)

        pt = PodTransport(host="h", user="u")
        pt.connect()
        pt.exec("long cmd", timeout=600.0)

        client.exec_command.assert_called_once_with("long cmd", timeout=600.0)

    def test_exec_without_connect_raises(self):
        pt = PodTransport(host="h", user="u")
        with pytest.raises(RuntimeError, match="not connected"):
            pt.exec("ls")


# --------------------------------------------------------------------------- #
# Upload / Download
# --------------------------------------------------------------------------- #


class TestUpload:
    @patch("validator.pod_transport.paramiko.SSHClient")
    def test_upload_calls_sftp_put(self, MockSSHClient):
        client, _ = _mock_ssh_client()
        MockSSHClient.return_value = client

        sftp_mock = MagicMock()
        client.open_sftp.return_value = sftp_mock

        pt = PodTransport(host="h", user="u")
        pt.connect()
        pt.upload("/local/file.py", "/remote/file.py")

        sftp_mock.put.assert_called_once_with("/local/file.py", "/remote/file.py")
        sftp_mock.close.assert_called_once()


class TestDownload:
    @patch("validator.pod_transport.paramiko.SSHClient")
    def test_download_calls_sftp_get(self, MockSSHClient, tmp_path):
        client, _ = _mock_ssh_client()
        MockSSHClient.return_value = client

        sftp_mock = MagicMock()
        client.open_sftp.return_value = sftp_mock

        local_dest = tmp_path / "subdir" / "results.json"

        pt = PodTransport(host="h", user="u")
        pt.connect()
        pt.download("/remote/results.json", local_dest)

        sftp_mock.get.assert_called_once_with(
            "/remote/results.json", str(local_dest),
        )
        sftp_mock.close.assert_called_once()
        assert local_dest.parent.exists()

    @patch("validator.pod_transport.paramiko.SSHClient")
    def test_download_creates_parent_dirs(self, MockSSHClient, tmp_path):
        client, _ = _mock_ssh_client()
        MockSSHClient.return_value = client

        sftp_mock = MagicMock()
        client.open_sftp.return_value = sftp_mock

        deeply_nested = tmp_path / "a" / "b" / "c" / "file.json"

        pt = PodTransport(host="h", user="u")
        pt.connect()
        pt.download("/remote/file.json", deeply_nested)

        assert deeply_nested.parent.exists()
