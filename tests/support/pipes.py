"""Shared pipe-backed fake OCI client and manager for the session suites."""

from __future__ import annotations

import os

from cacheon.eval.oci_process import OCIAttachedDiagnostic


class PipeClient:
    def __init__(self) -> None:
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        self.stdin = os.fdopen(request_write, "wb", buffering=0)
        self.stdout = os.fdopen(response_read, "rb", buffering=0)
        self.request_read = request_read
        self.response_write = response_write
        self.closed = self.finalized = self.aborted = False

    def finalize(self) -> None:
        self.finalized = self.closed = True

    def abort(self) -> None:
        self.aborted = self.closed = True

    def close(self) -> None:
        for stream in (self.stdin, self.stdout):
            if not stream.closed:
                stream.close()
        for fd in (self.request_read, self.response_write):
            try:
                os.close(fd)
            except OSError:
                pass


class DiagnosticPipeClient(PipeClient):
    def __init__(self, diagnostic: OCIAttachedDiagnostic) -> None:
        super().__init__()
        self._diagnostic = diagnostic

    def stderr_diagnostic(self) -> OCIAttachedDiagnostic:
        return self._diagnostic


class PipeManager:
    def __init__(self, client: PipeClient) -> None:
        self.client = client
        self.calls = 0

    def spawn_attached(self, _lease, _argv):
        self.calls += 1
        return self.client
