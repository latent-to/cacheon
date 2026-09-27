"""CPU tests for the OCI runtime session registry and engine kwarg additions."""

from __future__ import annotations

import pytest

from cacheon.eval.oci_backend import OCIBackendError, build_runtime_argv


@pytest.mark.parametrize("protocol", ("bogus", "resident"))
def test_runtime_argv_registers_only_ordinary_and_reference_sessions(
    protocol: str,
) -> None:
    # The full argv builder needs a resolved launch; the protocol registry is
    # testable through its guard clause alone.  The resident hot-swap session
    # left with the routing screen, so it is refused like any unknown protocol.
    with pytest.raises(OCIBackendError, match="not registered"):
        build_runtime_argv(
            lease=None,  # type: ignore[arg-type]
            resolved=None,  # type: ignore[arg-type]
            preflight=None,  # type: ignore[arg-type]
            model_root=None,  # type: ignore[arg-type]
            publication=None,  # type: ignore[arg-type]
            cache_root=None,  # type: ignore[arg-type]
            seccomp_path=None,  # type: ignore[arg-type]
            runtime=None,  # type: ignore[arg-type]
            session_protocol=protocol,
        )


class TestEngineKwargAdditions:
    def test_watchdog_timeout_accepted(self) -> None:
        from cacheon.eval.oci_session_protocol import _validate_engine_kwargs

        result = _validate_engine_kwargs({"watchdog_timeout": 1800})
        assert result == {"watchdog_timeout": 1800}

    def test_cuda_graph_bs_accepted_sorted(self) -> None:
        from cacheon.eval.oci_session_protocol import _validate_engine_kwargs

        result = _validate_engine_kwargs({"cuda_graph_bs": [1, 8, 256]})
        assert result == {"cuda_graph_bs": [1, 8, 256]}

    def test_cuda_graph_bs_rejects_unsorted_or_duplicates(self) -> None:
        from cacheon.eval.oci_session_protocol import (
            SessionProtocolError,
            _validate_engine_kwargs,
        )

        for bad in ([256, 8], [8, 8], [0], [], "256"):
            with pytest.raises(SessionProtocolError):
                _validate_engine_kwargs({"cuda_graph_bs": bad})
