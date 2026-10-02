"""Session-level worker errors on the replay reader keep their candidate attribution."""

import asyncio
import os
import time

import pytest

from cacheon.eval.oci_outer_session import OuterSessionCandidateError, OuterSessionProtocolError
from cacheon.eval.oci_session_protocol import MAX_CONTROL_BYTES, error_message, frame_message
from tests.test_oci_outer_session import _attached, _request


def test_replay_reader_keeps_candidate_attribution_of_a_session_level_error() -> None:
    class CandidateEngineFailure(RuntimeError):
        pass

    current = _request(1, request_id="4" * 32)
    for bound, expected in ((None, OuterSessionCandidateError), (_request(0), OuterSessionProtocolError)):
        transport, client = _attached()
        try:
            error = error_message(session_id=current.session_id, launch_digest=current.launch_digest, stage="batch",
                                  error=CandidateEngineFailure("rank 0 tree_cache: the cache served a page"), request=bound)
            os.write(client.response_write, frame_message(error, max_bytes=MAX_CONTROL_BYTES))
            with pytest.raises(expected):
                asyncio.run(transport.aread_response({current.request_id: current}, deadline=time.monotonic() + 1))
        finally:
            transport.abort()
            client.close()
