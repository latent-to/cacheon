from __future__ import annotations

import contextlib
import hashlib
import os
import struct
from types import SimpleNamespace

import pytest

from cacheon.eval import oci_session_worker as worker
from cacheon.eval.oci_session_protocol import (
    CONTROL_MAGIC,
    FRAME_HEADER_BYTES,
    MAX_CONTROL_BYTES,
    EngineSessionConfig,
    RuntimePreflightFacts,
    frame_message,
    make_init,
    parse_frame_bytes,
    preflight_accept_message,
    ready_message,
)


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _config() -> EngineSessionConfig:
    return EngineSessionConfig(
        model_path="/cacheon/input/model",
        dtype="bfloat16",
        deterministic=False,
        attention_backend=None,
        disable_cuda_graph=False,
        mem_fraction_static=0.8,
        log_level="warning",
        max_running_requests=16,
        tp_size=1,
        moe_runner_backend=None,
        disable_custom_all_reduce=False,
        engine_kwargs={},
    )


def _facts(config: EngineSessionConfig, launch: str) -> RuntimePreflightFacts:
    return RuntimePreflightFacts(
        launch_digest=launch,
        runtime_digest=_h("runtime"),
        stack_digest=_h("stack"),
        tree_digest=_h("tree"),
        engine_config_digest=config.digest,
        worker_distribution_digest=_h("worker"),
        model_revision_digest=_h("revision"),
        model_manifest_digest=_h("manifest"),
        model_content_digest=_h("content"),
        sglang_version="0.0.0.dev1",
        gpu_architectures=("sm120",),
        topology_digest=_h("topology"),
        loopback_only=True,
        read_only_inputs=True,
        private_writable_cache=True,
    )


def _read_exact(fd: int, size: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        chunk = os.read(fd, size - len(payload))
        assert chunk
        payload.extend(chunk)
    return bytes(payload)


def _read_frame(fd: int) -> bytes:
    header = _read_exact(fd, FRAME_HEADER_BYTES)
    assert header[:4] == CONTROL_MAGIC
    size = struct.unpack(">I", header[4:])[0]
    return header + _read_exact(fd, size)


def test_run_session_emits_preflight_then_the_exact_ready_envelope(
    monkeypatch,
):
    config = _config()
    session = "1" * 32
    launch = _h("launch")
    facts = _facts(config, launch)
    monkeypatch.setenv("CACHEON_LAUNCH_DIGEST", launch)
    monkeypatch.setenv("CACHEON_ENGINE_CONFIG_DIGEST", config.digest)
    init = frame_message(
        make_init(
            config,
            session_id=session,
            launch_digest=launch,
            expected_engine_config_digest=config.digest,
        ),
        max_bytes=MAX_CONTROL_BYTES,
    )
    accept = frame_message(
        preflight_accept_message(
            session_id=session, launch_digest=launch, facts=facts
        ),
        max_bytes=MAX_CONTROL_BYTES,
    )
    input_read, input_write = os.pipe()
    output_read, output_write = os.pipe()
    os.write(input_write, init + accept)
    os.close(input_write)
    monkeypatch.setattr(
        worker,
        "_validate_live_preflight",
        lambda _config, *, launch_digest: (
            facts,
            SimpleNamespace(root="tree", runtime_manifest=None),
        ),
    )

    @contextlib.contextmanager
    def engine_session(_config, _tree):
        yield SimpleNamespace(engine=object(), require_completion=lambda: None)

    monkeypatch.setattr(worker, "_engine_session", engine_session)
    try:
        assert worker.run_session(input_fd=input_read, output_fd=output_write) == 1
        preflight = parse_frame_bytes(
            _read_frame(output_read), max_bytes=MAX_CONTROL_BYTES
        )
        ready = parse_frame_bytes(
            _read_frame(output_read), max_bytes=MAX_CONTROL_BYTES
        )
        assert preflight["type"] == "preflight"
        assert ready == ready_message(session_id=session, launch_digest=launch)
    finally:
        os.close(input_read)
        os.close(output_write)
        os.close(output_read)


def test_pure_generation_request_disables_engine_logprob_work() -> None:
    from cacheon.eval.oci_session_protocol import BatchRequest, SessionProtocolError
    from cacheon.eval.oci_session_worker import _engine_outputs, _generate

    request = BatchRequest(
        "a" * 32, "b" * 64, "c" * 32, "d" * 32, 0, ("alpha", "beta"), 2, 0, 0.0
    )
    calls: list[dict] = []

    class _Engine:
        def generate(self, **kwargs):
            calls.append(kwargs)
            # A logprob-free engine response: token IDs only, no top-k
            # structure anywhere in meta_info.
            return [
                {"meta_info": {"output_ids": [10, 20], "prompt_tokens": 3}},
                {"meta_info": {"output_ids": [30, 40], "prompt_tokens": 3}},
            ]

    evidence = _generate(_Engine(), request)
    assert calls[0]["return_logprob"] is False
    assert calls[0]["top_logprobs_num"] == 0
    assert tuple(prompt.output_ids for prompt in evidence.prompts) == (
        (10, 20),
        (30, 40),
    )
    assert all(
        position == ()
        for prompt in evidence.prompts
        for position in prompt.top_logprobs
    )

    assert all(prompt.prompt_tokens == 3 for prompt in evidence.prompts)

    # A response without the engine's own prompt token count is an
    # infrastructure fault, never a gradable read.
    with pytest.raises(SessionProtocolError, match="prompt token count"):
        _engine_outputs(
            [{"meta_info": {"output_ids": [10, 20]}}] * 2, request=request
        )

    # Width zero never excuses a missing top-k on an eval-carrying request.
    topk_request = BatchRequest(
        "a" * 32, "b" * 64, "c" * 32, "d" * 32, 0, ("alpha",), 2, 2, 0.0
    )
    with pytest.raises(SessionProtocolError, match="token/top-k"):
        _engine_outputs(
            [{"meta_info": {"output_ids": [10, 20], "prompt_tokens": 3}}],
            request=topk_request,
        )


def test_width_zero_reference_scoring_skips_the_support_gather() -> None:
    # Teacher-NLL-only mode: no token_ids_logprob is asked of the teacher
    # engine and the evidence carries exact empty support rows, while the
    # target NLL and teacher argmax remain fully validated.
    from types import SimpleNamespace

    from cacheon.eval.oci_session_worker import _reference_role_evidence

    calls: list[dict] = []

    class _Teacher:
        def generate(self, **kwargs):
            calls.append(kwargs)
            # Prefix length 3, response length 2: teacher scores positions
            # from logprob_start_len; the worker slices the last 2 entries.
            return [
                {
                    "meta_info": {
                        "input_token_logprobs": [
                            (-0.5, 9), (-0.25, 11), (-0.75, 12),
                        ],
                        "input_top_logprobs": [
                            [(-0.4, 3)], [(-0.2, 11)], [(-0.6, 4)],
                        ],
                    }
                }
            ]

    role = SimpleNamespace(output_ids=(11, 12), supports=((), ()))
    rows = _reference_role_evidence(
        _Teacher(), [[1, 2, 3]], [role], vocab_size=128
    )
    assert "token_ids_logprob" not in calls[0]
    assert calls[0]["return_logprob"] is True and calls[0]["top_logprobs_num"] == 1
    tokens = rows[0].tokens
    assert [token.target_logprob for token in tokens] == [-0.25, -0.75]
    assert [token.true_argmax_token_id for token in tokens] == [11, 4]
    assert all(token.support_logprobs == () for token in tokens)
