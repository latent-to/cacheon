from __future__ import annotations

from contextlib import contextmanager

from optima.eval import oci_session_worker as worker
from optima.eval.oci_session_protocol import (
    MAX_CONTROL_BYTES,
    make_init,
    parse_frame_bytes,
)
from optima.eval.throughput_kl import EvalConfig


def _config() -> EvalConfig:
    return EvalConfig(
        model_path="/models/model",
        num_prompts=1,
        max_new_tokens=1,
        top_logprobs_num=1,
        warmup_iters=1,
        conditioning_iters=1,
        timed_iters=1,
        tp_size=1,
    )


def _runtime_attestation() -> dict[str, object]:
    return {
        "verified": True,
        "sglang_version": "0.0.test",
        "referee_source_digest": "sha256:" + "1" * 64,
        "model_revision": "2" * 40,
        "model_manifest_digest": "sha256:" + "3" * 64,
        "model_content_digest": "sha256:" + "4" * 64,
        "environment_sha256": "5" * 64,
        "gpu_count": 1,
        "gpu_architectures": ["sm120"],
        "topology_sha256": "6" * 64,
    }


def _patch_fds(monkeypatch, init):
    monkeypatch.setattr(worker, "_control_input_fd", lambda _fd: 98)
    monkeypatch.setattr(worker.os, "close", lambda _fd: None)
    monkeypatch.setattr(worker, "_read_json_frame", lambda *_a, **_k: init)


def test_worker_publishes_live_preflight_before_engine_or_candidate_entry(monkeypatch):
    events: list[str] = []
    session_id = "a" * 32
    init = make_init(_config(), mode="candidate", session_id=session_id)
    _patch_fds(monkeypatch, init)
    monkeypatch.setattr(
        worker,
        "_assert_live_session_sandbox",
        lambda **_kwargs: events.append("sandbox"),
    )

    def attest():
        events.append("attest")
        return _runtime_attestation()

    monkeypatch.setattr("optima.eval.oci_worker.attest_runtime", attest)

    def write(_fd, payload):
        message = parse_frame_bytes(payload, max_bytes=MAX_CONTROL_BYTES)
        events.append("write-" + str(message["type"]))

    monkeypatch.setattr(worker, "_write_all", write)

    @contextmanager
    def engine_context(*_args, **_kwargs):
        events.append("engine-context")
        raise RuntimeError("stop after proving order")
        yield  # pragma: no cover

    monkeypatch.setattr(worker, "_engine_context", engine_context)

    assert worker.run_session(input_fd=0, output_fd=99) == 1
    assert events[:4] == [
        "sandbox",
        "attest",
        "write-preflight",
        "engine-context",
    ]


def test_worker_attestation_failure_prevents_engine_or_candidate_entry(monkeypatch):
    events: list[str] = []
    init = make_init(_config(), mode="candidate", session_id="b" * 32)
    _patch_fds(monkeypatch, init)
    monkeypatch.setattr(worker, "_assert_live_session_sandbox", lambda **_k: None)

    def attest():
        events.append("attest")
        raise RuntimeError("runtime identity mismatch")

    monkeypatch.setattr("optima.eval.oci_worker.attest_runtime", attest)

    def forbidden(*_args, **_kwargs):
        events.append("engine-context")
        raise AssertionError("engine context must not be reached")

    monkeypatch.setattr(worker, "_engine_context", forbidden)
    monkeypatch.setattr(worker, "_write_all", lambda *_a, **_k: None)

    assert worker.run_session(input_fd=0, output_fd=99) == 1
    assert events == ["attest"]
