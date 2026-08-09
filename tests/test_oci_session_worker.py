from __future__ import annotations

import contextlib
import hashlib
import os
import struct
import sys
from types import SimpleNamespace

import pytest

from cacheon import discovery_overlay
from cacheon.discovery_overlay import (
    DiscoveryActivationReceipt,
    DiscoveryDriverOrigin,
    DiscoverySchedulerMember,
)
from cacheon.eval import engine_worker, oci_session_worker as worker
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


def _receipt(identity: str) -> DiscoveryActivationReceipt:
    return DiscoveryActivationReceipt(
        schema="cacheon.discovery-driver-activation.v1",
        overlay_identity_digest=identity,
        driver_pid=100,
        driver_origin=DiscoveryDriverOrigin(
            "sglang", "0.0.0.dev1", "sglang/__init__.py"
        ),
        scheduler_target_module="sglang.srt.managers.scheduler",
        scheduler_target_qualname="run_scheduler_process",
        tp_size=1,
        members=(
            DiscoverySchedulerMember(101, 0, 0, 0, 0, 0, 0, None),
        ),
    )


def _clear_discovery_environment(monkeypatch) -> None:
    for key in discovery_overlay.DISCOVERY_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def _select_discovery(monkeypatch, identity: str) -> None:
    _clear_discovery_environment(monkeypatch)
    monkeypatch.setenv(discovery_overlay.ARMED, "1")
    monkeypatch.setenv(discovery_overlay.EXPECTED_IDENTITY, identity)


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


def test_worker_derives_discovery_only_from_closed_validator_environment(
    tmp_path, monkeypatch
):
    identity = _h("overlay")
    publication = tmp_path / "publication"
    root = publication / worker.DISCOVERY_OVERLAY_RELPATH
    root.mkdir(parents=True)
    _select_discovery(monkeypatch, identity)
    monkeypatch.setattr(worker, "_artifact_root", lambda: publication)
    monkeypatch.setattr(worker, "_read_only_directory", lambda path: path == root)

    assert worker._requested_discovery_overlay(reference_mode=False) == (
        root,
        identity,
    )
    with pytest.raises(worker.SessionWorkerError, match="session role"):
        worker._requested_discovery_overlay(reference_mode=True)
    monkeypatch.setenv(discovery_overlay.DRIVER_PID, "999")
    with pytest.raises(worker.SessionWorkerError, match="transient"):
        worker._requested_discovery_overlay(reference_mode=False)

    _clear_discovery_environment(monkeypatch)
    assert worker._requested_discovery_overlay(reference_mode=False) is None
    monkeypatch.setenv(discovery_overlay.EXPECTED_IDENTITY, identity)
    with pytest.raises(worker.SessionWorkerError, match="ambient"):
        worker._requested_discovery_overlay(reference_mode=False)


def test_engine_session_arms_before_stock_import_and_receipts_before_ready(
    tmp_path, monkeypatch
):
    config = _config()
    identity = _h("overlay")
    publication = tmp_path / "publication"
    overlay_root = publication / worker.DISCOVERY_OVERLAY_RELPATH
    overlay_root.mkdir(parents=True)
    tree = SimpleNamespace(root=tmp_path / "tree", runtime_manifest=None)
    tree.root.mkdir()
    _select_discovery(monkeypatch, identity)
    monkeypatch.setenv("CACHEON_EXPECTED_SGLANG_VERSION", "0.0.0.dev1")
    monkeypatch.setattr(worker, "_artifact_root", lambda: publication)
    monkeypatch.setattr(worker, "_read_only_directory", lambda path: path == overlay_root)
    order: list[str] = []
    receipt = _receipt(identity)

    def arm(**kwargs):
        assert kwargs == {
            "expected_identity_digest": identity,
            "expected_members": 1,
        }
        assert os.environ[discovery_overlay.DRIVER_PID] == str(os.getpid())
        assert os.environ[discovery_overlay.OVERLAY_ROOT] == str(overlay_root)
        order.append("arm")

    monkeypatch.setattr(discovery_overlay, "arm_driver_activation", arm)
    monkeypatch.setattr(
        discovery_overlay,
        "install_process_role_hook",
        lambda: order.append("install-hook"),
    )
    monkeypatch.setattr(
        worker,
        "_prepare_descendant_bootstrap",
        lambda: order.append("bootstrap"),
    )
    monkeypatch.setattr(
        discovery_overlay,
        "clear_driver_activation",
        lambda: order.append("clear"),
    )

    @contextlib.contextmanager
    def isolated(*_args, **_kwargs):
        assert order == ["arm", "install-hook", "bootstrap"]
        order.append("engine")
        yield SimpleNamespace(engine=object(), require_completion=lambda: None)

    monkeypatch.setattr(engine_worker, "isolated_engine_session", isolated)
    stock = SimpleNamespace(__file__="/installed/sglang/__init__.py")
    monkeypatch.setitem(sys.modules, "sglang", stock)

    def require(module, root, **kwargs):
        assert module is stock and root == overlay_root
        assert kwargs == {
            "expected_identity_digest": identity,
            "expected_members": 1,
            "expected_sglang_version": "0.0.0.dev1",
        }
        order.append("activation-proof")
        return receipt

    monkeypatch.setattr(discovery_overlay, "require_driver_activation", require)
    with worker._engine_session(config, tree) as handle:
        assert handle.discovery_activation_receipt == receipt
        assert order == [
            "arm", "install-hook", "bootstrap", "engine", "activation-proof"
        ]
    assert order[-1] == "clear"


@pytest.mark.parametrize("session_protocol", ("ordinary", "reference"))
def test_non_discovery_arms_never_emit_activation(
    tmp_path, monkeypatch, session_protocol
):
    config = _config()
    tree = SimpleNamespace(root=tmp_path / "tree", runtime_manifest=None)
    tree.root.mkdir()
    _clear_discovery_environment(monkeypatch)
    monkeypatch.setenv("CACHEON_SESSION_PROTOCOL", session_protocol)
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    monkeypatch.setattr(
        discovery_overlay,
        "arm_driver_activation",
        lambda **_kwargs: pytest.fail("non-discovery arm attempted activation"),
    )
    monkeypatch.setattr(worker, "_prepare_descendant_bootstrap", lambda: None)

    @contextlib.contextmanager
    def isolated(*_args, **_kwargs):
        yield SimpleNamespace(engine=object(), require_completion=lambda: None)

    monkeypatch.setattr(engine_worker, "isolated_engine_session", isolated)
    with worker._engine_session(config, tree) as handle:
        assert handle.discovery_activation_receipt is None
        assert all(
            not os.environ.get(key)
            for key in discovery_overlay.DISCOVERY_ENVIRONMENT_KEYS
        )


def test_run_session_places_typed_activation_only_in_ready_frame(
    monkeypatch,
):
    config = _config()
    session = "1" * 32
    launch = _h("launch")
    facts = _facts(config, launch)
    receipt = _receipt(_h("overlay"))
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
        yield SimpleNamespace(
            engine=object(),
            require_completion=lambda: None,
            discovery_activation_receipt=receipt,
        )

    monkeypatch.setattr(worker, "_engine_session", engine_session)
    try:
        assert worker.run_session(input_fd=input_read, output_fd=output_write) == 1
        preflight = parse_frame_bytes(
            _read_frame(output_read), max_bytes=MAX_CONTROL_BYTES
        )
        ready = parse_frame_bytes(
            _read_frame(output_read), max_bytes=MAX_CONTROL_BYTES
        )
        assert "discovery_activation" not in preflight
        assert ready["discovery_activation"] == receipt.to_dict()
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
                {"meta_info": {"output_ids": [10, 20]}},
                {"meta_info": {"output_ids": [30, 40]}},
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

    # Width zero never excuses a missing top-k on an eval-carrying request.
    topk_request = BatchRequest(
        "a" * 32, "b" * 64, "c" * 32, "d" * 32, 0, ("alpha",), 2, 2, 0.0
    )
    with pytest.raises(SessionProtocolError, match="token/top-k"):
        _engine_outputs([{"meta_info": {"output_ids": [10, 20]}}], request=topk_request)


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
