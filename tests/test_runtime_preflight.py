"""Adversarial tests for the candidate-free stock runtime preflight."""

from __future__ import annotations

import json
import subprocess

import pytest

from optima.eval import runtime_preflight as rp
from optima.eval.runtime_preflight import (
    CommandResult,
    RuntimePreflightConfig,
    RuntimePreflightError,
    run_runtime_preflight,
)


IMAGE = "registry.example/sglang@sha256:" + "a" * 64
LOCAL_IMAGE_ID = "sha256:" + "b" * 64
DOCKER = "/usr/bin/docker"
CONTAINER_NAME = "optima-stock-preflight-" + "1" * 20


class Clock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


class ScriptedRunner:
    def __init__(self, results, *, clock: Clock | None = None, advances=()):
        self.results = list(results)
        self.calls = []
        self.clock = clock
        self.advances = list(advances)

    def __call__(
        self,
        argv,
        *,
        timeout_s,
        max_stdout_bytes,
        max_stderr_bytes,
    ):
        self.calls.append((
            tuple(argv),
            float(timeout_s),
            max_stdout_bytes,
            max_stderr_bytes,
        ))
        if self.clock is not None and self.advances:
            self.clock.value += self.advances.pop(0)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _config(**changes) -> RuntimePreflightConfig:
    values = {
        "image": IMAGE,
        "expected_sglang_version": "0.5.2",
        "uid": 65532,
        "gid": 65532,
        "docker_binary": DOCKER,
        "timeout_s": 60.0,
    }
    values.update(changes)
    return RuntimePreflightConfig(**values)


def _inspect(*, repo_digests=None, image_id=LOCAL_IMAGE_ID, volumes=None, extra=None):
    payload = {
        "Id": image_id,
        "RepoDigests": [IMAGE] if repo_digests is None else repo_digests,
        "Volumes": volumes,
    }
    if extra:
        payload.update(extra)
    return CommandResult(0, json.dumps(payload).encode(), b"")


def _container_payload(*, version="0.5.2", extra=None):
    payload = {
        "schema": rp.CONTAINER_RECEIPT_SCHEMA,
        "sglang_version": version,
        "python": {
            "implementation": "cpython",
            "version": "3.11.15",
            "abi": "cpython-311-x86_64-linux-gnu",
            "platform": "linux-x86_64",
            "machine": "x86_64",
        },
        "packages": {
            "cuda-python": "12.9.0",
            "flashinfer-python": "0.6.12",
            "nvidia-cuda-runtime-cu12": "12.9.79",
            "torch": "2.9.1",
            "triton": "3.5.1",
        },
        "cuda": {
            "cudart_library": "libcudart.so.12",
            "cuda_visible_devices": "",
            "nvidia_visible_devices": "void",
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _container(**changes):
    return CommandResult(0, json.dumps(_container_payload(**changes)).encode(), b"")


def _successful_runner(**container_changes):
    return ScriptedRunner([_inspect(), _container(**container_changes)])


def test_success_binds_manifest_local_id_and_returns_hashable_canonical_receipt():
    runner = _successful_runner()
    receipt = run_runtime_preflight(_config(), runner=runner, clock=Clock())

    assert receipt.requested_image == IMAGE
    assert receipt.requested_manifest_digest == "sha256:" + "a" * 64
    assert receipt.local_image_id == LOCAL_IMAGE_ID
    assert receipt.repo_digests == (IMAGE,)
    assert receipt.sglang_version == "0.5.2"
    assert receipt.uid == receipt.gid == 65532
    assert len(receipt.sha256) == 64
    assert receipt.sha256 == rp.hashlib.sha256(
        receipt.canonical_json.encode("ascii")
    ).hexdigest()
    assert hash(receipt)
    assert json.loads(receipt.canonical_json) == receipt.canonical_payload()


def test_exact_host_inspect_and_container_security_argv(monkeypatch):
    monkeypatch.setattr(rp.secrets, "token_hex", lambda size: "1" * (size * 2))
    runner = _successful_runner()
    config = _config()
    receipt = run_runtime_preflight(config, runner=runner, clock=Clock())

    inspect_argv = runner.calls[0][0]
    assert inspect_argv == (
        DOCKER,
        "image",
        "inspect",
        rp._INSPECT_FORMAT,
        IMAGE,
    )
    run_argv = runner.calls[1][0]
    container_name = CONTAINER_NAME
    assert run_argv == (
        DOCKER,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--read-only",
        "--runtime=runc",
        "--ipc=none",
        f"--name={container_name}",
        "--stop-timeout=1",
        "--no-healthcheck",
        "--user=65532:65532",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        "--security-opt=seccomp=builtin",
        "--pids-limit=32",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1.0",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m",
        "--workdir=/tmp",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=CUDA_VISIBLE_DEVICES=",
        "--log-driver=none",
        "--entrypoint=python3",
        LOCAL_IMAGE_ID,
        "-I",
        "-S",
        "-c",
        rp._CONTAINER_SCRIPT,
    )
    assert not any(
        arg == "--gpus" or arg.startswith(("--gpus=", "--mount=", "--volume="))
        or arg in {"-v", "--mount", "--volume"}
        for arg in run_argv
    )
    assert all("candidate" not in arg for arg in run_argv)
    assert "distributions(path=paths)" in rp._CONTAINER_SCRIPT
    assert "sys.path.append(path)" in rp._CONTAINER_SCRIPT
    assert receipt.security_argv_sha256 == rp.hashlib.sha256(
        json.dumps(run_argv, separators=(",", ":")).encode()
    ).hexdigest()


def test_each_preflight_uses_a_unique_trusted_container_name(monkeypatch):
    tokens = iter(("1" * 20, "2" * 20))
    monkeypatch.setattr(rp.secrets, "token_hex", lambda _size: next(tokens))
    runners = (_successful_runner(), _successful_runner())

    for runner in runners:
        run_runtime_preflight(_config(), runner=runner, clock=Clock())

    names = [
        next(arg for arg in runner.calls[1][0] if arg.startswith("--name="))
        for runner in runners
    ]
    assert names == [
        "--name=optima-stock-preflight-" + "1" * 20,
        "--name=optima-stock-preflight-" + "2" * 20,
    ]


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"image": "registry.example/sglang:latest"}, "name@sha256"),
        ({"image": "registry.example/sglang@sha256:" + "A" * 64}, "name@sha256"),
        ({"docker_binary": "docker"}, "absolute normalized"),
        ({"docker_binary": "//usr/bin/docker"}, "absolute normalized"),
        ({"docker_binary": "/usr/bin/../bin/docker"}, "absolute normalized"),
        ({"docker_binary": "/usr/bin/docker;rm"}, "absolute normalized"),
        ({"uid": 0}, "nonzero"),
        ({"uid": True}, "nonzero"),
        ({"gid": 0}, "nonzero"),
        ({"expected_sglang_version": "0.5.2;bad"}, "version"),
    ],
)
def test_config_rejects_mutable_image_unsafe_docker_and_root_ids(changes, match):
    with pytest.raises(RuntimePreflightError, match=match):
        _config(**changes)


def test_repo_digest_must_exactly_bind_requested_image_to_local_id():
    other = "registry.example/sglang@sha256:" + "d" * 64
    runner = ScriptedRunner([_inspect(repo_digests=[other])])

    with pytest.raises(RuntimePreflightError, match="not bound"):
        run_runtime_preflight(_config(), runner=runner, clock=Clock())
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "inspect_result, match",
    [
        (CommandResult(0, b"not-json", b""), "malformed JSON"),
        (_inspect(extra={"RepoTags": []}), "keys/type mismatch"),
        (_inspect(image_id="sha256:short"), "local image ID"),
        (_inspect(repo_digests=[IMAGE, IMAGE]), "invalid RepoDigests"),
        (_inspect(volumes={"/candidate-state": {}}), "Dockerfile volumes"),
    ],
)
def test_image_inspect_is_strict_and_bounded(inspect_result, match):
    runner = ScriptedRunner([inspect_result])
    with pytest.raises(RuntimePreflightError, match=match):
        run_runtime_preflight(_config(), runner=runner, clock=Clock())


@pytest.mark.parametrize(
    "raw, match",
    [
        (b"not-json", "malformed JSON"),
        (
            json.dumps(_container_payload()).encode() + b"\nextra",
            "malformed JSON",
        ),
        (
            json.dumps(_container_payload(extra={"candidate": "forbidden"})).encode(),
            "keys/type mismatch",
        ),
    ],
)
def test_container_receipt_rejects_malformed_or_extra_output(raw, match):
    runner = ScriptedRunner([_inspect(), CommandResult(0, raw, b"")])
    with pytest.raises(RuntimePreflightError, match=match):
        run_runtime_preflight(_config(), runner=runner, clock=Clock())


def test_wrong_sglang_version_is_a_validator_fault():
    runner = _successful_runner(version="0.5.3")
    with pytest.raises(RuntimePreflightError, match="installed sglang mismatch") as caught:
        run_runtime_preflight(_config(), runner=runner, clock=Clock())
    assert caught.value.validator_fault is True
    assert caught.value.retryable is False


def test_timeout_is_bounded_by_one_absolute_deadline(monkeypatch):
    monkeypatch.setattr(rp.secrets, "token_hex", lambda size: "1" * (size * 2))
    clock = Clock()
    runner = ScriptedRunner(
        [
            _inspect(),
            subprocess.TimeoutExpired((DOCKER, "run"), 1.0),
            CommandResult(0, b"removed\n", b""),
        ],
        clock=clock,
        advances=(59.0, 0.0),
    )
    with pytest.raises(RuntimePreflightError, match="timed out") as caught:
        run_runtime_preflight(_config(), runner=runner, clock=clock)
    assert runner.calls[0][1] == pytest.approx(60.0)
    assert runner.calls[1][1] == pytest.approx(1.0)
    assert runner.calls[2][0] == (
        DOCKER,
        "rm",
        "--force",
        "--volumes",
        CONTAINER_NAME,
    )
    assert runner.calls[2][1] == pytest.approx(5.0)
    assert caught.value.validator_fault is True


def test_nonzero_exit_stderr_and_runner_output_limit_fail_closed():
    for result, match in (
        (CommandResult(2, b"", b"daemon unavailable"), "exited 2"),
        (_inspect(), None),
    ):
        if match is not None:
            with pytest.raises(RuntimePreflightError, match=match):
                run_runtime_preflight(
                    _config(), runner=ScriptedRunner([result]), clock=Clock()
                )
    noisy = ScriptedRunner([
        _inspect(),
        CommandResult(0, json.dumps(_container_payload()).encode(), b"warning"),
        CommandResult(0, b"removed\n", b""),
    ])
    with pytest.raises(RuntimePreflightError, match="unexpected stderr"):
        run_runtime_preflight(_config(), runner=noisy, clock=Clock())

    oversized = ScriptedRunner([
        CommandResult(0, b"x" * (rp.MAX_INSPECT_STDOUT_BYTES + 1), b"")
    ])
    with pytest.raises(RuntimePreflightError, match="output bounds"):
        run_runtime_preflight(_config(), runner=oversized, clock=Clock())


def test_default_runner_invokes_subprocess_with_shell_false(monkeypatch):
    captured = {}

    def refuse(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        raise OSError("test stop")

    monkeypatch.setattr(rp.subprocess, "Popen", refuse)
    with pytest.raises(RuntimePreflightError, match="cannot execute"):
        rp._bounded_argv_runner(
            (DOCKER, "version"),
            timeout_s=1.0,
            max_stdout_bytes=16,
            max_stderr_bytes=16,
        )
    assert captured["argv"] == [DOCKER, "version"]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL


def test_injected_clock_and_runner_types_fail_as_validator_faults():
    with pytest.raises(RuntimePreflightError, match="clock") as clock_error:
        run_runtime_preflight(_config(), runner=_successful_runner(), clock=lambda: "bad")
    assert clock_error.value.validator_fault is True

    runner = ScriptedRunner([CommandResult(0, "not-bytes", b"")])
    with pytest.raises(RuntimePreflightError, match="field types") as runner_error:
        run_runtime_preflight(_config(), runner=runner, clock=Clock())
    assert runner_error.value.validator_fault is True


def test_deeply_nested_json_is_wrapped_as_validator_fault():
    nested = b"[" * 2000 + b"0" + b"]" * 2000
    runner = ScriptedRunner([CommandResult(0, nested, b"")])
    with pytest.raises(RuntimePreflightError, match="malformed JSON") as caught:
        run_runtime_preflight(_config(), runner=runner, clock=Clock())
    assert caught.value.validator_fault is True
