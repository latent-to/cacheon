"""Unit tests for validator.docker_eval -- all mock-based, no Docker or GPU."""

from __future__ import annotations

import io
import json
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from validator.chain import CommitmentRecord
from validator.docker_eval import (
    INTERNAL_NETWORK,
    RawPromptResult,
    capture_container_logs,
    ensure_eval_network,
    evaluate_challenger,
    pull_image,
    reset_gpu_state,
    run_baseline_if_needed,
    send_prompt,
    start_container,
    stop_and_remove,
    wait_for_health,
)
from validator.baseline import BaselineCache, BaselinePromptResult
from validator.eval_schema import ChatMessage, Prompt

pytestmark = pytest.mark.unit

_IMAGE = "registry.example.com/miner:v1"
_DIGEST = "sha256:" + "a" * 64


# --------------------------------------------------------------------------- #
# pull_image
# --------------------------------------------------------------------------- #


class TestPullImage:
    @patch("validator.docker_eval.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        pull_image(_IMAGE, _DIGEST)
        args = mock_run.call_args
        cmd = args[0][0]
        assert cmd[0] == "docker"
        assert cmd[1] == "pull"
        assert f"{_IMAGE}@{_DIGEST}" in cmd[2]

    @patch("validator.docker_eval.subprocess.run")
    def test_failure_raises(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not found")
        with pytest.raises(RuntimeError, match="docker pull failed"):
            pull_image(_IMAGE, _DIGEST)

    @patch("validator.docker_eval.subprocess.run")
    def test_timeout_raises(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker pull", timeout=300)
        with pytest.raises(subprocess.TimeoutExpired):
            pull_image(_IMAGE, _DIGEST, timeout_s=300)


# --------------------------------------------------------------------------- #
# ensure_eval_network
# --------------------------------------------------------------------------- #


class TestEnsureEvalNetwork:
    @patch("validator.docker_eval.subprocess.run")
    def test_noop_when_network_exists(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        ensure_eval_network()
        mock_run.assert_called_once()
        assert "inspect" in mock_run.call_args[0][0]

    @patch("validator.docker_eval.subprocess.run")
    def test_creates_internal_network_when_missing(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=1),  # inspect fails
            MagicMock(returncode=0),  # create succeeds
        ]
        ensure_eval_network()
        assert mock_run.call_count == 2
        create_cmd = mock_run.call_args_list[1][0][0]
        assert "create" in create_cmd
        assert "--internal" in create_cmd
        assert INTERNAL_NETWORK in create_cmd

    @patch("validator.docker_eval.subprocess.run")
    def test_raises_on_create_failure(self, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=1),  # inspect fails
            MagicMock(returncode=1, stderr="permission denied"),  # create fails
        ]
        with pytest.raises(RuntimeError, match="Failed to create Docker network"):
            ensure_eval_network()


# --------------------------------------------------------------------------- #
# start_container
# --------------------------------------------------------------------------- #


class TestStartContainer:
    @patch("validator.docker_eval._get_container_ip", return_value="172.18.0.2")
    @patch("validator.docker_eval.ensure_eval_network")
    @patch("validator.docker_eval.subprocess.run")
    def test_returns_id_and_url(self, mock_run, _mock_net, _mock_ip):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="abc123def456\n", stderr=""
        )
        cid, url = start_container(
            _IMAGE,
            _DIGEST,
            model_volume="/mnt/models",
        )
        assert cid == "abc123def456"
        assert url == "http://172.18.0.2:8000"

    @patch("validator.docker_eval._get_container_ip", return_value="172.18.0.2")
    @patch("validator.docker_eval.ensure_eval_network")
    @patch("validator.docker_eval.subprocess.run")
    def test_isolation_flags_present(self, mock_run, _mock_net, _mock_ip):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="container_id\n", stderr=""
        )
        start_container(
            _IMAGE,
            _DIGEST,
            model_volume="/mnt/models",
        )
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "--network" in cmd_str and "cacheon-internal" in cmd_str
        assert "--pids-limit" in cmd_str
        assert "--shm-size" in cmd_str
        assert "/mnt/models:/models:ro" in cmd_str
        assert "-p" not in cmd

    @patch("validator.docker_eval._get_container_ip", return_value="172.18.0.2")
    @patch("validator.docker_eval.ensure_eval_network")
    @patch("validator.docker_eval.subprocess.run")
    def test_gpus_always_all(self, mock_run, _mock_net, _mock_ip):
        """V1 hardcodes --gpus all."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="container_id\n", stderr=""
        )
        start_container(
            _IMAGE,
            _DIGEST,
            model_volume="/mnt/models",
        )
        cmd = mock_run.call_args[0][0]
        gpus_idx = cmd.index("--gpus") + 1
        assert cmd[gpus_idx] == "all"

    @patch("validator.docker_eval.ensure_eval_network")
    @patch("validator.docker_eval.subprocess.run")
    def test_failure_raises(self, mock_run, _mock_net):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        with pytest.raises(RuntimeError, match="docker run failed"):
            start_container(
                _IMAGE,
                _DIGEST,
                model_volume="/mnt/models",
            )

    @patch("validator.docker_eval.reset_gpu_state")
    @patch("validator.docker_eval.stop_and_remove")
    @patch(
        "validator.docker_eval._get_container_ip",
        side_effect=RuntimeError("no ip"),
    )
    @patch("validator.docker_eval.ensure_eval_network")
    @patch("validator.docker_eval.subprocess.run")
    def test_ip_lookup_failure_cleans_up(
        self, mock_run, _mock_net, _mock_ip, mock_stop_rm, mock_reset_gpu
    ):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="orphancontainerid\n", stderr=""
        )
        with pytest.raises(RuntimeError, match="no ip"):
            start_container(
                _IMAGE,
                _DIGEST,
                model_volume="/mnt/models",
            )
        mock_stop_rm.assert_called_once_with("orphancontainerid")
        mock_reset_gpu.assert_called_once()


# --------------------------------------------------------------------------- #
# stop_and_remove
# --------------------------------------------------------------------------- #


class TestStopAndRemove:
    @patch("validator.docker_eval.subprocess.run")
    def test_calls_stop_and_rm(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        stop_and_remove("abc123")
        assert mock_run.call_count == 2
        calls = [c[0][0] for c in mock_run.call_args_list]
        assert calls[0][1] == "stop"
        assert calls[1][1] == "rm"

    @patch("validator.docker_eval.subprocess.run")
    def test_never_raises_on_failure(self, mock_run):
        mock_run.side_effect = Exception("docker broken")
        stop_and_remove("abc123")

    @patch("validator.docker_eval.subprocess.run")
    def test_never_raises_on_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=30)
        stop_and_remove("abc123")


# --------------------------------------------------------------------------- #
# reset_gpu_state
# --------------------------------------------------------------------------- #


class TestResetGpuState:
    @patch("validator.docker_eval.subprocess.run")
    def test_calls_nvidia_smi(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        reset_gpu_state()
        cmd = mock_run.call_args[0][0]
        assert "nvidia-smi" in cmd
        assert "--gpu-reset" in cmd

    @patch("validator.docker_eval.subprocess.run")
    def test_never_raises(self, mock_run):
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        reset_gpu_state()


# --------------------------------------------------------------------------- #
# wait_for_health
# --------------------------------------------------------------------------- #


class TestWaitForHealth:
    @patch("validator.docker_eval.urlopen")
    @patch("validator.docker_eval.time.sleep")
    def test_success_on_first_poll(self, mock_sleep, mock_urlopen):
        mock_resp = MagicMock(status=200)
        mock_urlopen.return_value = mock_resp
        wait_for_health("http://172.18.0.2:8000", timeout_s=10)
        mock_sleep.assert_not_called()

    @patch("validator.docker_eval.urlopen")
    @patch("validator.docker_eval.time.sleep")
    @patch("validator.docker_eval.time.monotonic")
    def test_success_after_retries(self, mock_mono, mock_sleep, mock_urlopen):
        mock_mono.side_effect = [0, 0, 5, 5, 10, 10]
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionRefusedError("not ready")
            return MagicMock(status=200)

        mock_urlopen.side_effect = side_effect
        wait_for_health("http://172.18.0.2:8000", timeout_s=30, poll_interval_s=5)

    @patch("validator.docker_eval.urlopen")
    @patch("validator.docker_eval.time.sleep")
    @patch("validator.docker_eval.time.monotonic")
    def test_timeout_raises(self, mock_mono, mock_sleep, mock_urlopen):
        mock_mono.side_effect = [0, 0, 100, 100, 700]
        mock_urlopen.side_effect = ConnectionRefusedError("nope")
        with pytest.raises(TimeoutError, match="/health"):
            wait_for_health("http://172.18.0.2:8000", timeout_s=600)


# --------------------------------------------------------------------------- #
# send_prompt -- streaming
# --------------------------------------------------------------------------- #


def _make_sse_response(chunks: list[dict]) -> MagicMock:
    """Build a mock HTTP response that yields SSE lines."""
    lines: list[bytes] = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}\n".encode())
    lines.append(b"data: [DONE]\n")
    resp = MagicMock()
    resp.__iter__ = lambda self: iter(lines)
    resp.status = 200
    return resp


class TestSendPromptStreaming:
    @patch("validator.docker_eval.time.monotonic")
    @patch("validator.docker_eval.urlopen")
    def test_parses_tokens_and_measures_timing(self, mock_urlopen, mock_mono):
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {"content": "!"}}]},
        ]
        mock_urlopen.return_value = _make_sse_response(chunks)
        mock_mono.side_effect = [0.0, 0.1, 0.2, 0.3]

        result = send_prompt(
            "http://172.18.0.2:8000", [{"role": "user", "content": "hi"}], stream=True
        )

        assert result.tokens == ["Hello", " world", "!"]
        assert result.output_text == "Hello world!"
        assert result.output_tokens == 3
        assert result.ttft_s == pytest.approx(0.1)
        assert result.throughput_tps == pytest.approx(3 / 0.2)
        assert result.error is None
        assert result.top_logprobs is None

    @patch("validator.docker_eval.time.monotonic")
    @patch("validator.docker_eval.urlopen")
    def test_parses_logprobs_from_sse_chunks(self, mock_urlopen, mock_mono):
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"content": "Hello"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "Hello",
                                    "logprob": -0.01,
                                    "top_logprobs": [
                                        {"token": "Hello", "logprob": -0.01},
                                        {"token": "Hi", "logprob": -3.5},
                                    ],
                                },
                            ]
                        },
                    }
                ],
            },
            {
                "choices": [
                    {
                        "delta": {"content": " world"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": " world",
                                    "logprob": -0.02,
                                    "top_logprobs": [
                                        {"token": " world", "logprob": -0.02},
                                    ],
                                },
                            ]
                        },
                    }
                ],
            },
        ]
        mock_urlopen.return_value = _make_sse_response(chunks)
        mock_mono.side_effect = [0.0, 0.1, 0.2]

        result = send_prompt(
            "http://172.18.0.2:8000",
            [{"role": "user", "content": "hi"}],
            stream=True,
            logprobs=True,
        )

        assert result.tokens == ["Hello", " world"]
        assert result.top_logprobs is not None
        assert len(result.top_logprobs) == 2
        assert result.top_logprobs[0][0]["token"] == "Hello"
        assert result.error is None

    @patch("validator.docker_eval.time.monotonic")
    @patch("validator.docker_eval.urlopen")
    def test_batched_logprobs_align_with_tokens(self, mock_urlopen, mock_mono):
        """Two logprob entries in one chunk produce two aligned tokens."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"content": "AB"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "A",
                                    "top_logprobs": [{"token": "A", "logprob": -0.1}],
                                },
                                {
                                    "token": "B",
                                    "top_logprobs": [{"token": "B", "logprob": -0.2}],
                                },
                            ]
                        },
                    }
                ],
            },
        ]
        mock_urlopen.return_value = _make_sse_response(chunks)
        mock_mono.side_effect = [0.0, 0.1]

        result = send_prompt(
            "http://172.18.0.2:8000",
            [{"role": "user", "content": "hi"}],
            stream=True,
            logprobs=True,
        )

        assert result.error is None
        assert result.tokens == ["A", "B"]
        assert result.top_logprobs is not None
        assert len(result.top_logprobs) == 2
        assert result.top_logprobs[0][0]["token"] == "A"
        assert result.top_logprobs[1][0]["token"] == "B"
        assert result.output_text == "AB"

    @patch("validator.docker_eval.time.monotonic")
    @patch("validator.docker_eval.urlopen")
    def test_empty_content_with_logprobs_aligns(self, mock_urlopen, mock_mono):
        """delta.content="" but logprobs present: token comes from logprobs."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"content": "Hello"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "Hello",
                                    "top_logprobs": [
                                        {"token": "Hello", "logprob": -0.01}
                                    ],
                                },
                            ]
                        },
                    }
                ],
            },
            {
                "choices": [
                    {
                        "delta": {"content": ""},
                        "logprobs": {
                            "content": [
                                {
                                    "token": " ",
                                    "top_logprobs": [{"token": " ", "logprob": -0.05}],
                                },
                            ]
                        },
                    }
                ],
            },
        ]
        mock_urlopen.return_value = _make_sse_response(chunks)
        mock_mono.side_effect = [0.0, 0.1]

        result = send_prompt(
            "http://172.18.0.2:8000",
            [{"role": "user", "content": "hi"}],
            stream=True,
            logprobs=True,
        )

        assert result.error is None
        assert result.tokens == ["Hello", " "]
        assert len(result.top_logprobs) == 2

    @patch("validator.docker_eval.time.monotonic")
    @patch("validator.docker_eval.urlopen")
    def test_empty_stream_returns_error(self, mock_urlopen, mock_mono):
        mock_urlopen.return_value = _make_sse_response([])
        mock_mono.return_value = 0.0
        result = send_prompt(
            "http://172.18.0.2:8000", [{"role": "user", "content": "hi"}], stream=True
        )
        assert result.error == "no_tokens_in_stream"

    @patch("validator.docker_eval.urlopen")
    def test_connection_error_returns_error(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionRefusedError("refused")
        result = send_prompt(
            "http://172.18.0.2:8000", [{"role": "user", "content": "hi"}], stream=True
        )
        assert result.error is not None
        assert "request_failed" in result.error

    @patch("validator.docker_eval.time.monotonic")
    @patch("validator.docker_eval.urlopen")
    def test_mid_stream_error_returns_partial_result(self, mock_urlopen, mock_mono):
        def _failing_iter():
            yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n'
            raise ConnectionResetError("peer reset")

        resp = MagicMock()
        resp.__iter__ = lambda self: _failing_iter()
        resp.status = 200
        mock_urlopen.return_value = resp
        mock_mono.side_effect = [0.0, 0.1]

        result = send_prompt(
            "http://172.18.0.2:8000", [{"role": "user", "content": "hi"}], stream=True
        )

        assert result.error is not None
        assert "stream_error" in result.error
        assert result.tokens == ["Hello"]
        assert result.ttft_s == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# send_prompt -- non-streaming
# --------------------------------------------------------------------------- #


def _make_json_response(body: dict) -> MagicMock:
    """Build a mock HTTP response with a JSON body."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(body).encode()
    resp.status = 200
    return resp


class TestSendPromptNonStreaming:
    @patch("validator.docker_eval.time.monotonic")
    @patch("validator.docker_eval.urlopen")
    def test_parses_tokens_and_logprobs(self, mock_urlopen, mock_mono):
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello world"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "Hello",
                                "logprob": -0.01,
                                "top_logprobs": [
                                    {"token": "Hello", "logprob": -0.01},
                                    {"token": "Hi", "logprob": -3.5},
                                ],
                            },
                            {
                                "token": " world",
                                "logprob": -0.02,
                                "top_logprobs": [
                                    {"token": " world", "logprob": -0.02},
                                    {"token": " there", "logprob": -2.1},
                                ],
                            },
                        ]
                    },
                }
            ]
        }
        mock_urlopen.return_value = _make_json_response(body)
        mock_mono.side_effect = [0.0, 0.5]

        result = send_prompt(
            "http://172.18.0.2:8000",
            [{"role": "user", "content": "hi"}],
            stream=False,
            logprobs=True,
        )

        assert result.tokens == ["Hello", " world"]
        assert result.output_tokens == 2
        assert result.top_logprobs is not None
        assert len(result.top_logprobs) == 2
        assert result.output_text == "Hello world"
        assert result.error is None

    @patch("validator.docker_eval.time.monotonic")
    @patch("validator.docker_eval.urlopen")
    def test_no_choices_returns_error(self, mock_urlopen, mock_mono):
        mock_urlopen.return_value = _make_json_response({"choices": []})
        mock_mono.side_effect = [0.0, 0.1]
        result = send_prompt(
            "http://172.18.0.2:8000", [{"role": "user", "content": "hi"}], stream=False
        )
        assert result.error == "no_choices_in_response"

    @patch("validator.docker_eval.time.monotonic")
    @patch("validator.docker_eval.urlopen")
    def test_logprobs_content_null(self, mock_urlopen, mock_mono):
        body = {
            "choices": [{"message": {"content": "ok"}, "logprobs": {"content": None}}]
        }
        mock_urlopen.return_value = _make_json_response(body)
        mock_mono.side_effect = [0.0, 0.5]
        result = send_prompt(
            "http://172.18.0.2:8000",
            [{"role": "user", "content": "hi"}],
            stream=False,
            logprobs=True,
        )
        assert result.error is None
        assert result.tokens == []
        assert result.top_logprobs is None

    @patch("validator.docker_eval.time.monotonic")
    @patch("validator.docker_eval.urlopen")
    def test_logprobs_is_non_dict(self, mock_urlopen, mock_mono):
        body = {"choices": [{"message": {"content": "ok"}, "logprobs": "bad"}]}
        mock_urlopen.return_value = _make_json_response(body)
        mock_mono.side_effect = [0.0, 0.5]
        result = send_prompt(
            "http://172.18.0.2:8000",
            [{"role": "user", "content": "hi"}],
            stream=False,
            logprobs=True,
        )
        assert result.error is None
        assert result.tokens == []

    @patch("validator.docker_eval.time.monotonic")
    @patch("validator.docker_eval.urlopen")
    def test_logprobs_content_has_non_dict_entries(self, mock_urlopen, mock_mono):
        body = {
            "choices": [
                {
                    "message": {"content": "ok"},
                    "logprobs": {
                        "content": [None, "bad", {"token": "ok", "top_logprobs": []}]
                    },
                }
            ]
        }
        mock_urlopen.return_value = _make_json_response(body)
        mock_mono.side_effect = [0.0, 0.5]
        result = send_prompt(
            "http://172.18.0.2:8000",
            [{"role": "user", "content": "hi"}],
            stream=False,
            logprobs=True,
        )
        assert result.error is None
        assert result.tokens == ["ok"]
        assert result.output_tokens == 1


# --------------------------------------------------------------------------- #
# evaluate_challenger -- integration with mocks
# --------------------------------------------------------------------------- #


def _make_commitment(**overrides) -> CommitmentRecord:
    defaults = dict(
        uid=1,
        hotkey="hk_miner1",
        commit_block=100,
        image=_IMAGE,
        digest=_DIGEST,
        raw="{}",
    )
    defaults.update(overrides)
    return CommitmentRecord(**defaults)


def _make_baseline(n_prompts: int = 2) -> BaselineCache:
    return BaselineCache(
        cache_key="testkey",
        results=[
            BaselinePromptResult(
                tokens=["Hello", " world"],
                top_logprobs=[
                    [{"token": "Hello", "logprob": -0.01}],
                    [{"token": " world", "logprob": -0.02}],
                ],
                ttft_s=1.0,
                throughput_tps=100.0,
                output_tokens=2,
            )
            for _ in range(n_prompts)
        ],
    )


def _make_prompts(n: int = 2, n_warmup: int = 2) -> list[Prompt]:
    return [
        Prompt(messages=[ChatMessage(role="user", content=f"Q{i}")], max_tokens=256)
        for i in range(n + n_warmup)
    ]


class TestCaptureContainerLogs:
    @patch("validator.docker_eval.subprocess.run")
    def test_saves_logs_to_file(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="vLLM startup log\n", stderr="warning msg\n"
        )
        capture_container_logs("test-container", tmp_path, "uid3_abc_500")
        log_path = tmp_path / "container_logs" / "uid3_abc_500.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "vLLM startup log" in content
        assert "warning msg" in content

    @patch("validator.docker_eval.subprocess.run")
    def test_empty_output_skipped(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        capture_container_logs("test-container", tmp_path, "empty")
        assert not (tmp_path / "container_logs" / "empty.log").exists()

    @patch("validator.docker_eval.subprocess.run")
    def test_never_raises(self, mock_run, tmp_path):
        mock_run.side_effect = Exception("docker not found")
        capture_container_logs("test-container", tmp_path, "fail")


class TestEvaluateChallenger:
    @patch("validator.docker_eval.capture_container_logs")
    @patch("validator.docker_eval.reset_gpu_state")
    @patch("validator.docker_eval.stop_and_remove")
    @patch("validator.docker_eval.send_prompt")
    @patch("validator.docker_eval.wait_for_health")
    @patch(
        "validator.docker_eval.start_container",
        return_value=("cid123", "http://172.18.0.2:8000"),
    )
    @patch("validator.docker_eval.pull_image")
    def test_successful_eval(
        self,
        mock_pull,
        mock_start,
        mock_health,
        mock_send,
        mock_stop,
        mock_reset,
        mock_capture_logs,
    ):
        scored_r = RawPromptResult(
            prompt_index=0,
            output_text="Hello world",
            tokens=["Hello", " world"],
            top_logprobs=[
                [{"token": "Hello", "logprob": -0.01}],
                [{"token": " world", "logprob": -0.02}],
            ],
            ttft_s=0.5,
            throughput_tps=150.0,
            output_tokens=2,
        )
        warmup_r = RawPromptResult(
            prompt_index=0,
            output_text="warmup",
            tokens=["warmup"],
            top_logprobs=[
                [{"token": "warmup", "logprob": -0.1}],
            ],
            ttft_s=1.0,
            throughput_tps=50.0,
            output_tokens=1,
        )
        mock_send.side_effect = [
            warmup_r,
            warmup_r,
            scored_r,
            scored_r,
        ]

        record = evaluate_challenger(
            _make_commitment(),
            _make_prompts(n=2, n_warmup=2),
            _make_baseline(n_prompts=2),
            model_volume="/models",
            startup_timeout_s=600,
            per_prompt_timeout_s=120,
            n_warmup=2,
            current_block=500,
            state_dir="/tmp/test_state",
        )

        assert record.disqualified is False
        assert record.score > 0
        assert record.token_match_rate == 1.0
        assert record.ttft_improvement > 0
        assert record.throughput_improvement > 0
        assert record.per_prompt is not None
        assert len(record.per_prompt) == 2
        assert record.per_prompt[0]["ttft_s"] == 0.5
        assert record.per_prompt[0]["throughput_tps"] == 150.0
        mock_pull.assert_called_once()
        mock_stop.assert_called_once_with("cid123")
        mock_reset.assert_called_once()
        mock_capture_logs.assert_called_once()

    @patch("validator.docker_eval.capture_container_logs")
    @patch("validator.docker_eval.reset_gpu_state")
    @patch("validator.docker_eval.stop_and_remove")
    @patch("validator.docker_eval.wait_for_health")
    @patch(
        "validator.docker_eval.start_container",
        return_value=("cid123", "http://172.18.0.2:8000"),
    )
    @patch("validator.docker_eval.pull_image")
    def test_health_timeout_dqs(
        self,
        mock_pull,
        mock_start,
        mock_health,
        mock_stop,
        mock_reset,
        mock_capture_logs,
    ):
        mock_health.side_effect = TimeoutError("/health timeout")

        record = evaluate_challenger(
            _make_commitment(),
            _make_prompts(),
            _make_baseline(),
            model_volume="/models",
            startup_timeout_s=600,
            per_prompt_timeout_s=120,
            n_warmup=2,
            current_block=500,
        )

        assert record.disqualified is True
        assert "health timeout" in (record.disqualify_reason or "").lower()
        assert record.score == 0.0
        mock_stop.assert_called_once()
        mock_reset.assert_called_once()

    @patch("validator.docker_eval.capture_container_logs")
    @patch("validator.docker_eval.reset_gpu_state")
    @patch("validator.docker_eval.stop_and_remove")
    @patch("validator.docker_eval.send_prompt")
    @patch("validator.docker_eval.wait_for_health")
    @patch(
        "validator.docker_eval.start_container",
        return_value=("cid123", "http://172.18.0.2:8000"),
    )
    @patch("validator.docker_eval.pull_image")
    def test_correctness_fail_dqs(
        self,
        mock_pull,
        mock_start,
        mock_health,
        mock_send,
        mock_stop,
        mock_reset,
        mock_capture_logs,
    ):
        wrong_r = RawPromptResult(
            prompt_index=0,
            output_text="WRONG tokens",
            tokens=["WRONG", " tokens"],
            top_logprobs=[
                [{"token": "WRONG", "logprob": -0.01}],
                [{"token": " tokens", "logprob": -0.02}],
            ],
            ttft_s=0.5,
            throughput_tps=150.0,
            output_tokens=2,
        )
        warmup_r = RawPromptResult(
            prompt_index=0,
            output_text="w",
            tokens=["w"],
            top_logprobs=[
                [{"token": "w", "logprob": -0.1}],
            ],
            ttft_s=1.0,
            throughput_tps=50.0,
            output_tokens=1,
        )
        mock_send.side_effect = [
            warmup_r,
            warmup_r,
            wrong_r,
            wrong_r,
        ]

        record = evaluate_challenger(
            _make_commitment(),
            _make_prompts(n=2, n_warmup=2),
            _make_baseline(n_prompts=2),
            model_volume="/models",
            startup_timeout_s=600,
            per_prompt_timeout_s=120,
            n_warmup=2,
            current_block=500,
        )

        assert record.disqualified is True
        assert "correctness_fail" in (record.disqualify_reason or "")
        assert record.score == 0.0

    @patch("validator.docker_eval.capture_container_logs")
    @patch("validator.docker_eval.reset_gpu_state")
    @patch("validator.docker_eval.stop_and_remove")
    @patch("validator.docker_eval.send_prompt")
    @patch("validator.docker_eval.wait_for_health")
    @patch(
        "validator.docker_eval.start_container",
        return_value=("cid123", "http://172.18.0.2:8000"),
    )
    @patch("validator.docker_eval.pull_image")
    def test_prompt_error_dqs(
        self,
        mock_pull,
        mock_start,
        mock_health,
        mock_send,
        mock_stop,
        mock_reset,
        mock_capture_logs,
    ):
        warmup_r = RawPromptResult(
            prompt_index=0,
            output_text="w",
            tokens=["w"],
            top_logprobs=[
                [{"token": "w", "logprob": -0.1}],
            ],
            ttft_s=1.0,
            throughput_tps=50.0,
            output_tokens=1,
        )
        error_r = RawPromptResult(
            prompt_index=0,
            output_text="",
            tokens=[],
            top_logprobs=None,
            ttft_s=0.0,
            throughput_tps=0.0,
            output_tokens=0,
            error="request_timeout",
        )
        mock_send.side_effect = [
            warmup_r,
            warmup_r,
            error_r,
            error_r,
        ]

        record = evaluate_challenger(
            _make_commitment(),
            _make_prompts(n=2, n_warmup=2),
            _make_baseline(n_prompts=2),
            model_volume="/models",
            startup_timeout_s=600,
            per_prompt_timeout_s=120,
            n_warmup=2,
            current_block=500,
        )

        assert record.disqualified is True
        assert "prompt_errors" in (record.disqualify_reason or "")


# --------------------------------------------------------------------------- #
# run_baseline_if_needed -- error checking
# --------------------------------------------------------------------------- #


class TestRunBaselineErrorCheck:
    @patch("validator.docker_eval.stop_and_remove")
    @patch("validator.docker_eval.send_prompt")
    @patch("validator.docker_eval.wait_for_health")
    @patch(
        "validator.docker_eval.start_container",
        return_value=("cid_bl", "http://172.18.0.3:8000"),
    )
    @patch("validator.docker_eval.pull_image")
    def test_baseline_prompt_error_raises_and_does_not_cache(
        self,
        mock_pull,
        mock_start,
        mock_health,
        mock_send,
        mock_stop,
        tmp_path,
    ):
        warmup_r = RawPromptResult(
            prompt_index=0,
            output_text="w",
            tokens=["w"],
            top_logprobs=[
                [{"token": "w", "logprob": -0.1}],
            ],
            ttft_s=1.0,
            throughput_tps=50.0,
            output_tokens=1,
        )
        ok_r = RawPromptResult(
            prompt_index=0,
            output_text="Hello",
            tokens=["Hello"],
            top_logprobs=[
                [{"token": "Hello", "logprob": -0.01}],
            ],
            ttft_s=0.5,
            throughput_tps=100.0,
            output_tokens=1,
        )
        error_r = RawPromptResult(
            prompt_index=1,
            output_text="",
            tokens=[],
            top_logprobs=None,
            ttft_s=0.0,
            throughput_tps=0.0,
            output_tokens=0,
            error="request_timeout",
        )
        mock_send.side_effect = [
            warmup_r,
            warmup_r,
            ok_r,
            error_r,
        ]

        prompts = _make_prompts(n=2, n_warmup=2)

        with pytest.raises(RuntimeError, match="Baseline had prompt errors"):
            run_baseline_if_needed(
                prompts,
                baseline_image="vllm:latest",
                baseline_digest="sha256:" + "b" * 64,
                model_volume="/models",
                gpu_count=4,
                cache_dir=tmp_path,
                block_hash="0xabc",
                startup_timeout_s=600,
                per_prompt_timeout_s=120,
                n_warmup=2,
            )

        assert list(tmp_path.iterdir()) == []
