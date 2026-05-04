"""Unit tests for validator.eval_schema -- no GPU, no Docker, no bittensor."""

from __future__ import annotations

import pytest

from validator.eval_schema import (
    ChatMessage,
    EvaluationJob,
    EvaluationResult,
    PerPromptResult,
    Prompt,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# ChatMessage
# --------------------------------------------------------------------------- #


class TestChatMessage:
    def test_round_trip(self):
        msg = ChatMessage(role="user", content="Hello")
        assert ChatMessage.from_dict(msg.to_dict()) == msg

    def test_from_dict_coerces_types(self):
        msg = ChatMessage.from_dict({"role": 123, "content": 456})
        assert msg.role == "123"
        assert msg.content == "456"


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #


class TestPrompt:
    def test_round_trip(self):
        prompt = Prompt(
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Summarize this."),
            ],
            max_tokens=512,
        )
        restored = Prompt.from_dict(prompt.to_dict())
        assert restored == prompt

    def test_default_max_tokens(self):
        prompt = Prompt(messages=[ChatMessage(role="user", content="hi")])
        assert prompt.max_tokens == 256

    def test_from_dict_default_max_tokens(self):
        prompt = Prompt.from_dict(
            {
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert prompt.max_tokens == 256


# --------------------------------------------------------------------------- #
# EvaluationJob
# --------------------------------------------------------------------------- #


def _make_job(**overrides) -> EvaluationJob:
    defaults = dict(
        image="miner/vllm-opt:v1",
        digest="sha256:" + "a" * 64,
        prompts=[
            Prompt(
                messages=[
                    ChatMessage(role="user", content="What is 2+2?"),
                ]
            ),
        ],
    )
    defaults.update(overrides)
    return EvaluationJob(**defaults)


class TestEvaluationJob:
    def test_round_trip(self):
        job = _make_job()
        restored = EvaluationJob.from_dict(job.to_dict())
        assert restored == job

    def test_defaults(self):
        job = _make_job()
        assert job.model_volume == "/models"
        assert job.per_prompt_timeout_s == 120
        assert job.n_warmup == 2
        assert job.startup_timeout_s == 600

    def test_custom_values(self):
        job = _make_job(
            model_volume="/mnt/weights",
            per_prompt_timeout_s=60,
            n_warmup=1,
            startup_timeout_s=300,
        )
        assert job.model_volume == "/mnt/weights"
        assert job.per_prompt_timeout_s == 60
        assert job.n_warmup == 1
        assert job.startup_timeout_s == 300

    def test_from_dict_uses_defaults_for_missing_keys(self):
        data = {
            "image": "miner/server:v1",
            "digest": "sha256:" + "b" * 64,
            "prompts": [{"messages": [{"role": "user", "content": "hi"}]}],
        }
        job = EvaluationJob.from_dict(data)
        assert job.model_volume == "/models"
        assert job.n_warmup == 2
        assert job.per_prompt_timeout_s == 120

    def test_multiple_prompts(self):
        prompts = [
            Prompt(messages=[ChatMessage(role="user", content=f"Q{i}")], max_tokens=128)
            for i in range(10)
        ]
        job = _make_job(prompts=prompts)
        assert len(job.prompts) == 10
        restored = EvaluationJob.from_dict(job.to_dict())
        assert len(restored.prompts) == 10
        assert restored.prompts[5].messages[0].content == "Q5"


# --------------------------------------------------------------------------- #
# PerPromptResult
# --------------------------------------------------------------------------- #


class TestPerPromptResult:
    def test_round_trip(self):
        r = PerPromptResult(
            ttft_s=0.045,
            throughput_tps=120.5,
            output_tokens=256,
            token_match_rate=0.998,
        )
        assert PerPromptResult.from_dict(r.to_dict()) == r


# --------------------------------------------------------------------------- #
# EvaluationResult
# --------------------------------------------------------------------------- #


def _make_result(**overrides) -> EvaluationResult:
    defaults = dict(
        success=True,
        ttft_improvement=0.15,
        throughput_improvement=0.22,
        token_match_rate=0.997,
        median_ttft_s=0.042,
        median_throughput_tps=130.0,
        per_prompt=[
            PerPromptResult(
                ttft_s=0.04,
                throughput_tps=125.0,
                output_tokens=256,
                token_match_rate=0.998,
            ),
            PerPromptResult(
                ttft_s=0.044,
                throughput_tps=135.0,
                output_tokens=200,
                token_match_rate=0.995,
            ),
        ],
        aggregation="median",
        error=None,
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


class TestEvaluationResult:
    def test_round_trip(self):
        result = _make_result()
        restored = EvaluationResult.from_dict(result.to_dict())
        assert restored == result

    def test_failure_result(self):
        result = EvaluationResult(
            success=False,
            error="Container failed /health within 600s",
        )
        assert result.success is False
        assert result.ttft_improvement == 0.0
        assert result.throughput_improvement == 0.0
        assert result.token_match_rate == 0.0
        assert result.per_prompt == []
        assert result.aggregation == "median"

    def test_from_dict_defaults(self):
        data = {"success": True}
        result = EvaluationResult.from_dict(data)
        assert result.ttft_improvement == 0.0
        assert result.throughput_improvement == 0.0
        assert result.token_match_rate == 0.0
        assert result.per_prompt == []
        assert result.aggregation == "median"
        assert result.error is None

    def test_per_prompt_preserved_on_round_trip(self):
        per_prompt = [
            PerPromptResult(
                ttft_s=0.05 * i,
                throughput_tps=100.0 + i,
                output_tokens=256,
                token_match_rate=1.0,
            )
            for i in range(1, 11)
        ]
        result = _make_result(per_prompt=per_prompt)
        restored = EvaluationResult.from_dict(result.to_dict())
        assert len(restored.per_prompt) == 10
        assert restored.per_prompt[0].ttft_s == pytest.approx(0.05)
        assert restored.per_prompt[9].throughput_tps == pytest.approx(110.0)

    def test_error_field_none_when_success(self):
        result = _make_result(success=True, error=None)
        data = result.to_dict()
        assert data["error"] is None
        restored = EvaluationResult.from_dict(data)
        assert restored.error is None

    def test_error_field_preserved(self):
        result = _make_result(success=False, error="OOM killed")
        restored = EvaluationResult.from_dict(result.to_dict())
        assert restored.error == "OOM killed"
