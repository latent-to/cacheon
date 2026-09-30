"""Replay teacher scoring cannot accidentally call the legacy numeric judge."""

from types import SimpleNamespace

import pytest

from cacheon.eval.b300_qualification_commission import (
    B300QualificationCommissionError, _require_replay_quality,
)
from cacheon.eval.qualification_runner import _rollout
from cacheon.eval.reference_protocol import ReferenceRoleInput, ReferenceTokenEvidence


def test_teacher_only_rollout_preserves_tokens_without_numeric_judgement():
    def numeric_judge(**_):
        raise AssertionError("replay prompt is not a sealed numeric question")

    result = _rollout(
        profile=SimpleNamespace(hidden_tasks_per_prompt=0), prompt_digest="a" * 64,
        frame={"top_logprobs": ((), ())},
        role_input=ReferenceRoleInput((3, 7), ((), ())),
        role_evidence=SimpleNamespace(tokens=(
            ReferenceTokenEvidence(-0.25, 3, ()), ReferenceTokenEvidence(-0.5, 7, ()),
        )), hidden_judge=numeric_judge,
    )
    assert len(result.tokens) == 2
    assert result.hidden_tasks == ()


@pytest.mark.parametrize("required,count", [(True, 1), (True, 0), (False, 1)])
def test_replay_numeric_profile_is_rejected_before_engine_launch(required, count):
    policy = SimpleNamespace(topk_width=0, hidden_tasks_required=required,
                             hidden_tasks_per_prompt=count)
    with pytest.raises(B300QualificationCommissionError, match="without numeric hidden tasks"):
        _require_replay_quality(policy, {"temperature": "0"})


def test_replay_teacher_only_profile_is_supported():
    policy = SimpleNamespace(topk_width=0, hidden_tasks_required=False, hidden_tasks_per_prompt=0)
    _require_replay_quality(policy, {"temperature": "0"})


def test_zero_hidden_tasks_do_not_bind_coding_prompts_to_numeric_judge():
    from cacheon.eval.b300_qualification_commission import _bind_hidden_judge

    def judge(**kwargs):
        raise AssertionError("no numeric task was commissioned")

    judge.binding = object()
    judge.bind_prompt_plan = judge
    assert _bind_hidden_judge(judge, binding=judge.binding, tokenizer_digest="unused",
                             prompt_batches=(("coding prompt",),), workload_digest="unused",
                             hidden_tasks_per_prompt=0) is judge
