from __future__ import annotations

import hashlib

import pytest

from cacheon.eval.numeric_answer_judge import (
    HIDDEN_JUDGE_DOMAIN,
    HIDDEN_TASK_POLICY_DOMAIN,
    HIDDEN_TASK_POLICY_PAYLOAD,
    NumericAnswerHiddenJudge,
    NumericAnswerJudgeError,
    NumericAnswerJudgeInfrastructureError,
    NumericAnswerOccurrence,
    hidden_task_policy_digest,
    numeric_hidden_judge_digest,
)
from cacheon.eval.qualification_runner import (
    HiddenJudgeBinding,
    hidden_judge_output_digest,
)
from cacheon.stack_identity import canonical_digest


def _h(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _binding(
    authority: tuple[tuple[tuple[tuple[int, ...], ...], ...], ...],
    *,
    corpus: str | None = None,
    judge: str | None = None,
    policy: str | None = None,
) -> HiddenJudgeBinding:
    corpus_digest = _h("numeric-corpus") if corpus is None else corpus
    judge_digest = (
        numeric_hidden_judge_digest(
            hidden_corpus_commitment=corpus_digest,
            accepted_token_subsequences=authority,
        )
        if judge is None
        else judge
    )
    return HiddenJudgeBinding(
        corpus_digest,
        judge_digest,
        hidden_task_policy_digest() if policy is None else policy,
    )


def _one_prompt_judge(
    *,
    gold_text: str,
    output_texts: dict[tuple[int, ...], str],
) -> tuple[NumericAnswerHiddenJudge, str, str]:
    gold_ids = (101, 102)
    prompt, task = _h("prompt:one"), _h("task:one")
    authority = (((gold_ids,),),)
    decoded = {gold_ids: gold_text, **output_texts}
    judge = NumericAnswerHiddenJudge(
        _binding(authority),
        tokenizer_digest=_h("tokenizer"),
        decoder=decoded.__getitem__,
        accepted_token_subsequences=authority,
        occurrence_map={prompt: NumericAnswerOccurrence((task,), gold_ids)},
    )
    return judge, prompt, task


def test_existing_identity_formulas_and_exact_receipt() -> None:
    first_gold, second_gold = (101, 102), (201,)
    authority = (((first_gold,),), ((second_gold,),))
    corpus = _h("numeric-corpus")
    expected_judge = canonical_digest(
        HIDDEN_JUDGE_DOMAIN,
        {
            "accepted_token_subsequences": [
                [[[101, 102]]],
                [[[201]]],
            ],
            "hidden_corpus_commitment": corpus,
        },
    )
    expected_policy = canonical_digest(
        HIDDEN_TASK_POLICY_DOMAIN,
        {"algorithm": "gsm8k-numeric-extraction.v1", "tasks_per_prompt": 1},
    )
    assert HIDDEN_TASK_POLICY_PAYLOAD == {
        "algorithm": "gsm8k-numeric-extraction.v1",
        "tasks_per_prompt": 1,
    }
    assert hidden_task_policy_digest() == expected_policy
    assert numeric_hidden_judge_digest(
        hidden_corpus_commitment=corpus,
        accepted_token_subsequences=authority,
    ) == expected_judge

    first_prompt, second_prompt = _h("prompt:first"), _h("prompt:second")
    first_task, second_task = _h("task:first"), _h("task:second")
    output = (7, 8, 9)
    text = {
        first_gold: "worked solution\n#### $1,234.50.",
        second_gold: "answer is -7",
        output: "a distracting 999; the answer is 1,234.50001",
    }
    binding = HiddenJudgeBinding(corpus, expected_judge, expected_policy)
    judge = NumericAnswerHiddenJudge(
        binding,
        tokenizer_digest=_h("tokenizer"),
        decoder=text.__getitem__,
        accepted_token_subsequences=authority,
        occurrence_map={
            first_prompt: NumericAnswerOccurrence((first_task,), first_gold),
            second_prompt: NumericAnswerOccurrence((second_task,), second_gold),
        },
    )
    receipt = judge(
        prompt_digest=first_prompt,
        output_ids=output,
        task_digests=(first_task,),
    )

    assert judge.binding is binding
    assert judge.tokenizer_digest == _h("tokenizer")
    assert receipt.binding_digest == binding.digest
    assert receipt.prompt_digest == first_prompt
    assert receipt.output_ids_digest == hidden_judge_output_digest(first_prompt, output)
    assert receipt.task_digests == (first_task,)
    assert receipt.passed == (True,)


@pytest.mark.parametrize(
    ("gold", "output", "passed"),
    (
        ("#### 42", "answer is 42 then an unrelated 99", True),
        ("42", "work used 12 and finally 42", True),
        ("7", "answer is 7\nQuestion: answer is 99", True),
        ("1.0", "Answer: 1.00009", True),
        ("1.0", "Answer: 1.00011", False),
        ("-$2.5", "-$2.5000", True),
        ("$1,234.50.", "#### 1234.5.", True),
        ("0.75", "the answer is $0.7500", True),
        ("5", "there is no numeric answer", False),
    ),
)
def test_numeric_policy_cases(gold: str, output: str, passed: bool) -> None:
    output_ids = (9, 10)
    judge, prompt, task = _one_prompt_judge(
        gold_text=gold,
        output_texts={output_ids: output},
    )
    receipt = judge(
        prompt_digest=prompt,
        output_ids=output_ids,
        task_digests=(task,),
    )
    assert receipt.passed == (passed,)


def test_cue_is_preferred_to_later_uncued_number() -> None:
    output_ids = (77,)
    judge, prompt, task = _one_prompt_judge(
        gold_text="42",
        output_texts={output_ids: "answer is 42; diagnostic counter 314159"},
    )
    assert judge(
        prompt_digest=prompt,
        output_ids=output_ids,
        task_digests=(task,),
    ).passed == (True,)


def test_judge_has_no_target_identity_or_target_branch() -> None:
    output_ids = (8,)
    judge, prompt, task = _one_prompt_judge(
        gold_text="#### -11",
        output_texts={output_ids: "the answer is -11"},
    )
    results = {}
    for arbitrary_target_label in (
        "collective.some_future_registered_target",
        "singleton.entirely_different_profile",
    ):
        results[arbitrary_target_label] = judge(
            prompt_digest=prompt,
            output_ids=output_ids,
            task_digests=(task,),
        ).passed
    assert set(results.values()) == {(True,)}


def test_unknown_malformed_and_nonexact_tasks_are_rejected() -> None:
    output_ids = (9,)
    judge, prompt, task = _one_prompt_judge(
        gold_text="3",
        output_texts={output_ids: "3"},
    )
    with pytest.raises(NumericAnswerJudgeError, match="not sealed"):
        judge(
            prompt_digest=_h("unknown prompt"),
            output_ids=output_ids,
            task_digests=(task,),
        )
    with pytest.raises(NumericAnswerJudgeError, match="differ"):
        judge(
            prompt_digest=prompt,
            output_ids=output_ids,
            task_digests=(_h("wrong task"),),
        )
    with pytest.raises(NumericAnswerJudgeError, match="differ"):
        judge(
            prompt_digest=prompt,
            output_ids=output_ids,
            task_digests=tuple(reversed((task, _h("inserted task")))),
        )
    with pytest.raises(NumericAnswerJudgeError, match="exact tuple"):
        judge(
            prompt_digest=prompt,
            output_ids=output_ids,
            task_digests=[task],  # type: ignore[arg-type]
        )
    with pytest.raises(NumericAnswerJudgeError, match="output IDs"):
        judge(
            prompt_digest=prompt,
            output_ids=(True,),  # type: ignore[arg-type]
            task_digests=(task,),
        )


def test_binding_policy_and_tokenizer_drift_fail_closed() -> None:
    gold = (101,)
    authority = (((gold,),),)
    occurrence = {
        _h("prompt"): NumericAnswerOccurrence((_h("task"),), gold),
    }
    kwargs = {
        "tokenizer_digest": _h("tokenizer"),
        "decoder": lambda _ids: "1",
        "accepted_token_subsequences": authority,
        "occurrence_map": occurrence,
    }
    with pytest.raises(NumericAnswerJudgeError, match="hidden-judge binding"):
        NumericAnswerHiddenJudge(
            _binding(authority, judge=_h("drifted hidden judge")),
            **kwargs,
        )
    with pytest.raises(NumericAnswerJudgeError, match="hidden-task policy"):
        NumericAnswerHiddenJudge(
            _binding(authority, policy=_h("drifted task policy")),
            **kwargs,
        )
    with pytest.raises(NumericAnswerJudgeError, match="numeric tokenizer"):
        NumericAnswerHiddenJudge(
            _binding(authority),
            **{**kwargs, "tokenizer_digest": "not-a-digest"},
        )


def test_authority_requires_ordered_one_to_one_occurrence_coverage() -> None:
    first, second = (101,), (202,)
    authority = (((first,), (second,)),)
    prompt_a, prompt_b = _h("prompt:a"), _h("prompt:b")
    task_a, task_b = _h("task:a"), _h("task:b")
    common = {
        "binding": _binding(authority),
        "tokenizer_digest": _h("tokenizer"),
        "decoder": lambda _ids: "1",
        "accepted_token_subsequences": authority,
    }
    with pytest.raises(NumericAnswerJudgeError, match="cover every"):
        NumericAnswerHiddenJudge(
            **common,
            occurrence_map={prompt_a: NumericAnswerOccurrence((task_a,), first)},
        )
    with pytest.raises(NumericAnswerJudgeError, match="order differs"):
        NumericAnswerHiddenJudge(
            **common,
            occurrence_map={
                prompt_b: NumericAnswerOccurrence((task_b,), second),
                prompt_a: NumericAnswerOccurrence((task_a,), first),
            },
        )
    with pytest.raises(NumericAnswerJudgeError, match="exactly one sorted"):
        NumericAnswerOccurrence(tuple(sorted((task_a, task_b))), first)

    ambiguous = (((first, second),),)
    with pytest.raises(NumericAnswerJudgeError, match="exactly one sealed gold"):
        numeric_hidden_judge_digest(
            hidden_corpus_commitment=_h("numeric-corpus"),
            accepted_token_subsequences=ambiguous,
        )


def test_decoder_failure_is_infrastructure_not_candidate_failure() -> None:
    gold = (101,)
    output = (7,)
    prompt, task = _h("prompt"), _h("task")
    authority = (((gold,),),)

    def broken_decoder(token_ids: tuple[int, ...]) -> str:
        if token_ids == output:
            raise RuntimeError("decoder backend disappeared")
        return "4"

    judge = NumericAnswerHiddenJudge(
        _binding(authority),
        tokenizer_digest=_h("tokenizer"),
        decoder=broken_decoder,
        accepted_token_subsequences=authority,
        occurrence_map={prompt: NumericAnswerOccurrence((task,), gold)},
    )
    with pytest.raises(
        NumericAnswerJudgeInfrastructureError,
        match="output decoding failed",
    ) as raised:
        judge(
            prompt_digest=prompt,
            output_ids=output,
            task_digests=(task,),
        )
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_malformed_sealed_gold_is_infrastructure_failure() -> None:
    output = (7,)
    judge, prompt, task = _one_prompt_judge(
        gold_text="gold authority contains no number",
        output_texts={output: "1"},
    )
    with pytest.raises(
        NumericAnswerJudgeInfrastructureError,
        match="sealed numeric gold",
    ):
        judge(
            prompt_digest=prompt,
            output_ids=output,
            task_digests=(task,),
        )
