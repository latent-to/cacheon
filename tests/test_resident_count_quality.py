from __future__ import annotations

import hashlib
import json

import pytest

from cacheon.eval.count_quality import CountQualityPolicy
from cacheon.eval.numeric_answer_judge import (
    NumericAnswerHiddenJudge,
    NumericAnswerOccurrence,
    hidden_task_policy_digest,
    numeric_hidden_judge_digest,
)
from cacheon.eval.qualification import ReferenceManifest
from cacheon.eval.qualification_runner import HiddenJudgeBinding
from cacheon.eval.resident_count_quality import (
    ResidentCountPromptObservation,
    ResidentCountQualityEnvelope,
    ResidentCountQualityError,
    ResidentCountQualityInfrastructureError,
    ResidentCountQualityObservation,
    compare_resident_count_quality,
    rejudge_resident_count_observation,
)
from cacheon.stack_identity import canonical_digest


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _reference(binding: HiddenJudgeBinding, *, tokenizer: str | None = None) -> ReferenceManifest:
    return ReferenceManifest(
        *(_h(f"reference-{index}") for index in range(14)),
        _h("tokenizer") if tokenizer is None else tokenizer,
        binding.hidden_corpus_commitment,
        binding.hidden_judge_digest,
        _h("selection"),
    )


def _quality_fixture(total: int = 4):
    gold_rows = tuple((1000 + index,) for index in range(total))
    authority = (tuple((gold,) for gold in gold_rows),)
    corpus = _h("corpus")
    binding = HiddenJudgeBinding(
        corpus,
        numeric_hidden_judge_digest(
            hidden_corpus_commitment=corpus,
            accepted_token_subsequences=authority,
        ),
        hidden_task_policy_digest(),
    )
    prompts = tuple(_h(f"prompt-{index}") for index in range(total))
    tasks = tuple(_h(f"task-{index}") for index in range(total))
    outputs = tuple((2000 + index,) for index in range(total))
    decoded = {
        **{gold: f"#### {index}" for index, gold in enumerate(gold_rows)},
        **{output: f"answer is {index}" for index, output in enumerate(outputs)},
    }
    occurrences = {
        prompt: NumericAnswerOccurrence((task,), gold)
        for prompt, task, gold in zip(prompts, tasks, gold_rows, strict=True)
    }
    judge = NumericAnswerHiddenJudge(
        binding,
        tokenizer_digest=_h("tokenizer"),
        decoder=decoded.__getitem__,
        accepted_token_subsequences=authority,
        occurrence_map=occurrences,
    )
    envelope = ResidentCountQualityEnvelope(
        _reference(binding),
        binding,
        _h("prompt-plan"),
        _h("generation-shape"),
        _h("full-admission"),
        total,
    )
    return judge, envelope, prompts, tasks, outputs, decoded


def _observation(
    role: str,
    judge: NumericAnswerHiddenJudge,
    envelope: ResidentCountQualityEnvelope,
    prompts: tuple[str, ...],
    tasks: tuple[str, ...],
    outputs: tuple[tuple[int, ...], ...],
) -> ResidentCountQualityObservation:
    rows = []
    for ordinal, (prompt, task, output) in enumerate(
        zip(prompts, tasks, outputs, strict=True)
    ):
        receipt = judge(
            prompt_digest=prompt,
            output_ids=output,
            task_digests=(task,),
        )
        rows.append(
            ResidentCountPromptObservation(
                ordinal,
                prompt,
                (task,),
                output,
                receipt,
            )
        )
    return ResidentCountQualityObservation(
        role,
        envelope,
        _h(f"{role}-execution"),
        tuple(rows),
    )


def test_observation_roundtrip_rejudges_and_has_no_stored_count() -> None:
    judge, envelope, prompts, tasks, outputs, _ = _quality_fixture()
    stock = _observation("stock", judge, envelope, prompts, tasks, outputs)
    reopened = ResidentCountQualityObservation.from_canonical_bytes(stock.canonical_bytes)

    assert reopened == stock
    assert reopened.correct == 4
    assert rejudge_resident_count_observation(reopened, judge) == 4
    assert "correct" not in reopened.to_dict()
    assert "stock_correct" not in reopened.canonical_bytes.decode()


def test_exact_61_reference_passes_drop_9_and_fails_drop_10() -> None:
    judge, envelope, prompts, tasks, outputs, decoded = _quality_fixture(64)
    bad = []
    for index in range(64):
        token = (9000 + index,)
        decoded[token] = "no numeric answer"
        bad.append(token)

    stock_outputs = tuple(outputs[index] if index < 61 else bad[index] for index in range(64))
    candidate_52 = tuple(outputs[index] if index < 52 else bad[index] for index in range(64))
    candidate_51 = tuple(outputs[index] if index < 51 else bad[index] for index in range(64))
    stock = _observation("stock", judge, envelope, prompts, tasks, stock_outputs)
    passing = _observation("candidate", judge, envelope, prompts, tasks, candidate_52)
    failing = _observation("candidate", judge, envelope, prompts, tasks, candidate_51)
    policy = CountQualityPolicy(10)

    pass_result = compare_resident_count_quality(stock, passing, judge=judge, policy=policy)
    fail_result = compare_resident_count_quality(stock, failing, judge=judge, policy=policy)

    assert (pass_result.evidence.stock_correct, pass_result.evidence.candidate_correct) == (61, 52)
    assert pass_result.verdict.decision == "PASS"
    assert pass_result.verdict.observed_drop == 9
    assert fail_result.evidence.candidate_correct == 51
    assert fail_result.verdict.decision == "FAIL"
    assert fail_result.verdict.observed_drop == 10


def test_unparseable_candidate_is_incorrect_not_infrastructure() -> None:
    judge, envelope, prompts, tasks, outputs, decoded = _quality_fixture()
    wrong = (9999,)
    decoded[wrong] = "no numeric answer"
    stock = _observation("stock", judge, envelope, prompts, tasks, outputs)
    candidate = _observation(
        "candidate",
        judge,
        envelope,
        prompts,
        tasks,
        (wrong, *outputs[1:]),
    )
    result = compare_resident_count_quality(
        stock,
        candidate,
        judge=judge,
        policy=CountQualityPolicy(2),
    )
    assert result.evidence.candidate_correct == 3
    assert result.verdict.decision == "PASS"


def test_decoder_failure_during_reopen_is_infrastructure() -> None:
    judge, envelope, prompts, tasks, outputs, _ = _quality_fixture()
    stock = _observation("stock", judge, envelope, prompts, tasks, outputs)

    broken = NumericAnswerHiddenJudge(
        judge.binding,
        tokenizer_digest=judge.tokenizer_digest,
        decoder=lambda _ids: (_ for _ in ()).throw(RuntimeError("decoder unavailable")),
        accepted_token_subsequences=judge._accepted_token_subsequences,
        occurrence_map=judge._occurrences,
    )
    with pytest.raises(ResidentCountQualityInfrastructureError, match="rejudged"):
        rejudge_resident_count_observation(stock, broken)


@pytest.mark.parametrize(
    "drift",
    ("prompt-plan", "generation-shape", "admission", "tokenizer"),
)
def test_stock_candidate_envelope_drift_fails_before_counting(drift: str) -> None:
    judge, envelope, prompts, tasks, outputs, _ = _quality_fixture()
    stock = _observation("stock", judge, envelope, prompts, tasks, outputs)
    reference = envelope.reference
    if drift == "tokenizer":
        reference = _reference(envelope.judge_binding, tokenizer=_h("other-tokenizer"))
    changed = ResidentCountQualityEnvelope(
        reference,
        envelope.judge_binding,
        _h("other-prompt-plan") if drift == "prompt-plan" else envelope.prompt_plan_digest,
        _h("other-shape") if drift == "generation-shape" else envelope.generation_shape_digest,
        _h("other-admission") if drift == "admission" else envelope.admission_policy_digest,
        envelope.expected_prompt_count,
    )
    candidate = _observation("candidate", judge, changed, prompts, tasks, outputs)
    with pytest.raises(ResidentCountQualityInfrastructureError, match="envelopes differ"):
        compare_resident_count_quality(
            stock,
            candidate,
            judge=judge,
            policy=CountQualityPolicy(10),
        )


def test_prompt_order_tamper_and_noncanonical_json_fail_closed() -> None:
    judge, envelope, prompts, tasks, outputs, _ = _quality_fixture()
    stock = _observation("stock", judge, envelope, prompts, tasks, outputs)
    value = stock.to_dict()
    value["prompts"] = list(reversed(value["prompts"]))
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ResidentCountQualityError, match="ordered"):
        ResidentCountQualityObservation.from_canonical_bytes(payload)

    duplicate = stock.canonical_bytes.decode().replace(
        '"role":"stock"', '"role":"stock","role":"stock"'
    ).encode()
    with pytest.raises(ResidentCountQualityError, match="repeats key"):
        ResidentCountQualityObservation.from_canonical_bytes(duplicate)


def test_generic_quality_schema_contains_no_target_or_post_eval_authority() -> None:
    judge, envelope, prompts, tasks, outputs, _ = _quality_fixture()
    candidate = _observation("candidate", judge, envelope, prompts, tasks, outputs)
    text = candidate.canonical_bytes.decode().lower()
    for forbidden in (
        "msa",
        "arnorm",
        "all_reduce",
        "settlement",
        "incentive",
        "weight",
        "crown",
    ):
        assert forbidden not in text


def test_prompt_receipt_must_bind_retained_output() -> None:
    judge, envelope, prompts, tasks, outputs, _ = _quality_fixture()
    receipt = judge(
        prompt_digest=prompts[0],
        output_ids=outputs[0],
        task_digests=(tasks[0],),
    )
    with pytest.raises(ResidentCountQualityError, match="differs"):
        ResidentCountPromptObservation(
            0,
            prompts[0],
            (tasks[0],),
            outputs[1],
            receipt,
        )
