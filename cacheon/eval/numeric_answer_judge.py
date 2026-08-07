"""Validator-owned numeric hidden-judge semantics for sealed qualification prompts.

The judge in this module deliberately does not load a tokenizer, dataset, model,
or deployment configuration.  Its caller supplies the exact tokenizer identity,
an already-constructed decoder, and the sealed answer authority.  This keeps the
grading policy reusable while leaving authority loading and path ownership at the
deployment boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import re

from cacheon.eval.qualification_runner import (
    HiddenJudgeBinding,
    HiddenJudgeReceipt,
    QualificationRunnerError,
    hidden_judge_output_digest,
)
from cacheon.stack_identity import StackIdentityError, canonical_digest, require_sha256_hex


HIDDEN_JUDGE_DOMAIN = "cacheon.private-b300-fe-hidden-judge.v1"
HIDDEN_TASK_POLICY_DOMAIN = "cacheon.private-b300-hidden-task-policy.v1"
HIDDEN_TASK_POLICY_PAYLOAD = {
    "algorithm": "gsm8k-numeric-extraction.v1",
    "tasks_per_prompt": 1,
}


class NumericAnswerJudgeError(QualificationRunnerError):
    """The sealed numeric judge authority or one invocation is malformed."""


class NumericAnswerJudgeInfrastructureError(NumericAnswerJudgeError):
    """Trusted decoding or sealed-gold interpretation failed."""


TokenDecoder = Callable[[tuple[int, ...]], str]
NestedAcceptedTokenSubsequences = tuple[
    tuple[tuple[tuple[int, ...], ...], ...], ...
]


def _digest(value: object, *, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except StackIdentityError as exc:
        raise NumericAnswerJudgeError(str(exc)) from None


def _token_ids(value: object, *, field: str) -> tuple[int, ...]:
    if type(value) is not tuple or not value or any(
        type(token) is not int or token < 0 for token in value
    ):
        raise NumericAnswerJudgeError(
            f"{field} must be a non-empty tuple of non-negative token IDs"
        )
    return value


def _output_ids(value: object) -> tuple[int, ...]:
    if type(value) is not tuple or any(
        type(token) is not int or token < 0 for token in value
    ):
        raise NumericAnswerJudgeError(
            "numeric hidden-judge output IDs must be a tuple of non-negative integers"
        )
    return value


def _normalize_authority(
    value: object,
) -> NestedAcceptedTokenSubsequences:
    if type(value) is not tuple or not value:
        raise NumericAnswerJudgeError(
            "accepted-token-subsequence authority must contain sealed batches"
        )
    batches: list[tuple[tuple[tuple[int, ...], ...], ...]] = []
    for batch in value:
        if type(batch) is not tuple or not batch:
            raise NumericAnswerJudgeError(
                "accepted-token-subsequence authority has an empty or malformed batch"
            )
        prompts: list[tuple[tuple[int, ...], ...]] = []
        for accepted in batch:
            if type(accepted) is not tuple or len(accepted) != 1:
                raise NumericAnswerJudgeError(
                    "each numeric prompt must have exactly one sealed gold sequence"
                )
            prompts.append(
                (
                    _token_ids(
                        accepted[0],
                        field="accepted token subsequence",
                    ),
                )
            )
        batches.append(tuple(prompts))
    return tuple(batches)


def hidden_task_policy_digest() -> str:
    """Return the already-sealed one-numeric-task policy identity."""

    return canonical_digest(HIDDEN_TASK_POLICY_DOMAIN, HIDDEN_TASK_POLICY_PAYLOAD)


def numeric_hidden_judge_digest(
    *,
    hidden_corpus_commitment: str,
    accepted_token_subsequences: NestedAcceptedTokenSubsequences,
) -> str:
    """Recompute the existing numeric judge identity without resealing it."""

    corpus = _digest(
        hidden_corpus_commitment,
        field="numeric hidden corpus commitment",
    )
    authority = _normalize_authority(accepted_token_subsequences)
    return canonical_digest(
        HIDDEN_JUDGE_DOMAIN,
        {
            "accepted_token_subsequences": [
                [
                    [list(sequence) for sequence in accepted]
                    for accepted in batch
                ]
                for batch in authority
            ],
            "hidden_corpus_commitment": corpus,
        },
    )


@dataclass(frozen=True)
class NumericAnswerOccurrence:
    """One selected prompt's exact task and aligned sealed gold sequence."""

    task_digests: tuple[str, ...]
    gold_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        tasks = self.task_digests
        if (
            type(tasks) is not tuple
            or len(tasks) != 1
            or tasks != tuple(sorted(set(tasks)))
        ):
            raise NumericAnswerJudgeError(
                "numeric answer occurrence must contain exactly one sorted hidden task"
            )
        canonical_tasks = tuple(
            _digest(task, field="numeric hidden task") for task in tasks
        )
        gold = _token_ids(self.gold_token_ids, field="numeric gold token sequence")
        object.__setattr__(self, "task_digests", canonical_tasks)
        object.__setattr__(self, "gold_token_ids", gold)


# These expressions intentionally preserve the established GSM8K numeric
# policy.  Broadening them would change grading calibration without changing
# the already-sealed task-policy digest.
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")
_CUE_RE = re.compile(
    r"(?:answer\s+is|answer:|####)\s*(-?\$?\d[\d,]*\.?\d*)",
    re.IGNORECASE,
)


def _parse_numeric_answer(text: str) -> float | None:
    scoped = text.split("\nQuestion:", 1)[0]
    cue_matches = tuple(_CUE_RE.finditer(scoped))
    if cue_matches:
        raw = cue_matches[-1].group(1)
    else:
        matches = tuple(_NUMBER_RE.finditer(scoped))
        if not matches:
            return None
        raw = matches[-1].group(0)
    normalized = raw.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(normalized)
    except ValueError:
        return None


class NumericAnswerHiddenJudge:
    """Grade sealed numeric answers under the existing prompt-authority identity.

    ``occurrence_map`` iteration order is significant: its rows must align
    one-for-one with the flattened nested answer authority.  Requiring alignment,
    rather than only equal multisets, prevents a caller from permuting gold answers
    among prompt digests while retaining the same judge digest.
    """

    def __init__(
        self,
        binding: HiddenJudgeBinding,
        *,
        tokenizer_digest: str,
        decoder: TokenDecoder,
        accepted_token_subsequences: NestedAcceptedTokenSubsequences,
        occurrence_map: Mapping[str, NumericAnswerOccurrence],
    ) -> None:
        if type(binding) is not HiddenJudgeBinding:
            raise NumericAnswerJudgeError("numeric hidden-judge binding is not exact")
        if not callable(decoder):
            raise NumericAnswerJudgeError("numeric token decoder is not callable")
        tokenizer = _digest(tokenizer_digest, field="numeric tokenizer")
        authority = _normalize_authority(accepted_token_subsequences)
        if binding.hidden_task_policy_digest != hidden_task_policy_digest():
            raise NumericAnswerJudgeError(
                "numeric hidden-task policy differs from the sealed policy"
            )
        expected_judge = numeric_hidden_judge_digest(
            hidden_corpus_commitment=binding.hidden_corpus_commitment,
            accepted_token_subsequences=authority,
        )
        if binding.hidden_judge_digest != expected_judge:
            raise NumericAnswerJudgeError(
                "numeric answer authority differs from its hidden-judge binding"
            )
        if not isinstance(occurrence_map, Mapping):
            raise NumericAnswerJudgeError("numeric occurrence map is not a mapping")
        occurrence_rows = tuple(occurrence_map.items())
        gold_rows = tuple(
            accepted[0]
            for batch in authority
            for accepted in batch
        )
        if len(occurrence_rows) != len(gold_rows):
            raise NumericAnswerJudgeError(
                "numeric occurrence map does not cover every sealed gold sequence"
            )
        frozen: dict[str, NumericAnswerOccurrence] = {}
        for (prompt_value, occurrence), sealed_gold in zip(
            occurrence_rows, gold_rows, strict=True
        ):
            prompt = _digest(prompt_value, field="numeric qualification prompt")
            if type(occurrence) is not NumericAnswerOccurrence:
                raise NumericAnswerJudgeError("numeric occurrence row is not exact")
            if prompt in frozen:
                raise NumericAnswerJudgeError("numeric occurrence repeats a prompt")
            if occurrence.gold_token_ids != sealed_gold:
                raise NumericAnswerJudgeError(
                    "numeric occurrence order differs from the sealed answer authority"
                )
            frozen[prompt] = occurrence

        self.binding = binding
        self.tokenizer_digest = tokenizer
        self._decoder = decoder
        self._accepted_token_subsequences = authority
        self._occurrences = frozen

    def _decode(self, token_ids: tuple[int, ...], *, field: str) -> str:
        try:
            decoded = self._decoder(token_ids)
        except Exception as exc:
            raise NumericAnswerJudgeInfrastructureError(
                f"numeric {field} decoding failed"
            ) from exc
        if type(decoded) is not str:
            raise NumericAnswerJudgeInfrastructureError(
                f"numeric {field} decoder did not return text"
            )
        return decoded

    def __call__(
        self,
        *,
        prompt_digest: str,
        output_ids: tuple[int, ...],
        task_digests: tuple[str, ...],
    ) -> HiddenJudgeReceipt:
        prompt = _digest(prompt_digest, field="numeric qualification prompt")
        output = _output_ids(output_ids)
        if type(task_digests) is not tuple:
            raise NumericAnswerJudgeError("numeric hidden tasks must be an exact tuple")
        occurrence = self._occurrences.get(prompt)
        if occurrence is None:
            raise NumericAnswerJudgeError("numeric qualification prompt is not sealed")
        if task_digests != occurrence.task_digests:
            raise NumericAnswerJudgeError(
                "numeric hidden tasks differ from the sealed prompt occurrence"
            )

        gold_text = self._decode(occurrence.gold_token_ids, field="gold")
        output_text = self._decode(output, field="output")
        gold = _parse_numeric_answer(gold_text)
        if gold is None:
            raise NumericAnswerJudgeInfrastructureError(
                "sealed numeric gold does not contain a numeric answer"
            )
        candidate = _parse_numeric_answer(output_text)
        passed = candidate is not None and abs(candidate - gold) <= 1e-4
        return HiddenJudgeReceipt(
            self.binding.digest,
            prompt,
            hidden_judge_output_digest(prompt, output),
            task_digests,
            (passed,),
        )


__all__ = [
    "HIDDEN_JUDGE_DOMAIN",
    "HIDDEN_TASK_POLICY_DOMAIN",
    "HIDDEN_TASK_POLICY_PAYLOAD",
    "NestedAcceptedTokenSubsequences",
    "NumericAnswerHiddenJudge",
    "NumericAnswerJudgeError",
    "NumericAnswerJudgeInfrastructureError",
    "NumericAnswerOccurrence",
    "TokenDecoder",
    "hidden_task_policy_digest",
    "numeric_hidden_judge_digest",
]
