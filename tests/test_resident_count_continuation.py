from __future__ import annotations

import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.evidence_store import (
    EvidenceArtifactRef,
    publish_evidence,
)
from cacheon.eval.qualification_continuation import (
    RESIDENT_COUNT_QUALITY_PAYLOAD_SCHEMA,
    QualificationContinuation,
    QualificationContinuationStore,
)
from cacheon.eval.resident_count_continuation import (
    MAX_RESIDENT_COUNT_EXECUTION_ARTIFACT_BYTES,
    RESIDENT_COUNT_EXECUTION_ARTIFACT_DOMAIN,
    RESIDENT_COUNT_EXECUTION_ARTIFACT_SCHEMA,
    ResidentCountQualityContinuationHold,
    publish_resident_count_quality_continuation,
    reopen_resident_count_quality_continuation,
)
from cacheon.eval.resident_count_quality_execution import (
    ResidentCountQualityExecutionEvidence,
)
from cacheon.eval.resident_count_quality import (
    publish_resident_count_observation,
)
from cacheon.eval.resident_count_quality_execution import (
    ResidentCountQualityExecutionResult,
    execute_candidate_count_quality,
)
from cacheon.stack_identity import canonical_json_bytes
from tests.test_resident_count_quality_execution import _fixture, _h


def _scope(
    root: Path,
    *,
    identity: str = "one",
) -> QualificationContinuation:
    return QualificationContinuationStore(root / "continuation").scope(
        request_digest=_h(f"request-{identity}"),
        authority_digest=_h(f"authority-{identity}"),
        source_digest=_h(f"source-{identity}"),
    )


def _execute(
    *,
    profile: str = "profile-one",
    count: int = 1,
) -> tuple[object, object, tuple[ResidentCountQualityExecutionResult, ...], int, int]:
    plan, judge, pair, factory_a, factory_b = _fixture(
        4,
        barrier=False,
        profile=profile,
    )
    try:
        rows = tuple(
            execute_candidate_count_quality(
                plan,
                pair=pair,
                judge=judge,
                deadline=10**10,
            )
            for _ in range(count)
        )
    finally:
        pair.close()
    return plan, judge, rows, factory_a.calls, factory_b.calls


def _publish(
    root: Path,
    continuation: QualificationContinuation,
    execution: ResidentCountQualityExecutionResult,
    plan,
    judge,
):
    return publish_resident_count_quality_continuation(
        root / "evidence",
        continuation,
        execution,
        plan=plan,
        fixed_stock_authority_digest=_h("fixed-stock"),
        pair_binding=plan.pair_binding,
        judge=judge,
    )


def _reopen(root: Path, continuation: QualificationContinuation, plan, judge):
    return reopen_resident_count_quality_continuation(
        root / "evidence",
        continuation,
        plan=plan,
        fixed_stock_authority_digest=_h("fixed-stock"),
        pair_binding=plan.pair_binding,
        judge=judge,
    )


def _artifact_path(root: Path, reference: EvidenceArtifactRef) -> Path:
    return (
        root
        / "evidence"
        / reference.domain
        / reference.sha256[:2]
        / reference.sha256
    )


def _rewrite_checkpoint(
    continuation: QualificationContinuation,
    checkpoint,
) -> None:
    payload = {
        "checkpoint": continuation._codec.encode(checkpoint),
        "mode": "resident_count",
        "schema": RESIDENT_COUNT_QUALITY_PAYLOAD_SCHEMA,
    }
    path = continuation.directory / "resident_count.json"
    path.chmod(0o600)
    path.write_bytes(continuation._record_bytes("resident_count", payload))
    path.chmod(0o400)


def test_two_synthetic_identities_publish_separate_products_and_reopen_without_pair_calls(
    tmp_path: Path,
) -> None:
    identities = []
    for profile in ("profile-one", "profile-two"):
        root = (tmp_path / profile).resolve()
        root.mkdir(mode=0o700)
        plan, judge, rows, calls_a, calls_b = _execute(profile=profile)
        execution = rows[0]
        continuation = _scope(root, identity=profile)

        checkpoint = _publish(root, continuation, execution, plan, judge)
        record_path = continuation.directory / "resident_count.json"
        original_record = record_path.read_bytes()
        assert _publish(root, continuation, execution, plan, judge) == checkpoint
        assert record_path.read_bytes() == original_record

        restarted = _scope(root, identity=profile)
        assert _reopen(root, restarted, plan, judge) == execution
        assert (calls_a, calls_b) == (1, 1)
        assert checkpoint.raw_execution_evidence.domain == (
            RESIDENT_COUNT_EXECUTION_ARTIFACT_DOMAIN
        )
        assert checkpoint.raw_execution_evidence.schema == (
            RESIDENT_COUNT_EXECUTION_ARTIFACT_SCHEMA
        )
        assert checkpoint.raw_execution_evidence_semantic_digest == (
            execution.evidence.digest
        )
        assert checkpoint.candidate_observation != checkpoint.raw_execution_evidence
        assert checkpoint.candidate_observation_semantic_digest == (
            execution.observation.digest
        )
        assert checkpoint.execution_plan_digest == plan.digest
        assert checkpoint.fixed_stock_authority_digest == _h("fixed-stock")
        assert checkpoint.pair_binding_digest == plan.pair_binding.digest
        identities.append(
            (
                checkpoint.raw_execution_evidence.sha256,
                checkpoint.candidate_observation.sha256,
                checkpoint.pair_binding_digest,
            )
        )

        for artifact in (
            checkpoint.raw_execution_evidence,
            checkpoint.candidate_observation,
        ):
            text = _artifact_path(root, artifact).read_text().lower()
            for forbidden in ("/users/", "/root/", "host_path", "pathlib"):
                assert forbidden not in text

    assert identities[0] != identities[1]
    parameters = inspect.signature(
        reopen_resident_count_quality_continuation
    ).parameters
    assert "evaluator" not in parameters and "model" not in parameters


def test_raw_artifact_without_checkpoint_is_not_completion(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    plan, judge, rows, _, _ = _execute()
    execution = rows[0]
    codec = ContinuationCodec((ResidentCountQualityExecutionEvidence,))
    payload = canonical_json_bytes(codec.encode(execution.evidence))
    reference = publish_evidence(
        root / "evidence",
        payload,
        domain=RESIDENT_COUNT_EXECUTION_ARTIFACT_DOMAIN,
        media_type="application/json",
        schema=RESIDENT_COUNT_EXECUTION_ARTIFACT_SCHEMA,
        max_bytes=MAX_RESIDENT_COUNT_EXECUTION_ARTIFACT_BYTES,
    )
    continuation = _scope(root)

    assert _artifact_path(root, reference).is_file()
    assert _reopen(root, continuation, plan, judge) is None
    assert not (continuation.directory / "resident_count.json").exists()


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    (
        ("raw", "missing"),
        ("raw", "mutated"),
        ("observation", "missing"),
        ("observation", "mutated"),
    ),
)
def test_checkpoint_referencing_missing_or_mutated_artifact_holds_without_repair(
    tmp_path: Path,
    artifact: str,
    mutation: str,
) -> None:
    root = (tmp_path / f"{artifact}-{mutation}").resolve()
    root.mkdir(mode=0o700)
    plan, judge, rows, _, _ = _execute()
    execution = rows[0]
    continuation = _scope(root)
    checkpoint = _publish(root, continuation, execution, plan, judge)
    reference = (
        checkpoint.raw_execution_evidence
        if artifact == "raw"
        else checkpoint.candidate_observation
    )
    path = _artifact_path(root, reference)
    if mutation == "missing":
        path.unlink()
    else:
        raw = path.read_bytes()
        path.chmod(0o600)
        path.write_bytes((b"[" if raw[:1] == b"{" else b"{") + raw[1:])
        path.chmod(0o400)

    with pytest.raises(ResidentCountQualityContinuationHold) as raised:
        _reopen(root, continuation, plan, judge)
    assert raised.value.decision == "HOLD"
    with pytest.raises(ResidentCountQualityContinuationHold):
        _publish(root, continuation, execution, plan, judge)
    assert not path.exists() if mutation == "missing" else path.read_bytes() != raw


def test_separately_stored_observation_must_equal_independent_raw_regrade(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    plan, judge, rows, _, _ = _execute(count=2)
    first, second = rows
    assert first.evidence.digest != second.evidence.digest
    assert first.observation.digest != second.observation.digest
    continuation = _scope(root)
    checkpoint = _publish(root, continuation, first, plan, judge)
    foreign_observation = publish_resident_count_observation(
        root / "evidence",
        second.observation,
    )
    forged = replace(
        checkpoint,
        candidate_observation=foreign_observation,
        candidate_observation_semantic_digest=second.observation.digest,
    )
    _rewrite_checkpoint(continuation, forged)

    with pytest.raises(
        ResidentCountQualityContinuationHold,
        match="independent raw regrade",
    ):
        _reopen(root, continuation, plan, judge)


def test_resigned_semantic_or_caller_authority_mismatch_holds(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    plan, judge, rows, _, _ = _execute()
    continuation = _scope(root)
    checkpoint = _publish(root, continuation, rows[0], plan, judge)
    _rewrite_checkpoint(
        continuation,
        replace(
            checkpoint,
            raw_execution_evidence_semantic_digest=_h("foreign-raw-semantic"),
        ),
    )
    with pytest.raises(
        ResidentCountQualityContinuationHold,
        match="semantic digest differs",
    ):
        _reopen(root, continuation, plan, judge)

    clean_root = (tmp_path / "clean").resolve()
    clean_root.mkdir(mode=0o700)
    clean = _scope(clean_root)
    _publish(clean_root, clean, rows[0], plan, judge)
    with pytest.raises(
        ResidentCountQualityContinuationHold,
        match="caller-supplied authority",
    ):
        reopen_resident_count_quality_continuation(
            clean_root / "evidence",
            clean,
            plan=plan,
            fixed_stock_authority_digest=_h("other-fixed-stock"),
            pair_binding=plan.pair_binding,
            judge=judge,
        )


@pytest.mark.parametrize("legacy_stage", ("resident_count", "quality"))
def test_old_incomplete_or_legacy_collision_cannot_auto_promote(
    tmp_path: Path,
    legacy_stage: str,
) -> None:
    root = (tmp_path / legacy_stage).resolve()
    root.mkdir(mode=0o700)
    plan, judge, rows, _, _ = _execute()
    execution = rows[0]
    continuation = _scope(root)
    observation = publish_resident_count_observation(
        root / "evidence",
        execution.observation,
    )
    old_checkpoint = {
        "type": (
            "cacheon.eval.qualification_continuation."
            "ResidentCountQualityCheckpoint"
        ),
        "value": {
            "candidate_observation": observation.to_dict(),
            "candidate_observation_semantic_digest": execution.observation.digest,
            "execution_plan_digest": plan.digest,
            "fixed_stock_authority_digest": _h("fixed-stock"),
        },
    }
    continuation._record(
        legacy_stage,
        {"checkpoint": old_checkpoint, "mode": "resident_count"},
    )
    path = continuation.directory / f"{legacy_stage}.json"
    original = path.read_bytes()

    with pytest.raises(ResidentCountQualityContinuationHold):
        _reopen(root, continuation, plan, judge)
    with pytest.raises(ResidentCountQualityContinuationHold):
        _publish(root, continuation, execution, plan, judge)
    assert path.read_bytes() == original
    assert not (continuation.directory / (
        "quality.json" if legacy_stage == "resident_count" else "resident_count.json"
    )).exists()


def test_concurrent_foreign_publishers_never_overwrite_winning_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    plan, judge, rows, _, _ = _execute(count=2)
    continuation = _scope(root)

    def publish(row: ResidentCountQualityExecutionResult):
        try:
            return "ok", _publish(root, continuation, row, plan, judge)
        except ResidentCountQualityContinuationHold as exc:
            return "hold", exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(publish, rows))

    assert sorted(status for status, _ in outcomes) == ["hold", "ok"]
    record_path = continuation.directory / "resident_count.json"
    winner_bytes = record_path.read_bytes()
    reopened = _reopen(root, continuation, plan, judge)
    assert reopened in rows
    assert record_path.read_bytes() == winner_bytes
    with pytest.raises(ResidentCountQualityContinuationHold):
        _publish(
            root,
            continuation,
            rows[0] if reopened == rows[1] else rows[1],
            plan,
            judge,
        )
    assert record_path.read_bytes() == winner_bytes


def test_concurrent_identical_publishers_converge_on_byte_identical_checkpoint(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    plan, judge, rows, _, _ = _execute()
    continuation = _scope(root)

    with ThreadPoolExecutor(max_workers=2) as executor:
        checkpoints = tuple(
            executor.map(
                lambda _index: _publish(
                    root,
                    continuation,
                    rows[0],
                    plan,
                    judge,
                ),
                range(2),
            )
        )

    assert checkpoints[0] == checkpoints[1]
    assert _reopen(root, continuation, plan, judge) == rows[0]


def test_checkpoint_record_is_closed_and_contains_no_aggregate_score(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    plan, judge, rows, _, _ = _execute()
    continuation = _scope(root)
    _publish(root, continuation, rows[0], plan, judge)
    record = json.loads(
        (continuation.directory / "resident_count.json").read_text()
    )
    checkpoint = record["payload"]["checkpoint"]["value"]

    assert set(checkpoint) == {
        "candidate_observation",
        "candidate_observation_semantic_digest",
        "execution_plan_digest",
        "fixed_stock_authority_digest",
        "pair_binding_digest",
        "raw_execution_evidence",
        "raw_execution_evidence_semantic_digest",
        "schema",
    }
    text = json.dumps(record, sort_keys=True).lower()
    for forbidden in ("correct", "score", "target_id", "profile-one", "/users/"):
        assert forbidden not in text
