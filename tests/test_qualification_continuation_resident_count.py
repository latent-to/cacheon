"""Resident-count checkpoint continuation contracts."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.qualification import SelectionEntropyReceipt
from cacheon.eval.qualification_continuation import (
    QualificationContinuation,
    QualificationContinuationError,
    QualificationContinuationStore,
    QualityContinuation,
    ResidentCountQualityCheckpoint,
)
from tests.test_marginal_runtime import _case, _prepared
from tests.test_qualification_continuation import (
    _TypedExecutor,
    _pristine_reference_execution,
)
from tests.test_qualification_runner import _quiescence


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reference(
    label: str = "candidate-observation",
    *,
    domain: str = "cacheon.resident-count-observation",
    schema: str = "cacheon.resident-count-observation.v1",
) -> EvidenceArtifactRef:
    return EvidenceArtifactRef(
        domain=domain,
        sha256=_digest(label),
        size=17,
        media_type="application/json",
        schema=schema,
    )


def _checkpoint(
    *,
    observation: str = "candidate-observation",
    semantic: str = "candidate-observation-semantic",
    raw: str = "raw-execution-evidence",
    raw_semantic: str = "raw-execution-evidence-semantic",
    plan: str = "resident-count-execution-plan",
    stock: str = "fixed-stock-authority",
    pair: str = "resident-pair-binding",
) -> ResidentCountQualityCheckpoint:
    return ResidentCountQualityCheckpoint(
        raw_execution_evidence=_reference(
            raw,
            domain="cacheon.resident-count-execution",
            schema="cacheon.resident-count-execution.v1",
        ),
        raw_execution_evidence_semantic_digest=_digest(raw_semantic),
        candidate_observation=_reference(observation),
        candidate_observation_semantic_digest=_digest(semantic),
        execution_plan_digest=_digest(plan),
        fixed_stock_authority_digest=_digest(stock),
        pair_binding_digest=_digest(pair),
    )


def _scope(
    root: Path,
    *,
    request: str = "request",
    authority: str = "authority",
    source: str = "source",
) -> QualificationContinuation:
    return QualificationContinuationStore(root).scope(
        request_digest=_digest(request),
        authority_digest=_digest(authority),
        source_digest=_digest(source),
    )


def _resident_count_path(continuation: QualificationContinuation) -> Path:
    return continuation.directory / "resident_count.json"


def _quality_path(continuation: QualificationContinuation) -> Path:
    return continuation.directory / "quality.json"


def _rewrite_payload(
    continuation: QualificationContinuation, payload: object
) -> None:
    path = _resident_count_path(continuation)
    path.chmod(0o600)
    path.write_bytes(continuation._record_bytes("resident_count", payload))
    path.chmod(0o400)


def _pristine_quality(root: Path) -> QualityContinuation:
    case = _case(root / "runtime")
    prepared = _prepared(case)
    baseline_execution = _TypedExecutor(root / "artifacts").execute(
        prepared.baseline_launch,
        case.baseline_binding.launch_binding,
        case.mount,
        prepared.baseline_session_plan,
        deadline=999.0,
    )
    reference_execution = _pristine_reference_execution(baseline_execution)
    return QualityContinuation(
        teardown_before=_quiescence(1, 3.0),
        entropy=SelectionEntropyReceipt(
            _digest("entropy-source"),
            _digest("commitment"),
            _digest("entropy"),
            _digest("entropy-authority"),
        ),
        entropy_observed=3.5,
        requests=tuple(
            row.request for row in reference_execution.session.exchanges
        ),
        reference_execution=reference_execution,
        teardown_after=_quiescence(2, 6.0),
        t_nonce="pending",
        t_operation_digest=_digest("t-operation"),
    )


def test_resident_count_checkpoint_round_trip_idempotence_and_conflict(
    tmp_path: Path,
) -> None:
    continuation = _scope(tmp_path / "store")
    checkpoint = _checkpoint()

    assert continuation.load_resident_count_quality() is None
    continuation.record_resident_count_quality(checkpoint)
    original = _resident_count_path(continuation).read_bytes()
    continuation.record_resident_count_quality(checkpoint)

    assert _resident_count_path(continuation).read_bytes() == original
    assert continuation.load_resident_count_quality() == checkpoint
    assert continuation.load_quality() is None
    with pytest.raises(QualificationContinuationError, match="other content"):
        continuation.record_resident_count_quality(_checkpoint(plan="other-plan"))
    with pytest.raises(QualificationContinuationError, match="exact checkpoint"):
        continuation.record_resident_count_quality(  # type: ignore[arg-type]
            SimpleNamespace()
        )


def test_checkpoint_constructor_rejects_untyped_and_noncanonical_fields() -> None:
    reference = _reference()
    digest = _digest("binding")

    with pytest.raises(QualificationContinuationError, match="evidence reference"):
        ResidentCountQualityCheckpoint(  # type: ignore[arg-type]
            raw_execution_evidence=reference,
            raw_execution_evidence_semantic_digest=digest,
            candidate_observation=reference.to_dict(),
            candidate_observation_semantic_digest=digest,
            execution_plan_digest=digest,
            fixed_stock_authority_digest=digest,
            pair_binding_digest=digest,
        )
    with pytest.raises(QualificationContinuationError, match="exactly str"):
        ResidentCountQualityCheckpoint(  # type: ignore[arg-type]
            raw_execution_evidence=reference,
            raw_execution_evidence_semantic_digest=digest,
            candidate_observation=reference,
            candidate_observation_semantic_digest=1,
            execution_plan_digest=digest,
            fixed_stock_authority_digest=digest,
            pair_binding_digest=digest,
        )
    with pytest.raises(QualificationContinuationError, match="lowercase 64-hex"):
        ResidentCountQualityCheckpoint(
            raw_execution_evidence=reference,
            raw_execution_evidence_semantic_digest=digest,
            candidate_observation=reference,
            candidate_observation_semantic_digest=digest,
            execution_plan_digest="F" * 64,
            fixed_stock_authority_digest=digest,
            pair_binding_digest=digest,
        )


def test_actual_record_copy_isolated_by_request_authority_and_source(
    tmp_path: Path,
) -> None:
    source = _scope(tmp_path / "store")
    source.record_resident_count_quality(_checkpoint())
    source_path = _resident_count_path(source)

    for changed in (
        {"request": "other-request"},
        {"authority": "other-authority"},
        {"source": "other-source"},
    ):
        target = _scope(tmp_path / "store", **changed)
        shutil.copyfile(source_path, _resident_count_path(target))
        with pytest.raises(QualificationContinuationError, match="sealed identity"):
            target.load_resident_count_quality()


@pytest.mark.parametrize("first", ("quality", "resident_count"))
def test_pristine_and_resident_count_coexist_and_reopen_in_both_write_orders(
    tmp_path: Path, first: str
) -> None:
    continuation = _scope(tmp_path / "store")
    pristine = _pristine_quality(tmp_path / "fixture")
    pristine = replace(
        pristine,
        t_nonce=continuation.arm_evaluator("t", pristine.t_operation_digest),
    )
    resident_count = _checkpoint()

    if first == "quality":
        continuation.record_quality(pristine)
        preserved_path = _quality_path(continuation)
        preserved = preserved_path.read_bytes()
        assert continuation.load_resident_count_quality() is None
        continuation.record_resident_count_quality(resident_count)
    else:
        continuation.record_resident_count_quality(resident_count)
        preserved_path = _resident_count_path(continuation)
        preserved = preserved_path.read_bytes()
        assert continuation.load_quality() is None
        continuation.record_quality(pristine)

    assert preserved_path.read_bytes() == preserved
    assert continuation.load_quality() == pristine
    assert continuation.load_resident_count_quality() == resident_count
    assert json.loads(_quality_path(continuation).read_text())["stage"] == "quality"
    assert (
        json.loads(_resident_count_path(continuation).read_text())["stage"]
        == "resident_count"
    )


def test_legacy_resident_count_collision_requires_explicit_migration(
    tmp_path: Path,
) -> None:
    legacy = _scope(tmp_path / "legacy")
    pristine = _pristine_quality(tmp_path / "fixture")
    legacy._record(
        "quality",
        {
            "mode": "resident_count",
            "checkpoint": legacy._codec.encode(_checkpoint()),
        },
    )
    original = _quality_path(legacy).read_bytes()
    error = "legacy quality.json.*explicit migration.*ambiguous"
    with pytest.raises(QualificationContinuationError, match=error):
        legacy.load_resident_count_quality()
    with pytest.raises(QualificationContinuationError, match=error):
        legacy.load_quality()
    with pytest.raises(QualificationContinuationError, match=error):
        legacy.record_resident_count_quality(_checkpoint())
    with pytest.raises(QualificationContinuationError, match=error):
        legacy.record_quality(pristine)

    assert _quality_path(legacy).read_bytes() == original
    assert not _resident_count_path(legacy).exists()


def test_resident_count_record_rejects_live_and_dangling_symlinks(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint()
    for dangling in (False, True):
        continuation = _scope(tmp_path / f"store-{dangling}")
        target = tmp_path / f"target-{dangling}.json"
        if not dangling:
            target.write_text("sentinel")
        path = _resident_count_path(continuation)
        path.symlink_to(target)

        with pytest.raises(QualificationContinuationError, match="regular file"):
            continuation.load_resident_count_quality()
        with pytest.raises(QualificationContinuationError, match="regular file"):
            continuation.record_resident_count_quality(checkpoint)
        if not dangling:
            assert target.read_text() == "sentinel"


def test_canonically_resigned_mode_digest_and_reference_tampers_fail(
    tmp_path: Path,
) -> None:
    mode_case = _scope(tmp_path / "mode")
    mode_case.record_resident_count_quality(_checkpoint())
    mode_payload = json.loads(_resident_count_path(mode_case).read_text())["payload"]
    mode_payload["mode"] = "pristine_t"
    _rewrite_payload(mode_case, mode_payload)
    with pytest.raises(QualificationContinuationError, match="resident count shape"):
        mode_case.load_resident_count_quality()

    digest_case = _scope(tmp_path / "digest")
    digest_case.record_resident_count_quality(_checkpoint())
    digest_payload = json.loads(_resident_count_path(digest_case).read_text())[
        "payload"
    ]
    digest_payload["checkpoint"]["value"]["execution_plan_digest"] = "F" * 64
    _rewrite_payload(digest_case, digest_payload)
    with pytest.raises(QualificationContinuationError, match="lowercase 64-hex"):
        digest_case.load_resident_count_quality()

    reference_case = _scope(tmp_path / "reference")
    reference_case.record_resident_count_quality(_checkpoint())
    reference_payload = json.loads(_resident_count_path(reference_case).read_text())[
        "payload"
    ]
    reference_payload["checkpoint"]["value"]["candidate_observation"]["size"] = -1
    _rewrite_payload(reference_case, reference_payload)
    with pytest.raises(QualificationContinuationError, match="size is invalid"):
        reference_case.load_resident_count_quality()

    unsigned_case = _scope(tmp_path / "unsigned")
    unsigned_case.record_resident_count_quality(_checkpoint())
    unsigned_record = json.loads(_resident_count_path(unsigned_case).read_text())
    unsigned_record["payload"]["checkpoint"]["value"][
        "fixed_stock_authority_digest"
    ] = _digest("mutated-authority")
    unsigned_path = _resident_count_path(unsigned_case)
    unsigned_path.chmod(0o600)
    unsigned_path.write_text(
        json.dumps(unsigned_record, sort_keys=True, separators=(",", ":"))
    )
    unsigned_path.chmod(0o400)
    with pytest.raises(QualificationContinuationError, match="sealed identity"):
        unsigned_case.load_resident_count_quality()


def test_serialized_checkpoint_is_closed_path_free_and_score_free(
    tmp_path: Path,
) -> None:
    continuation = _scope(tmp_path / "store")
    continuation.record_resident_count_quality(_checkpoint())
    record = json.loads(_resident_count_path(continuation).read_text())
    payload = record["payload"]
    encoded = payload["checkpoint"]
    value = encoded["value"]

    assert set(payload) == {"checkpoint", "mode", "schema"}
    assert payload["mode"] == "resident_count"
    assert set(encoded) == {"type", "value"}
    assert set(value) == {
        "raw_execution_evidence",
        "raw_execution_evidence_semantic_digest",
        "candidate_observation",
        "candidate_observation_semantic_digest",
        "execution_plan_digest",
        "fixed_stock_authority_digest",
        "pair_binding_digest",
        "schema",
    }
    for field in ("raw_execution_evidence", "candidate_observation"):
        assert set(value[field]) == {
            "domain",
            "media_type",
            "schema",
            "sha256",
            "size",
        }

    text = _resident_count_path(continuation).read_text().lower()
    for forbidden in ("path", "target", "model", "reservation", "correct", "score"):
        assert forbidden not in text

    def integers(item: object) -> list[int]:
        if type(item) is int:
            return [item]
        if type(item) is dict:
            return [number for child in item.values() for number in integers(child)]
        if type(item) is list:
            return [number for child in item for number in integers(child)]
        return []

    assert integers(record) == [17, 17]
