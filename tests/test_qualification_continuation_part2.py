from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval.evidence_store import EvidenceArtifactRef
from cacheon.eval.qualification_continuation import (
    QualificationContinuation,
    QualificationContinuationError,
    QualificationContinuationStore,
    ResidentCountQualityCheckpoint,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reference(label: str = "candidate-observation") -> EvidenceArtifactRef:
    return EvidenceArtifactRef(
        domain="cacheon.resident-count-observation",
        sha256=_digest(label),
        size=17,
        media_type="application/json",
        schema="cacheon.resident-count-observation.v1",
    )


def _checkpoint(
    *,
    observation: str = "candidate-observation",
    semantic: str = "candidate-observation-semantic",
    plan: str = "resident-count-execution-plan",
    stock: str = "fixed-stock-authority",
) -> ResidentCountQualityCheckpoint:
    return ResidentCountQualityCheckpoint(
        candidate_observation=_reference(observation),
        candidate_observation_semantic_digest=_digest(semantic),
        execution_plan_digest=_digest(plan),
        fixed_stock_authority_digest=_digest(stock),
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


def _quality_path(continuation: QualificationContinuation) -> Path:
    return continuation.directory / "quality.json"


def _rewrite_payload(
    continuation: QualificationContinuation, payload: object
) -> None:
    path = _quality_path(continuation)
    path.chmod(0o600)
    path.write_bytes(continuation._record_bytes("quality", payload))
    path.chmod(0o400)


def test_resident_count_checkpoint_round_trip_idempotence_and_conflict(
    tmp_path: Path,
) -> None:
    continuation = _scope(tmp_path / "store")
    checkpoint = _checkpoint()

    assert continuation.load_resident_count_quality() is None
    continuation.record_resident_count_quality(checkpoint)
    original = _quality_path(continuation).read_bytes()
    continuation.record_resident_count_quality(checkpoint)

    assert _quality_path(continuation).read_bytes() == original
    assert continuation.load_resident_count_quality() == checkpoint
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
            candidate_observation=reference.to_dict(),
            candidate_observation_semantic_digest=digest,
            execution_plan_digest=digest,
            fixed_stock_authority_digest=digest,
        )
    with pytest.raises(QualificationContinuationError, match="exactly str"):
        ResidentCountQualityCheckpoint(  # type: ignore[arg-type]
            candidate_observation=reference,
            candidate_observation_semantic_digest=1,
            execution_plan_digest=digest,
            fixed_stock_authority_digest=digest,
        )
    with pytest.raises(QualificationContinuationError, match="lowercase 64-hex"):
        ResidentCountQualityCheckpoint(
            candidate_observation=reference,
            candidate_observation_semantic_digest=digest,
            execution_plan_digest="F" * 64,
            fixed_stock_authority_digest=digest,
        )


def test_actual_record_copy_isolated_by_request_authority_and_source(
    tmp_path: Path,
) -> None:
    source = _scope(tmp_path / "store")
    source.record_resident_count_quality(_checkpoint())
    source_path = _quality_path(source)

    for changed in (
        {"request": "other-request"},
        {"authority": "other-authority"},
        {"source": "other-source"},
    ):
        target = _scope(tmp_path / "store", **changed)
        shutil.copyfile(source_path, _quality_path(target))
        with pytest.raises(QualificationContinuationError, match="sealed identity"):
            target.load_resident_count_quality()


def test_legacy_loader_refuses_resident_count_mode_and_reverse_shape(
    tmp_path: Path,
) -> None:
    resident = _scope(tmp_path / "resident")
    resident.record_resident_count_quality(_checkpoint())
    with pytest.raises(QualificationContinuationError, match="not closed"):
        resident.load_quality()

    legacy = _scope(tmp_path / "legacy")
    legacy._record(
        "quality",
        {
            "audit_completed": "2",
            "audit_started": "1",
            "audit_witnesses": [],
            "entropy": {},
            "entropy_observed": "1",
            "reference_execution": {},
            "requests": [],
            "teardown_after": {},
            "teardown_before": {},
        },
    )
    with pytest.raises(QualificationContinuationError, match="resident count shape"):
        legacy.load_resident_count_quality()


def test_canonically_resigned_mode_digest_and_reference_tampers_fail(
    tmp_path: Path,
) -> None:
    mode_case = _scope(tmp_path / "mode")
    mode_case.record_resident_count_quality(_checkpoint())
    mode_payload = json.loads(_quality_path(mode_case).read_text())["payload"]
    mode_payload["mode"] = "pristine_t"
    _rewrite_payload(mode_case, mode_payload)
    with pytest.raises(QualificationContinuationError, match="resident count shape"):
        mode_case.load_resident_count_quality()

    digest_case = _scope(tmp_path / "digest")
    digest_case.record_resident_count_quality(_checkpoint())
    digest_payload = json.loads(_quality_path(digest_case).read_text())["payload"]
    digest_payload["checkpoint"]["value"]["execution_plan_digest"] = "F" * 64
    _rewrite_payload(digest_case, digest_payload)
    with pytest.raises(QualificationContinuationError, match="lowercase 64-hex"):
        digest_case.load_resident_count_quality()

    reference_case = _scope(tmp_path / "reference")
    reference_case.record_resident_count_quality(_checkpoint())
    reference_payload = json.loads(_quality_path(reference_case).read_text())["payload"]
    reference_payload["checkpoint"]["value"]["candidate_observation"]["size"] = -1
    _rewrite_payload(reference_case, reference_payload)
    with pytest.raises(QualificationContinuationError, match="size is invalid"):
        reference_case.load_resident_count_quality()

    unsigned_case = _scope(tmp_path / "unsigned")
    unsigned_case.record_resident_count_quality(_checkpoint())
    unsigned_record = json.loads(_quality_path(unsigned_case).read_text())
    unsigned_record["payload"]["checkpoint"]["value"][
        "fixed_stock_authority_digest"
    ] = _digest("mutated-authority")
    unsigned_path = _quality_path(unsigned_case)
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
    record = json.loads(_quality_path(continuation).read_text())
    payload = record["payload"]
    encoded = payload["checkpoint"]
    value = encoded["value"]

    assert set(payload) == {"checkpoint", "mode"}
    assert payload["mode"] == "resident_count"
    assert set(encoded) == {"type", "value"}
    assert set(value) == {
        "candidate_observation",
        "candidate_observation_semantic_digest",
        "execution_plan_digest",
        "fixed_stock_authority_digest",
    }
    assert set(value["candidate_observation"]) == {
        "domain",
        "media_type",
        "schema",
        "sha256",
        "size",
    }

    text = _quality_path(continuation).read_text().lower()
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

    assert integers(record) == [17]
