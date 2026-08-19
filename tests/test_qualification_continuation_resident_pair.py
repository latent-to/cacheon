"""Resident-pair speed continuation contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheon.eval.qualification_continuation import (
    QualificationContinuation,
    QualificationContinuationError,
    QualificationContinuationStore,
)
from cacheon.eval.resident_pair_crossover import (
    ResidentPairCrossoverEvidence,
    ResidentPairCrossoverPlan,
    run_resident_pair_crossover,
)
from cacheon.stack_identity import canonical_json_bytes
from tests.test_resident_pair_crossover import _setup


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture
def resident_pairs():
    pairs = []
    yield pairs
    for pair in pairs:
        pair.close()


def _run(
    root: Path, resident_pairs: list, label: str
) -> tuple[ResidentPairCrossoverPlan, ResidentPairCrossoverEvidence]:
    plan, pair, clock, *_ = _setup(root, resident_pairs)
    plan = replace(plan, candidate_bundle_digest=_digest(label))
    evidence = run_resident_pair_crossover(
        plan,
        pair=pair,
        deadline=clock() + 120.0,
        clock=clock,
    )
    return plan, evidence


def _scope(root: Path, *, request: str = "request") -> QualificationContinuation:
    return QualificationContinuationStore(root / "continuation").scope(
        request_digest=_digest(request),
        authority_digest=_digest("qualification-authority"),
        source_digest=_digest("sealed-source"),
    )


def _path(continuation: QualificationContinuation) -> Path:
    return continuation.directory / "speed.json"


def _resign_payload(
    continuation: QualificationContinuation, payload: object
) -> None:
    path = _path(continuation)
    path.write_bytes(continuation._record_bytes("speed", payload))
    path.chmod(0o400)


def test_resident_pair_roundtrip_rerecord_conflict_and_restart(
    tmp_path: Path, resident_pairs: list
) -> None:
    plan, evidence = _run(tmp_path / "first-pair", resident_pairs, "bundle-one")
    other_plan, other = _run(
        tmp_path / "second-pair", resident_pairs, "bundle-two"
    )
    assert plan.digest != other_plan.digest

    continuation = _scope(tmp_path)
    assert continuation.load_resident_pair_speed(plan) is None
    continuation.record_resident_pair_speed(evidence)
    original = _path(continuation).read_bytes()
    continuation.record_resident_pair_speed(evidence)

    assert _path(continuation).read_bytes() == original
    assert canonical_json_bytes(json.loads(original)) == original
    assert _scope(tmp_path).load_resident_pair_speed(plan) == evidence
    with pytest.raises(QualificationContinuationError, match="other content"):
        continuation.record_resident_pair_speed(other)


def test_resident_pair_reopen_rejects_foreign_plan_and_untyped_inputs(
    tmp_path: Path, resident_pairs: list
) -> None:
    plan, evidence = _run(tmp_path / "first-pair", resident_pairs, "bundle-one")
    foreign, _ = _run(tmp_path / "foreign-pair", resident_pairs, "bundle-two")
    continuation = _scope(tmp_path)
    continuation.record_resident_pair_speed(evidence)

    with pytest.raises(QualificationContinuationError, match="invalid"):
        continuation.load_resident_pair_speed(foreign)
    with pytest.raises(QualificationContinuationError, match="exact crossover evidence"):
        continuation.record_resident_pair_speed(SimpleNamespace())  # type: ignore[arg-type]
    with pytest.raises(QualificationContinuationError, match="exact crossover plan"):
        continuation.load_resident_pair_speed(SimpleNamespace())  # type: ignore[arg-type]
    assert plan.candidate_bundle_digest != foreign.candidate_bundle_digest


@pytest.mark.parametrize("tamper", ("raw_slice", "headline"))
def test_canonically_resigned_raw_slice_and_headline_tampers_fail_regrade(
    tmp_path: Path, resident_pairs: list, tamper: str
) -> None:
    plan, evidence = _run(tmp_path / "pair", resident_pairs, "bundle-one")
    source = _scope(tmp_path / "source")
    source.record_resident_pair_speed(evidence)
    payload = copy.deepcopy(json.loads(_path(source).read_text())["payload"])
    value = payload["crossover"]["value"]
    if tamper == "raw_slice":
        batch = value["request_slices"][0]["new_batches"][0]
        batch["token_numerator"] -= 1
    else:
        value["final_verdict"]["detail"] = "forged stored headline"

    forged = _scope(tmp_path / tamper)
    _resign_payload(forged, payload)
    assert canonical_json_bytes(json.loads(_path(forged).read_bytes())) == _path(
        forged
    ).read_bytes()
    with pytest.raises(QualificationContinuationError, match="invalid"):
        forged.load_resident_pair_speed(plan)


def test_legacy_and_resident_pair_speed_modes_are_mutually_exclusive(
    tmp_path: Path, resident_pairs: list
) -> None:
    plan, evidence = _run(tmp_path / "pair", resident_pairs, "bundle-one")
    resident_pair = _scope(tmp_path / "resident-pair")
    resident_pair.record_resident_pair_speed(evidence)

    with pytest.raises(QualificationContinuationError, match="resident shape"):
        resident_pair.load_resident_speed()
    with pytest.raises(QualificationContinuationError, match="marginal shape"):
        resident_pair.load_marginal_speed(SimpleNamespace())  # type: ignore[arg-type]

    legacy_resident = _scope(tmp_path / "legacy-resident")
    legacy_resident._record("speed", {"mode": "resident", "crossover": {}})
    with pytest.raises(QualificationContinuationError, match="resident pair shape"):
        legacy_resident.load_resident_pair_speed(plan)

    legacy_marginal = _scope(tmp_path / "legacy-marginal")
    legacy_marginal._record(
        "speed",
        {
            "mode": "marginal",
            "baseline_before": {},
            "candidates": [],
            "baseline_after": {},
            "candidates_repeat": [],
            "baseline_third": None,
        },
    )
    with pytest.raises(QualificationContinuationError, match="resident pair shape"):
        legacy_marginal.load_resident_pair_speed(plan)


def test_resident_pair_payload_is_closed_generic_and_plan_bound(
    tmp_path: Path, resident_pairs: list
) -> None:
    plan, evidence = _run(tmp_path / "first-pair", resident_pairs, "bundle-one")
    other_plan, other = _run(
        tmp_path / "second-pair", resident_pairs, "bundle-two"
    )
    payloads = []
    for index, value in enumerate((evidence, other)):
        continuation = _scope(tmp_path / str(index))
        continuation.record_resident_pair_speed(value)
        payloads.append(json.loads(_path(continuation).read_text())["payload"])

    assert all(set(payload) == {"crossover", "mode"} for payload in payloads)
    assert all(payload["mode"] == "resident_pair" for payload in payloads)
    assert all(set(payload["crossover"]) == {"type", "value"} for payload in payloads)
    assert payloads[0]["crossover"]["value"]["candidate_bundle_digest"] == (
        plan.candidate_bundle_digest
    )
    assert payloads[1]["crossover"]["value"]["candidate_bundle_digest"] == (
        other_plan.candidate_bundle_digest
    )

    text = json.dumps(payloads, sort_keys=True).lower()
    for forbidden in (
        "/users/",
        "/root/",
        "target",
        "model",
        "reservation",
        "aggregate",
        "score",
        "arnorm",
        "msa",
        "all_reduce",
    ):
        assert forbidden not in text
