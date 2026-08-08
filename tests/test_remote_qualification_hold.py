from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from cacheon.chain.remote_evaluation_dispatcher import (
    AuthenticatedRemoteEvaluationResponse,
    RemoteEvaluationDispatcherError,
    RemoteEvaluationRequest,
    RemoteWorkerCredential,
    _request_body_for_qualification,
    reopen_remote_response,
    seal_remote_request,
    seal_remote_response,
)
from cacheon.chain.remote_qualification_hold import (
    RemoteQualificationHoldProduct,
    RemoteQualificationHoldReason,
    capture_remote_qualification_hold,
    remote_qualification_hold_from_dict,
    remote_qualification_hold_to_dict,
)
from cacheon.eval.qualification_intake import QualificationReservation
from cacheon.stack_identity import canonical_json_bytes, sha256_hex


def _fixtures():
    path = Path(__file__).with_name("test_remote_evaluation_dispatcher.py")
    specification = importlib.util.spec_from_file_location(
        "cacheon_remote_hold_test_fixtures", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _qualification_request(tmp_path: Path, *, count: int = 1):
    fixtures = _fixtures()
    fixtures._published_rows(tmp_path, count)
    service = fixtures.ArenaService(fixtures._manifest(), fixtures._Provider())
    cursor = fixtures._Cursor(
        (fixtures.BLOCK, fixtures._block_hash(fixtures.BLOCK))
    )
    coordinator = fixtures._coordinator(
        tmp_path,
        service,
        cursor,
        qualification_max_members=count,
    )
    assert all(coordinator.run_screen_once() is not None for _ in range(count))
    claim = coordinator.claim_qualification()
    assert claim is not None
    credential = RemoteWorkerCredential("qualification-hold-key", b"h" * 32)
    identity = fixtures._transport_identity(coordinator, credential)
    request = seal_remote_request(
        claim.lease,
        coordinator.readiness,
        service.manifest.service_id,
        identity,
        credential,
        _request_body_for_qualification(coordinator, claim),
    )
    return request, identity, credential


def _hold(request: RemoteEvaluationRequest) -> RemoteQualificationHoldProduct:
    return capture_remote_qualification_hold(
        request,
        reason=RemoteQualificationHoldReason.GRAPH_EVIDENCE_INCOMPLETE,
        diagnostic_digest=_h("graph-hold-diagnostic"),
    )


def test_hold_roundtrip_is_path_free_and_binds_the_exact_request(
    tmp_path: Path,
) -> None:
    request, identity, credential = _qualification_request(tmp_path)
    product = _hold(request)
    response = seal_remote_response(request, product, identity, credential)
    wire = response.to_dict()

    reopened_response = AuthenticatedRemoteEvaluationResponse.from_dict(wire)
    assert reopen_remote_response(
        request, reopened_response, identity, credential
    ) == product
    assert remote_qualification_hold_from_dict(
        remote_qualification_hold_to_dict(product)
    ) == product
    reservations = tuple(
        QualificationReservation.from_dict(row["reservation"])
        for row in request.body["candidates"]
    )
    assert product.request_digest == request.digest
    assert product.service_identity == request.service_identity
    assert product.service_digest == request.body["service_digest"]
    assert product.worker_readiness_digest == request.worker_readiness_digest
    assert product.ready_receipt_digest == request.ready_receipt_digest
    assert product.ready_epoch == request.ready_epoch
    assert product.screen_lane == request.body["screen_lane"]
    assert product.reservation_digests == tuple(
        row.reservation_digest for row in reservations
    )
    assert product.selected_delta_digests == tuple(
        row.selected_delta_digest for row in reservations
    )
    assert product.candidate_digests == tuple(
        row["candidate_digest"] for row in request.body["candidates"]
    )
    encoded = json.dumps(product.to_dict(), sort_keys=True)
    assert str(tmp_path) not in encoded
    assert not any("path" in key for key in product.to_dict())


def test_hold_projection_is_target_neutral_for_a_two_candidate_cohort(
    tmp_path: Path,
) -> None:
    request, identity, credential = _qualification_request(tmp_path, count=2)
    product = _hold(request)

    assert len(product.reservation_digests) == 2
    assert len(product.selected_delta_digests) == 2
    assert len(product.candidate_digests) == 2
    assert reopen_remote_response(
        request,
        seal_remote_response(request, product, identity, credential),
        identity,
        credential,
    ) == product


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("request_digest", _h("foreign-request")),
        ("service_identity", "foreign-service@" + _h("foreign-service-id")),
        ("service_digest", _h("foreign-service")),
        ("worker_readiness_digest", _h("foreign-readiness")),
        ("ready_receipt_digest", _h("foreign-ready-receipt")),
        ("ready_epoch", 101),
        ("screen_lane", "reproduction"),
        ("reservation_digests", (_h("foreign-reservation"),)),
        ("selected_delta_digests", (_h("foreign-delta"),)),
        ("candidate_digests", (_h("foreign-candidate"),)),
    ),
)
def test_signed_hold_rejects_every_request_authority_drift(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    request, identity, credential = _qualification_request(tmp_path)
    drifted = dataclasses.replace(_hold(request), **{field: replacement})
    response = seal_remote_response(request, drifted, identity, credential)

    with pytest.raises(
        RemoteEvaluationDispatcherError,
        match="differs from its request authority",
    ):
        reopen_remote_response(request, response, identity, credential)


@pytest.mark.parametrize("tamper", ("reason", "diagnostic", "payload_digest", "mac"))
def test_hold_rejects_authenticated_field_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    request, identity, credential = _qualification_request(tmp_path)
    response = seal_remote_response(request, _hold(request), identity, credential)
    if tamper in {"reason", "diagnostic"}:
        payload = response.payload
        if tamper == "reason":
            payload["reason"] = "graph_evidence_unavailable"
        else:
            payload["diagnostic_digest"] = _h("tampered-diagnostic")
        payload_bytes = canonical_json_bytes(payload)
        tampered_product = remote_qualification_hold_from_dict(payload)
        response = dataclasses.replace(
            response,
            payload_bytes=payload_bytes,
            payload_sha256=sha256_hex(payload_bytes),
            payload_digest=tampered_product.digest,
        )
    elif tamper == "payload_digest":
        response = dataclasses.replace(
            response, payload_digest=_h("tampered-payload-digest")
        )
    else:
        response = dataclasses.replace(response, auth_tag="f" * 64)

    with pytest.raises(
        RemoteEvaluationDispatcherError,
        match="remote response authentication failed",
    ):
        reopen_remote_response(request, response, identity, credential)


def test_hold_reason_and_response_algebra_are_closed(tmp_path: Path) -> None:
    request, identity, credential = _qualification_request(tmp_path)
    product = _hold(request)
    response = seal_remote_response(request, product, identity, credential)
    wire = product.to_dict()

    with pytest.raises(RemoteEvaluationDispatcherError, match="is invalid"):
        remote_qualification_hold_from_dict(
            {**wire, "reason": "arbitrary operator message /private/path"}
        )
    with pytest.raises(RemoteEvaluationDispatcherError, match="fields are not closed"):
        remote_qualification_hold_from_dict({**wire, "evidence_path": "/tmp/x"})
    with pytest.raises(RemoteEvaluationDispatcherError, match="diagnostic digest"):
        remote_qualification_hold_from_dict(
            {**wire, "diagnostic_digest": "not-a-digest"}
        )
    with pytest.raises(RemoteEvaluationDispatcherError, match="response is malformed"):
        dataclasses.replace(response, payload_kind="arena_screen_receipt")
    with pytest.raises(RemoteEvaluationDispatcherError, match="response is malformed"):
        dataclasses.replace(response, stage="screen")
