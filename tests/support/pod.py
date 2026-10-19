"""The GPU pod and its adapter, played on CPU through the production spool and response sealers.

The end-to-end modules answer each carrier the standing supervisor writes: the pod
verifies the authenticated request and its publication, then seals a verdict,
a candidate-failure product or a worker HOLD, and finalizes it into the
carrier's own spool. The verdict and its measurements are test-supplied; every
identity the CPU checks is derived from the carrier and the commissioned
artifacts, as ``tests/test_validator_end_to_end.py`` states.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cacheon.chain import remote_worker_spool as spool
from cacheon.chain.evaluation_coordinator import WorkerReadiness
from cacheon.chain.qualification_request import require_commissioned_incumbent
from cacheon.chain.remote_qualification_hold import (
    RemoteQualificationHoldReason, capture_remote_qualification_hold,
)
from cacheon.chain.remote_evaluation_dispatcher import (
    RemoteEvaluationRequest, capture_remote_qualification_product, seal_remote_response,
)
from cacheon.chain.remote_worker_registration import registration_credential, registration_transport_identity
from cacheon.engine_tree import inspect_contribution, materialize_engine_tree
from cacheon.eval.b300_publication_intake import resolve_cohort_publications
from cacheon.eval.b300_registered_qualification_inputs import ATTRIBUTION_SCHEMA
from cacheon.eval.candidate_failure_product import candidate_failure_digest, publish_candidate_failure
from cacheon.eval.evidence_store import publish_evidence
from cacheon.eval.oci_session_protocol import SlotAuditPolicy
from cacheon.eval.qualification import QualificationDecision
from cacheon.eval.qualification_intake import (
    QualificationAuthorityManifest, QualificationIntakeBatch, QualificationIntakeOutcome,
    QualificationReservation,
)
from cacheon.eval.remote_run_forensics import append_event, journal_path
from cacheon.settlement import SettlementQualification
from cacheon.stack_identity import canonical_digest, canonical_json_bytes
from cacheon.stack_manifest import ProposalContributionRef
from cacheon.stack_plan import plan_candidate_stack, plan_marginal_arm
from tests.support.e2e import private


def pod_request(root: Path, sealed, carrier: Path):
    """The pod side of one carrier: the authenticated request and its publication."""

    registration = sealed.registration
    identity = registration_transport_identity(registration)
    credential = registration_credential(registration, Path(registration["credential_path"]))
    request = spool.verify_request(spool.load_json(carrier / "request.json"), carrier, registration,
                                   identity=identity, credential=credential)
    wire = RemoteEvaluationRequest.from_dict(spool.authenticated_wire_request(
        request, carrier, identity=identity, credential=credential))
    require_commissioned_incumbent(wire.body, SimpleNamespace(
        incumbent_stack=sealed.incumbent, incumbent_tree_digest=sealed.incumbent_tree.tree_digest))
    (publication,) = resolve_cohort_publications(
        request, carrier, wire.body["candidates"], private(root / "pod" / "publications"))
    return SimpleNamespace(carrier=carrier, request=request, wire=wire, publication=publication,
                           identity=identity, credential=credential)


def pod_answer(root: Path, sealed, carrier: Path, decision: QualificationDecision, *, speedup: str = "1.06",
               incumbent_tree_digest: str | None = None, tamper=None, raised: str = "") -> str:
    """Play the GPU pod: verify the authenticated request, then seal its verdict.

    ``speedup`` is the PASS's measured speedup over the commissioned baseline;
    ``incumbent_tree_digest`` plays a pod that measured against another incumbent tree;
    ``tamper`` rewrites the finalized result before RESULT_READY, as a hostile pod or shuttle would;
    ``raised`` makes the FAIL the pod's candidate-failure product carrying that rank failure.
    """

    pod = pod_request(root, sealed, carrier)
    request, wire, publication = pod.request, pod.wire, pod.publication
    (reservation,) = (QualificationReservation.from_dict(row["reservation"]) for row in wire.body["candidates"])

    def pod_output(name: str) -> str:
        return canonical_digest("e2e.pod-output", {"field": name, "request": request["request_id"]})

    evidence = private(root / "pod" / "evidence")
    authority = QualificationAuthorityManifest(
        "registered", pod_output("authority"), pod_output("source"), pod_output("commitment"),
        pod_output("secret-reference"), (reservation.selected_delta_digest,), (reservation,))
    if raised:
        attempt, failure = publish_candidate_failure(
            evidence, authority_manifest_digest=authority.digest, source_digest=authority.source_digest,
            culprit_reservation_digest=reservation.reservation_digest, target_id=reservation.target_id,
            selected_delta_digest=reservation.selected_delta_digest, arm_digest=pod_output("arm"),
            launch_digest=pod_output("launch"), failure_kind="candidate_exception", failure=raised)
    else:
        attempt = publish_evidence(evidence, canonical_json_bytes({
            "decision": decision.value, "request_id": request["request_id"],
            "speed_witness": {"resident_policy": {"version": 17}}}),  # the live paired-replay policy
            domain="qualification.cohort-attempt", media_type="application/json",
            schema="cacheon.qualification.cohort-attempt.v1")
    settlement = None
    if decision is QualificationDecision.PASS:
        catalog, context, incumbent = sealed.catalog, sealed.context, sealed.incumbent
        inspected = inspect_contribution(publication.root, catalog=catalog)
        contribution = ProposalContributionRef(
            reservation.target_id, inspected.target_spec_digest, publication.content_hash,
            inspected.selected_payload_digest, canonical_digest(ATTRIBUTION_SCHEMA, {
                "finalized_block": reservation.finalized_block,
                "finalized_event_index": reservation.finalized_event_index,
                "finalized_event_subindex": reservation.finalized_event_subindex,
                "hotkey": reservation.hotkey, "reservation_digest": reservation.reservation_digest,
                "submission_digest": reservation.submission_digest}))
        candidate = plan_candidate_stack(incumbent, contribution, catalog=catalog, expected_context=context)
        tree = materialize_engine_tree(
            candidate, context=context, catalog=catalog,
            resolver={**sealed.resolver, ("proposal", contribution.artifact_digest): publication.root},
            destination=root / "pod" / "trees" / candidate.digest)
        arm = plan_marginal_arm(incumbent, contribution, catalog=catalog,
                                incumbent_tree_digest=sealed.incumbent_tree.tree_digest,
                                candidate_tree_digest=tree.tree_digest, expected_context=context)
        audit = SlotAuditPolicy(pod_output("audit-seed")[:32], 100_000, 32, reservation.target_members, 1)
        settlement = SettlementQualification(
            lane="registered", arena_digest=incumbent.arena_digest,
            reservation_digest=reservation.reservation_digest, finalized_block=reservation.finalized_block,
            event_index=reservation.finalized_event_index, event_subindex=reservation.finalized_event_subindex,
            hotkey=reservation.hotkey, target_id=reservation.target_id, members=reservation.target_members,
            selected_delta_digest=reservation.selected_delta_digest,
            qualification_authority_digest=authority.digest, qualification_plan_digest=pod_output("plan"),
            qualification_attempt_digest=attempt.sha256, qualification_report_digest=pod_output("report"),
            selection_commitment_digest=authority.commitment_digest,
            selection_secret_commitment_digest=pod_output("secret-commitment"),
            selection_evidence_digest=pod_output("selection-evidence"), arm_digest=arm.digest,
            incumbent_stack_digest=arm.baseline_before.stack_digest,
            incumbent_tree_digest=arm.baseline_before.tree_digest,
            candidate_stack_digest=arm.challenger.stack_digest, candidate_tree_digest=arm.challenger.tree_digest,
            speedup=speedup, incumbent_manifest=incumbent, candidate_manifest=arm.candidate,
            audit_control_digest=audit.control.digest, audit_policy=audit,
            audit_evidence_digest=pod_output("audit-evidence"))
    outcome = QualificationIntakeOutcome(
        reservation.reservation_digest, reservation.selected_delta_digest, authority.digest, decision,
        "qualified" if settlement else "candidate_exception" if raised else "speed_regression", False,
        attempt_artifact_sha256=attempt.sha256, settlement_qualification=settlement,
        report_digest=candidate_failure_digest(failure) if raised else pod_output("report"))
    product = capture_remote_qualification_product(
        batch=QualificationIntakeBatch(authority.digest, (outcome,), attempt), authority_manifest=authority,
        incumbent_stack=sealed.incumbent,
        incumbent_tree_digest=incumbent_tree_digest or sealed.incumbent_tree.tree_digest,
        screen_lane=wire.body["screen_lane"], service_digest=wire.body["service_digest"],
        readiness=WorkerReadiness(**sealed.registration["worker_readiness"]), evidence_root=evidence,
        evidence_references=(attempt,))
    write_response(pod, seal_remote_response(wire, product, pod.identity, pod.credential), tamper)
    return reservation.reservation_digest


def pod_hold(root: Path, sealed, carrier: Path) -> None:
    """The worker HOLD the pod adapter returns when the qualification worker raised."""

    pod = pod_request(root, sealed, carrier)
    product = capture_remote_qualification_hold(
        pod.wire, reason=RemoteQualificationHoldReason.QUALIFICATION_WORKER_ERROR,
        failure_type="RuntimeError", failure_message="qualification worker raised")
    write_response(pod, seal_remote_response(pod.wire, product, pod.identity, pod.credential))


def write_response(pod, response, tamper=None) -> None:
    """The adapter's terminal write: the sealed response, its journal, then RESULT_READY."""

    request_id = pod.request["request_id"]
    result = private(pod.carrier.parent.parent / "results" / request_id)  # the carrier's own spool
    (result / "response.json").write_bytes(spool.spool_canonical_json(response.to_dict()) + b"\n")
    append_event(journal_path(result), request_id, "adapter.terminal", "completed")
    spool.finalize_adapter_response(pod.request, pod.carrier, result, identity=pod.identity,
                                    credential=pod.credential)
    if tamper is not None:
        tamper(result)
    spool.atomic_bytes(result / "RESULT_READY", (request_id + "\n").encode(), mode=0o400)
