"""Append-only audit corrections consumed by the ordinary qualification runner.

The correction extends an audit continuation; it never replaces its original
record or changes the measured candidate, workload, or speed evidence. Private
operators supply a completed corrected execution from their commissioned audit
authority. Normal authenticated request handling consumes the resulting child
continuation and executes only missing downstream work.
"""

from __future__ import annotations

from dataclasses import fields, replace
import json

from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.engine_launch import EngineLaunchSpec
from cacheon.eval.evidence_store import EvidenceArtifactRef, publish_evidence, reopen_evidence
from cacheon.eval.oci_backend import EngineExecutionEvidence
from cacheon.eval.oci_outer_session import SessionExecutionPlan
from cacheon.eval.oci_session_protocol import SlotAuditPolicy
from cacheon.eval.qualification_continuation import (
    AuditContinuation, QualificationContinuation, QualificationContinuationError,
    QualificationContinuationStore,
)
from cacheon.eval.resident_audit_authority import ResidentAuditExecutionAuthority
from cacheon.stack_identity import canonical_digest, canonical_json_bytes, sha256_hex


DOMAIN = "qualification.audit-recovery"
SCHEMA = "cacheon.qualification.audit-recovery.v1"
# A validator repair changes its own distribution/native build identities, not
# the bundle, model, image, topology, numeric policy, or serving configuration.
_VALIDATOR_FIELDS = frozenset({
    "runtime_digest", "base_engine_digest", "controller_distribution_digest",
    "worker_distribution_digest", "validator_overlay_digest", "native_build_spec_digest",
    "arena_digest", "stack_digest", "tree_digest",
})


def _same_contributions(value, launch, replacement_stack):
    """Rebind only validator identities in the original materialized bytes."""
    from cacheon.engine_tree import _logical_tree_digest
    from cacheon.eval.marginal_runtime import _expected_contributions, _tree_metadata
    from cacheon.stack_manifest import EvaluationStackManifest

    candidate = value.prepared.candidates[0]
    stack = EvaluationStackManifest.from_dict(replacement_stack)
    old = candidate.arm.candidate.to_dict()
    new = stack.to_dict()
    # A standalone validator audit has its own attribution record. It cannot
    # change executable contributions or the original miner's settlement claim.
    entries = lambda row: {key: {k: v for k, v in ref.items() if k != "attribution_digest"}
                           for key, ref in row["entries"].items()}
    identities = {"arena_digest", "runtime_digest", "base_engine_digest"}
    if (launch.stack_digest != stack.digest or entries(old) != entries(new)
            or any(old[k] != new[k] for k in old if k not in identities | {"entries"})
            or any(new[k] != getattr(launch, k) for k in identities)):
        return False
    tree = candidate.binding.tree
    metadata_path = "metadata/cacheon_engine_tree.json"
    metadata = _tree_metadata(tree)
    metadata["stack_digest"] = stack.digest
    metadata["contributions"] = _expected_contributions(stack)
    encoded = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    rows = tuple(replace(row, size=len(encoded), sha256=sha256_hex(encoded))
                 if row.path == metadata_path else row for row in tree.files)
    return launch.tree_digest == _logical_tree_digest(rows)


def _workload(plan: SessionExecutionPlan) -> str:
    from cacheon.eval.qualification_runner import _encode_record

    return canonical_digest("cacheon.qualification.recovery-workload.v1", {
        field.name: _encode_record(getattr(plan, field.name))
        for field in fields(plan)
        if field.name not in {"launch_digest", "expected_preflight", "audit_policy", "engine_config"}
    })


def _operation(value, lifecycle) -> str:
    from cacheon.eval.qualification import cohort_trajectory_digest
    from cacheon.eval.qualification_runner import qualification_authority_digest

    return canonical_digest("cacheon.qualification.audit-operation.v1", {
        "authority": qualification_authority_digest(value),
        "source": value.prepared.source.digest,
        "trajectory": cohort_trajectory_digest(lifecycle),
    })


def _speed(value, continuation):
    from cacheon.eval.crossover_runtime import ResidentMarginalLifecycleEvidence
    from cacheon.eval.qualification_runner import ResidentSpeedWitness

    speed = continuation.load_resident_speed()
    if speed is None:
        raise QualificationContinuationError("audit recovery has no retained speed")
    lifecycle = ResidentMarginalLifecycleEvidence(value.prepared, value.resident_speed_plan, speed)
    witness = ResidentSpeedWitness.from_evidence(speed, value.resident_speed_plan)
    return speed, lifecycle, witness


def _checked(value, payload):
    from cacheon.eval.qualification_runner import AuditWitness, qualification_authority_digest
    from cacheon.eval.qualification import QualificationDecision

    expected_keys = {"authority", "source", "speed", "original_final", "original_audit",
                     "reason", "launch", "policy", "execution", "workload", "stack"}
    if type(payload) is not dict or set(payload) != expected_keys:
        raise QualificationContinuationError("audit recovery record is not closed")
    if (payload["authority"] != qualification_authority_digest(value)
            or payload["source"] != value.prepared.source.digest
            or not isinstance(payload["reason"], str) or not payload["reason"].strip()):
        raise QualificationContinuationError("audit recovery differs from its original authority")
    original = value.resident_audit_plan
    old = AuditWitness.from_dict(payload["original_audit"])
    if (old.selected_delta_digest != value.candidates[0].selected_delta_digest
            or old.policy != original.audit_policy):
        raise QualificationContinuationError("recovery names another original audit")
    if payload["execution"] is None:
        if (any(payload[name] is not None for name in ("launch", "policy", "workload", "stack"))
                or old.decision is not QualificationDecision.PASS
                or old.candidate_launch_digest != original.launch.digest
                or old.runtime_resource_policy_digest != value.expected_runtime_resource_policy_digest):
            raise QualificationContinuationError("quality recovery requires a complete unchanged audit")
        return old, None
    launch = EngineLaunchSpec.from_dict(payload["launch"])
    if (any(getattr(launch, field.name) != getattr(original.launch, field.name)
            for field in fields(launch) if field.name not in _VALIDATOR_FIELDS)
            or not _same_contributions(value, launch, payload["stack"])):
        raise QualificationContinuationError("audit recovery changed bundle, model, or runtime envelope")
    policy = SlotAuditPolicy.from_dict(payload["policy"])
    if any(getattr(policy, field.name) != getattr(original.audit_policy, field.name)
           for field in fields(policy) if field.name != "validator_seed"):
        raise QualificationContinuationError("audit recovery changed its comparison policy")
    if payload["workload"] != _workload(original.plan):
        raise QualificationContinuationError("audit recovery changed its workload")
    execution = ContinuationCodec((EngineExecutionEvidence,)).decode(payload["execution"])
    if (type(execution) is not EngineExecutionEvidence
            or execution.launch_digest != launch.digest
            or execution.resource_policy_digest != value.expected_runtime_resource_policy_digest):
        raise QualificationContinuationError("audit recovery execution differs from its authority")
    witness = AuditWitness.from_execution(
        execution, selected_delta_digest=value.candidates[0].selected_delta_digest, policy=policy,
    )
    if witness.decision is not QualificationDecision.PASS:
        raise QualificationContinuationError("replacement audit has not passed")
    old_bars = {(row.slot, row.mode, row.min_ratio) for row in old.receipts}
    if old_bars and {(row.slot, row.mode, row.min_ratio) for row in witness.receipts} != old_bars:
        raise QualificationContinuationError("audit recovery changed a numerical threshold")
    return witness, execution


def reopen_audit_recovery(root, reference: EvidenceArtifactRef, *, expected):
    """Revalidate the correction from its retained execution and original plan."""
    from cacheon.eval.qualification_runner import _canonical_payload

    if reference.domain != DOMAIN or reference.schema != SCHEMA:
        raise QualificationContinuationError("audit recovery reference has another schema")
    payload = _canonical_payload(reopen_evidence(root, reference))
    witness, execution = _checked(expected, payload)
    return payload, witness, execution


def authorize_recovery(value, continuation, *, reason, authority=None, execution=None,
                       replacement_stack=None):
    """Retain completed work and authorize only missing downstream execution.

    Called by the private operator after authenticating the original request and
    proving its executor quiescent. The replacement must be a completed audit of
    the unchanged candidate and workload; this API cannot create speed or T.
    """
    from cacheon.eval.qualification_runner import qualification_authority_digest

    if (type(continuation) is not QualificationContinuation
            or ((authority is None) != (execution is None))
            or (authority is not None and type(authority) is not ResidentAuditExecutionAuthority)
            or continuation.authority_digest != qualification_authority_digest(value)
            or continuation.source_digest != value.prepared.source.digest):
        raise QualificationContinuationError("audit recovery requires the original sealed scope")
    speed, lifecycle, speed_witness = _speed(value, continuation)
    old = continuation.load_audit(_operation(value, lifecycle))
    if old is None or len(old.audit_witnesses) != 1 or continuation.load_quality() is not None:
        raise QualificationContinuationError("audit recovery needs one audit and no completed quality")
    final = continuation.load_final()
    if final is not None:
        from cacheon.eval.qualification_runner import reopen_qualification_stage_exit
        terminal = reopen_qualification_stage_exit(value.evidence_root, final, expected=value)
        if terminal.stage != "audit":
            raise QualificationContinuationError("audit recovery cannot overturn a speed verdict")
    if (authority is not None and authority.physical_allocation_digest
            != value.resident_audit_plan.physical_allocation_digest):
        raise QualificationContinuationError("audit recovery changed physical allocation")
    payload = {
        "authority": continuation.authority_digest, "source": continuation.source_digest,
        "speed": speed_witness.evidence_digest,
        "original_final": None if final is None else final.to_dict(),
        "original_audit": old.audit_witnesses[0][1].to_dict(), "reason": reason,
        "launch": None if authority is None else authority.launch.to_dict(),
        "policy": None if authority is None else authority.audit_policy.to_dict(),
        "execution": (None if execution is None else
                      ContinuationCodec((EngineExecutionEvidence,)).encode(execution)),
        "workload": None if authority is None else _workload(authority.plan),
        "stack": (None if authority is None else
                  (replacement_stack or value.prepared.candidates[0].arm.candidate).to_dict()),
    }
    witness, execution = _checked(value, payload)
    if execution is not None and execution.device_receipts[0].started_monotonic_s < speed.completed_monotonic_s:
        raise QualificationContinuationError("replacement audit predates completed speed")
    reference = publish_evidence(value.evidence_root, canonical_json_bytes(payload),
                                 domain=DOMAIN, media_type="application/json", schema=SCHEMA)
    child = _child(continuation, reference)
    child.record_resident_speed(speed)
    operation = _operation(value, lifecycle)
    existing = child.load_audit(operation)
    if existing is None:
        nonce = child.arm_evaluator("audit", operation)
        child.record_audit(AuditContinuation(
            nonce, operation, ((witness.selected_delta_digest, witness),),
            old.audit_started if execution is None else execution.device_receipts[0].started_monotonic_s,
            old.audit_completed if execution is None else execution.device_receipts[-1].completed_monotonic_s,
            old.audit_last_completed if execution is None else execution.device_receipts[-1].completed_monotonic_s,
        ))
    # Publish the pointer last. A partial preparation cannot become runnable.
    continuation._record("recovery", {"reference": reference.to_dict()})
    return reference


def _child(continuation, reference):
    store = QualificationContinuationStore(continuation.directory / "recoveries" / reference.sha256)
    child = store.scope(request_digest=continuation.request_digest,
                        authority_digest=continuation.authority_digest,
                        source_digest=continuation.source_digest)
    child.recovery_reference = reference
    return child


def resolve_audit_recovery(value, continuation):
    """Resolve a published child only after verifying all retained parent facts."""
    payload = continuation._load("recovery")
    if payload is None:
        return continuation
    if type(payload) is not dict or set(payload) != {"reference"}:
        raise QualificationContinuationError("audit recovery pointer is malformed")
    reference = EvidenceArtifactRef.from_dict(payload["reference"])
    proof, witness, _execution = reopen_audit_recovery(value.evidence_root, reference, expected=value)
    speed, lifecycle, speed_witness = _speed(value, continuation)
    original = continuation.load_audit(_operation(value, lifecycle))
    final = continuation.load_final()
    if (continuation.load_quality() is not None
            or proof["speed"] != speed_witness.evidence_digest
            or original is None or original.audit_witnesses[0][1].to_dict() != proof["original_audit"]
            or proof["original_final"] != (None if final is None else final.to_dict())):
        raise QualificationContinuationError("audit recovery parent evidence changed")
    child = _child(continuation, reference)
    restored = child.load_audit(_operation(value, lifecycle))
    if (child.load_resident_speed() != speed or restored is None
            or dict(restored.audit_witnesses) != {witness.selected_delta_digest: witness}):
        raise QualificationContinuationError("audit recovery child is incomplete or changed")
    return child
