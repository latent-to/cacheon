"""Portable completed stages for the qualification worker.

The original physical speed plan remains measurement authority. A replacement
worker may execute missing stages under its own commissioned allocation; it
cannot rename old devices or include an unfinished stage in a score.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from cacheon.eval.continuation_codec import ContinuationCodec
from cacheon.eval.engine_launch import EngineLaunchSpec, PhysicalHardwareBinding
from cacheon.eval.qualification_continuation import (
    QualificationContinuationError, QualificationContinuationStore,
)
from cacheon.eval.runtime_preflight import RuntimePreflightReceipt
from cacheon.stack_identity import canonical_digest, canonical_json_bytes


SCHEMA = "cacheon.qualification.worker-checkpoint.v1"
RECOVERY_DOMAIN = "qualification.worker-recovery"
RECOVERY_SCHEMA = "cacheon.qualification.worker-recovery.v1"
MAX_CHECKPOINT_BYTES = 128 << 20
_COMPLETE = {
    "speed": ("speed",),
    "audit": ("audit_armed", "audit_completed"),
    "quality": ("t_armed", "quality"),
}


class CheckpointComputationChanged(QualificationContinuationError):
    """The replacement would measure other work, so the saved stages cannot score it.

    Only executable, native build, launch envelope, stack, workload, portable contract
    and replay GPU configuration comparisons raise it; damage stays the plain error.
    """


def _clock_domain():
    """Distinguish Linux boots and time namespaces without altering timestamps."""
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + os.readlink("/proc/self/ns/time")


def _codec():
    from cacheon.eval.calibration import CalibrationContext, CalibrationManifest, CalibrationThresholdPolicy
    from cacheon.eval.qualification import SelectionCommitment
    from cacheon.eval.qualification_runner import CandidateQualificationAuthority

    return ContinuationCodec((PhysicalHardwareBinding, RuntimePreflightReceipt,
        CalibrationContext, CalibrationManifest, CalibrationThresholdPolicy,
        SelectionCommitment, CandidateQualificationAuthority))


def _binding(launch, binding):
    codec = _codec()
    return {"launch": launch.to_dict(),
            "physical": codec.encode(binding.physical_hardware),
            "preflight": codec.encode(binding.runtime_preflight_receipt)}


def _device_contract(policy):
    from cacheon.eval.qualification_runner import _encode_record

    row = _encode_record(policy)
    for gpu in row["expected_gpus"]:
        for field in ("physical_id", "uuid", "pci_bus_id"):
            gpu.pop(field)
    return row


def _contract_fields(value, *, portable=False):
    """Separate measured work from the commissioning identities that wrap it."""
    from cacheon.eval.qualification_runner import _encode_record
    from cacheon.eval.scoring import marginal_workload_digest

    candidates = _encode_record(value.candidates)
    for candidate in candidates:
        profile = candidate["profile"]
        for field in ("calibration_context_digest", "calibration_digest"):
            profile.pop(field)
        for field in ("pristine_launch_digest", "logical_hardware_digest"):
            profile["reference"].pop(field)
        if portable:
            profile.pop("runtime_resource_policy_digest")
            for field in ("arena_digest", "pristine_stack_digest", "pristine_tree_digest",
                          "controller_distribution_digest"):
                profile["reference"].pop(field)
    commitment = _encode_record(value.commitment)
    commitment.pop("reference_manifest_digest")
    if portable:
        commitment.pop("source_plan_digest")
        commitment.pop("secret_commitment")  # The original secret and commitment are restored together.
    threshold = value.calibration_threshold_policy.to_dict()
    threshold.pop("context")
    speed_policy = value.resident_speed_plan.policy.to_dict()
    speed_policy.pop("calibration_digest")
    speed_policy.pop("calibration_context_digest")
    result = {
        "source": value.prepared.source.digest,
        "candidates": candidates, "commitment": commitment,
        "model": value.model_mount.digest,
        "calibration_threshold": threshold,
        "audit": [row.to_dict() for row in value.audit_policies],
        "reference": [value.pristine_stack.digest, value.reference_engine_config.digest],
        "workload": marginal_workload_digest(value.resident_speed_plan.baseline.session_plan),
        "speed_policy": speed_policy,
    }
    if portable:
        def stack(manifest):
            return {key: item for key, item in manifest.to_dict().items() if key != "arena_digest"}
        for audit in result["audit"]:
            audit.pop("validator_seed")
        source = value.prepared.source
        result.update(
            source={"incumbent": stack(source.incumbent), "candidate": stack(source.candidate),
                    "transition": source.transition.to_dict()},
            model={key: getattr(value.model_mount, key) for key in
                   ("model_revision_digest", "model_manifest_digest", "model_content_digest")},
            reference=[stack(value.pristine_stack), value.reference_engine_config.digest])
    return result


def _contract(value):
    return canonical_digest(SCHEMA + ".work", _contract_fields(value))


def capture_origin(value, continuation, executor):
    """Bind portable state to the plan and policies before paid execution."""
    from cacheon.eval.qualification_runner import qualification_authority_digest
    from cacheon.eval.evidence_store import reopen_evidence

    plan = value.resident_speed_plan
    arms = []
    for arm in (plan.baseline, plan.candidate):
        arms.append({**_binding(arm.launch, arm.binding),
                     "namespace": arm.executor_namespace_digest,
                     "runtime_policy": arm.runtime_resource_policy_digest,
                     "configuration": arm.device_configuration_digest})
    codec = _codec()
    return {
        "authority": qualification_authority_digest(value),
        "request": continuation.request_digest,
        "source": value.prepared.source.digest,
        "boot": _clock_domain(),
        "contract": _contract(value),
        "devices": [_device_contract(executor.device_policy)],
        "baseline": _binding(value.prepared.baseline_launch,
                             value.prepared.incumbent_binding.launch_binding),
        "pristine": _binding(value.pristine_launch, value.pristine_binding),
        "arms": arms,
        "candidates": [codec.encode(row) for row in value.candidates],
        "commitment": codec.encode(value.commitment),
        "selection_secret": value.selection_secret.hex(),
        "calibration": [codec.encode(row) for row in (
            value.calibration_context, value.calibration_manifest, value.calibration_threshold_policy)],
        "calibration_ref": value.calibration_artifact_ref.to_dict(),
        "calibration_bytes": json.loads(reopen_evidence(value.evidence_root, value.calibration_artifact_ref)),
        "speed_policy": value.resident_speed_plan.policy.to_dict(),
        "audit_policies": [row.to_dict() for row in value.audit_policies],
    }


def _restore_binding(template, row):
    """Reproduce the original tree while retaining the replacement's local paths."""
    from cacheon.engine_tree import reopen_materialized_engine_tree, _write_materialized_tree

    codec = _codec()
    launch = EngineLaunchSpec.from_dict(row["launch"])
    tree = reopen_materialized_engine_tree(template.materialized_tree_root,
        expected_tree_digest=template.native_build_spec.tree_digest)
    if tree.tree_digest != launch.tree_digest:
        # Recommissioning changes the arena and therefore the stack metadata.
        # The materializer must reproduce the old tree from the unchanged code
        # bytes; the old digest is checked, never assigned to different bytes.
        destination = tree.root.parent / ("checkpoint-" + launch.tree_digest)
        if destination.exists():
            tree = reopen_materialized_engine_tree(destination, expected_tree_digest=launch.tree_digest)
        else:
            metadata_path = "metadata/cacheon_engine_tree.json"
            metadata = json.loads((tree.root / metadata_path).read_bytes())
            tree = _write_materialized_tree(destination, stack_digest=launch.stack_digest,
                files={item.path: (tree.root / item.path).read_bytes()
                       for item in tree.files if item.path != metadata_path},
                runtime_manifest=tree.runtime_manifest, contributions=metadata["contributions"])
            if tree.tree_digest != launch.tree_digest:
                raise CheckpointComputationChanged("worker checkpoint executable tree changed")
    native = replace(template.native_build_spec, tree_digest=tree.tree_digest)
    if native.digest != launch.native_build_spec_digest:
        raise CheckpointComputationChanged("worker checkpoint native build environment changed")
    binding = replace(template, materialized_tree_root=tree.root, native_build_spec=native,
                      controller_distribution_digest=launch.controller_distribution_digest,
                      physical_hardware=codec.decode(row["physical"]),
                      runtime_preflight_receipt=codec.decode(row["preflight"]))
    return launch, binding


def restore_speed_plan(plan, origin):
    """Reopen physical measurement identities without relabeling retained timings."""
    from cacheon.eval.oci_backend import expected_runtime_preflight
    from cacheon.eval.resident_speed_policy import ResidentSpeedPolicy
    arms = []
    for template, row in zip((plan.baseline, plan.candidate), origin["arms"], strict=True):
        launch, binding = _restore_binding(template.binding, row)
        old, new = launch.to_dict(), template.launch.to_dict()
        old["hardware"].pop("device_policy_digest")
        new["hardware"].pop("device_policy_digest")
        for field in ("arena_digest", "stack_digest", "tree_digest", "native_build_spec_digest",
                      "resource_policy_digest", "controller_distribution_digest"):
            old.pop(field)
            new.pop(field)
        if old != new:
            raise CheckpointComputationChanged("worker checkpoint changed its executable envelope")
        arms.append(replace(template, launch=launch, binding=binding,
                            session_plan=replace(template.session_plan, launch_digest=launch.digest,
                                expected_preflight=expected_runtime_preflight(launch, binding.runtime_preflight_receipt)),
                            executor_namespace_digest=row["namespace"],
                            runtime_resource_policy_digest=row["runtime_policy"],
                            device_configuration_digest=row["configuration"]))
    return replace(plan, baseline=arms[0], candidate=arms[1],
                   policy=ResidentSpeedPolicy.from_dict(origin["speed_policy"]))


def restore_origin(value, origin):
    """Reconstruct the original read-only authority using current immutable trees."""
    from cacheon.eval.oci_backend import expected_runtime_preflight
    from cacheon.eval.qualification_runner import qualification_authority_digest
    from cacheon.eval.resident_audit_authority import ResidentAuditExecutionAuthority
    from cacheon.eval.evidence_store import EvidenceArtifactRef, publish_evidence
    from cacheon.eval.scoring import marginal_workload_digest
    from cacheon.engine_tree import reopen_materialized_engine_tree
    from cacheon.stack_manifest import EvaluationStackManifest
    from cacheon.stack_plan import StackArmIdentity

    def stack(template, launch):
        original = EvaluationStackManifest.from_dict({**template.to_dict(), "arena_digest": launch.arena_digest})
        if original.digest != launch.stack_digest:
            raise CheckpointComputationChanged("worker checkpoint incumbent or candidate changed")
        return original

    def materialized(template, binding):
        return replace(template, launch_binding=binding,
            tree=reopen_materialized_engine_tree(binding.materialized_tree_root,
                expected_tree_digest=binding.native_build_spec.tree_digest))

    def session(template, launch, binding):
        return replace(template, launch_digest=launch.digest,
                       expected_preflight=expected_runtime_preflight(launch, binding.runtime_preflight_receipt))

    codec = _codec()
    # Otherwise CausalQualificationInput raises QualificationRunnerError, killing the epoch.
    if (marginal_workload_digest(value.resident_speed_plan.baseline.session_plan)
            != codec.decode(origin["candidates"][0]).profile.reference.workload_digest):
        raise CheckpointComputationChanged("worker checkpoint workload changed")
    speed_plan = restore_speed_plan(value.resident_speed_plan, origin)
    candidate = speed_plan.candidate
    base_launch, base_binding = _restore_binding(
        value.prepared.incumbent_binding.launch_binding, origin["baseline"])
    source = replace(value.prepared.source,
        incumbent=stack(value.prepared.source.incumbent, base_launch),
        candidate=stack(value.prepared.source.candidate, candidate.launch),
        baseline_before=StackArmIdentity(base_launch.stack_digest, base_launch.tree_digest),
        baseline_after=StackArmIdentity(base_launch.stack_digest, base_launch.tree_digest),
        challenger=StackArmIdentity(candidate.launch.stack_digest, candidate.launch.tree_digest))
    prepared_candidate, = value.prepared.candidates
    prepared = replace(value.prepared, source=source,
        baseline_launch=base_launch,
        baseline_session_plan=session(value.prepared.baseline_session_plan, base_launch, base_binding),
        incumbent_binding=materialized(value.prepared.incumbent_binding, base_binding),
        candidates=(replace(prepared_candidate, arm=source, launch=candidate.launch,
                    binding=materialized(prepared_candidate.binding, candidate.binding),
                    session_plan=candidate.session_plan),))
    pristine_launch, pristine_binding = _restore_binding(value.pristine_binding, origin["pristine"])
    audit = value.resident_audit_plan
    audit_policy = audit.audit_policy
    if "audit_policies" in origin:
        from cacheon.eval.oci_session_protocol import SlotAuditPolicy
        audit_policy, = (SlotAuditPolicy.from_dict(row) for row in origin["audit_policies"])
    elif "selection_secret" in origin:
        from cacheon.eval.b300_registered_qualification import AUDIT_SEED_DOMAIN

        audit_policy = replace(audit_policy, validator_seed=hashlib.sha256(
            AUDIT_SEED_DOMAIN + bytes.fromhex(origin["selection_secret"])
            + bytes.fromhex(value.candidates[0].selected_delta_digest)).hexdigest()[:32])
    context, calibration, threshold = (codec.decode(row) for row in origin["calibration"])
    calibration_ref = EvidenceArtifactRef.from_dict(origin["calibration_ref"])
    if "calibration_bytes" in origin:
        published = publish_evidence(value.evidence_root, canonical_json_bytes(origin["calibration_bytes"]),
            domain=calibration_ref.domain, media_type=calibration_ref.media_type, schema=calibration_ref.schema)
        if published != calibration_ref:
            raise QualificationContinuationError("worker checkpoint calibration bytes changed")
    restored = replace(value, prepared=prepared,
        model_mount=replace(value.model_mount, arena_digest=base_launch.arena_digest),
        candidates=tuple(codec.decode(row) for row in origin["candidates"]),
        commitment=codec.decode(origin["commitment"]),
        selection_secret=(bytes.fromhex(origin["selection_secret"])
                          if "selection_secret" in origin else value.selection_secret),
        calibration_context=context, calibration_manifest=calibration,
        calibration_threshold_policy=threshold, calibration_artifact_ref=calibration_ref,
        audit_policies=(audit_policy,),
        resident_speed_plan=speed_plan,
        resident_audit_plan=ResidentAuditExecutionAuthority.derive(
            candidate.launch, candidate.binding, candidate.session_plan,
            audit_policy=audit_policy, max_new_tokens=audit.plan.max_new_tokens,
            top_logprobs_num=audit.plan.top_logprobs_num,
            executor_namespace_digest=candidate.executor_namespace_digest,
            runtime_resource_policy_digest=candidate.runtime_resource_policy_digest,
            device_configuration_digest=candidate.device_configuration_digest),
        pristine_stack=stack(value.pristine_stack, pristine_launch),
        pristine_launch=pristine_launch, pristine_binding=pristine_binding,
        reference_preflight=expected_runtime_preflight(pristine_launch,
                                                       pristine_binding.runtime_preflight_receipt),
        expected_launch_resource_policy_digest=candidate.launch.resource_policy_digest,
        expected_runtime_resource_policy_digest=candidate.runtime_resource_policy_digest,
        expected_device_policy_digest=base_launch.hardware.device_policy_digest)
    if _contract_fields(value, portable=True) != _contract_fields(restored, portable=True):
        raise CheckpointComputationChanged("worker checkpoint changed the sealed evaluation")
    if _contract(restored) != origin["contract"] or restored.prepared.source.digest != origin["source"]:
        raise QualificationContinuationError("worker checkpoint sealed evaluation cannot be reconstructed")
    if qualification_authority_digest(restored) != origin["authority"]:
        raise QualificationContinuationError("worker checkpoint original authority cannot be reconstructed")
    return restored


def publish_snapshot(continuation):
    """Publish complete immutable records for the standing CPU transfer loop."""
    origin = getattr(continuation, "checkpoint_origin", None)
    if origin is None:
        return
    from cacheon.chain.remote_worker_spool import atomic_bytes, atomic_json

    stages = dict(getattr(continuation, "restored_stages", {}))
    for name, records in _COMPLETE.items():
        paths = [continuation.directory / (record + ".json") for record in records]
        if all(path.is_file() for path in paths):
            stages[name] = {"origin": origin,
                            "records": {path.stem: json.loads(path.read_bytes()) for path in paths}}
    body = {"schema": SCHEMA, "origin": origin, "stages": stages}
    raw = canonical_json_bytes(body)
    if len(raw) > MAX_CHECKPOINT_BYTES:
        raise QualificationContinuationError("worker checkpoint exceeds its retained byte bound")
    data = gzip.compress(raw, mtime=0)
    digest = hashlib.sha256(data).hexdigest()
    root = continuation.checkpoint_root
    root.mkdir(mode=0o700, exist_ok=True)
    blob = root / (digest + ".json.gz")
    if not blob.exists():
        atomic_bytes(blob, data, mode=0o400)
    atomic_json(root / (continuation.request_digest + ".json"),
                {"sha256": digest, "size": len(data), "request_digest": continuation.request_digest},
                mode=0o400)


def read_snapshot(path, digest):
    """Reopen the CPU-selected snapshot before constructing any GPU work."""
    with Path(path).open("rb") as handle:
        data = handle.read(MAX_CHECKPOINT_BYTES + 1)
    if len(data) > MAX_CHECKPOINT_BYTES or hashlib.sha256(data).hexdigest() != digest:
        raise QualificationContinuationError("worker checkpoint archive changed")
    import io

    with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
        raw = handle.read(MAX_CHECKPOINT_BYTES + 1)
    if len(raw) > MAX_CHECKPOINT_BYTES:
        raise QualificationContinuationError("worker checkpoint expands beyond its byte bound")
    value = json.loads(raw)
    if (set(value) != {"schema", "origin", "stages"} or value["schema"] != SCHEMA
            or canonical_json_bytes(value) != raw or set(value["stages"]) - set(_COMPLETE)):
        raise QualificationContinuationError("worker checkpoint is not canonical")
    return value


def restore_stages(value, continuation, snapshot):
    """Reopen each completed stage under its own original plan and clock."""
    stages = snapshot["stages"]
    if stages and ("quality" in stages and "audit" not in stages
                   or "audit" in stages and "speed" not in stages):
        raise QualificationContinuationError("worker checkpoint lacks its completed predecessors")
    restored = {}
    for stage, row in stages.items():
        origin = row["origin"]
        original = restore_origin(value, origin)
        parent = QualificationContinuationStore(continuation.directory / "parents").scope(
            request_digest=origin["request"], authority_digest=origin["authority"],
            source_digest=origin["source"])
        if set(row["records"]) != set(_COMPLETE[stage]):
            raise QualificationContinuationError("worker checkpoint stage records are incomplete")
        for name, record in row["records"].items():
            encoded = parent._record_bytes(name, record["payload"])
            if encoded != canonical_json_bytes(record):
                raise QualificationContinuationError("worker checkpoint changed an original stage record")
            parent._record(name, record["payload"])
        restored[stage] = SimpleNamespace(value=original, continuation=parent, origin=origin,
            execution=restore_origin(value, origin.get("execution", origin)))
    continuation.restored_stages = {name: stages[name] for name in restored}
    return restored


def configure_continuation(value, continuation, executor):
    """Install the authenticated imported checkpoint in the ordinary runner."""
    from cacheon.chain.remote_worker_spool import load_json
    continuation.checkpoint_value = value
    continuation.checkpoint_origin = capture_origin(value, continuation, executor)
    replacement = getattr(continuation, "replacement_execution", None)
    if replacement is not None:
        continuation.checkpoint_origin["execution"] = capture_origin(replacement, continuation, executor)
    pointer = continuation.directory.parent / "imports" / (continuation.request_digest + ".json")
    continuation.worker_recovery = {}
    if pointer.exists():
        imported = load_json(pointer)
        snapshot = read_snapshot(pointer.parent / (imported["sha256"] + ".json.gz"), imported["sha256"])
        stages = restore_stages(value, continuation, snapshot)
        if "speed" in stages:
            continuation.worker_recovery = stages
            _publish_recovery(value, continuation, imported["sha256"], snapshot["stages"])
    publish_snapshot(continuation)


def _publish_recovery(value, continuation, digest, stages):
    from cacheon.eval.evidence_store import publish_evidence
    from cacheon.eval.qualification_runner import qualification_authority_digest

    stages = {name: row for name, row in stages.items() if name in ("speed", "audit", "quality")}
    public = {"schema": RECOVERY_SCHEMA, "checkpoint": digest,
        "authority": qualification_authority_digest(value),
        "origins": {name: _public_origin(row["origin"]) for name, row in stages.items()},
        "records": {name: {stage: record["record_digest"] for stage, record in row["records"].items()}
                    for name, row in stages.items()},
        "replacement_boot": continuation.checkpoint_origin["boot"],
        "replacement": _public_origin(continuation.checkpoint_origin.get("execution", continuation.checkpoint_origin))}
    continuation.worker_recovery_reference = publish_evidence(value.evidence_root,
        canonical_json_bytes(public), domain=RECOVERY_DOMAIN, media_type="application/json", schema=RECOVERY_SCHEMA)


def _public_origin(origin):
    """Keep the private selection secret out of the exported result product."""
    row = {key: value for key, value in origin.items()
           if key not in {"selection_secret", "calibration_bytes", "execution"}}
    if "execution" in origin:
        row["execution"] = _public_origin(origin["execution"])
    return row


def prepare_work(work, store, request_digest):
    """Retain the original qualification policy while commissioning missing roles."""
    from cacheon.chain.remote_worker_spool import load_json
    from cacheon.eval.qualification_intake import QualificationAuthorityManifest, QualificationPlanFactory

    pointer = store.root / "imports" / (request_digest + ".json")
    if not pointer.exists():
        return work
    imported = load_json(pointer)
    snapshot = read_snapshot(pointer.parent / (imported["sha256"] + ".json.gz"), imported["sha256"])
    stage = snapshot["stages"].get("speed")
    if stage is None:
        return work
    origin = stage["origin"]
    secret = bytes.fromhex(origin["selection_secret"])
    replacement = work.factory.build()
    original = restore_origin(replacement, origin)
    manifest = QualificationAuthorityManifest.seal(original,
        reservations=work.factory.manifest.reservations,
        selection_secret_reference=work.factory.manifest.selection_secret_reference)
    def load_secret(reference):
        if reference != manifest.selection_secret_reference:
            raise QualificationContinuationError("replacement selection reference changed")
        return secret
    def build(selected):
        if selected != secret:
            raise QualificationContinuationError("replacement selection secret changed")
        return original
    replacements = getattr(store, "worker_replacements", {})
    replacements[request_digest] = replacement
    store.worker_replacements = replacements
    return replace(work, factory=QualificationPlanFactory(manifest, load_secret, build,
                                                         work.factory.audit_control_delta_digests))


def reopen_recovery(root, reference, *, expected):
    """Rebuild prior stage authorities while independently regrading a result."""
    from cacheon.eval.evidence_store import reopen_evidence
    from cacheon.eval.qualification_runner import qualification_authority_digest

    if (reference.domain, reference.schema) != (RECOVERY_DOMAIN, RECOVERY_SCHEMA):
        raise QualificationContinuationError("worker recovery has another evidence domain")
    raw = reopen_evidence(root, reference)
    body = json.loads(raw)
    if (canonical_json_bytes(body) != raw or body["schema"] != RECOVERY_SCHEMA
            or body["authority"] != qualification_authority_digest(expected)
            or "speed" not in body["origins"]):
        raise QualificationContinuationError("worker recovery differs from its replacement authority")
    values = {stage: restore_origin(expected, origin.get("execution", origin))
              for stage, origin in body["origins"].items()}
    values["replacement"] = restore_origin(expected, body["replacement"])
    return body, values


def result_execution_authorities(expected, reference, clocks):
    """Bind result clocks and actual audit/T launches to the retained handoff."""
    if reference is None:
        if clocks:
            raise QualificationContinuationError("multiple clocks lack a worker recovery record")
        return expected.resident_audit_plan, expected
    body, values = reopen_recovery(expected.evidence_root, reference, expected=expected)
    domains = tuple(body["origins"][stage]["boot"] if stage in body["origins"]
                    else body["replacement_boot"] for stage in ("speed", "audit", "quality"))
    if tuple(clocks) != domains[:len(clocks)] or len(clocks) not in (1, 2, 3):
        raise QualificationContinuationError("result clock domains differ from its worker handoff")
    audit = values.get("audit", values["replacement"])
    quality = values.get("quality", values["replacement"])
    return audit.resident_audit_plan, quality


def reference_execution_matches(expected, actual, session):
    """Keep frozen reference math while binding T to its actual replacement launch."""
    from cacheon.eval.qualification import ReferenceManifest

    actual = expected if actual is None else actual
    if type(actual) is not ReferenceManifest:
        return False
    # result_execution_authorities has already reconstructed each exact tree
    # and launch. These are commissioning identities, not different T math.
    movable = {"pristine_launch_digest", "logical_hardware_digest", "arena_digest",
               "pristine_stack_digest", "pristine_tree_digest", "controller_distribution_digest"}
    return (all(getattr(expected, name) == getattr(actual, name)
                for name in expected.__dataclass_fields__ if not name.startswith("_") and name not in movable)
            and session.reference_manifest_digest == actual.digest
            and session.launch_digest == actual.pristine_launch_digest)
