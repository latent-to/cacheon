"""Measured-system identity is blind to controller revisions.

The acceptance requirement (operator order, 2026-08-10): a controller code
change must never invalidate sealed calibration or force re-derivation of
measurement evidence. Before this contract, CalibrationContext bound the
controller distribution digest twice -- directly, and transitively through
the full reference-manifest digest (whose launch digest embeds it again) --
so every commit re-bought every sealed authority.
"""

import hashlib

from cacheon.eval.calibration import CalibrationContext
from cacheon.eval.qualification import ReferenceManifest


def _manifest(controller: str, launch: str, model_content: str) -> ReferenceManifest:
    def digest(name: str) -> str:
        return hashlib.sha256(name.encode()).hexdigest()

    return ReferenceManifest(
        digest("s"),
        digest("t"),
        launch,
        digest("r"),
        digest("b"),
        digest("a"),
        digest("c"),
        controller,
        digest("w"),
        digest("1"),
        digest("2"),
        model_content,
        digest("h"),
        digest("d"),
        digest("k"),
        digest("e"),
        digest("j"),
        digest("p"),
    )


def _context(reference: ReferenceManifest) -> CalibrationContext:
    return CalibrationContext(
        reference.measured_digest,
        reference.arena_digest,
        reference.runtime_digest,
        reference.base_engine_digest,
        reference.model_revision_digest,
        reference.model_manifest_digest,
        reference.model_content_digest,
        reference.logical_hardware_digest,
        reference.workload_digest,
        hashlib.sha256(b"verification-policy").hexdigest(),
    )


def test_controller_revision_change_keeps_calibration_identity():
    before = _manifest("1" * 64, "4" * 64, "9" * 64)
    after = _manifest("f" * 64, "5" * 64, "9" * 64)

    # The provenance digest still moves with the controller: full manifests
    # remain distinguishable records of what ran.
    assert before.digest != after.digest

    # The measured identity does not move, so the sealed calibration context
    # -- and everything keyed to it -- survives the code change.
    assert before.measured_digest == after.measured_digest
    assert _context(before) == _context(after)
    assert _context(before).digest == _context(after).digest


def test_measured_system_change_moves_calibration_identity():
    before = _manifest("1" * 64, "4" * 64, "9" * 64)
    changed_model = _manifest("1" * 64, "4" * 64, "8" * 64)
    assert before.measured_digest != changed_model.measured_digest
    assert _context(before).digest != _context(changed_model).digest


def test_calibration_context_has_no_controller_field():
    assert "controller_distribution_digest" not in (
        CalibrationContext.__dataclass_fields__
    )


def test_engine_launch_measured_digest_is_controller_blind():
    import dataclasses

    from tests.test_engine_launch import _launch

    before = _launch(
        stack_digest=hashlib.sha256(b"stack").hexdigest(),
        tree_digest=hashlib.sha256(b"tree").hexdigest(),
    )
    after = dataclasses.replace(
        before,
        controller_distribution_digest=hashlib.sha256(b"new-rev").hexdigest(),
    )
    assert before.digest != after.digest
    assert before.measured_digest == after.measured_digest
