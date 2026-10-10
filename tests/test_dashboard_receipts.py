from __future__ import annotations

import json
from pathlib import Path

from cacheon.chain.baseline_band import (
    qualification_evidence_roots,
    qualification_speed,
)
from cacheon.eval.evidence_store import (
    prepare_evidence_root,
    publish_canonical_json_evidence,
)


def _witness_rate(role: str, seconds: float) -> dict[str, object]:
    return {
        "role": role,
        "timed_tokens": 786432,
        "timed_seconds": str(seconds * 6),
        "conditioning_seconds": str(seconds),
        "windows": [
            {"batch_index": index, "seconds": str(seconds), "tokens": 131072}
            for index in range(2, 8)
        ],
    }


def test_a_retained_batch_cell_witness_renders_no_rates(tmp_path: Path) -> None:
    """Its C/B lane ratio was never the credited gain, so the reader reports nothing for it."""
    root = prepare_evidence_root(tmp_path / "evidence")
    reference = publish_canonical_json_evidence(
        root,
        {"decision": "FAIL", "reason": "candidate_slower",
         "speed_witness": {"rates": [_witness_rate("B", 62.787),
                                     _witness_rate("C", 69.471)]}},
        domain="qualification.stage-exit",
        schema="cacheon.qualification.stage-exit.v1",
    )

    assert qualification_speed(json.dumps(reference.to_dict()), (tmp_path / "empty", root)) is None


def test_qualification_speed_absence_is_none_not_an_error(tmp_path: Path) -> None:
    root = prepare_evidence_root(tmp_path / "evidence")
    reference = publish_canonical_json_evidence(
        root, {"speed_witness": {"rates": []}},
        domain="qualification.stage-exit",
        schema="cacheon.qualification.stage-exit.v1",
    )
    missing = dict(reference.to_dict(), sha256="0" * 64)

    assert qualification_speed(None, (root,)) is None
    assert qualification_speed("", (root,)) is None
    assert qualification_speed("not json", (root,)) is None
    assert qualification_speed(json.dumps(missing), (root,)) is None
    assert qualification_speed(json.dumps(reference.to_dict()), ()) is None
    assert qualification_speed(json.dumps(reference.to_dict()), (root,)) is None


def test_qualification_evidence_roots_scan_rotated_stores(tmp_path: Path) -> None:
    state = tmp_path / "state"
    old = state / "qualification-evidence-aaaa"
    new = state / "qualification-evidence-bbbb"
    for directory in (old, new):
        directory.mkdir(parents=True)
    present_extra = tmp_path / "standing"
    present_extra.mkdir()

    roots = qualification_evidence_roots(
        state, (present_extra, tmp_path / "absent"))

    assert roots == (new, old, present_extra)
    assert qualification_evidence_roots(tmp_path / "nowhere") == ()
