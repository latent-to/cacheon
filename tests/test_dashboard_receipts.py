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
from dashboard.receipts import screen_stages


def _receipt(stages: list[dict[str, object]]) -> str:
    return json.dumps({"candidate_digest": "ab" * 32, "results": stages})


def test_screen_stages_render_every_graded_check() -> None:
    stages = screen_stages(_receipt([
        {"stage": "static", "grade": "pass", "reason": "static_verified",
         "elapsed_ms": 18, "evidence_digest": "cd" * 32},
        {"stage": "abbreviated_serving", "grade": "fail",
         "reason": "speedup 0.9958x vs required 1.0103x, noise 0.0052, confident",
         "elapsed_ms": 176772},
    ]))

    assert stages == [
        {"stage": "static", "grade": "pass", "reason": "static_verified",
         "elapsed_ms": 18},
        {"stage": "abbreviated_serving", "grade": "fail",
         "reason": "speedup 0.9958x vs required 1.0103x, noise 0.0052, confident",
         "elapsed_ms": 176772},
    ]


def test_screen_stages_tolerate_prehardening_and_corrupt_receipts() -> None:
    # Pre-hardening receipts carry no reasons; the stage and grade still show.
    legacy = screen_stages(_receipt([{"stage": "abbreviated_serving",
                                      "grade": "pass"}]))
    assert legacy == [{"stage": "abbreviated_serving", "grade": "pass",
                       "reason": None, "elapsed_ms": None}]
    assert screen_stages(None) is None
    assert screen_stages("not json") is None
    assert screen_stages(json.dumps({"no_results": True})) is None
    assert screen_stages(json.dumps({"results": "scalar"})) is None
    assert screen_stages(json.dumps({"results": ["scalar-row"]})) is None


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


def test_qualification_speed_reads_the_retained_stage_exit(tmp_path: Path) -> None:
    root = prepare_evidence_root(tmp_path / "evidence")
    reference = publish_canonical_json_evidence(
        root,
        {"decision": "FAIL", "reason": "candidate_slower",
         "speed_witness": {"rates": [_witness_rate("B", 62.787),
                                     _witness_rate("C", 69.471)]}},
        domain="qualification.stage-exit",
        schema="cacheon.qualification.stage-exit.v1",
    )

    speed = qualification_speed(
        json.dumps(reference.to_dict()), (tmp_path / "empty", root))

    assert speed is not None
    assert [lane["role"] for lane in speed["lanes"]] == ["B", "C"]
    assert speed["lanes"][0]["tokens_per_second"] == 2087.6
    assert speed["lanes"][1]["tokens_per_second"] == 1886.7
    assert speed["lanes"][0]["window_scatter"] == 0.0
    assert speed["lanes"][0]["conditioning_ratio"] == 1.0
    assert speed["speedup"] == 0.9038


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
    # Retained but rate-free artifacts render as an empty lane list.
    assert qualification_speed(
        json.dumps(reference.to_dict()), (root,)) == {"lanes": []}


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
