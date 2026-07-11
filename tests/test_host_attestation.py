from __future__ import annotations
import copy
import dataclasses
import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace
import pytest
from optima.eval import host_attestation as ha
from optima.eval.host_attestation import (
    HostAttestationError,
    publish_host_attestation,
    verify_host_attestation,
)
from optima.arenas import MINIMAX_M3_B300_TP4_DECODE_V1, derive_prompt_seed
from optima.commit_reveal import make_chain_scope
from optima.competition import ResolvedCompetition
from optima.eval.qualification import QualificationReport
from tests.quality_report_helpers import (
    calibrated_test_arena,
    evaluation_report,
    patch_qualification_registry,
)
ARENA = calibrated_test_arena(MINIMAX_M3_B300_TP4_DECODE_V1)
IMAGE = ARENA.validator_image
CONFIG_HASH = "b" * 64
POLICY_HASH = "c" * 64
BUNDLE_HASH = "e" * 64
CHAIN_SCOPE = make_chain_scope(genesis_hash="0x" + "9" * 64, netuid=120)
VALIDATOR_HOTKEY = "validator-hotkey"
MINER_HOTKEY = "miner-hotkey"
EVALUATION_ID = "8" * 64
SEED_BLOCK = 100
SEED_ROUND_ID = 1
SEED_BLOCK_HASH = "0x" + "5" * 64
PROMPT_SEED = derive_prompt_seed(
    ARENA,
    bundle_hash=BUNDLE_HASH,
    round_id=SEED_ROUND_ID,
    block_hash=SEED_BLOCK_HASH,
)


@pytest.fixture
def publication_root(tmp_path: Path) -> Path:
    root = tmp_path / "controller-publication"
    root.mkdir(mode=0o700)
    (root / "published").mkdir(mode=0o700)
    return root


@pytest.fixture(autouse=True)
def _test_quality_arena(monkeypatch):
    patch_qualification_registry(monkeypatch, ARENA)


def _evaluation_report(*, candidate_rate: float = 108.0):
    return evaluation_report(ARENA, candidate_rate=candidate_rate)


def _prepared_qualification(*, candidate_rate: float = 108.0):
    return QualificationReport.prepare_evidence(
        _evaluation_report(candidate_rate=candidate_rate),
        competition=ResolvedCompetition(
            target="attention.decode",
            mode="slot",
            members=("attention.decode",),
            crownable=True,
        ),
        arena=ARENA,
        bundle_hash=BUNDLE_HASH,
        prompt_seed=PROMPT_SEED,
        seed_round_id=SEED_ROUND_ID,
        seed_block=SEED_BLOCK,
        seed_block_hash=SEED_BLOCK_HASH,
        chain_scope=CHAIN_SCOPE,
        validator_hotkey=VALIDATOR_HOTKEY,
        evaluation_id=EVALUATION_ID,
        miner_hotkey=MINER_HOTKEY,
        settlement_round_id=SEED_ROUND_ID,
        evaluation_block=SEED_BLOCK,
    )


def _context() -> dict:
    return _prepared_qualification().attestation_context()


def _runtime() -> dict:
    return {
        "schema": "optima-stock-runtime-preflight-v1",
        "requested_image": IMAGE,
        "requested_manifest_digest": IMAGE.rsplit("@", 1)[1],
        "local_image_id": "sha256:" + "6" * 64,
        "repo_digests": [IMAGE],
        "docker_binary": "/usr/bin/docker",
        "uid": 65532,
        "gid": 65532,
        "sglang_version": ARENA.sglang_version,
        "python": {
            "implementation": "cpython",
            "version": "3.11.12",
            "abi": "cpython-311-x86_64-linux-gnu",
            "platform": "linux-x86_64",
            "machine": "x86_64",
        },
        "packages": {
            "cuda-python": "13.0.1",
            "flashinfer-python": "0.4.1",
            "nvidia-cuda-runtime-cu12": "12.9.79",
            "torch": "2.8.0",
            "triton": "3.4.0",
        },
        "cuda": {
            "cudart_library": "libcudart.so.12",
            "cuda_visible_devices": "",
            "nvidia_visible_devices": "void",
        },
        "security_argv_sha256": "7" * 64,
    }


def _telemetry() -> list[dict]:
    return [
        {
            "physical_id": gpu,
            "uuid": f"GPU-00000000-0000-0000-0000-{gpu:012d}",
            "pstate": "P0",
            "temperature_c": 35,
            "gpu_utilization_percent": 0,
            "memory_utilization_percent": 0,
            "current_graphics_clock_mhz": 210,
            "current_memory_clock_mhz": 3996,
            "power_draw_mw": 80_000,
        }
        for gpu in range(4)
    ]


def _receipt(sequence: int, arm: str, phase: str) -> dict:
    started = float(sequence * 10)
    return {
        "schema": "optima.device-state-receipt.v1",
        "sequence": sequence,
        "arm": arm,
        "phase": phase,
        "selected_physical_gpu_ids": [0, 1, 2, 3],
        "configuration_sha256": CONFIG_HASH,
        "policy_sha256": POLICY_HASH,
        "started_monotonic_s": started,
        "completed_monotonic_s": started + 1.0,
        "consecutive_idle_samples": 2,
        "samples": [
            {
                "monotonic_s": started + offset,
                "telemetry": _telemetry(),
                "processes": [],
                "idle": True,
                "idle_reason": "no processes; temperature/utilization within policy",
                "active_envelope_passed": False,
                "active_envelope_reason": "not evaluated",
            }
            for offset in (0.25, 0.75)
        ],
    }


def _active_receipt(sequence: int, arm: str) -> dict:
    started = float(sequence * 10)
    processes = [
        {
            "physical_id": gpu,
            "pid": 10_000 + gpu,
            "kind": "C",
            "process_name": "sglang::scheduler",
        }
        for gpu in range(4)
    ]
    return {
        "schema": "optima.device-state-active-receipt.v2",
        "sequence": sequence,
        "arm": arm,
        "event": "final-warmup-conditioning",
        "selected_physical_gpu_ids": [0, 1, 2, 3],
        "configuration_sha256": CONFIG_HASH,
        "policy_sha256": POLICY_HASH,
        "started_monotonic_s": started,
        "completed_monotonic_s": started + 1.0,
        "consecutive_active_samples": 2,
        "release_sample_index": 2,
        "post_release_ready_samples": 1,
        "samples": [
            {
                "monotonic_s": started + offset,
                "telemetry": _telemetry(),
                "processes": processes,
                "idle": False,
                "idle_reason": "active scheduler processes",
                "active_envelope_passed": True,
                "active_envelope_reason": "active envelope satisfied",
            }
            for offset in (0.2, 0.4, 0.75)
        ],
    }


def _receipts(*, first_sequence: int = 1) -> list[dict]:
    result = []
    sequence = first_sequence
    for arm in ("baseline-1", "candidate-1", "bookend-1"):
        result.append(_receipt(sequence, arm, "pre"))
        sequence += 1
        result.append(_active_receipt(sequence, arm))
        sequence += 1
        result.append(_receipt(sequence, arm, "post"))
        sequence += 1
    return result


def _publish(
    root: Path, *, context=None, runtime=None, receipts=None, qualification=None
):
    prepared = _prepared_qualification()
    return publish_host_attestation(
        root,
        context=_context() if context is None else context,
        runtime_preflight=_runtime() if runtime is None else runtime,
        device_receipts=_receipts() if receipts is None else receipts,
        qualification_evidence=(
            prepared.evidence_dict() if qualification is None else qualification
        ),
    )


def test_happy_path_publishes_frozen_canonical_sidecar_outside_candidate_tree(
    publication_root,
):
    reference = _publish(publication_root)
    path = Path(reference.path)
    raw = path.read_bytes()
    payload = json.loads(raw)

    assert reference.sha256 == "sha256:" + hashlib.sha256(raw).hexdigest()
    expected_runtime = json.dumps(
        _runtime(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    assert reference.runtime_preflight_sha256 == (
        "sha256:" + hashlib.sha256(expected_runtime).hexdigest()
    )
    assert reference.qualification_evidence_sha256 == (
        _prepared_qualification().qualification_evidence_sha256
    )
    assert reference.receipt_count == 9
    assert reference.device_configuration_sha256 == CONFIG_HASH
    assert reference.device_policy_sha256 == POLICY_HASH
    assert reference.arms == ("baseline-1", "candidate-1", "bookend-1")
    assert payload == {
        "schema": "optima-host-attestation-v4",
        "context": _context(),
        "runtime_preflight": _runtime(),
        "device_receipts": _receipts(),
        "qualification_evidence": _prepared_qualification().evidence_dict(),
    }
    assert raw.endswith(b"\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert publication_root / "published" not in path.parents
    with pytest.raises(dataclasses.FrozenInstanceError):
        reference.sha256 = "changed"


def test_missing_bookend_is_rejected(publication_root):
    with pytest.raises(HostAttestationError, match="9..96|bookend"):
        _publish(publication_root, receipts=_receipts()[:6])


@pytest.mark.parametrize("attack", ("reordered", "mismatched"))
def test_reordered_or_mismatched_adjacent_pairs_are_rejected(publication_root, attack):
    receipts = _receipts()
    if attack == "reordered":
        receipts[0], receipts[1] = receipts[1], receipts[0]
    else:
        receipts[1]["arm"] = "baseline-2"
    with pytest.raises(HostAttestationError, match="sequence|timestamp|same-arm"):
        _publish(publication_root, receipts=receipts)


def test_reordered_bracket_stages_are_rejected_even_with_valid_pairs(publication_root):
    receipts = []
    sequence = 1
    for arm in ("candidate-1", "baseline-1", "bookend-1"):
        receipts.append(_receipt(sequence, arm, "pre"))
        sequence += 1
        receipts.append(_active_receipt(sequence, arm))
        sequence += 1
        receipts.append(_receipt(sequence, arm, "post"))
        sequence += 1
    with pytest.raises(HostAttestationError, match="ordered complete"):
        _publish(publication_root, receipts=receipts)


def test_device_configuration_change_is_rejected(publication_root):
    receipts = _receipts()
    receipts[6]["configuration_sha256"] = "8" * 64
    with pytest.raises(HostAttestationError, match="changed across the bracket"):
        _publish(publication_root, receipts=receipts)


@pytest.mark.parametrize(
    "attack",
    (
        "event",
        "pre_verdict",
        "post_verdict",
        "release_boundary",
        "ready_count",
        "missing_gpu_process",
    ),
)
def test_active_conditioning_receipt_fails_closed(publication_root, attack):
    receipts = _receipts()
    active = receipts[1]
    if attack == "event":
        active["event"] = "unrelated-work"
    elif attack == "pre_verdict":
        active["samples"][1]["active_envelope_passed"] = False
    elif attack == "post_verdict":
        active["samples"][-1]["active_envelope_passed"] = False
    elif attack == "release_boundary":
        active["release_sample_index"] = 1
    elif attack == "ready_count":
        active["post_release_ready_samples"] = 0
    else:
        active["samples"][-1]["processes"] = active["samples"][-1][
            "processes"
        ][:-1]
    with pytest.raises(
        HostAttestationError,
        match="final-warmup|active-envelope|ready|release|every selected GPU|must be an integer",
    ):
        _publish(publication_root, receipts=receipts)


def test_nonfinite_nested_number_is_rejected(publication_root):
    receipts = _receipts()
    receipts[2]["samples"][0]["monotonic_s"] = float("nan")
    with pytest.raises(HostAttestationError, match="finite"):
        _publish(publication_root, receipts=receipts)
    assert not (publication_root / "host_attestations").exists()


@pytest.mark.parametrize("attack", ("symlink", "hardlink"))
def test_existing_symlink_or_hardlink_is_rejected(publication_root, tmp_path, attack):
    reference = _publish(publication_root)
    path = Path(reference.path)
    raw = path.read_bytes()
    path.unlink()
    external = tmp_path / f"external-{attack}.json"
    external.write_bytes(raw)
    os.chmod(external, 0o444)
    if attack == "symlink":
        path.symlink_to(external)
    else:
        os.link(external, path)

    with pytest.raises(HostAttestationError, match="unsafe|shape"):
        _publish(publication_root)


def test_existing_same_digest_corruption_is_rejected(publication_root):
    reference = _publish(publication_root)
    path = Path(reference.path)
    os.chmod(path, 0o600)
    path.write_bytes(b"corrupt\n")
    os.chmod(path, 0o444)

    with pytest.raises(HostAttestationError, match="shape|corrupt"):
        _publish(publication_root)


def test_canonical_existing_publication_is_reused_without_replacement(publication_root):
    first = _publish(publication_root)
    before = Path(first.path).stat()
    second = _publish(
        publication_root,
        context=dict(reversed(list(_context().items()))),
        runtime=dict(reversed(list(_runtime().items()))),
        receipts=copy.deepcopy(_receipts()),
    )
    after = Path(second.path).stat()

    assert first == second
    assert (before.st_dev, before.st_ino, before.st_mtime_ns) == (
        after.st_dev, after.st_ino, after.st_mtime_ns
    )
    assert not list(Path(first.path).parent.glob(".tmp-*"))


def test_new_publication_fsyncs_regular_file_and_directories(
    publication_root, monkeypatch,
):
    observed = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        observed.append(stat.S_IFMT(os.fstat(fd).st_mode))
        return real_fsync(fd)

    monkeypatch.setattr(ha.os, "fsync", recording_fsync)
    _publish(publication_root)

    assert stat.S_IFREG in observed
    assert observed.count(stat.S_IFDIR) >= 2


def test_private_root_and_candidate_published_subtree_are_enforced(tmp_path, monkeypatch):
    group_writable = tmp_path / "group-writable"
    group_writable.mkdir(mode=0o700)
    os.chmod(group_writable, 0o720)
    with pytest.raises(HostAttestationError, match="permissions"):
        _publish(group_writable)

    candidate_root = tmp_path / "published"
    candidate_root.mkdir(mode=0o700)
    with pytest.raises(HostAttestationError, match="candidate-mounted"):
        _publish(candidate_root)

    nonowner = tmp_path / "nonowner"
    nonowner.mkdir(mode=0o700)
    real_uid = os.geteuid()
    monkeypatch.setattr(ha.os, "geteuid", lambda: real_uid + 1)
    with pytest.raises(HostAttestationError, match="controller-owned"):
        _publish(nonowner)


def test_runtime_schema_size_and_receipt_count_are_bounded(publication_root):
    runtime = _runtime()
    runtime["extra"] = True
    with pytest.raises(HostAttestationError, match="fields"):
        _publish(publication_root, runtime=runtime)

    runtime = _runtime()
    runtime["packages"]["torch"] = "x" * 100_000
    with pytest.raises(HostAttestationError, match="bounded|bound"):
        _publish(publication_root, runtime=runtime)

    receipts = _receipts()
    while len(receipts) <= ha.MAX_DEVICE_RECEIPTS:
        sequence = receipts[-1]["sequence"] + 1
        arm_number = len(receipts) // 3 + 1
        receipts.extend((
            _receipt(sequence, f"bookend-{arm_number}", "pre"),
            _active_receipt(sequence + 1, f"bookend-{arm_number}"),
            _receipt(sequence + 2, f"bookend-{arm_number}", "post"),
        ))
    with pytest.raises(HostAttestationError, match="9..96"):
        _publish(publication_root, receipts=receipts)


def test_failed_attempt_sequence_gaps_are_allowed_but_reuse_is_rejected(
    publication_root,
):
    # Failed arms remain in the launcher's diagnostic stream but are deliberately
    # omitted from crown evidence. The next successful B/C/B' bracket therefore
    # retains exact triplets whose trusted-guard sequences can start above one.
    gapped = _receipts(first_sequence=41)
    retained_arms = ("baseline-2", "candidate-3", "bookend-4")
    for group_index, arm in enumerate(retained_arms):
        for receipt in gapped[group_index * 3:(group_index + 1) * 3]:
            receipt["arm"] = arm
    for index, receipt in enumerate(gapped):
        receipt["sequence"] += index
        receipt["started_monotonic_s"] += index * 10.0
        receipt["completed_monotonic_s"] += index * 10.0
        for sample in receipt["samples"]:
            sample["monotonic_s"] += index * 10.0
    accepted = _publish(publication_root, receipts=gapped)
    assert accepted.receipt_count == 9

    reused = _receipts()
    reused[1]["sequence"] = reused[0]["sequence"]
    with pytest.raises(HostAttestationError, match="increasing and unique"):
        _publish(publication_root, receipts=reused)


def test_per_stage_retained_arm_ordinals_must_increase(publication_root):
    receipts = []
    sequence = 1
    for arm in ("baseline-2", "baseline-1", "candidate-1", "bookend-1"):
        receipts.append(_receipt(sequence, arm, "pre"))
        receipts.append(_active_receipt(sequence + 1, arm))
        receipts.append(_receipt(sequence + 2, arm, "post"))
        sequence += 3
    with pytest.raises(HostAttestationError, match="strictly increasing"):
        _publish(publication_root, receipts=receipts)


@pytest.mark.parametrize("field", ("requested_image", "sglang_version"))
def test_runtime_receipt_must_bind_the_same_arena_context(publication_root, field):
    runtime = _runtime()
    if field == "requested_image":
        runtime[field] = "other/sglang@sha256:" + "9" * 64
        runtime["requested_manifest_digest"] = "sha256:" + "9" * 64
        runtime["repo_digests"] = [runtime[field]]
    else:
        runtime[field] = "0.5.3"
    with pytest.raises(HostAttestationError, match="differs.*context"):
        _publish(publication_root, runtime=runtime)


def test_coherent_result_edit_cannot_reuse_an_existing_sidecar(publication_root):
    original = _publish(publication_root)
    edited = _prepared_qualification(candidate_rate=109.0)
    edited_context = edited.attestation_context()

    with pytest.raises(HostAttestationError, match="qualification evidence"):
        _publish(
            publication_root,
            context=_context(),
            qualification=edited.evidence_dict(),
        )
    with pytest.raises(HostAttestationError, match="settlement context"):
        verify_host_attestation(
            publication_root,
            original.sha256,
            expected_context=edited_context,
        )

    changed = _publish(
        publication_root,
        context=edited_context,
        qualification=edited.evidence_dict(),
    )
    assert changed.sha256 != original.sha256
    assert (
        changed.qualification_evidence_sha256
        == edited.qualification_evidence_sha256
    )


@pytest.mark.parametrize(
    "attack",
    ("miner", "round", "target", "mode", "score", "decision", "quality"),
)
def test_settlement_projection_rewrite_cannot_reuse_sidecar(
    publication_root, attack,
):
    original = _publish(publication_root)
    rewritten = _context()
    if attack == "miner":
        rewritten["miner_hotkey"] = "different-miner"
    elif attack == "round":
        rewritten["settlement_round_id"] += 1
        rewritten["evaluation_block"] += ARENA.settlement.round_blocks
    elif attack == "target":
        rewritten["target"] = "norm.rmsnorm"
        rewritten["member_slots"] = ["norm.rmsnorm"]
    elif attack == "mode":
        rewritten["target"] = "sglang.inference.bundle.v1"
        rewritten["mode"] = "system"
        rewritten["member_slots"] = []
    elif attack == "score":
        rewritten["score"] = 1.09
    elif attack == "decision":
        rewritten["passed_timed_quality"] = False
        rewritten["passed_quality"] = False
        rewritten["crownable"] = False
        rewritten["score"] = 0.0
    else:
        rewritten["quality_evidence"] = "coherently rewritten quality summary"

    with pytest.raises(HostAttestationError, match="settlement context"):
        verify_host_attestation(
            publication_root,
            original.sha256,
            expected_context=rewritten,
        )


@pytest.mark.parametrize("field", ("chain_scope", "validator_hotkey", "evaluation_id"))
def test_cross_chain_validator_or_evaluation_transplant_fails(
    publication_root, field,
):
    original = _publish(publication_root)
    transplanted = _context()
    if field == "chain_scope":
        transplanted[field] = (
            ARENA.settlement.chain_scope_scheme + ":sha256:" + "1" * 64
        )
    elif field == "validator_hotkey":
        transplanted[field] = "different-validator"
    else:
        transplanted[field] = "1" * 64

    with pytest.raises(HostAttestationError, match="provenance"):
        _publish(publication_root, context=transplanted)
    with pytest.raises(HostAttestationError, match="settlement context"):
        verify_host_attestation(
            publication_root,
            original.sha256,
            expected_context=transplanted,
        )


def test_verify_reopens_hashes_and_revalidates_expected_context(publication_root):
    published = _publish(publication_root)

    assert verify_host_attestation(
        publication_root, published.sha256, expected_context=_context()
    ) == published
    assert verify_host_attestation(
        publication_root, published, expected_context=_context()
    ) == published

    changed = _context()
    changed["bundle_hash"] = "9" * 64
    with pytest.raises(HostAttestationError, match="settlement context"):
        verify_host_attestation(
            publication_root, published.sha256, expected_context=changed
        )


def test_verify_never_trusts_reference_path_or_metadata(publication_root):
    published = _publish(publication_root)
    forged = dataclasses.replace(
        published,
        path=str(publication_root / "published" / "candidate-controlled.json"),
    )
    with pytest.raises(HostAttestationError, match="reference metadata/path"):
        verify_host_attestation(
            publication_root, forged, expected_context=_context()
        )


def test_verify_rejects_missing_symlink_hardlink_and_noncanonical_bytes(
    publication_root, tmp_path,
):
    published = _publish(publication_root)
    path = Path(published.path)
    raw = path.read_bytes()
    path.unlink()
    with pytest.raises(HostAttestationError, match="missing/unsafe"):
        verify_host_attestation(
            publication_root, published.sha256, expected_context=_context()
        )

    external = tmp_path / "external-verify.json"
    external.write_bytes(raw)
    os.chmod(external, 0o444)
    path.symlink_to(external)
    with pytest.raises(HostAttestationError, match="missing/unsafe"):
        verify_host_attestation(
            publication_root, published.sha256, expected_context=_context()
        )
    path.unlink()

    os.link(external, path)
    with pytest.raises(HostAttestationError, match="shape"):
        verify_host_attestation(
            publication_root, published.sha256, expected_context=_context()
        )
    path.unlink()

    payload = json.loads(raw)
    noncanonical = json.dumps(payload, indent=2).encode("ascii") + b"\n"
    noncanonical_digest = hashlib.sha256(noncanonical).hexdigest()
    noncanonical_path = path.parent / f"sha256-{noncanonical_digest}.json"
    noncanonical_path.write_bytes(noncanonical)
    os.chmod(noncanonical_path, 0o444)
    with pytest.raises(HostAttestationError, match="byte-canonical"):
        verify_host_attestation(
            publication_root,
            "sha256:" + noncanonical_digest,
            expected_context=_context(),
        )


def test_atomic_publication_detects_staged_name_replacement_race(
    publication_root, monkeypatch,
):
    real_link = os.link

    def racing_link(source, destination, *, src_dir_fd, dst_dir_fd,
                    follow_symlinks):
        os.unlink(source, dir_fd=src_dir_fd)
        replacement = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=src_dir_fd,
        )
        os.write(replacement, b"raced\n")
        os.close(replacement)
        return real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(ha.os, "link", racing_link)
    with pytest.raises(HostAttestationError, match="does not bind the staged inode"):
        _publish(publication_root)
