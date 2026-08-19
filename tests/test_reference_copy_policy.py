"""The validator reference corpus demotes resubmitted public bundles."""

from __future__ import annotations

import dataclasses

import pytest

import cacheon.chain.reference_copy_policy as policy_module
from cacheon.chain.reference_copy_policy import (
    reconcile_reference_copies,
    reference_copy_match,
    validator_reference_fingerprints,
)
from cacheon.copy_fingerprint import fingerprint_submitted_delta


@pytest.fixture(autouse=True)
def _fresh_corpus_cache():
    policy_module._CACHE = None
    policy_module._LIBRARY_CACHE = None
    yield
    policy_module._CACHE = None
    policy_module._LIBRARY_CACHE = None


@dataclasses.dataclass
class _Row:
    reservation_id: str
    status: str
    delta_fingerprint: object


class _Store:
    def __init__(self, rows):
        self.rows = rows
        self.marked = []

    def all(self):
        return tuple(self.rows)

    def mark_reference_copy(self, reservation_id, reference_name):
        self.marked.append((reservation_id, reference_name))


def test_corpus_includes_the_published_fused_epilogue_fixture() -> None:
    names = [name for name, _fp in validator_reference_fingerprints()]
    assert "stack_fused_epilogue_atomic" in names
    assert len(names) == len(set(names))


def test_resubmitted_public_fixture_is_an_authoritative_reference_copy() -> None:
    fingerprint = fingerprint_submitted_delta(
        policy_module._repo_root() / "tests/fixtures/stack_fused_epilogue_atomic"
    )
    assert reference_copy_match(fingerprint) == "stack_fused_epilogue_atomic"
    assert reference_copy_match(None) is None
    assert reference_copy_match(object()) is None


def test_reconcile_marks_only_live_matching_rows() -> None:
    fixture = fingerprint_submitted_delta(
        policy_module._repo_root() / "tests/fixtures/stack_fused_epilogue_atomic"
    )
    other = fingerprint_submitted_delta(
        policy_module._repo_root() / "tests/fixtures/stack_msa_singleton"
    )
    store = _Store(
        [
            _Row("resubmission", "promoted", fixture),
            _Row("already-failed", "failed", fixture),
            _Row("expired-row", "expired", fixture),
            _Row("no-fingerprint", "promoted", None),
            _Row("msa-copy", "published", other),
        ]
    )

    dispositions = reconcile_reference_copies(store)

    assert dispositions == (
        ("resubmission", "stack_fused_epilogue_atomic"),
        ("msa-copy", "stack_msa_singleton"),
    )
    assert store.marked == list(dispositions)


def test_library_corpus_fingerprints_kernel_files() -> None:
    table = policy_module.validator_library_fingerprints()
    assert table, "cacheon_kernels produced no fingerprints"
    assert any(rel.endswith(".py") for rel in table.values())
    assert all(len(fp) == 64 for fp in table)


def _doctored(base, library_fp):
    return dataclasses.replace(
        base,
        exact_payload_digest="1" * 64,
        selected_delta_digest="2" * 64,
        normalized_delta_digest="3" * 64,
        containment_fingerprints=tuple(sorted({library_fp})),
    )


def test_submission_containing_library_code_is_demoted() -> None:
    base = fingerprint_submitted_delta(
        policy_module._repo_root() / "tests/fixtures/stack_msa_singleton"
    )
    table = policy_module.validator_library_fingerprints()
    library_fp, rel = next(iter(sorted(table.items())))
    doctored = _doctored(base, library_fp)

    assert policy_module.library_copy_match(doctored) == rel
    assert policy_module.library_copy_match(base) is None
    assert policy_module.library_copy_match(None) is None

    store = _Store([_Row("returns-our-code", "promoted", doctored)])
    dispositions = reconcile_reference_copies(store)
    marker = "library-" + rel.replace("/", "-")
    assert dispositions == (("returns-our-code", marker[:80]),)


def test_lease_fenced_mark_defers_without_killing_the_tick() -> None:
    from cacheon.chain.intake import IntakeError

    fixture = fingerprint_submitted_delta(
        policy_module._repo_root() / "tests/fixtures/stack_fused_epilogue_atomic"
    )

    class _FencedStore(_Store):
        def mark_reference_copy(self, reservation_id, reference_name):
            if reservation_id == "leased":
                raise IntakeError("active evaluation lease fences reservation mutation")
            super().mark_reference_copy(reservation_id, reference_name)

    store = _FencedStore(
        [
            _Row("leased", "promoted", fixture),
            _Row("free", "promoted", fixture),
        ]
    )
    dispositions = reconcile_reference_copies(store)
    assert dispositions == (("free", "stack_fused_epilogue_atomic"),)


def test_library_kill_flag_off_is_inert(monkeypatch) -> None:
    base = fingerprint_submitted_delta(
        policy_module._repo_root() / "tests/fixtures/stack_msa_singleton"
    )
    table = policy_module.validator_library_fingerprints()
    library_fp = next(iter(sorted(table)))
    doctored = _doctored(base, library_fp)
    monkeypatch.setattr(policy_module, "_LIBRARY_KILL_ENABLED", False)
    assert policy_module.library_copy_match(doctored) is None
    store = _Store([_Row("spared", "promoted", doctored)])
    assert reconcile_reference_copies(store) == ()
