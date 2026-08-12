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
    yield
    policy_module._CACHE = None


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
