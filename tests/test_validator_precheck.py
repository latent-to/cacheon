"""Unit tests for validator.precheck — composition of fetch + sandbox."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from validator.challengers import PrecheckOutcome, PrecheckResult
from validator.precheck import make_fetch_precheck
from validator.policy_fetch import FetchOutcome, FetchResult

pytestmark = pytest.mark.unit


def _make_commit(repo: str = "owner/name", revision: str = "a" * 40):
    com = MagicMock()
    com.repo = repo
    com.revision = revision
    return com


class TestMakeFetchPrecheck:
    def test_fetch_ok_sandbox_pass(self, tmp_path: Path):
        policy_path = tmp_path / "policy.py"
        policy_path.write_text("pass")

        def fetch_fn(repo, revision):
            return FetchResult(outcome=FetchOutcome.OK, path=policy_path)

        def sandbox_check(source):
            return MagicMock(ok=True)

        precheck = make_fetch_precheck(fetch_fn, sandbox_check)
        result = precheck(_make_commit())

        assert result.outcome is PrecheckOutcome.OK

    def test_fetch_ok_sandbox_fail(self, tmp_path: Path):
        policy_path = tmp_path / "policy.py"
        policy_path.write_text("import os")

        def fetch_fn(repo, revision):
            return FetchResult(outcome=FetchOutcome.OK, path=policy_path)

        def sandbox_check(source):
            return MagicMock(ok=False, reason="blocked import: os")

        precheck = make_fetch_precheck(fetch_fn, sandbox_check)
        result = precheck(_make_commit())

        assert result.outcome is PrecheckOutcome.REJECTED
        assert result.reason.startswith("ast_blocked:")
        assert "blocked import: os" in result.reason

    def test_fetch_rejected_propagates(self):
        def fetch_fn(repo, revision):
            return FetchResult(
                outcome=FetchOutcome.REJECTED,
                reason="revision_unavailable (foo)",
            )

        precheck = make_fetch_precheck(fetch_fn)
        result = precheck(_make_commit())

        assert result.outcome is PrecheckOutcome.REJECTED
        assert "revision_unavailable" in result.reason

    def test_fetch_deferred_propagates(self):
        def fetch_fn(repo, revision):
            return FetchResult(
                outcome=FetchOutcome.DEFERRED,
                reason="hf_http_503",
            )

        precheck = make_fetch_precheck(fetch_fn)
        result = precheck(_make_commit())

        assert result.outcome is PrecheckOutcome.DEFERRED
        assert "hf_http_503" in result.reason

    def test_fetch_ok_missing_path(self):
        def fetch_fn(repo, revision):
            return FetchResult(outcome=FetchOutcome.OK, path=None)

        precheck = make_fetch_precheck(fetch_fn)
        result = precheck(_make_commit())

        assert result.outcome is PrecheckOutcome.REJECTED
        assert "fetch returned OK without path" in result.reason

    def test_non_utf8_policy_rejected(self, tmp_path: Path):
        policy_path = tmp_path / "policy.py"
        policy_path.write_bytes(b"\xff\xfe\x00\x00")  # UTF-32 LE BOM

        def fetch_fn(repo, revision):
            return FetchResult(outcome=FetchOutcome.OK, path=policy_path)

        precheck = make_fetch_precheck(fetch_fn)
        result = precheck(_make_commit())

        assert result.outcome is PrecheckOutcome.REJECTED
        assert "unreadable" in result.reason
        assert "UnicodeDecodeError" in result.reason
