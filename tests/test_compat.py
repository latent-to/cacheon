from __future__ import annotations

import sys

import pytest
from types import ModuleType

from cacheon.compat import PINNED_SGLANG, run_checks


def _version_check(monkeypatch, version: str, expected=PINNED_SGLANG):
    sglang = ModuleType("sglang")
    sglang.__version__ = version
    monkeypatch.setitem(sys.modules, "sglang", sglang)

    checks = run_checks(expected)

    return next(
        row for row in checks if row.name == f"sglang installed (pinned {expected})"
    )


@pytest.mark.parametrize("expected", (PINNED_SGLANG, "0.5.19"))
def test_compat_accepts_the_exact_arena_pin(monkeypatch, expected) -> None:
    row = _version_check(monkeypatch, expected, expected)

    assert row.ok
    assert row.detail == f"found {expected}"


def test_compat_rejects_an_installed_sglang_version_outside_the_pin(monkeypatch) -> None:
    version = "0.0.0.dev1+g56e290315"

    row = _version_check(monkeypatch, version)

    assert not row.ok
    assert row.detail == f"found {version}  <-- DIFFERS from pin"
