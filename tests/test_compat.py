from __future__ import annotations

import sys
from types import ModuleType

from cacheon.compat import PINNED_SGLANG, run_checks


def _version_check(monkeypatch, version: str):
    sglang = ModuleType("sglang")
    sglang.__version__ = version
    monkeypatch.setitem(sys.modules, "sglang", sglang)

    checks = run_checks()

    return next(
        row for row in checks if row.name == f"sglang installed (pinned {PINNED_SGLANG})"
    )


def test_compat_accepts_the_exact_canonical_sglang_pin(monkeypatch) -> None:
    row = _version_check(monkeypatch, PINNED_SGLANG)

    assert row.ok
    assert row.detail == f"found {PINNED_SGLANG}"


def test_compat_rejects_an_installed_sglang_version_outside_the_pin(monkeypatch) -> None:
    version = "0.0.0.dev1+g56e290315"

    row = _version_check(monkeypatch, version)

    assert not row.ok
    assert row.detail == f"found {version}  <-- DIFFERS from pin"


def _forward_context_check(monkeypatch, accessor):
    sglang = ModuleType("sglang")
    sglang.__version__ = PINNED_SGLANG
    sglang.__path__ = []
    srt = ModuleType("sglang.srt")
    srt.__path__ = []
    model_executor = ModuleType("sglang.srt.model_executor")
    model_executor.__path__ = []
    forward_context = ModuleType("sglang.srt.model_executor.forward_context")
    forward_context.get_attn_backend = accessor
    sglang.srt = srt
    srt.model_executor = model_executor
    model_executor.forward_context = forward_context
    for name, module in (
        ("sglang", sglang),
        ("sglang.srt", srt),
        ("sglang.srt.model_executor", model_executor),
        ("sglang.srt.model_executor.forward_context", forward_context),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    checks = run_checks()
    return next(
        row
        for row in checks
        if row.name == "attention backend accessor: get_attn_backend"
    )


def test_compat_requires_callable_attention_backend_accessor(monkeypatch) -> None:
    assert _forward_context_check(monkeypatch, lambda: object()).ok
    assert not _forward_context_check(monkeypatch, None).ok
