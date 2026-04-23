"""Compose HF policy fetching + AST sandbox into a ``PrecheckFn``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from inference_engine.sandbox import CheckResult, check as sandbox_check

from .challengers import PrecheckFn, PrecheckOutcome, PrecheckResult
from .policy_fetch import FetchOutcome, FetchResult


def make_fetch_precheck(
    fetch_fn: Callable[[str, str], FetchResult],
    sandbox_check_fn: Callable[[str], CheckResult] = sandbox_check,
) -> PrecheckFn:
    """Build a ``PrecheckFn`` that fetches source and runs the AST sandbox.

    Args:
        fetch_fn: callable ``(repo, revision) -> FetchResult``.
        sandbox_check_fn: callable ``(source) -> CheckResult``. Injected
            for tests; production uses ``inference_engine.sandbox.check``.

    Returns:
        A ``PrecheckFn`` compatible with ``validator.challengers.select_challengers``.
    """

    def precheck(com) -> PrecheckResult:
        fetch_result = fetch_fn(com.repo, com.revision)

        if fetch_result.outcome is FetchOutcome.REJECTED:
            return PrecheckResult(
                outcome=PrecheckOutcome.REJECTED,
                reason=fetch_result.reason or "fetch rejected",
            )

        if fetch_result.outcome is FetchOutcome.DEFERRED:
            return PrecheckResult(
                outcome=PrecheckOutcome.DEFERRED,
                reason=fetch_result.reason or "fetch deferred",
            )

        # FetchOutcome.OK — run the sandbox
        path = fetch_result.path
        if path is None:
            return PrecheckResult(
                outcome=PrecheckOutcome.REJECTED,
                reason="ast_blocked: fetch returned OK without path",
            )

        try:
            source = Path(path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            return PrecheckResult(
                outcome=PrecheckOutcome.REJECTED,
                reason=f"ast_blocked: policy.py unreadable ({type(exc).__name__})",
            )

        check_result = sandbox_check_fn(source)

        if not check_result.ok:
            violation = check_result.reason or "sandbox check failed"
            return PrecheckResult(
                outcome=PrecheckOutcome.REJECTED,
                reason=f"ast_blocked: {violation}",
            )

        return PrecheckResult(outcome=PrecheckOutcome.OK)

    return precheck
