"""Every reason mainnet has written to a miner-visible row carries a stated cause and next step.

``cacheon/chain/miner_feedback.py`` cites this table as its check. The rendered miner report,
over a live deployment's stores, is proven in ``tests/test_validator_operator.py``.
"""

from __future__ import annotations

import pytest

from cacheon.chain.miner_feedback import _guidance

# Every reason mainnet has actually written to a miner-visible row, read from
# the live intake database on 2026-08-23. This is the ground-truth floor: these
# have already been shown to real miners, so a regression that drops one is a
# regression against people, not against a hypothetical.
_OBSERVED_IN_PRODUCTION = (
    "speed_regression",
    "qualified",
    "graph_member_not_applicable",
    "graph_eager_failed",
    "speed_threshold_not_met",
    "candidate_slower",
    "candidate_kernel_does_not_compile",
    "finalized_block_sla_expired",
    "screen_rejected",
    "missing_eval_cost_payment",
    "screen_receipt_service_rotated",
    "screen_promoted",
    "copy_of:2aa6cf9a9b38f59e5ff55ca8383cb31f2aa34578cf852098d3824737b5c2cb23",
    "copy_of:validator_reference:library-collective-fused_ar_rmsnorm.py",
    "fetch:archive member is excluded from bundle identity: source/kernels/._moe.py",
    "manifest:unsupported capability field init_blocks",
    "systemic_release_cap:3",
    # Shown bare to miners on 2026-09-16, 09-17 and 09-18, with no explanation.
    "slot_audit_failed",
    "audit_not_covered",
)


@pytest.mark.parametrize("code", _OBSERVED_IN_PRODUCTION)
def test_reason_seen_in_production_is_explained(code: str) -> None:
    guidance = _guidance(code)
    assert guidance is not None, f"{code} has been shown to a miner with no explanation"
    assert guidance["cause"].strip()
    assert guidance["next_step"].strip()
