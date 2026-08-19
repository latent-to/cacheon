"""Operator-granted eval-cost admission credits.

One credit lets one reveal from the granted hotkey through the eval-cost gate
without an on-chain payment, exactly as if a payment had been consumed: the
intake store claims it inside the same admission transaction that admits the
row, it is spendable once, and it stays auditable through the reservation that
spent it. The DDL is shared between the store (which creates the table on
open) and the ops grant path here, which writes through its own connection
because the running intake controller holds the store's exclusive lock.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

EVAL_COST_CREDITS_DDL = """
CREATE TABLE IF NOT EXISTS eval_cost_credits (
    credit_id TEXT PRIMARY KEY,
    hotkey TEXT NOT NULL,
    coldkey TEXT NOT NULL DEFAULT '',
    amount_tao_rao INTEGER NOT NULL CHECK(amount_tao_rao>0),
    note TEXT NOT NULL DEFAULT '',
    granted_at TEXT NOT NULL,
    reservation_id TEXT NOT NULL DEFAULT '',
    spent_block INTEGER NOT NULL DEFAULT 0 CHECK(spent_block>=0)
) STRICT;
"""


@dataclass(frozen=True)
class EvalCostCredit:
    """One operator-granted fee waiver, spent by at most one reservation."""

    credit_id: str
    hotkey: str
    coldkey: str
    amount_tao_rao: int
    note: str
    granted_at: str
    reservation_id: str
    spent_block: int

    @property
    def spent(self) -> bool:
        return bool(self.reservation_id)


def _error(message: str) -> Exception:
    from cacheon.chain.intake import IntakeError

    return IntakeError(message)


def _require_credit_text(value: object, *, field: str, required: bool) -> str:
    if (
        not isinstance(value, str)
        or (required and not value)
        or value.strip() != value
        or len(value) > 256
        or any(char in value for char in "\x00\r\n")
    ):
        raise _error(f"eval-cost credit {field} is malformed")
    return value


def _credit_connection(path: str | Path) -> sqlite3.Connection:
    requested = Path(path).expanduser()
    if requested.is_symlink() or not requested.is_file():
        raise _error("intake database does not exist")
    db = sqlite3.connect(requested, isolation_level=None, timeout=30.0)
    db.row_factory = sqlite3.Row
    db.executescript(EVAL_COST_CREDITS_DDL)
    return db


def grant_eval_cost_credit(
    path: str | Path,
    *,
    hotkey: str,
    coldkey: str = "",
    amount_tao_rao: int = 1_000_000_000,
    note: str = "",
) -> str:
    """Record one artificial eval-cost credit and return its identifier.

    The next fee-gated reveal from ``hotkey`` that carries no payment pointer
    is admitted against this credit instead of failing
    ``missing_eval_cost_payment``. Safe to run while the intake controller is
    live: the credit is claimed inside the controller's own admission
    transaction, never here.
    """

    _require_credit_text(hotkey, field="hotkey", required=True)
    _require_credit_text(coldkey, field="coldkey", required=False)
    if type(amount_tao_rao) is not int or amount_tao_rao <= 0:
        raise _error("eval-cost credit amount must be a positive integer")
    if (
        not isinstance(note, str)
        or len(note) > 2_048
        or any(char in note for char in "\x00\r\n")
    ):
        raise _error("eval-cost credit note is malformed")
    from datetime import datetime, timezone

    credit_id = os.urandom(32).hex()
    granted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db = _credit_connection(path)
    try:
        db.execute(
            "INSERT INTO eval_cost_credits("
            "credit_id,hotkey,coldkey,amount_tao_rao,note,granted_at)"
            " VALUES(?,?,?,?,?,?)",
            (credit_id, hotkey, coldkey, amount_tao_rao, note, granted_at),
        )
    finally:
        db.close()
    return credit_id


def list_eval_cost_credits(
    path: str | Path, *, hotkey: str = ""
) -> tuple[EvalCostCredit, ...]:
    """Return recorded credits, oldest first, optionally for one hotkey."""

    _require_credit_text(hotkey, field="hotkey", required=False)
    db = _credit_connection(path)
    try:
        rows = db.execute(
            "SELECT credit_id,hotkey,coldkey,amount_tao_rao,note,granted_at,"
            "reservation_id,spent_block FROM eval_cost_credits"
            + (" WHERE hotkey=?" if hotkey else "")
            + " ORDER BY granted_at, credit_id",
            (hotkey,) if hotkey else (),
        ).fetchall()
    finally:
        db.close()
    return tuple(
        EvalCostCredit(
            credit_id=row["credit_id"],
            hotkey=row["hotkey"],
            coldkey=row["coldkey"],
            amount_tao_rao=row["amount_tao_rao"],
            note=row["note"],
            granted_at=row["granted_at"],
            reservation_id=row["reservation_id"],
            spent_block=row["spent_block"],
        )
        for row in rows
    )


__all__ = [
    "EVAL_COST_CREDITS_DDL",
    "EvalCostCredit",
    "grant_eval_cost_credit",
    "list_eval_cost_credits",
]
