"""Reserved durable schema for the extracted V2 economics stores.

The V2 finite-debt and incentive-composition economics logic was extracted
from the tree on 2026-08-09 (restorable from Git history).  Every production
intake database already carries the schema-4/5/6 metadata value and the V2
tables those migrations created, so the migrations and their exact DDL are
retained here verbatim.  Existing databases keep validating, and fresh
databases keep producing byte-identical schemas, without any V2 logic in the
tree.  Reintroducing V2 economics should move this schema layer back into its
owning store modules.
"""

from __future__ import annotations

import sqlite3


class FiniteDebtStoreError(RuntimeError):
    """Durable finite-debt authority is malformed, stale, or inconsistent."""


class IncentiveCompositionStoreError(RuntimeError):
    """Composition authority is malformed, stale, or inconsistent."""


class DebtPublicationError(RuntimeError):
    """A debt publication confirmation or boundary schedule is invalid."""


_FINITE_DEBT_SCHEMA_VERSION = 4
_COMPOSITION_SCHEMA_VERSION = 5
ACTIVE_INTAKE_SCHEMA_VERSION = "6"

_TABLE = "debt_weight_publication_confirmations"
_JOURNAL_TABLE = "debt_weight_publication_journal"
_COLUMNS = {
    "record_digest",
    "chain_scope_digest",
    "publication_kind",
    "policy_digest",
    "projection_digest",
    "weight_projection_digest",
    "effective_block",
    "effective_block_hash",
    "confirmed_block",
    "confirmed_block_hash",
    "record_json",
}
_JOURNAL_COLUMNS = {
    "sequence",
    "record_digest",
    "prior_record_digest",
    "binding_digest",
    "weight_projection_digest",
    "record_json",
    "binding_json",
}




# --- finite-debt schema (schema version 4) ---

_FINITE_DEBT_TABLE_DEFINITIONS = (
    """
    CREATE TABLE finite_debt_reward_events (
        sequence INTEGER PRIMARY KEY,
        event_digest TEXT NOT NULL UNIQUE,
        previous_event_digest TEXT NOT NULL,
        chain_scope_digest TEXT NOT NULL,
        event_type TEXT NOT NULL,
        block INTEGER NOT NULL CHECK(block>=0),
        block_hash TEXT NOT NULL,
        payload_json TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_policy_activations (
        activation_digest TEXT PRIMARY KEY,
        chain_scope_digest TEXT NOT NULL,
        policy_digest TEXT NOT NULL UNIQUE,
        policy_json TEXT NOT NULL,
        activation_block INTEGER NOT NULL CHECK(activation_block>=0),
        activation_block_hash TEXT NOT NULL,
        previous_policy_digest TEXT NOT NULL,
        seeded_clocks_json TEXT NOT NULL,
        activation_json TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_family_clocks (
        clock_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        policy_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(policy_digest),
        family_id TEXT NOT NULL,
        accepted_crown_block INTEGER NOT NULL CHECK(accepted_crown_block>=0),
        accepted_crown_block_hash TEXT NOT NULL,
        event_index INTEGER NOT NULL CHECK(event_index>=0),
        event_subindex INTEGER NOT NULL CHECK(event_subindex>=0),
        reservation_digest TEXT NOT NULL,
        source TEXT NOT NULL
            CHECK(source IN ('seed','crown','crown_no_debt','invalidation')),
        claim_digest TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL
            REFERENCES finite_debt_reward_events(event_digest),
        UNIQUE(policy_digest,family_id,accepted_crown_block,event_index,
               event_subindex,reservation_digest)
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_claims (
        claim_digest TEXT PRIMARY KEY,
        policy_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(policy_digest),
        family_id TEXT NOT NULL,
        candidate_digest TEXT NOT NULL UNIQUE,
        retained_evidence_digest TEXT NOT NULL,
        hotkey TEXT NOT NULL,
        accepted_crown_block INTEGER NOT NULL CHECK(accepted_crown_block>=0),
        accepted_crown_block_hash TEXT NOT NULL,
        event_index INTEGER NOT NULL CHECK(event_index>=0),
        event_subindex INTEGER NOT NULL CHECK(event_subindex>=0),
        reservation_digest TEXT NOT NULL,
        settlement_block INTEGER NOT NULL CHECK(settlement_block>=0),
        settlement_block_hash TEXT NOT NULL,
        settlement_event_digest TEXT NOT NULL UNIQUE,
        principal_units INTEGER NOT NULL CHECK(principal_units>0),
        claim_json TEXT NOT NULL,
        issuance_reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_claim_balances (
        claim_digest TEXT NOT NULL REFERENCES finite_debt_claims(claim_digest),
        revision INTEGER NOT NULL CHECK(revision>=0),
        balance_digest TEXT NOT NULL UNIQUE,
        principal_units INTEGER NOT NULL CHECK(principal_units>0),
        paid_units INTEGER NOT NULL CHECK(paid_units>=0),
        forfeited_units INTEGER NOT NULL CHECK(forfeited_units>=0),
        remaining_units INTEGER NOT NULL CHECK(remaining_units>=0),
        status TEXT NOT NULL CHECK(status IN ('open','paid','expired','cancelled')),
        terminal_block INTEGER,
        terminal_reason TEXT NOT NULL,
        balance_json TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL
            REFERENCES finite_debt_reward_events(event_digest),
        PRIMARY KEY(claim_digest,revision),
        CHECK(paid_units+forfeited_units+remaining_units=principal_units)
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_reward_epochs (
        epoch_digest TEXT PRIMARY KEY,
        chain_scope_digest TEXT NOT NULL,
        activation_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(activation_digest),
        policy_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(policy_digest),
        epoch_index INTEGER NOT NULL CHECK(epoch_index>0),
        start_block INTEGER NOT NULL CHECK(start_block>=0),
        effective_block INTEGER NOT NULL CHECK(effective_block>start_block),
        effective_block_hash TEXT NOT NULL,
        projection_digest TEXT NOT NULL UNIQUE,
        projection_json TEXT NOT NULL,
        publication_record_digest TEXT NOT NULL UNIQUE,
        payout_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest),
        payout_units INTEGER NOT NULL CHECK(payout_units>=0),
        reserve_units INTEGER NOT NULL CHECK(reserve_units>=0),
        epoch_json TEXT NOT NULL,
        UNIQUE(policy_digest,epoch_index),
        UNIQUE(policy_digest,effective_block),
        CHECK(payout_units+reserve_units=1000000)
    ) STRICT
    """,
    """
    CREATE TABLE finite_debt_epoch_allocations (
        epoch_digest TEXT NOT NULL
            REFERENCES finite_debt_reward_epochs(epoch_digest),
        claim_digest TEXT NOT NULL REFERENCES finite_debt_claims(claim_digest),
        hotkey TEXT NOT NULL,
        units INTEGER NOT NULL CHECK(units>0),
        PRIMARY KEY(epoch_digest,claim_digest)
    ) STRICT
    """,
)

_FINITE_DEBT_INDEX_DEFINITIONS = (
    "CREATE INDEX finite_debt_activations_block ON "
    "finite_debt_policy_activations(activation_block,activation_digest)",
    "CREATE INDEX finite_debt_clocks_latest ON "
    "finite_debt_family_clocks(policy_digest,family_id,accepted_crown_block DESC,"
    "event_index DESC,event_subindex DESC,reservation_digest DESC)",
    "CREATE INDEX finite_debt_balances_latest ON "
    "finite_debt_claim_balances(claim_digest,revision DESC)",
)

_FINITE_DEBT_IMMUTABLE_TABLES = (
    "finite_debt_reward_events",
    "finite_debt_policy_activations",
    "finite_debt_family_clocks",
    "finite_debt_claims",
    "finite_debt_claim_balances",
    "finite_debt_reward_epochs",
    "finite_debt_epoch_allocations",
)

_FINITE_DEBT_REQUIRED_COLUMNS = {
    "finite_debt_reward_events": {
        "sequence", "event_digest", "previous_event_digest", "chain_scope_digest",
        "event_type", "block", "block_hash", "payload_json",
    },
    "finite_debt_policy_activations": {
        "activation_digest", "chain_scope_digest", "policy_digest", "policy_json",
        "activation_block", "activation_block_hash", "previous_policy_digest",
        "seeded_clocks_json", "activation_json", "reward_event_digest",
    },
    "finite_debt_family_clocks": {
        "clock_sequence", "policy_digest", "family_id", "accepted_crown_block",
        "accepted_crown_block_hash", "event_index", "event_subindex",
        "reservation_digest", "source", "claim_digest", "reward_event_digest",
    },
    "finite_debt_claims": {
        "claim_digest", "policy_digest", "family_id", "candidate_digest",
        "retained_evidence_digest", "hotkey", "accepted_crown_block",
        "accepted_crown_block_hash", "event_index", "event_subindex",
        "reservation_digest", "settlement_block", "settlement_block_hash",
        "settlement_event_digest", "principal_units", "claim_json",
        "issuance_reward_event_digest",
    },
    "finite_debt_claim_balances": {
        "claim_digest", "revision", "balance_digest", "principal_units",
        "paid_units", "forfeited_units", "remaining_units", "status",
        "terminal_block", "terminal_reason", "balance_json", "reward_event_digest",
    },
    "finite_debt_reward_epochs": {
        "epoch_digest", "chain_scope_digest", "activation_digest", "policy_digest",
        "epoch_index", "start_block", "effective_block", "effective_block_hash",
        "projection_digest", "projection_json", "publication_record_digest",
        "payout_event_digest", "payout_units", "reserve_units", "epoch_json",
    },
    "finite_debt_epoch_allocations": {
        "epoch_digest", "claim_digest", "hotkey", "units",
    },
}


def _verify_finite_debt_schema(db: sqlite3.Connection) -> None:
    tables = {
        row["name"]: row
        for row in db.execute("PRAGMA table_list")
        if row["type"] == "table"
    }
    for table, columns in _FINITE_DEBT_REQUIRED_COLUMNS.items():
        row = tables.get(table)
        if row is None or row["strict"] != 1:
            raise FiniteDebtStoreError(f"{table} is missing or not STRICT")
        observed = {item["name"] for item in db.execute(f"PRAGMA table_info({table})")}
        if observed != columns:
            raise FiniteDebtStoreError(f"{table} columns differ from schema 4")
    triggers = {
        row["name"]
        for row in db.execute(
            "SELECT name FROM sqlite_schema WHERE type='trigger'"
        )
    }
    required_triggers = {
        f"{table}_reject_{action}"
        for table in _FINITE_DEBT_IMMUTABLE_TABLES
        for action in ("update", "delete")
    }
    if not required_triggers.issubset(triggers):
        raise FiniteDebtStoreError("finite-debt immutability triggers are incomplete")


def migrate_schema3_to4(db: sqlite3.Connection) -> None:
    """Create only empty finite-debt tables and advance metadata 3 -> 4."""

    schema = db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
    if schema is None:
        raise FiniteDebtStoreError("intake schema metadata is absent")
    # Schema 5 is an additive composition extension.  Reopening it must still
    # verify every schema-4 authority table before the schema-5 verifier runs.
    if schema["value"] in {str(_FINITE_DEBT_SCHEMA_VERSION), "5", "6", "7"}:
        _verify_finite_debt_schema(db)
        return
    if schema["value"] != "3":
        raise FiniteDebtStoreError("finite-debt migration requires intake schema 3")
    existing = {
        row["name"]
        for row in db.execute("PRAGMA table_list")
        if row["type"] == "table"
    }
    try:
        db.execute("BEGIN IMMEDIATE")
        for definition in _FINITE_DEBT_TABLE_DEFINITIONS:
            table = definition.split("CREATE TABLE ", 1)[1].split(" ", 1)[0]
            if table not in existing:
                db.execute(definition)
            elif db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
                raise FiniteDebtStoreError(
                    "schema-3 database contains non-authoritative finite-debt rows"
                )
        for definition in _FINITE_DEBT_INDEX_DEFINITIONS:
            db.execute(definition.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS "))
        for table in _FINITE_DEBT_IMMUTABLE_TABLES:
            for action in ("UPDATE", "DELETE"):
                name = f"{table}_reject_{action.lower()}"
                db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {name} BEFORE {action} ON {table} "
                    "BEGIN SELECT RAISE(ABORT,'finite-debt rows are immutable'); END"
                )
        _verify_finite_debt_schema(db)
        db.execute(
            "UPDATE metadata SET value=? WHERE key='schema' AND value='3'",
            (str(_FINITE_DEBT_SCHEMA_VERSION),),
        )
        if db.execute("SELECT changes() AS n").fetchone()["n"] != 1:
            raise FiniteDebtStoreError("intake schema changed during migration")
        db.execute("COMMIT")
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise


# --- incentive-composition schema (schema version 5) ---

_COMPOSITION_TABLE_DEFINITIONS = (
    """
    CREATE TABLE incentive_composition_activations (
        activation_digest TEXT PRIMARY KEY,
        chain_scope_digest TEXT NOT NULL,
        composition_policy_digest TEXT NOT NULL UNIQUE,
        core_activation_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(activation_digest),
        core_policy_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(policy_digest),
        selection_report_digest TEXT NOT NULL,
        policy_json TEXT NOT NULL,
        activation_block INTEGER NOT NULL CHECK(activation_block>=0),
        activation_block_hash TEXT NOT NULL,
        activation_json TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE incentive_discovery_wins (
        win_digest TEXT PRIMARY KEY,
        composition_policy_digest TEXT NOT NULL
            REFERENCES incentive_composition_activations(composition_policy_digest),
        candidate_digest TEXT NOT NULL UNIQUE,
        reservation_digest TEXT NOT NULL UNIQUE,
        proposal_digest TEXT NOT NULL UNIQUE,
        retained_evidence_digest TEXT NOT NULL UNIQUE,
        arm_digest TEXT NOT NULL UNIQUE,
        selected_delta_digest TEXT NOT NULL UNIQUE,
        candidate_tree_digest TEXT NOT NULL UNIQUE,
        hotkey TEXT NOT NULL,
        settlement_block INTEGER NOT NULL CHECK(settlement_block>=0),
        settlement_block_hash TEXT NOT NULL,
        settlement_event_digest TEXT NOT NULL UNIQUE,
        win_json TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE incentive_discovery_dispositions (
        disposition_digest TEXT PRIMARY KEY,
        composition_policy_digest TEXT NOT NULL
            REFERENCES incentive_composition_activations(composition_policy_digest),
        proposal_digest TEXT NOT NULL UNIQUE,
        review_digest TEXT NOT NULL UNIQUE,
        win_digest TEXT NOT NULL UNIQUE
            REFERENCES incentive_discovery_wins(win_digest),
        candidate_digest TEXT NOT NULL UNIQUE,
        retained_evidence_digest TEXT NOT NULL UNIQUE,
        hotkey TEXT NOT NULL,
        decision TEXT NOT NULL
            CHECK(decision='bounty_only'),
        disposition_json TEXT NOT NULL,
        win_block INTEGER NOT NULL CHECK(win_block>=0),
        authority_block INTEGER NOT NULL CHECK(authority_block>=0),
        authority_block_hash TEXT NOT NULL,
        claim_digest TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE incentive_discovery_claims (
        claim_digest TEXT PRIMARY KEY,
        composition_policy_digest TEXT NOT NULL
            REFERENCES incentive_composition_activations(composition_policy_digest),
        disposition_digest TEXT NOT NULL UNIQUE
            REFERENCES incentive_discovery_dispositions(disposition_digest),
        hotkey TEXT NOT NULL,
        awarded_block INTEGER NOT NULL CHECK(awarded_block>=0),
        expires_block INTEGER NOT NULL CHECK(expires_block>awarded_block),
        principal_units INTEGER NOT NULL CHECK(principal_units>0),
        claim_json TEXT NOT NULL,
        issuance_reward_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest)
    ) STRICT
    """,
    """
    CREATE TABLE incentive_discovery_balances (
        claim_digest TEXT NOT NULL
            REFERENCES incentive_discovery_claims(claim_digest),
        revision INTEGER NOT NULL CHECK(revision>=0),
        balance_digest TEXT NOT NULL UNIQUE,
        principal_units INTEGER NOT NULL CHECK(principal_units>0),
        paid_units INTEGER NOT NULL CHECK(paid_units>=0),
        forfeited_units INTEGER NOT NULL CHECK(forfeited_units>=0),
        remaining_units INTEGER NOT NULL CHECK(remaining_units>=0),
        status TEXT NOT NULL
            CHECK(status IN ('open','paid','expired','cancelled')),
        terminal_block INTEGER,
        terminal_reason TEXT NOT NULL,
        balance_json TEXT NOT NULL,
        reward_event_digest TEXT NOT NULL
            REFERENCES finite_debt_reward_events(event_digest),
        PRIMARY KEY(claim_digest,revision),
        CHECK(paid_units+forfeited_units+remaining_units=principal_units)
    ) STRICT
    """,
    """
    CREATE TABLE incentive_composed_epochs (
        epoch_digest TEXT PRIMARY KEY,
        chain_scope_digest TEXT NOT NULL,
        activation_digest TEXT NOT NULL
            REFERENCES incentive_composition_activations(activation_digest),
        composition_policy_digest TEXT NOT NULL
            REFERENCES incentive_composition_activations(composition_policy_digest),
        core_policy_digest TEXT NOT NULL
            REFERENCES finite_debt_policy_activations(policy_digest),
        epoch_index INTEGER NOT NULL CHECK(epoch_index>0),
        start_block INTEGER NOT NULL CHECK(start_block>=0),
        effective_block INTEGER NOT NULL CHECK(effective_block>start_block),
        effective_block_hash TEXT NOT NULL,
        projection_digest TEXT NOT NULL UNIQUE,
        projection_json TEXT NOT NULL,
        publication_record_digest TEXT NOT NULL UNIQUE,
        payout_event_digest TEXT NOT NULL UNIQUE
            REFERENCES finite_debt_reward_events(event_digest),
        discovery_payout_units INTEGER NOT NULL CHECK(discovery_payout_units>=0),
        core_payout_units INTEGER NOT NULL CHECK(core_payout_units>=0),
        reserve_units INTEGER NOT NULL CHECK(reserve_units>=0),
        epoch_json TEXT NOT NULL,
        UNIQUE(composition_policy_digest,epoch_index),
        UNIQUE(composition_policy_digest,effective_block),
        CHECK(discovery_payout_units+core_payout_units+reserve_units=1000000)
    ) STRICT
    """,
    """
    CREATE TABLE incentive_composed_allocations (
        epoch_digest TEXT NOT NULL
            REFERENCES incentive_composed_epochs(epoch_digest),
        reward_class TEXT NOT NULL CHECK(reward_class IN ('discovery','core')),
        claim_digest TEXT NOT NULL,
        hotkey TEXT NOT NULL,
        units INTEGER NOT NULL CHECK(units>0),
        PRIMARY KEY(epoch_digest,reward_class,claim_digest)
    ) STRICT
    """,
)

_COMPOSITION_INDEX_DEFINITIONS = (
    "CREATE INDEX incentive_composition_activation_block ON "
    "incentive_composition_activations(activation_block,activation_digest)",
    "CREATE INDEX incentive_discovery_balances_latest ON "
    "incentive_discovery_balances(claim_digest,revision DESC)",
)

_COMPOSITION_IMMUTABLE_TABLES = (
    "incentive_composition_activations",
    "incentive_discovery_wins",
    "incentive_discovery_dispositions",
    "incentive_discovery_claims",
    "incentive_discovery_balances",
    "incentive_composed_epochs",
    "incentive_composed_allocations",
)

_COMPOSITION_REQUIRED_COLUMNS = {
    "incentive_composition_activations": {
        "activation_digest", "chain_scope_digest", "composition_policy_digest",
        "core_activation_digest", "core_policy_digest", "selection_report_digest",
        "policy_json", "activation_block", "activation_block_hash",
        "activation_json", "reward_event_digest",
    },
    "incentive_discovery_wins": {
        "win_digest", "composition_policy_digest", "candidate_digest",
        "reservation_digest", "proposal_digest", "retained_evidence_digest",
        "arm_digest", "selected_delta_digest", "candidate_tree_digest", "hotkey",
        "settlement_block", "settlement_block_hash", "settlement_event_digest",
        "win_json", "reward_event_digest",
    },
    "incentive_discovery_dispositions": {
        "disposition_digest", "composition_policy_digest", "proposal_digest",
        "review_digest", "win_digest", "candidate_digest",
        "retained_evidence_digest", "hotkey", "decision", "disposition_json",
        "win_block", "authority_block", "authority_block_hash", "claim_digest",
        "reward_event_digest",
    },
    "incentive_discovery_claims": {
        "claim_digest", "composition_policy_digest", "disposition_digest", "hotkey",
        "awarded_block", "expires_block", "principal_units", "claim_json",
        "issuance_reward_event_digest",
    },
    "incentive_discovery_balances": {
        "claim_digest", "revision", "balance_digest", "principal_units", "paid_units",
        "forfeited_units", "remaining_units", "status", "terminal_block",
        "terminal_reason", "balance_json", "reward_event_digest",
    },
    "incentive_composed_epochs": {
        "epoch_digest", "chain_scope_digest", "activation_digest",
        "composition_policy_digest", "core_policy_digest", "epoch_index",
        "start_block", "effective_block", "effective_block_hash",
        "projection_digest", "projection_json", "publication_record_digest",
        "payout_event_digest", "discovery_payout_units", "core_payout_units",
        "reserve_units", "epoch_json",
    },
    "incentive_composed_allocations": {
        "epoch_digest", "reward_class", "claim_digest", "hotkey", "units",
    },
}


def _verify_composition_schema(db: sqlite3.Connection) -> None:
    tables = {
        row["name"]: row
        for row in db.execute("PRAGMA table_list")
        if row["type"] == "table"
    }
    for table, columns in _COMPOSITION_REQUIRED_COLUMNS.items():
        row = tables.get(table)
        if row is None or row["strict"] != 1:
            raise IncentiveCompositionStoreError(f"{table} is missing or not STRICT")
        observed = {item["name"] for item in db.execute(f"PRAGMA table_info({table})")}
        if observed != columns:
            raise IncentiveCompositionStoreError(
                f"{table} columns differ from schema 5"
            )
    triggers = {
        row["name"]
        for row in db.execute("SELECT name FROM sqlite_schema WHERE type='trigger'")
    }
    required = {
        f"{table}_reject_{action}"
        for table in _COMPOSITION_IMMUTABLE_TABLES
        for action in ("update", "delete")
    }
    if not required.issubset(triggers):
        raise IncentiveCompositionStoreError(
            "incentive-composition immutability triggers are incomplete"
        )


def migrate_schema4_to5(db: sqlite3.Connection) -> None:
    """Create only empty composition tables and advance metadata 4 -> 5."""

    schema = db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
    if schema is None:
        raise IncentiveCompositionStoreError("intake schema metadata is absent")
    if schema["value"] in {str(_COMPOSITION_SCHEMA_VERSION), "6", "7"}:
        _verify_composition_schema(db)
        return
    if schema["value"] != "4":
        raise IncentiveCompositionStoreError(
            "incentive-composition migration requires intake schema 4"
        )
    existing = {
        row["name"]
        for row in db.execute("PRAGMA table_list")
        if row["type"] == "table"
    }
    try:
        db.execute("BEGIN IMMEDIATE")
        for definition in _COMPOSITION_TABLE_DEFINITIONS:
            table = definition.split("CREATE TABLE ", 1)[1].split(" ", 1)[0]
            if table not in existing:
                db.execute(definition)
            elif db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None:
                raise IncentiveCompositionStoreError(
                    "schema-4 database contains non-authoritative composition rows"
                )
        for definition in _COMPOSITION_INDEX_DEFINITIONS:
            db.execute(definition.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS "))
        for table in _COMPOSITION_IMMUTABLE_TABLES:
            for action in ("UPDATE", "DELETE"):
                name = f"{table}_reject_{action.lower()}"
                db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {name} BEFORE {action} ON {table} "
                    "BEGIN SELECT RAISE(ABORT,'incentive composition rows are immutable'); END"
                )
        _verify_composition_schema(db)
        db.execute(
            "UPDATE metadata SET value=? WHERE key='schema' AND value='4'",
            (str(_COMPOSITION_SCHEMA_VERSION),),
        )
        if db.execute("SELECT changes() AS n").fetchone()["n"] != 1:
            raise IncentiveCompositionStoreError(
                "intake schema changed during composition migration"
            )
        db.execute("COMMIT")
    except Exception:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise


# --- debt-publication schema (schema version 6) ---

def ensure_debt_publication_schema(db: sqlite3.Connection) -> None:
    """Create or verify the additive immutable confirmation table."""

    if not isinstance(db, sqlite3.Connection):
        raise DebtPublicationError("debt publication schema requires SQLite")
    db.execute(
        "CREATE TABLE IF NOT EXISTS debt_weight_publication_confirmations("
        "record_digest TEXT PRIMARY KEY,chain_scope_digest TEXT NOT NULL,"
        "publication_kind TEXT NOT NULL,policy_digest TEXT NOT NULL,"
        "projection_digest TEXT NOT NULL UNIQUE,"
        "weight_projection_digest TEXT NOT NULL UNIQUE,"
        "effective_block INTEGER NOT NULL,"
        "effective_block_hash TEXT NOT NULL,confirmed_block INTEGER NOT NULL,"
        "confirmed_block_hash TEXT NOT NULL,record_json TEXT NOT NULL,"
        "UNIQUE(publication_kind,policy_digest,effective_block)) STRICT"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS debt_weight_publication_journal("
        "sequence INTEGER PRIMARY KEY,record_digest TEXT NOT NULL UNIQUE,"
        "prior_record_digest TEXT NOT NULL,binding_digest TEXT NOT NULL,"
        "weight_projection_digest TEXT NOT NULL,record_json TEXT NOT NULL,"
        "binding_json TEXT NOT NULL) STRICT"
    )
    for action in ("UPDATE", "DELETE"):
        db.execute(
            f"CREATE TRIGGER IF NOT EXISTS {_TABLE}_reject_{action.lower()} "
            f"BEFORE {action} ON {_TABLE} BEGIN SELECT "
            "RAISE(ABORT,'debt publication confirmations are immutable'); END"
        )
        db.execute(
            f"CREATE TRIGGER IF NOT EXISTS {_JOURNAL_TABLE}_reject_{action.lower()} "
            f"BEFORE {action} ON {_JOURNAL_TABLE} BEGIN SELECT "
            "RAISE(ABORT,'debt publication journal rows are immutable'); END"
        )
    row = db.execute(
        "SELECT strict FROM pragma_table_list WHERE name=?", (_TABLE,)
    ).fetchone()
    columns = {item["name"] for item in db.execute(f"PRAGMA table_info({_TABLE})")}
    if row is None or row["strict"] != 1 or columns != _COLUMNS:
        raise DebtPublicationError("debt publication confirmation schema differs")
    journal_row = db.execute(
        "SELECT strict FROM pragma_table_list WHERE name=?", (_JOURNAL_TABLE,)
    ).fetchone()
    journal_columns = {
        item["name"] for item in db.execute(f"PRAGMA table_info({_JOURNAL_TABLE})")
    }
    if (
        journal_row is None
        or journal_row["strict"] != 1
        or journal_columns != _JOURNAL_COLUMNS
    ):
        raise DebtPublicationError("debt publication journal schema differs")
