"""Commit-reveal + king-of-the-hill scoring — the anti-copy mechanism.

The problem in any open competition where submissions are evaluated in the open:
a lazy miner copies the current leader's submission (it's just code shipped to
the validator) and resubmits it, splitting reward for no work. Two mechanisms
defeat that here:

1. **Commit-reveal.** A miner first posts a *commitment* — a hash of
   ``(content_hash, hotkey, salt)`` — during the commit window, before any bundle
   is revealed. Later, in the reveal window, they post ``(content_hash, salt)``.
   A reveal is only accepted if it matches a commitment that *that hotkey* posted
   earlier. So you cannot reveal a bundle you didn't already commit to — and you
   couldn't have committed to a competitor's bundle you hadn't seen yet. Copying
   at reveal time is therefore impossible; the copier has no matching commitment.
   If two miners independently committed to the *same* content, the earliest
   commitment (lowest sequence) is the original; later identical ones are copies
   and earn nothing.

2. **Improvement-over-best (king of the hill).** A standing *champion* (the best
   validated bundle so far) holds the title and the emission. A challenger only
   takes the title if its score beats the champion's by a margin (which absorbs
   measurement noise). A copy ties the champion — it never clears the margin — so
   it earns zero. The only way to earn is to genuinely beat the best.

This module is pure-Python and persists to a JSON ledger so it can be tested and
reasoned about without a GPU. In a real Bittensor subnet the commitments live
on-chain, the bundles are fetched from a content-addressed store, and ``hotkey``
is the miner's SS58 address; the semantics here are the same.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("optima.ledger")

# Bump when the on-disk ledger format changes in a way older code cannot read.
SCHEMA_VERSION = 10
MAX_LEDGER_BYTES = 256 * 1024 * 1024

RETRY_KIND_NO_DECISION = "no_decision"
RETRY_KIND_INFRASTRUCTURE = "infrastructure"
RETRY_KINDS = frozenset({RETRY_KIND_NO_DECISION, RETRY_KIND_INFRASTRUCTURE})
RETRY_STATE_AUTOMATIC = "automatic"
RETRY_STATE_HELD = "held"
RETRY_STATE_IN_PROGRESS = "in_progress"
RETRY_STATES = frozenset({
    RETRY_STATE_AUTOMATIC,
    RETRY_STATE_HELD,
    RETRY_STATE_IN_PROGRESS,
})


def make_commitment(content_hash: str, hotkey: str, salt: str) -> str:
    """The value a miner posts in the commit window."""
    return hashlib.sha256(f"{content_hash}:{hotkey}:{salt}".encode("utf-8")).hexdigest()


def make_chain_scope(*, genesis_hash: str, netuid: int,
                     scheme: str = "genesis-netuid-v1") -> str:
    """Immutable ledger namespace for one concrete chain and subnet."""
    if not scheme or not genesis_hash or type(netuid) is not int or netuid < 0:
        raise ValueError("chain scope requires scheme, genesis hash, and non-negative netuid")
    material = json.dumps(
        {"scheme": scheme, "genesis_hash": genesis_hash, "netuid": netuid},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return f"{scheme}:sha256:{hashlib.sha256(material).hexdigest()}"


def _atomic_write_json(path: Path, data: object) -> None:
    """Write JSON durably: serialize to a sibling temp file, then atomically rename
    it over the target. A crash mid-write leaves the previous file intact — never a
    truncated half-file. ``os.replace`` is atomic on a single filesystem."""
    path = Path(path)
    payload = json.dumps(data, indent=2, allow_nan=False).encode("utf-8")
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.{os.getpid()}.", dir=path.parent
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _read_ledger_json(path: Path) -> object:
    """Read one bounded, stable, owner-controlled ledger without following links."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_size < 0
            or before.st_size > MAX_LEDGER_BYTES
        ):
            raise ValueError(
                "ledger must be one bounded owner-controlled regular file"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("ledger was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise ValueError("ledger grew beyond its inspected size")
        after = os.fstat(fd)
        stable = (
            "st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_size",
            "st_mtime_ns", "st_ctime_ns",
        )
        if any(getattr(before, key) != getattr(after, key) for key in stable):
            raise ValueError("ledger changed while being read")
    finally:
        os.close(fd)

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate ledger JSON key {key!r}")
            value[key] = item
        return value

    return json.loads(
        b"".join(chunks),
        object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid ledger JSON constant {value}")
        ),
    )


def _only_fields(cls: type, d: dict) -> dict:
    """Keep only keys that name a field of ``cls``. Unknown keys (written by a newer
    schema) are dropped, and missing keys fall back to the dataclass defaults — so a
    record can gain optional fields without breaking older or newer ledger files."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in d.items() if k in names}


@dataclass
class Commitment:
    hotkey: str
    commitment: str
    round_id: int
    seq: int  # monotonic; commit order = anti-copy priority


@dataclass
class Reveal:
    hotkey: str
    content_hash: str
    salt: str
    round_id: int
    commit_seq: int
    original: bool = True
    fingerprint: str = ""  # reformat-invariant near-copy fingerprint (auto-demotes a match)
    structural_fingerprint: str = ""  # rename/constant-tweak skeleton — ADVISORY only (review)
    # Per-slot reformat-invariant fingerprints (slot -> hash). The LOAD-BEARING copy
    # compare: matching ANY single slot demotes, so padding a stolen bundle with an
    # extra unrelated op cannot perturb the whole-bundle ``fingerprint`` into freshness.
    slot_fingerprints: dict[str, str] = field(default_factory=dict)
    # Per-slot PATH-INDEPENDENT fingerprints of each substantial closure file
    # (slot -> sorted hashes). Catches relocation/padding WITHIN a slot: the ledger
    # demotes on set CONTAINMENT (all of a prior reveal's files appear here), which a
    # stolen body moved into an imported module cannot evade and a merely-shared
    # vendored utility cannot trigger.
    slot_file_fingerprints: dict[str, list[str]] = field(default_factory=dict)
    # Non-component products (for example a bounded SGLang system patch) use a
    # validator-owned competition target rather than inventing a fake slot.
    product_fingerprints: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Score:
    hotkey: str
    content_hash: str
    round_id: int
    score: float
    kl_mean: float
    passed: bool
    sglang_version: str = ""  # the pin this speedup was measured against (re-baseline key)
    # ``slot`` is retained for reading/writing pre-target ledgers.  New producers
    # populate the three fields below from validator-owned competition resolution;
    # atomic submissions deliberately leave ``slot`` empty so they can never mint
    # member-slot champions.
    slot: str = ""
    target: str = ""
    mode: str = ""
    member_slots: tuple[str, ...] = ()
    # Immutable scoring namespace. Empty means a pre-arena legacy row and is never
    # eligible for a registered-arena championship.
    arena_name: str = ""
    arena_fingerprint: str = ""
    arena_bracket: str = ""
    regime: str = ""
    validator_image: str = ""
    referee_source_digest: str = ""
    referee_tree_digest: str = ""
    model_revision: str = ""
    model_manifest_digest: str = ""
    model_content_digest: str = ""
    # Content-addressed, validator-owned evidence for the exact stock runtime
    # preflight and the pre/post device-state receipts around every B/C/B' arm.
    host_attestation_sha256: str = ""
    prompt_seed: int = 0
    prompt_engine_version: str = ""
    prompt_seed_scheme: str = ""
    seed_round_id: int = 0
    seed_block: int = 0
    seed_block_hash: str = ""
    quality_evidence: str = ""
    chain_scope: str = ""
    validator_hotkey: str = ""
    evaluation_id: str = ""
    miner_hotkey: str = ""
    settlement_round_id: int = 0
    evaluation_block: int = 0
    passed_quality: bool = False
    passed_timed_quality: bool = False
    passed_warmup_quality: bool = False
    passed_speedup: bool = False
    confident: bool = False
    crownable: bool = False
    qualification_evidence_sha256: str = ""


@dataclass
class Champion:
    content_hash: str
    hotkey: str
    score: float
    round_id: int
    sglang_version: str = ""  # the pin the title was won under; a different current pin = stale
    target: str = ""
    mode: str = ""
    member_slots: tuple[str, ...] = ()
    arena_name: str = ""
    arena_fingerprint: str = ""
    arena_bracket: str = ""
    regime: str = ""
    validator_image: str = ""
    referee_source_digest: str = ""
    referee_tree_digest: str = ""
    model_revision: str = ""
    model_manifest_digest: str = ""
    model_content_digest: str = ""
    host_attestation_sha256: str = ""
    prompt_seed: int = 0
    prompt_engine_version: str = ""
    prompt_seed_scheme: str = ""
    seed_round_id: int = 0
    seed_block: int = 0
    seed_block_hash: str = ""
    quality_evidence: str = ""
    chain_scope: str = ""
    validator_hotkey: str = ""
    evaluation_id: str = ""
    miner_hotkey: str = ""
    settlement_round_id: int = 0
    evaluation_block: int = 0
    passed_quality: bool = False
    passed_timed_quality: bool = False
    passed_warmup_quality: bool = False
    passed_speedup: bool = False
    confident: bool = False
    crownable: bool = False
    qualification_evidence_sha256: str = ""


@dataclass(frozen=True)
class EvalRecord:
    """The typed result of evaluating one bundle — the audit row and the dedup key.
    ``score`` / ``passed`` / ``mean_kl`` mirror the king-of-the-hill ``Score`` atom; the
    rest is the fidelity detail an eval actually produces. Keyed in the ledger by
    ``(hotkey, bundle_hash)`` so an already-scored submission is never re-run. Add a
    field when a producer needs it — ``schema_version`` + the tolerant load make that safe.
    """
    hotkey: str
    bundle_hash: str
    slot: str
    round_id: int
    score: float
    passed: bool
    throughput: float = 0.0
    mean_kl: float = 0.0
    gsm8k_acc: float = -1.0  # -1 = not measured
    dq_reason: str = ""
    # Additive settlement identity. ``slot`` remains the legacy singleton field;
    # load/record normalization maps old rows to target=slot, mode=slot.
    target: str = ""
    mode: str = ""
    member_slots: tuple[str, ...] = ()
    arena_name: str = ""
    arena_fingerprint: str = ""
    arena_bracket: str = ""
    regime: str = ""
    sglang_version: str = ""
    validator_image: str = ""
    referee_source_digest: str = ""
    referee_tree_digest: str = ""
    model_revision: str = ""
    model_manifest_digest: str = ""
    model_content_digest: str = ""
    host_attestation_sha256: str = ""
    prompt_seed: int = 0
    prompt_engine_version: str = ""
    prompt_seed_scheme: str = ""
    seed_round_id: int = 0
    seed_block: int = 0
    seed_block_hash: str = ""
    quality_evidence: str = ""
    chain_scope: str = ""
    validator_hotkey: str = ""
    evaluation_id: str = ""
    miner_hotkey: str = ""
    settlement_round_id: int = 0
    evaluation_block: int = 0
    passed_quality: bool = False
    passed_timed_quality: bool = False
    passed_warmup_quality: bool = False
    passed_speedup: bool = False
    confident: bool = False
    crownable: bool = False
    qualification_evidence_sha256: str = ""
    # Development/verify command rows are useful audit receipts but must never
    # suppress a later validator-owned OCI qualification of the same bundle.
    development_only: bool = False


@dataclass(frozen=True)
class RetryRecord:
    """Persistent non-terminal evaluation state.

    Legacy rows default to an automatic no-decision so a schema upgrade cannot
    silently disqualify a miner because the validator previously failed to record
    why a retry was needed.
    """

    hotkey: str
    bundle_hash: str
    arena_bracket: str
    chain_scope: str
    attempts: int
    next_block: int
    last_reason: str
    kind: str = RETRY_KIND_NO_DECISION
    state: str = RETRY_STATE_AUTOMATIC
    no_decision_attempts: int = 0
    infrastructure_attempts: int = 0
    lease_id: str = ""
    lease_block: int = 0


@dataclass(frozen=True)
class ValidatorFaultHold:
    """Durable circuit breaker for an ambiguous/controller-owned evaluation fault.

    This state is deliberately separate from miner retry accounting. The active GPU
    lease is rolled back to its previously completed retry counters and no automatic
    pass may replay the work until a trusted operator explicitly releases this hold.
    """

    hotkey: str
    bundle_hash: str
    arena_bracket: str
    chain_scope: str
    evaluation_id: str
    created_block: int
    reason: str


@dataclass(frozen=True)
class PendingSettlement:
    """One authoritative score/eval pair awaiting durable settlement.

    The evidence digest makes this a disposition for an exact qualification,
    rather than a broad invitation to reconsider every historical score from the
    same round. Recovery processes rows one at a time in the ledger's canonical
    chain-reveal order. This is important: whether two reveals happen to be seen
    in one validator pass or two must not change which challenger faces the
    incumbent's dethrone margin.
    """

    hotkey: str
    content_hash: str
    round_id: int
    target: str
    arena_bracket: str
    evidence_sha256: str
    chain_scope: str


def _retry_from_raw(raw: dict) -> RetryRecord:
    """Load retry rows while conservatively accounting for pre-lease schemas."""
    values = _only_fields(RetryRecord, raw)
    if "no_decision_attempts" not in raw or "infrastructure_attempts" not in raw:
        attempts = values.get("attempts", 0)
        state = values.get("state", RETRY_STATE_AUTOMATIC)
        completed = (
            attempts - 1 if state == RETRY_STATE_IN_PROGRESS else attempts
        )
        completed = max(0, completed) if type(completed) is int else completed
        if values.get("kind", RETRY_KIND_NO_DECISION) == RETRY_KIND_INFRASTRUCTURE:
            values["infrastructure_attempts"] = completed
            values["no_decision_attempts"] = 0
        else:
            values["no_decision_attempts"] = completed
            values["infrastructure_attempts"] = 0
    return RetryRecord(**values)


def _validate_retry_limits(
    *,
    max_automatic_infrastructure_attempts: int,
    max_automatic_no_decision_attempts: int,
    max_total_attempts: int,
) -> None:
    values = (
        max_automatic_infrastructure_attempts,
        max_automatic_no_decision_attempts,
        max_total_attempts,
    )
    if (
        any(type(value) is not int or value <= 0 for value in values)
        or max_total_attempts
        < max(
            max_automatic_infrastructure_attempts,
            max_automatic_no_decision_attempts,
        )
    ):
        raise ValueError("invalid retry attempt budgets")


def _retry_limit_reached(
    retry: RetryRecord,
    *,
    max_automatic_infrastructure_attempts: int,
    max_automatic_no_decision_attempts: int,
    max_total_attempts: int,
) -> bool:
    return bool(
        retry.infrastructure_attempts
        >= max_automatic_infrastructure_attempts
        or retry.no_decision_attempts
        >= max_automatic_no_decision_attempts
        or retry.attempts >= max_total_attempts
    )


@dataclass
class SettleResult:
    champion: Optional[Champion]
    weights: dict[str, float]
    title_changed: bool
    challenger_score: float
    rejected_copies: list[str] = field(default_factory=list)  # hotkeys
    champion_stale: bool = False  # champion was crowned under a different sglang pin -> re-baseline


@dataclass
class PerTargetSettleResult:
    """One champion per validator-owned competition target.

    A singleton slot is a target with one member.  An atomic multi-slot bundle is
    one target with several members and therefore receives exactly one title and
    one emission share—never one title per member.
    """

    champions: dict[str, Champion]  # target -> champion
    weights: dict[str, float]  # hotkey -> emission share across targets with a champion
    title_changes: dict[str, bool]  # target -> did the title change this round
    stale_targets: list[str] = field(default_factory=list)
    rejected_copies: list[str] = field(default_factory=list)
    arena_bracket: str = ""  # empty only for the quarantined legacy namespace

    @property
    def stale_slots(self) -> list[str]:
        """Compatibility spelling for callers written before atomic targets."""
        return self.stale_targets


# Public compatibility alias.  The behavior now settles target IDs; for legacy
# singleton ledgers every target ID is identical to its historical slot ID.
PerSlotSettleResult = PerTargetSettleResult


def _normalize_competition_identity(
    *,
    slot: str = "",
    target: str = "",
    mode: str = "",
    member_slots: tuple[str, ...] | list[str] = (),
    allow_empty: bool = True,
) -> tuple[str, str, tuple[str, ...], str]:
    """Return canonical ``(target, mode, members, legacy_slot)``.

    This is deliberately ledger-format normalization, not competition policy.
    Policy is resolved from the fetched manifest before a row reaches the ledger.
    Its one inference exists solely so old JSON rows remain readable:
    ``slot=s`` becomes ``target=s, mode=slot, members=(s,)``.
    """

    slot = str(slot or "")
    target = str(target or "")
    mode = str(mode or "")
    members = tuple(str(member) for member in (member_slots or ()))

    if not target and slot:
        return slot, "slot", (slot,), slot
    if not target:
        if allow_empty and not mode and not members:
            return "", "", (), ""
        raise ValueError("competition identity requires a non-empty target")

    if mode == "slot":
        members = members or (target,)
        if members != (target,):
            raise ValueError(
                "slot competition identity requires member_slots == (target,)"
            )
        if slot and slot != target:
            raise ValueError("legacy slot disagrees with slot competition target")
        return target, mode, members, target

    if mode == "atomic":
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError(
                "atomic competition identity requires at least two unique members"
            )
        # Never retain a member-shaped legacy key for an atomic score.
        return target, mode, members, ""

    if mode == "system":
        if slot or members:
            raise ValueError(
                "system competition identity must not contain component slots"
            )
        return target, mode, (), ""

    raise ValueError("competition identity mode must be 'slot', 'atomic', or 'system'")


def _arena_identity(arena) -> dict[str, str]:
    """Extract and validate a registered immutable arena scope."""
    if arena is None:
        return {
            "arena_name": "",
            "arena_fingerprint": "",
            "arena_bracket": "",
            "regime": "",
            "validator_image": "",
            "referee_source_digest": "",
            "referee_tree_digest": "",
            "model_revision": "",
            "model_manifest_digest": "",
            "model_content_digest": "",
        }
    from optima.arenas import get_arena

    registered = get_arena(str(arena.name))
    if registered.fingerprint != arena.fingerprint:
        raise ValueError(
            f"arena {arena.name!r} does not match the registered profile"
        )
    return {
        "arena_name": registered.name,
        "arena_fingerprint": registered.fingerprint,
        "arena_bracket": registered.bracket,
        "regime": registered.workload.regime,
        "validator_image": registered.validator_image,
        "referee_source_digest": registered.referee_source_digest,
        "referee_tree_digest": registered.referee_tree_digest,
        "model_revision": registered.model_revision,
        "model_manifest_digest": registered.model_manifest_digest,
        "model_content_digest": registered.model_content_digest,
    }


def _record_matches_arena(record, arena_fields: dict[str, str]) -> bool:
    return all(getattr(record, key, "") == value for key, value in arena_fields.items())


def _crown_evidence_tuple(record) -> tuple:
    """Canonical evidence shared by Score, EvalRecord, and Champion.

    Content/bundle key, hotkey, round and the EvalRecord ``passed`` bit are checked
    separately at their differently named fields. Everything else that can alter
    competition identity, runtime scope, prompt provenance, host evidence, or the
    quality decision must agree byte-for-byte across all three authorities.
    """

    return (
        getattr(record, "target", ""),
        getattr(record, "mode", ""),
        tuple(getattr(record, "member_slots", ())),
        getattr(record, "score", None),
        getattr(record, "sglang_version", ""),
        getattr(record, "arena_name", ""),
        getattr(record, "arena_fingerprint", ""),
        getattr(record, "arena_bracket", ""),
        getattr(record, "regime", ""),
        getattr(record, "validator_image", ""),
        getattr(record, "referee_source_digest", ""),
        getattr(record, "referee_tree_digest", ""),
        getattr(record, "model_revision", ""),
        getattr(record, "model_manifest_digest", ""),
        getattr(record, "model_content_digest", ""),
        getattr(record, "host_attestation_sha256", ""),
        getattr(record, "prompt_seed", None),
        getattr(record, "prompt_engine_version", ""),
        getattr(record, "prompt_seed_scheme", ""),
        getattr(record, "seed_round_id", None),
        getattr(record, "seed_block", None),
        getattr(record, "seed_block_hash", ""),
        getattr(record, "quality_evidence", ""),
        getattr(record, "chain_scope", ""),
        getattr(record, "validator_hotkey", ""),
        getattr(record, "evaluation_id", ""),
        getattr(record, "miner_hotkey", ""),
        getattr(record, "settlement_round_id", None),
        getattr(record, "evaluation_block", None),
        getattr(record, "passed_quality", None),
        getattr(record, "passed_timed_quality", None),
        getattr(record, "passed_warmup_quality", None),
        getattr(record, "passed_speedup", None),
        getattr(record, "confident", None),
        getattr(record, "crownable", None),
        getattr(record, "qualification_evidence_sha256", ""),
    )


def _score_evidence_sha256(score: Score) -> str:
    """Content address one exact score disposition.

    This deliberately includes the row identity outside ``_crown_evidence_tuple``:
    two miners or two rounds with otherwise identical measurements are distinct
    settlement work, while JSON canonicalization keeps the digest stable across a
    ledger save/load (tuples round-trip as arrays and normalize back to tuples).
    """

    payload = json.dumps(
        {
            "hotkey": score.hotkey,
            "content_hash": score.content_hash,
            "round_id": score.round_id,
            "passed": score.passed,
            "evidence": _crown_evidence_tuple(score),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _host_attestation_context(
    champion: Champion, arena
) -> dict[str, object]:
    """Exact context the retained host sidecar must independently bind."""

    from optima.eval.host_attestation import host_attestation_context

    return host_attestation_context(
        arena,
        bundle_hash=champion.content_hash,
        prompt_seed=champion.prompt_seed,
        seed_round_id=champion.seed_round_id,
        seed_block=champion.seed_block,
        seed_block_hash=champion.seed_block_hash,
        chain_scope=champion.chain_scope,
        validator_hotkey=champion.validator_hotkey,
        evaluation_id=champion.evaluation_id,
        miner_hotkey=champion.miner_hotkey,
        settlement_round_id=champion.settlement_round_id,
        evaluation_block=champion.evaluation_block,
        target=champion.target,
        mode=champion.mode,
        member_slots=champion.member_slots,
        score=champion.score,
        passed_quality=champion.passed_quality,
        passed_timed_quality=champion.passed_timed_quality,
        passed_warmup_quality=champion.passed_warmup_quality,
        passed_speedup=champion.passed_speedup,
        confident=champion.confident,
        crownable=champion.crownable,
        quality_evidence=champion.quality_evidence,
        qualification_evidence_sha256=(
            champion.qualification_evidence_sha256
        ),
    )


def _champion_from_score(
    challenger: Score, *, current_sglang_version: str = ""
) -> Champion:
    """Project an authoritative score into the exact title row it proposes."""

    return Champion(
        content_hash=challenger.content_hash,
        hotkey=challenger.hotkey,
        score=challenger.score,
        round_id=challenger.round_id,
        sglang_version=current_sglang_version or challenger.sglang_version,
        target=challenger.target,
        mode=challenger.mode,
        member_slots=challenger.member_slots,
        arena_name=challenger.arena_name,
        arena_fingerprint=challenger.arena_fingerprint,
        arena_bracket=challenger.arena_bracket,
        regime=challenger.regime,
        validator_image=challenger.validator_image,
        referee_source_digest=challenger.referee_source_digest,
        referee_tree_digest=challenger.referee_tree_digest,
        model_revision=challenger.model_revision,
        model_manifest_digest=challenger.model_manifest_digest,
        model_content_digest=challenger.model_content_digest,
        host_attestation_sha256=challenger.host_attestation_sha256,
        prompt_seed=challenger.prompt_seed,
        prompt_engine_version=challenger.prompt_engine_version,
        prompt_seed_scheme=challenger.prompt_seed_scheme,
        seed_round_id=challenger.seed_round_id,
        seed_block=challenger.seed_block,
        seed_block_hash=challenger.seed_block_hash,
        quality_evidence=challenger.quality_evidence,
        chain_scope=challenger.chain_scope,
        validator_hotkey=challenger.validator_hotkey,
        evaluation_id=challenger.evaluation_id,
        miner_hotkey=challenger.miner_hotkey,
        settlement_round_id=challenger.settlement_round_id,
        evaluation_block=challenger.evaluation_block,
        passed_quality=challenger.passed_quality,
        passed_timed_quality=challenger.passed_timed_quality,
        passed_warmup_quality=challenger.passed_warmup_quality,
        passed_speedup=challenger.passed_speedup,
        confident=challenger.confident,
        crownable=challenger.crownable,
        qualification_evidence_sha256=(
            challenger.qualification_evidence_sha256
        ),
    )


def _normalized_record(cls: type, raw: dict):
    values = _only_fields(cls, raw)
    target, mode, members, slot = _normalize_competition_identity(
        slot=values.get("slot", ""),
        target=values.get("target", ""),
        mode=values.get("mode", ""),
        member_slots=values.get("member_slots", ()),
    )
    values.update(target=target, mode=mode, member_slots=members, slot=slot)
    return cls(**values)


def _normalized_champion(raw: dict, *, target_hint: str = "") -> Champion:
    """Restore tuple/canonical competition identity from JSON champion rows."""
    values = _only_fields(Champion, raw)
    target = str(values.get("target") or target_hint)
    mode = str(values.get("mode") or ("slot" if target else ""))
    members = values.get("member_slots", ())
    if target:
        target, mode, members, _ = _normalize_competition_identity(
            target=target,
            mode=mode,
            member_slots=members,
        )
        values.update(target=target, mode=mode, member_slots=members)
    else:
        values["member_slots"] = tuple(members or ())
    return Champion(**values)


class RevealError(ValueError):
    pass


class LedgerAttestationError(RuntimeError):
    """A crown-shaped row cannot be checked against retained host evidence."""

    validator_fault = True
    retryable = False


class PendingSettlementError(RuntimeError):
    """Persisted settlement work is incomplete or internally inconsistent.

    This is a controller/storage fault, never a miner disposition.  In particular
    callers must not clear the pending row after this exception.
    """

    validator_fault = True
    retryable = False


class Ledger:
    def __init__(self) -> None:
        self.commitments: list[Commitment] = []
        self.reveals: list[Reveal] = []
        self.scores: list[Score] = []
        self.evals: dict[str, EvalRecord] = {}
        self.retries: dict[str, RetryRecord] = {}
        self.validator_fault_holds: dict[str, ValidatorFaultHold] = {}
        self.pending_settlements: dict[str, PendingSettlement] = {}
        self.champion: Optional[Champion] = None  # winner-take-all baseline (single best)
        # Keys are competition targets. Legacy singleton targets equal their slot
        # names. This map is now a quarantined legacy namespace: registered arenas
        # use ``arena_champions[arena.bracket][target]`` and never inherit these
        # historical titles.
        self.champions: dict[str, Champion] = {}
        self.arena_champions: dict[str, dict[str, Champion]] = {}
        self.chain_scope: str = ""
        # External validator authority. This is bound from the active signing/
        # reconciliation hotkey at every production entrypoint; it is never
        # inferred from a Score, Champion, or retained sidecar.
        self.validator_hotkey: str = ""
        self._seq = 0

    @staticmethod
    def _pending_key(pending: PendingSettlement) -> str:
        material = json.dumps(
            {
                "arena_bracket": pending.arena_bracket,
                "round_id": pending.round_id,
                "hotkey": pending.hotkey,
                "content_hash": pending.content_hash,
                "target": pending.target,
                "evidence_sha256": pending.evidence_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    # ---- persistence ----

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        p = Path(path)
        led = cls()
        try:
            data = _read_ledger_json(p)
        except FileNotFoundError:
            return led
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            RecursionError,
            OSError,
            ValueError,
        ) as exc:
            raise LedgerAttestationError(
                f"existing ledger {p} is unreadable; refusing to start fresh: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise LedgerAttestationError(
                f"existing ledger {p} is not a JSON object; refusing to start fresh"
            )
        ver = data.get("schema_version", 1)
        if type(ver) is not int or ver < 1:
            raise LedgerAttestationError(
                f"existing ledger {p} has an invalid schema_version"
            )
        if ver > SCHEMA_VERSION:
            raise ValueError(
                f"ledger {p} is schema v{ver}, newer than this build supports (v{SCHEMA_VERSION}); "
                "upgrade optima before reading it"
            )
        raw_pending_present = bool(data.get("pending_settlements"))
        if ver < 9 and raw_pending_present:
            # v8 pending digests predate validator/evaluation/result binding.
            # Recomputing or silently clearing them would broaden their authority.
            raise PendingSettlementError(
                "pre-v9 pending settlement cannot be migrated safely; "
                "preserve the ledger and requalify under the current referee"
            )
        try:
            led.commitments = [
                Commitment(**_only_fields(Commitment, c))
                for c in data.get("commitments", [])
            ]
            led.reveals = [
                Reveal(**_only_fields(Reveal, r)) for r in data.get("reveals", [])
            ]
            commitments_by_seq: dict[int, Commitment] = {}
            for commitment in led.commitments:
                if (
                    not isinstance(commitment.hotkey, str)
                    or not commitment.hotkey
                    or len(commitment.hotkey) > 256
                    or re.fullmatch(r"[0-9a-f]{64}", commitment.commitment) is None
                    or type(commitment.round_id) is not int
                    or commitment.round_id < 0
                    or type(commitment.seq) is not int
                    or commitment.seq < 0
                    or commitment.seq in commitments_by_seq
                ):
                    raise ValueError("invalid commitment row")
                commitments_by_seq[commitment.seq] = commitment
            seen_reveals: set[tuple[str, str, int]] = set()
            for reveal in led.reveals:
                commitment = commitments_by_seq.get(reveal.commit_seq)
                reveal_key = (reveal.hotkey, reveal.content_hash, reveal.commit_seq)
                if (
                    not isinstance(reveal.hotkey, str)
                    or not reveal.hotkey
                    or len(reveal.hotkey) > 256
                    or re.fullmatch(r"[0-9a-f]{64}", reveal.content_hash) is None
                    or not isinstance(reveal.salt, str)
                    or len(reveal.salt) > 1024
                    or type(reveal.round_id) is not int
                    or reveal.round_id < 0
                    or type(reveal.commit_seq) is not int
                    or type(reveal.original) is not bool
                    or reveal_key in seen_reveals
                    or commitment is None
                    or commitment.hotkey != reveal.hotkey
                    or commitment.round_id != reveal.round_id
                    or commitment.commitment
                    != make_commitment(
                        reveal.content_hash, reveal.hotkey, reveal.salt
                    )
                    or not isinstance(reveal.slot_fingerprints, dict)
                    or not isinstance(reveal.slot_file_fingerprints, dict)
                    or not isinstance(reveal.product_fingerprints, dict)
                ):
                    raise ValueError("invalid reveal row")
                seen_reveals.add(reveal_key)
            led.scores = [_normalized_record(Score, s) for s in data.get("scores", [])]
            # Persisted dictionary keys are a cache, not authority. Rebuild them from
            # the typed row so a hand-edited key cannot suppress or alias an eval.
            led.evals = {}
            for raw in (data.get("evals") or {}).values():
                record = _normalized_record(EvalRecord, raw)
                if ver < 9 and record.arena_bracket:
                    # Older schemas did not bind the complete validator,
                    # evaluation-lease, and canonical qualification evidence.
                    # Requalify once rather than guessing and permanently
                    # poisoning production dedup with a non-authoritative row.
                    record = dataclasses.replace(record, development_only=True)
                key = led._eval_key(
                    record.hotkey, record.bundle_hash, record.arena_bracket
                )
                if key in led.evals:
                    raise ValueError(f"duplicate canonical eval row {key!r}")
                led.evals[key] = record
            led.retries = {}
            for raw in (data.get("retries") or {}).values():
                retry = _retry_from_raw(raw)
                completed_attempts = (
                    retry.no_decision_attempts + retry.infrastructure_attempts
                )
                if (
                    not retry.hotkey
                    or not retry.bundle_hash
                    or not retry.arena_bracket
                    or not retry.chain_scope
                    or type(retry.attempts) is not int
                    or type(retry.next_block) is not int
                    or retry.attempts <= 0
                    or retry.next_block < 0
                    or not retry.last_reason
                    or len(retry.last_reason) > 2000
                    or retry.kind not in RETRY_KINDS
                    or retry.state not in RETRY_STATES
                    or type(retry.no_decision_attempts) is not int
                    or retry.no_decision_attempts < 0
                    or type(retry.infrastructure_attempts) is not int
                    or retry.infrastructure_attempts < 0
                    or type(retry.lease_block) is not int
                    or retry.lease_block < 0
                    or (
                        retry.state == RETRY_STATE_IN_PROGRESS
                        and (
                            retry.attempts != completed_attempts + 1
                            or re.fullmatch(r"[0-9a-f]{64}", retry.lease_id) is None
                        )
                    )
                    or (
                        retry.state != RETRY_STATE_IN_PROGRESS
                        and (
                            retry.attempts != completed_attempts
                            or retry.lease_id
                            or retry.lease_block != 0
                        )
                    )
                ):
                    raise ValueError("invalid retry row")
                key = led._eval_key(
                    retry.hotkey, retry.bundle_hash, retry.arena_bracket
                )
                if key in led.retries:
                    raise ValueError(f"duplicate canonical retry row {key!r}")
                led.retries[key] = retry
            led.validator_fault_holds = {}
            for raw in (data.get("validator_fault_holds") or {}).values():
                hold = ValidatorFaultHold(
                    **_only_fields(ValidatorFaultHold, raw)
                )
                if (
                    not hold.hotkey
                    or re.fullmatch(r"[0-9a-f]{64}", hold.bundle_hash) is None
                    or not hold.arena_bracket
                    or not hold.chain_scope
                    or re.fullmatch(r"[0-9a-f]{64}", hold.evaluation_id) is None
                    or type(hold.created_block) is not int
                    or hold.created_block < 0
                    or not hold.reason
                    or len(hold.reason) > 2000
                ):
                    raise ValueError("invalid validator-fault hold row")
                key = led._eval_key(
                    hold.hotkey, hold.bundle_hash, hold.arena_bracket
                )
                if key in led.validator_fault_holds:
                    raise ValueError(
                        f"duplicate canonical validator-fault hold {key!r}"
                    )
                led.validator_fault_holds[key] = hold
            champ = data.get("champion")
            led.champion = _normalized_champion(champ) if champ else None
            led.champions = {
                slot: _normalized_champion(c, target_hint=slot)
                for slot, c in (data.get("champions") or {}).items() if c
            }
            led.arena_champions = {
                bracket: {
                    target: _normalized_champion(c, target_hint=target)
                    for target, c in (champions or {}).items() if c
                }
                for bracket, champions in (data.get("arena_champions") or {}).items()
                if isinstance(champions, dict)
            }
        except (TypeError, ValueError, AttributeError) as exc:
            if raw_pending_present:
                # An unresolved disposition makes this ledger irreplaceable
                # recovery authority. Quarantining it and returning an empty
                # ledger would silently forget already-paid GPU work.
                raise PendingSettlementError(
                    f"pending ledger has invalid authoritative rows: {exc}"
                ) from exc
            raise LedgerAttestationError(
                f"existing ledger {p} has invalid typed rows; refusing to start "
                f"fresh: {exc}"
            ) from exc
        raw_scope = data.get("chain_scope", "")
        if (
            not isinstance(raw_scope, str)
            or (
                raw_scope
                and re.fullmatch(
                    r"[0-9A-Za-z._-]{1,128}:sha256:[0-9a-f]{64}",
                    raw_scope,
                ) is None
            )
        ):
            raise LedgerAttestationError("ledger chain scope is malformed")
        led.chain_scope = raw_scope
        raw_validator = data.get("validator_hotkey", "")
        if not isinstance(raw_validator, str) or any(
            char in raw_validator for char in "\x00\r\n"
        ) or len(raw_validator) > 256 or raw_validator.strip() != raw_validator:
            raise LedgerAttestationError("ledger validator hotkey is malformed")
        led.validator_hotkey = raw_validator
        raw_pending = data.get("pending_settlements", {})
        if not isinstance(raw_pending, dict):
            raise PendingSettlementError(
                "pending settlement store must be a mapping"
            )
        for raw in raw_pending.values():
            try:
                pending = PendingSettlement(
                    **_only_fields(PendingSettlement, raw)
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise PendingSettlementError(
                    f"invalid pending settlement row: {exc}"
                ) from exc
            if (
                not pending.hotkey
                or re.fullmatch(r"[0-9a-f]{64}", pending.content_hash) is None
                or type(pending.round_id) is not int
                or pending.round_id < 0
                or not pending.target
                or not pending.arena_bracket
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}", pending.evidence_sha256
                ) is None
                or not pending.chain_scope
            ):
                raise PendingSettlementError("invalid pending settlement identity")
            key = led._pending_key(pending)
            if key in led.pending_settlements:
                raise PendingSettlementError(
                    f"duplicate canonical pending settlement row {key!r}"
                )
            led.pending_settlements[key] = pending
        if any(retry.chain_scope != led.chain_scope for retry in led.retries.values()):
            raise LedgerAttestationError(
                "ledger has retry rows from a different chain scope"
            )
        if any(
            hold.chain_scope != led.chain_scope
            for hold in led.validator_fault_holds.values()
        ):
            raise LedgerAttestationError(
                "ledger has validator-fault holds from a different chain scope"
            )
        if any(
            pending.chain_scope != led.chain_scope
            for pending in led.pending_settlements.values()
        ):
            raise PendingSettlementError(
                "pending settlement chain scope differs from ledger authority"
            )
        # Pending is load-bearing recovery state. Never accept a marker that no
        # longer binds an exact authoritative Score + EvalRecord pair; doing so
        # could either clear unsettled work or broaden a later competition.
        for pending in led.pending_settlements.values():
            led._score_for_pending(pending)
        raw_seq = data.get("seq", len(led.commitments))
        commit_seqs = [commitment.seq for commitment in led.commitments]
        if (
            type(raw_seq) is not int
            or raw_seq < 0
            or any(type(value) is not int or value < 0 for value in commit_seqs)
            or len(set(commit_seqs)) != len(commit_seqs)
            or (commit_seqs and raw_seq <= max(commit_seqs))
        ):
            raise LedgerAttestationError("ledger commitment sequence is invalid")
        led._seq = raw_seq
        return led

    def save(self, path: str | Path) -> None:
        data = {
            "schema_version": SCHEMA_VERSION,
            "commitments": [asdict(c) for c in self.commitments],
            "reveals": [asdict(r) for r in self.reveals],
            "scores": [asdict(s) for s in self.scores],
            "evals": {k: asdict(v) for k, v in self.evals.items()},
            "retries": {k: asdict(v) for k, v in self.retries.items()},
            "validator_fault_holds": {
                key: asdict(value)
                for key, value in self.validator_fault_holds.items()
            },
            "pending_settlements": {
                key: asdict(value)
                for key, value in self.pending_settlements.items()
            },
            "champion": asdict(self.champion) if self.champion else None,
            "champions": {slot: asdict(c) for slot, c in self.champions.items()},
            "arena_champions": {
                bracket: {target: asdict(c) for target, c in champions.items()}
                for bracket, champions in self.arena_champions.items()
            },
            "chain_scope": self.chain_scope,
            "validator_hotkey": self.validator_hotkey,
            "seq": self._seq,
        }
        _atomic_write_json(Path(path), data)

    def bind_chain_scope(self, scope: str) -> None:
        """Bind this ledger to one genesis/netuid namespace, once and fail closed."""
        if (
            not isinstance(scope, str)
            or re.fullmatch(
                r"[0-9A-Za-z._-]{1,128}:sha256:[0-9a-f]{64}", scope
            ) is None
        ):
            raise ValueError("invalid chain scope identity")
        if self.chain_scope and self.chain_scope != scope:
            raise LedgerAttestationError(
                f"ledger belongs to a different chain scope: "
                f"{self.chain_scope!r} != {scope!r}"
            )
        if not self.chain_scope:
            has_history = bool(
                self.commitments
                or self.reveals
                or self.scores
                or self.evals
                or bool(self.retries)
                or bool(self.validator_fault_holds)
                or bool(self.pending_settlements)
                or self.champion
                or self.champions
                or self.arena_champions
            )
            if has_history:
                raise LedgerAttestationError(
                    "legacy ledger contains commitments/reveals/scoring history without "
                    "a chain scope; migrate it explicitly instead of adopting it into "
                    "this chain"
                )
            self.chain_scope = scope

    def bind_validator_hotkey(self, hotkey: str) -> None:
        """Bind this ledger to the independently-known active validator."""

        if (
            not isinstance(hotkey, str)
            or not hotkey
            or len(hotkey) > 256
            or hotkey.strip() != hotkey
            or any(char in hotkey for char in "\x00\r\n")
        ):
            raise LedgerAttestationError(
                "active validator hotkey must be a bounded non-empty identity"
            )
        if self.validator_hotkey and self.validator_hotkey != hotkey:
            raise LedgerAttestationError(
                "ledger belongs to a different validator: "
                f"{self.validator_hotkey!r} != {hotkey!r}"
            )
        self.validator_hotkey = hotkey

    # ---- commit phase ----

    def commit(self, hotkey: str, commitment: str, round_id: int) -> int:
        seq = self._seq
        self._seq += 1
        self.commitments.append(Commitment(hotkey, commitment, round_id, seq))
        return seq

    # ---- reveal phase ----

    def reveal(self, hotkey: str, content_hash: str, salt: str, round_id: int,
               fingerprint: str = "", structural_fingerprint: str = "",
               slot_fingerprints: Optional[dict[str, str]] = None,
               slot_file_fingerprints: Optional[dict[str, list[str]]] = None,
               product_fingerprints: Optional[dict[str, list[str]]] = None) -> Reveal:
        """Verify a reveal against this hotkey's prior commitments; record it.

        Raises RevealError if no commitment by this hotkey matches. The commitment
        match is per-round (you commit and reveal within a round). Copy detection is
        **cumulative across ALL rounds** and matches on any of:

        * the exact ``content_hash``;
        * the whole-bundle reformat-invariant ``fingerprint``;
        * any single slot of ``slot_fingerprints`` (a stolen slot inside a bundle
          PADDED with an extra op);
        * per-slot file-set CONTAINMENT via ``slot_file_fingerprints`` — every
          substantial closure file of one reveal appearing in the other (a stolen
          body RELOCATED into an imported module, or a slot padded with extra
          files). Containment, not intersection, so two honest miners vendoring
          the same public utility next to their own distinct kernels never match.

        (All from ``optima.copy_fingerprint``.) Earliest commit (lowest seq) by a
        DIFFERENT hotkey is the original; this reveal is a copy if such an earlier
        one exists.
        """
        target = make_commitment(content_hash, hotkey, salt)
        match = min(
            (c for c in self.commitments
             if c.hotkey == hotkey and c.round_id == round_id and c.commitment == target),
            key=lambda c: c.seq,
            default=None,
        )
        if match is None:
            raise RevealError(
                f"no commitment by {hotkey!r} in round {round_id} matches the revealed bundle"
            )

        # Copy detection: a DIFFERENT hotkey's earlier reveal of the same content
        # (exact hash) OR the same normalized structure (near-copy fingerprint), in
        # ANY round, makes the later commit the copy. Same-hotkey re-reveals of one's
        # own work are never copies. Earliest commit_seq wins.
        slot_fps = {s: fp for s, fp in (slot_fingerprints or {}).items() if fp}
        file_fps = {s: sorted(v) for s, v in (slot_file_fingerprints or {}).items() if v}
        product_fps = {
            target: sorted(set(values))
            for target, values in (product_fingerprints or {}).items()
            if target and values
        }

        def _same(r: Reveal) -> bool:
            if r.hotkey == hotkey:
                return False
            if r.content_hash == content_hash:
                return True
            if fingerprint and r.fingerprint == fingerprint:
                return True
            # Per-slot compare: one stolen slot demotes, however the rest of the
            # bundle was padded/perturbed.
            if any(r.slot_fingerprints.get(s) == fp for s, fp in slot_fps.items()):
                return True
            # Per-slot file-set CONTAINMENT (either direction — commit order decides
            # who is original): all of one bundle's substantial files for a slot
            # appearing inside the other's = the same work, wherever the copier
            # relocated it and whatever they padded around it.
            for s, mine in file_fps.items():
                theirs = set(r.slot_file_fingerprints.get(s, ()))
                if theirs and (theirs <= set(mine) or set(mine) <= theirs):
                    return True
            for target, mine in product_fps.items():
                theirs = set(r.product_fingerprints.get(target, ()))
                if theirs.intersection(mine):
                    return True
            return False

        prior = [r for r in self.reveals if _same(r)]
        original = all(match.seq < r.commit_seq for r in prior) if prior else True
        if prior and original:
            # This reveal predates earlier-recorded ones; demote them.
            for r in prior:
                r.original = False

        rev = Reveal(
            hotkey=hotkey,
            content_hash=content_hash,
            salt=salt,
            round_id=round_id,
            commit_seq=match.seq,
            original=original,
            fingerprint=fingerprint,
            structural_fingerprint=structural_fingerprint,
            slot_fingerprints=slot_fps,
            slot_file_fingerprints=file_fps,
            product_fingerprints=product_fps,
        )
        self.reveals.append(rev)
        return rev

    # ---- emission policy ----

    def current_weights(
        self,
        per_slot: bool = True,
        *,
        arena=None,
        host_attestation_verifier=None,
        validator_hotkey: str | None = None,
    ) -> dict[str, float]:
        """The emission weights implied by the CURRENT champion state (no re-settle).

        THE single swap point for emission policy: every weight consumer (the chain
        validator loop, ``optima set-weights``) reads this instead of re-deriving
        winner-take-all inline. Today: per-target championships split emission equally
        across targets (a hotkey holding k of n targets earns k/n); ``per_slot`` is
        the compatibility spelling retained by the CLI and ``False`` selects the
        single-champion baseline. The planned relative-improvement +
        time-decay scheme replaces THIS method's body, nothing else.
        """
        if arena is not None:
            if not self.chain_scope:
                raise ValueError(
                    "registered-arena weights require a genesis/netuid-bound ledger"
                )
            if (
                not validator_hotkey
                or validator_hotkey != self.validator_hotkey
            ):
                raise LedgerAttestationError(
                    "registered-arena weights require the independently-known "
                    "active validator bound to this ledger"
                )
            arena_fields = _arena_identity(arena)
            champions = {
                target: champion
                for target, champion in self.arena_champions.get(
                    arena_fields["arena_bracket"], {}
                ).items()
                if self._valid_arena_champion(
                    champion, target=target, arena=arena,
                    arena_fields=arena_fields,
                    host_attestation_verifier=host_attestation_verifier,
                    validator_hotkey=validator_hotkey,
                )
            }
        else:
            # Backward-compatible inspection of pre-arena ledgers. Production
            # weight callers always provide an arena, so legacy titles can never
            # silently become current in a registered bracket.
            champions = self.champions
        if per_slot and champions:
            share = 1.0 / len(champions)
            weights: dict[str, float] = {}
            for champ in champions.values():
                weights[champ.hotkey] = weights.get(champ.hotkey, 0.0) + share
            return weights
        if arena is not None:
            return {}
        if self.champion:
            return {self.champion.hotkey: 1.0}
        return {}

    def current_weights_across_arenas(
        self,
        arenas,
        *,
        host_attestation_verifier,
        validator_hotkey: str,
    ) -> dict[str, float]:
        """One chain-scoped emission vector across every registered arena target.

        Per-arena daemons must never overwrite each other on-chain.  This projection
        treats each live ``(arena bracket, competition target)`` title as one equal
        emission unit under ``equal-per-target-koth-v1`` and independently verifies
        every retained crown before aggregation.
        """

        from optima.arenas import get_arena

        values = tuple(arenas)
        if not values:
            raise LedgerAttestationError("global weights require at least one arena")
        by_bracket = {}
        policies = set()
        for arena in values:
            registered = get_arena(arena.name)
            if registered.fingerprint != arena.fingerprint:
                raise LedgerAttestationError(
                    f"global weights received an unregistered arena {arena.name!r}"
                )
            if arena.bracket in by_bracket:
                raise LedgerAttestationError(
                    f"global weights received duplicate arena bracket {arena.bracket!r}"
                )
            by_bracket[arena.bracket] = registered
            policies.add(registered.settlement.emission_policy)
        if policies != {"equal-per-target-koth-v1"}:
            raise LedgerAttestationError(
                "registered arenas do not share the supported global emission policy"
            )
        if (
            not self.chain_scope
            or not validator_hotkey
            or validator_hotkey != self.validator_hotkey
        ):
            raise LedgerAttestationError(
                "global weights require the exact chain-scoped active validator"
            )

        live: list[Champion] = []
        for arena in sorted(by_bracket.values(), key=lambda value: value.bracket):
            arena_fields = _arena_identity(arena)
            for target, champion in sorted(
                self.arena_champions.get(arena.bracket, {}).items()
            ):
                if not self._valid_arena_champion(
                    champion,
                    target=target,
                    arena=arena,
                    arena_fields=arena_fields,
                    host_attestation_verifier=host_attestation_verifier,
                    validator_hotkey=validator_hotkey,
                ):
                    raise LedgerAttestationError(
                        "persisted champion is no longer authoritative for "
                        f"{arena.bracket!r}/{target!r}; refusing to redistribute "
                        "its emission across the remaining titles"
                    )
                live.append(champion)
        if not live:
            return {}
        share = 1.0 / len(live)
        weights: dict[str, float] = {}
        for champion in live:
            weights[champion.hotkey] = weights.get(champion.hotkey, 0.0) + share
        return weights

    def _valid_arena_champion(
        self,
        champion: Champion,
        *,
        target: str,
        arena,
        arena_fields: dict[str, str],
        host_attestation_verifier=None,
        validator_hotkey: str | None = None,
    ) -> bool:
        """Validate persisted title rows before they can influence emissions."""
        try:
            normalized = _normalize_competition_identity(
                target=champion.target,
                mode=champion.mode,
                member_slots=champion.member_slots,
            )
        except (TypeError, ValueError):
            return False
        from optima.arenas import derive_prompt_seed

        provenance_ok = bool(
            champion.prompt_seed_scheme == arena.workload.prompt_seed_scheme
            and type(champion.seed_round_id) is int
            and type(champion.seed_block) is int
            and champion.seed_round_id >= 0
            and champion.seed_block >= 0
            and champion.seed_round_id
            == champion.seed_block // arena.settlement.round_blocks
            and re.fullmatch(r"0x[0-9a-f]{64}", champion.seed_block_hash)
            and champion.prompt_seed
            == derive_prompt_seed(
                arena,
                bundle_hash=champion.content_hash,
                round_id=champion.seed_round_id,
                block_hash=champion.seed_block_hash,
            )
            and champion.miner_hotkey == champion.hotkey
            and type(champion.settlement_round_id) is int
            and type(champion.evaluation_block) is int
            and champion.settlement_round_id == champion.round_id
            and champion.settlement_round_id
            == champion.evaluation_block // arena.settlement.round_blocks
            and champion.evaluation_block >= champion.seed_block
        )
        champion_evidence = _crown_evidence_tuple(champion)
        matching_score = any(
            score.hotkey == champion.hotkey
            and score.content_hash == champion.content_hash
            and score.round_id == champion.round_id
            and score.passed
            and _crown_evidence_tuple(score) == champion_evidence
            and _record_matches_arena(score, arena_fields)
            for score in self.scores
        )
        matching_eval = any(
            record.hotkey == champion.hotkey
            and record.bundle_hash == champion.content_hash
            and record.round_id == champion.round_id
            and record.passed
            and record.development_only is False
            and _crown_evidence_tuple(record) == champion_evidence
            and _record_matches_arena(record, arena_fields)
            for record in self.evals.values()
        )
        structurally_valid = bool(
            normalized[0] == target
            and champion.target == target
            and _record_matches_arena(champion, arena_fields)
            and champion.sglang_version == arena.sglang_version
            and math.isfinite(champion.score)
            and champion.score > 1.0
            and type(champion.prompt_seed) is int
            and champion.prompt_seed > 0
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}", champion.host_attestation_sha256
            )
            and bool(champion.quality_evidence)
            and champion.chain_scope == self.chain_scope
            and bool(validator_hotkey)
            and validator_hotkey == self.validator_hotkey
            and champion.validator_hotkey == validator_hotkey
            and re.fullmatch(r"[0-9a-f]{64}", champion.evaluation_id)
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                champion.qualification_evidence_sha256,
            )
            and champion.passed_quality is True
            and champion.passed_timed_quality is True
            and champion.passed_warmup_quality is True
            and champion.passed_speedup is True
            and champion.confident is True
            and champion.crownable is True
            and provenance_ok
            and matching_score
            and matching_eval
            and self._is_original(
                champion.hotkey, champion.content_hash, champion.round_id
            )
        )
        if not structurally_valid:
            return False
        if not callable(host_attestation_verifier):
            raise LedgerAttestationError(
                "registered-arena crown validation requires retained-host verification"
            )
        # The concrete verifier owns nofollow/hash/context checking and raises a
        # validator fault for missing or corrupt retained evidence. Never turn
        # that infrastructure failure into a silent no-crown or miner terminal.
        retained = host_attestation_verifier(
            champion.host_attestation_sha256,
            _host_attestation_context(champion, arena),
        )
        if not retained:
            raise LedgerAttestationError(
                "retained-host verifier returned no authoritative evidence"
            )
        if (
            getattr(retained, "qualification_evidence_sha256", "")
            != champion.qualification_evidence_sha256
        ):
            raise LedgerAttestationError(
                "retained-host verifier returned a different qualification evidence"
            )
        return True

    def structural_near_copies(self, structural_fingerprint: str, hotkey: str) -> list[str]:
        """ADVISORY: prior reveals by OTHER hotkeys whose structural skeleton matches
        (rename/constant-tweak similarity). Returned for review/flagging — NOT used to
        demote, since the skeleton can collide on genuinely-distinct simple kernels."""
        if not structural_fingerprint:
            return []
        return sorted({
            r.hotkey for r in self.reveals
            if r.hotkey != hotkey and r.structural_fingerprint == structural_fingerprint
        })

    # ---- scoring ----

    def record_score(self, hotkey: str, content_hash: str, round_id: int,
                     score: float, kl_mean: float, passed: bool, sglang_version: str = "",
                     slot: str = "", *, target: str = "", mode: str = "",
                     member_slots: tuple[str, ...] | list[str] = (), arena=None,
                     prompt_seed: int = 0, prompt_engine_version: str = "",
                     prompt_seed_scheme: str = "", seed_round_id: int = 0,
                     seed_block: int = 0, seed_block_hash: str = "",
                     quality_evidence: str = "",
                     host_attestation_sha256: str = "",
                     validator_hotkey: str = "",
                     evaluation_id: str = "",
                     miner_hotkey: str = "",
                     settlement_round_id: int = 0,
                     evaluation_block: int = 0,
                     passed_quality: bool = False,
                     passed_timed_quality: bool = False,
                     passed_warmup_quality: bool = False,
                     passed_speedup: bool = False,
                     confident: bool = False,
                     crownable: bool = False,
                     qualification_evidence_sha256: str = "") -> Score:
        if (
            not isinstance(hotkey, str)
            or not hotkey
            or len(hotkey) > 256
            or hotkey.strip() != hotkey
            or any(char in hotkey for char in "\x00\r\n")
            or not isinstance(content_hash, str)
            or not content_hash
            or type(round_id) is not int
            or round_id < 0
            or type(passed) is not bool
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or isinstance(kl_mean, bool)
            or not isinstance(kl_mean, (int, float))
            or not math.isfinite(float(score))
            or not math.isfinite(float(kl_mean))
            or float(kl_mean) < 0
        ):
            raise ValueError("score and kl_mean must be finite; kl_mean non-negative")
        target, mode, members, legacy_slot = _normalize_competition_identity(
            slot=slot,
            target=target,
            mode=mode,
            member_slots=member_slots,
        )
        arena_fields = _arena_identity(arena)
        if arena is not None and sglang_version != arena.sglang_version:
            raise ValueError(
                "score sglang_version must equal the registered arena pin"
            )
        if arena is not None:
            from optima.arenas import derive_prompt_seed

            if re.fullmatch(r"[0-9a-f]{64}", content_hash) is None:
                raise ValueError(
                    "arena score content hash must be canonical lowercase SHA-256"
                )
            if not self.chain_scope:
                raise ValueError(
                    "arena score requires a genesis/netuid-bound ledger"
                )
            if not passed or score <= 1.0:
                raise ValueError("arena score rows must be crownable wins above 1.0")
            if prompt_seed <= 0:
                raise ValueError("arena score requires a positive post-commit prompt seed")
            if prompt_engine_version != arena.workload.prompt_engine_version:
                raise ValueError("arena score prompt engine version mismatch")
            if prompt_seed_scheme != arena.workload.prompt_seed_scheme:
                raise ValueError("arena score prompt seed scheme mismatch")
            if (type(seed_round_id) is not int or type(seed_block) is not int
                    or seed_round_id < 0 or seed_block < 0
                    or seed_round_id != seed_block // arena.settlement.round_blocks):
                raise ValueError("arena score seed round/block provenance mismatch")
            if re.fullmatch(r"0x[0-9a-f]{64}", seed_block_hash) is None:
                raise ValueError("arena score requires a canonical finalized block hash")
            expected_seed = derive_prompt_seed(
                arena,
                bundle_hash=content_hash,
                round_id=seed_round_id,
                block_hash=seed_block_hash,
            )
            if prompt_seed != expected_seed:
                raise ValueError("arena score prompt seed does not match its provenance")
            if not quality_evidence:
                raise ValueError("arena score requires controller quality evidence")
            if re.fullmatch(
                r"sha256:[0-9a-f]{64}", host_attestation_sha256
            ) is None:
                raise ValueError(
                    "arena score requires retained trusted-host attestation"
                )
            if (
                not self.validator_hotkey
                or validator_hotkey != self.validator_hotkey
            ):
                raise ValueError(
                    "arena score validator differs from external ledger authority"
                )
            if re.fullmatch(r"[0-9a-f]{64}", evaluation_id) is None:
                raise ValueError(
                    "arena score requires the exact persisted evaluation lease ID"
                )
            if miner_hotkey != hotkey:
                raise ValueError(
                    "arena score miner identity differs from the scored hotkey"
                )
            if (
                type(settlement_round_id) is not int
                or type(evaluation_block) is not int
                or settlement_round_id < 0
                or evaluation_block < seed_block
                or settlement_round_id != round_id
                or settlement_round_id
                != evaluation_block // arena.settlement.round_blocks
            ):
                raise ValueError(
                    "arena score settlement round/evaluation block provenance mismatch"
                )
            crown_projection = (
                passed_quality,
                passed_timed_quality,
                passed_warmup_quality,
                passed_speedup,
                confident,
                crownable,
            )
            if any(type(value) is not bool for value in crown_projection):
                raise ValueError("arena score qualification decisions must be booleans")
            if crown_projection != (True, True, True, True, True, True):
                raise ValueError(
                    "arena score must project one fully-qualified crownable outcome"
                )
            if re.fullmatch(
                r"sha256:[0-9a-f]{64}", qualification_evidence_sha256
            ) is None:
                raise ValueError(
                    "arena score requires canonical qualification evidence"
                )
        recorded = Score(
            hotkey, content_hash, round_id, score, kl_mean, passed,
            sglang_version, legacy_slot, target, mode, members, **arena_fields,
            prompt_seed=prompt_seed,
            prompt_engine_version=prompt_engine_version,
            prompt_seed_scheme=prompt_seed_scheme,
            seed_round_id=seed_round_id,
            seed_block=seed_block,
            seed_block_hash=seed_block_hash,
            host_attestation_sha256=host_attestation_sha256,
            quality_evidence=quality_evidence[:4096],
            chain_scope=self.chain_scope,
            validator_hotkey=validator_hotkey,
            evaluation_id=evaluation_id,
            miner_hotkey=miner_hotkey,
            settlement_round_id=settlement_round_id,
            evaluation_block=evaluation_block,
            passed_quality=passed_quality,
            passed_timed_quality=passed_timed_quality,
            passed_warmup_quality=passed_warmup_quality,
            passed_speedup=passed_speedup,
            confident=confident,
            crownable=crownable,
            qualification_evidence_sha256=qualification_evidence_sha256,
        )
        self.scores.append(recorded)
        return recorded

    # ---- full eval records (audit trail + dedup; the rich superset of a Score) ----

    @staticmethod
    def _eval_key(hotkey: str, bundle_hash: str, arena_bracket: str = "") -> str:
        # The same bundle must be evaluated independently in every arena. Empty
        # scope preserves the exact historical key for old ledgers.
        prefix = f"{arena_bracket}|" if arena_bracket else ""
        return f"{prefix}{hotkey}:{bundle_hash}"

    def record_eval(self, rec: EvalRecord) -> None:
        """Store the full eval record, keyed by (hotkey, bundle_hash). Recording the
        same submission again overwrites it (evaluations are deterministic)."""
        normalized = _normalized_record(EvalRecord, asdict(rec))
        self.evals[self._eval_key(
            rec.hotkey, rec.bundle_hash, normalized.arena_bracket
        )] = normalized

    # ---- durable pending-settlement recovery ----

    def _score_for_pending(self, pending: PendingSettlement) -> Score:
        """Return the exact authoritative score bound by ``pending``.

        The EvalRecord is independent authority for both the evidence tuple and
        production provenance. A missing/mutated pair is a storage/controller
        fault; it must never be treated as a completed miner disposition.
        """

        if pending.chain_scope != self.chain_scope or not self.chain_scope:
            raise PendingSettlementError(
                "pending settlement is outside the ledger chain scope"
            )
        record = self.eval_for(
            pending.hotkey,
            pending.content_hash,
            arena_bracket=pending.arena_bracket,
        )
        if (
            record is None
            or record.hotkey != pending.hotkey
            or record.bundle_hash != pending.content_hash
            or record.round_id != pending.round_id
            or record.target != pending.target
            or record.arena_bracket != pending.arena_bracket
            or record.chain_scope != pending.chain_scope
            or not self.validator_hotkey
            or record.validator_hotkey != self.validator_hotkey
            or re.fullmatch(r"[0-9a-f]{64}", record.evaluation_id) is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                record.qualification_evidence_sha256,
            ) is None
            or record.passed is not True
            or record.development_only is not False
            or not math.isfinite(record.score)
            or record.score <= 1.0
        ):
            raise PendingSettlementError(
                "pending settlement lacks its authoritative EvalRecord"
            )
        matches = [
            score
            for score in self.scores
            if score.hotkey == pending.hotkey
            and score.content_hash == pending.content_hash
            and score.round_id == pending.round_id
            and score.target == pending.target
            and score.arena_bracket == pending.arena_bracket
            and score.chain_scope == pending.chain_scope
            and score.passed is True
            and _score_evidence_sha256(score) == pending.evidence_sha256
            and _crown_evidence_tuple(score) == _crown_evidence_tuple(record)
        ]
        if not matches:
            raise PendingSettlementError(
                "pending settlement lacks its exact Score/EvalRecord pair"
            )
        return matches[-1]

    def mark_pending_settlement(self, score: Score) -> PendingSettlement:
        """Mark one just-recorded production qualification for settlement.

        The caller persists this marker in the same atomic ledger save as the
        Score and EvalRecord, before attempting any settlement.
        """

        if not any(candidate is score for candidate in self.scores):
            raise PendingSettlementError(
                "only a Score owned by this ledger can become pending"
            )
        pending = PendingSettlement(
            hotkey=score.hotkey,
            content_hash=score.content_hash,
            round_id=score.round_id,
            target=score.target,
            arena_bracket=score.arena_bracket,
            evidence_sha256=_score_evidence_sha256(score),
            chain_scope=score.chain_scope,
        )
        if (
            not pending.hotkey
            or re.fullmatch(r"[0-9a-f]{64}", pending.content_hash) is None
            or type(pending.round_id) is not int
            or pending.round_id < 0
            or not pending.target
            or not pending.arena_bracket
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", pending.evidence_sha256
            ) is None
            or not pending.chain_scope
        ):
            raise PendingSettlementError(
                "authoritative score has an invalid pending-settlement identity"
            )
        # Validate the independently-recorded EvalRecord before creating the
        # durable disposition. This rejects development/non-crown rows by design.
        self._score_for_pending(pending)
        self.pending_settlements[self._pending_key(pending)] = pending
        return pending

    def pending_settlements_for(
        self, *, arena_bracket: str, chain_scope: str
    ) -> tuple[PendingSettlement, ...]:
        """Return exact pending rows in deterministic chain-reveal order."""

        if (
            not arena_bracket
            or not chain_scope
            or chain_scope != self.chain_scope
        ):
            raise PendingSettlementError(
                "pending settlement lookup scope does not match the ledger"
            )
        reveal_order: dict[tuple[str, str], int] = {}
        for reveal in self.reveals:
            key = (reveal.hotkey, reveal.content_hash)
            previous = reveal_order.get(key)
            if previous is None or reveal.commit_seq < previous:
                reveal_order[key] = reveal.commit_seq

        rows = tuple(sorted(
            (
                pending
                for pending in self.pending_settlements.values()
                if pending.arena_bracket == arena_bracket
                and pending.chain_scope == chain_scope
            ),
            key=lambda pending: (
                pending.round_id,
                reveal_order.get(
                    (pending.hotkey, pending.content_hash),
                    1 << 63,
                ),
                pending.hotkey,
                pending.content_hash,
                pending.target,
                pending.evidence_sha256,
            ),
        ))
        for pending in rows:
            self._score_for_pending(pending)
        return rows

    def verify_pending_settlements(
        self,
        pending_rows: tuple[PendingSettlement, ...],
        *,
        arena,
        host_attestation_verifier,
        validator_hotkey: str,
    ) -> frozenset[str]:
        """Verify retained evidence for every pending disposition.

        Verification happens even for a challenger that ultimately fails the
        dethrone threshold. Thus no pending row clears merely because its score
        was numerically irrelevant before the standalone host evidence was read.
        The returned digests are the only candidate scores settlement may inspect.
        """

        if not pending_rows:
            raise PendingSettlementError("cannot verify an empty pending batch")
        rounds = {pending.round_id for pending in pending_rows}
        brackets = {pending.arena_bracket for pending in pending_rows}
        if len(rounds) != 1 or brackets != {arena.bracket}:
            raise PendingSettlementError(
                "pending batch must contain exactly one arena round"
            )
        arena_fields = _arena_identity(arena)
        evidence: set[str] = set()
        for pending in pending_rows:
            score = self._score_for_pending(pending)
            proposed = _champion_from_score(
                score, current_sglang_version=arena.sglang_version
            )
            if not self._valid_arena_champion(
                proposed,
                target=score.target,
                arena=arena,
                arena_fields=arena_fields,
                host_attestation_verifier=host_attestation_verifier,
                validator_hotkey=validator_hotkey,
            ):
                raise PendingSettlementError(
                    "pending settlement failed authoritative crown verification"
                )
            evidence.add(pending.evidence_sha256)
        return frozenset(evidence)

    def clear_pending_settlements(
        self, pending_rows: tuple[PendingSettlement, ...]
    ) -> None:
        """Clear exactly a previously-settled batch, never a broad round range."""

        if not pending_rows:
            raise PendingSettlementError("cannot clear an empty pending batch")
        for pending in pending_rows:
            key = self._pending_key(pending)
            if self.pending_settlements.get(key) != pending:
                raise PendingSettlementError(
                    "pending settlement changed before durable completion"
                )
        for pending in pending_rows:
            del self.pending_settlements[self._pending_key(pending)]

    def begin_retry_attempt(
        self,
        *,
        hotkey: str,
        bundle_hash: str,
        arena_bracket: str,
        current_block: int,
        reason: str,
        max_automatic_infrastructure_attempts: int,
        max_automatic_no_decision_attempts: int,
        max_total_attempts: int,
    ) -> RetryRecord:
        """Acquire one evaluation lease and count it before GPU work starts.

        The caller MUST durably ``save`` the ledger before launching the evaluator.
        A crash then leaves an ``in_progress`` row which cannot be automatically
        leased again.  If a migrated row already exhausts current policy, this
        method converts it to ``held`` and returns it without incrementing.
        """
        _validate_retry_limits(
            max_automatic_infrastructure_attempts=(
                max_automatic_infrastructure_attempts
            ),
            max_automatic_no_decision_attempts=(
                max_automatic_no_decision_attempts
            ),
            max_total_attempts=max_total_attempts,
        )
        if not self.chain_scope:
            raise ValueError("retry state requires a chain-scoped ledger")
        if (
            not hotkey or not bundle_hash or not arena_bracket or not reason
            or type(current_block) is not int or current_block < 0
        ):
            raise ValueError("invalid retry state")
        key = self._eval_key(hotkey, bundle_hash, arena_bracket)
        if key in self.validator_fault_holds:
            raise ValueError(
                "evaluation is held after a validator fault; trusted release required"
            )
        previous = self.retries.get(key)
        if previous is not None and previous.state == RETRY_STATE_IN_PROGRESS:
            raise ValueError(
                "evaluation attempt is already in progress; operator recovery required"
            )
        if previous is not None and previous.state == RETRY_STATE_HELD:
            raise ValueError("held retry requires an operator release")
        if previous is not None and current_block < previous.next_block:
            raise ValueError(
                f"retry backoff is active until block {previous.next_block}"
            )
        if previous is not None and _retry_limit_reached(
            previous,
            max_automatic_infrastructure_attempts=(
                max_automatic_infrastructure_attempts
            ),
            max_automatic_no_decision_attempts=(
                max_automatic_no_decision_attempts
            ),
            max_total_attempts=max_total_attempts,
        ):
            held = dataclasses.replace(
                previous,
                state=RETRY_STATE_HELD,
                lease_id="",
                lease_block=0,
                last_reason=(
                    "automatic retry budget exhausted before lease: " + reason
                )[:2000],
            )
            self.retries[key] = held
            return held

        attempts = 1 if previous is None else previous.attempts + 1
        no_decision_attempts = (
            0 if previous is None else previous.no_decision_attempts
        )
        infrastructure_attempts = (
            0 if previous is None else previous.infrastructure_attempts
        )
        kind = RETRY_KIND_NO_DECISION if previous is None else previous.kind
        lease_material = json.dumps(
            {
                "chain_scope": self.chain_scope,
                "arena_bracket": arena_bracket,
                "hotkey": hotkey,
                "bundle_hash": bundle_hash,
                "attempts": attempts,
                "block": current_block,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        lease_id = hashlib.sha256(lease_material).hexdigest()
        retry = RetryRecord(
            hotkey=hotkey,
            bundle_hash=bundle_hash,
            arena_bracket=arena_bracket,
            chain_scope=self.chain_scope,
            attempts=attempts,
            next_block=current_block,
            last_reason=reason[:2000],
            kind=kind,
            state=RETRY_STATE_IN_PROGRESS,
            no_decision_attempts=no_decision_attempts,
            infrastructure_attempts=infrastructure_attempts,
            lease_id=lease_id,
            lease_block=current_block,
        )
        self.retries[key] = retry
        return retry

    def complete_retry_attempt(
        self,
        *,
        hotkey: str,
        bundle_hash: str,
        arena_bracket: str,
        lease_id: str,
        kind: str,
        current_block: int,
        reason: str,
        base_backoff_blocks: int,
        max_backoff_blocks: int,
        max_automatic_infrastructure_attempts: int,
        max_automatic_no_decision_attempts: int,
        max_total_attempts: int,
    ) -> RetryRecord:
        """Finish the current lease as a retry without incrementing total attempts."""
        _validate_retry_limits(
            max_automatic_infrastructure_attempts=(
                max_automatic_infrastructure_attempts
            ),
            max_automatic_no_decision_attempts=(
                max_automatic_no_decision_attempts
            ),
            max_total_attempts=max_total_attempts,
        )
        if (
            kind not in RETRY_KINDS
            or not reason
            or type(current_block) is not int
            or current_block < 0
            or type(base_backoff_blocks) is not int
            or base_backoff_blocks <= 0
            or type(max_backoff_blocks) is not int
            or max_backoff_blocks < base_backoff_blocks
        ):
            raise ValueError("invalid retry completion")
        key = self._eval_key(hotkey, bundle_hash, arena_bracket)
        retry = self.retries.get(key)
        if (
            retry is None
            or retry.state != RETRY_STATE_IN_PROGRESS
            or not lease_id
            or retry.lease_id != lease_id
        ):
            raise ValueError("retry completion does not match the active lease")
        if current_block < retry.lease_block:
            raise ValueError("retry completion predates its lease")

        no_decision_attempts = retry.no_decision_attempts + int(
            kind == RETRY_KIND_NO_DECISION
        )
        infrastructure_attempts = retry.infrastructure_attempts + int(
            kind == RETRY_KIND_INFRASTRUCTURE
        )
        exponent = min(retry.attempts - 1, 30)
        delay = min(max_backoff_blocks, base_backoff_blocks * (2 ** exponent))
        completed = dataclasses.replace(
            retry,
            next_block=current_block + delay,
            last_reason=reason[:2000],
            kind=kind,
            state=RETRY_STATE_AUTOMATIC,
            no_decision_attempts=no_decision_attempts,
            infrastructure_attempts=infrastructure_attempts,
            lease_id="",
            lease_block=0,
        )
        if _retry_limit_reached(
            completed,
            max_automatic_infrastructure_attempts=(
                max_automatic_infrastructure_attempts
            ),
            max_automatic_no_decision_attempts=(
                max_automatic_no_decision_attempts
            ),
            max_total_attempts=max_total_attempts,
        ):
            completed = dataclasses.replace(completed, state=RETRY_STATE_HELD)
        self.retries[key] = completed
        return completed

    def complete_retry_terminal(
        self,
        *,
        hotkey: str,
        bundle_hash: str,
        arena_bracket: str,
        lease_id: str,
    ) -> RetryRecord:
        """Finish an active lease with success or terminal DQ and clear retry state."""
        key = self._eval_key(hotkey, bundle_hash, arena_bracket)
        retry = self.retries.get(key)
        if (
            retry is None
            or retry.state != RETRY_STATE_IN_PROGRESS
            or not lease_id
            or retry.lease_id != lease_id
        ):
            raise ValueError("terminal completion does not match the active lease")
        del self.retries[key]
        return retry

    def hold_validator_fault(
        self,
        *,
        hotkey: str,
        bundle_hash: str,
        arena_bracket: str,
        lease_id: str,
        current_block: int,
        reason: str,
    ) -> ValidatorFaultHold:
        """Roll back one active lease and durably stop automatic GPU replay.

        Completed miner retry counters are preserved. The current controller-owned
        attempt is not added to either retry budget.
        """

        key = self._eval_key(hotkey, bundle_hash, arena_bracket)
        retry = self.retries.get(key)
        if (
            retry is None
            or retry.state != RETRY_STATE_IN_PROGRESS
            or retry.lease_id != lease_id
            or not lease_id
            or type(current_block) is not int
            or current_block < 0
            or not reason
        ):
            raise ValueError("validator-fault hold does not match the active lease")
        completed_attempts = (
            retry.no_decision_attempts + retry.infrastructure_attempts
        )
        if completed_attempts:
            self.retries[key] = dataclasses.replace(
                retry,
                attempts=completed_attempts,
                next_block=current_block,
                last_reason=(
                    "prior miner retry counters preserved across validator fault"
                ),
                state=RETRY_STATE_AUTOMATIC,
                lease_id="",
                lease_block=0,
            )
        else:
            del self.retries[key]
        hold = ValidatorFaultHold(
            hotkey=hotkey,
            bundle_hash=bundle_hash,
            arena_bracket=arena_bracket,
            chain_scope=self.chain_scope,
            evaluation_id=lease_id,
            created_block=current_block,
            reason=reason[:2000],
        )
        self.validator_fault_holds[key] = hold
        return hold

    def validator_fault_for(
        self, hotkey: str, bundle_hash: str, *, arena_bracket: str
    ) -> Optional[ValidatorFaultHold]:
        return self.validator_fault_holds.get(
            self._eval_key(hotkey, bundle_hash, arena_bracket)
        )

    def validator_faults_for_scope(
        self, *, arena_bracket: str, chain_scope: str
    ) -> tuple[ValidatorFaultHold, ...]:
        """List controller-owned holds in one exact chain/arena namespace."""

        if not arena_bracket or not chain_scope or chain_scope != self.chain_scope:
            raise ValueError(
                "validator-fault inspection chain scope does not match the ledger"
            )
        return tuple(sorted(
            (
                hold
                for hold in self.validator_fault_holds.values()
                if hold.arena_bracket == arena_bracket
                and hold.chain_scope == chain_scope
            ),
            key=lambda hold: (
                hold.created_block,
                hold.hotkey,
                hold.bundle_hash,
                hold.evaluation_id,
            ),
        ))

    def release_validator_fault(
        self,
        hotkey: str,
        bundle_hash: str,
        *,
        arena_bracket: str,
        chain_scope: str,
    ) -> ValidatorFaultHold:
        """Trusted circuit-breaker release after controller repair/audit."""

        if chain_scope != self.chain_scope or not chain_scope:
            raise ValueError("validator-fault release chain scope mismatch")
        key = self._eval_key(hotkey, bundle_hash, arena_bracket)
        hold = self.validator_fault_holds.get(key)
        if hold is None or hold.chain_scope != chain_scope:
            raise KeyError("validator-fault hold does not exist in this chain/arena")
        del self.validator_fault_holds[key]
        return hold

    def record_retry(
        self,
        *,
        hotkey: str,
        bundle_hash: str,
        arena_bracket: str,
        kind: str,
        current_block: int,
        reason: str,
        base_backoff_blocks: int,
        max_backoff_blocks: int,
        max_automatic_infrastructure_attempts: int,
        max_automatic_no_decision_attempts: int,
        max_total_attempts: int,
    ) -> RetryRecord:
        """Compatibility path for callers not yet using persist-before-GPU leases.

        New production callers must use ``begin_retry_attempt`` followed by one of
        the completion methods. This fallback still keeps cumulative counters and
        all arena-pinned caps, but cannot make a process crash visible before return.
        """
        lease = self.begin_retry_attempt(
            hotkey=hotkey,
            bundle_hash=bundle_hash,
            arena_bracket=arena_bracket,
            current_block=current_block,
            reason="compatibility retry attempt (not pre-leased)",
            max_automatic_infrastructure_attempts=(
                max_automatic_infrastructure_attempts
            ),
            max_automatic_no_decision_attempts=(
                max_automatic_no_decision_attempts
            ),
            max_total_attempts=max_total_attempts,
        )
        if lease.state == RETRY_STATE_HELD:
            return lease
        return self.complete_retry_attempt(
            hotkey=hotkey,
            bundle_hash=bundle_hash,
            arena_bracket=arena_bracket,
            lease_id=lease.lease_id,
            kind=kind,
            current_block=current_block,
            reason=reason,
            base_backoff_blocks=base_backoff_blocks,
            max_backoff_blocks=max_backoff_blocks,
            max_automatic_infrastructure_attempts=(
                max_automatic_infrastructure_attempts
            ),
            max_automatic_no_decision_attempts=(
                max_automatic_no_decision_attempts
            ),
            max_total_attempts=max_total_attempts,
        )

    def retries_for_scope(
        self, *, arena_bracket: str, chain_scope: str
    ) -> tuple[RetryRecord, ...]:
        """Return retry rows only after an exact chain+arena scope check."""
        if not arena_bracket or not chain_scope or chain_scope != self.chain_scope:
            raise ValueError("retry inspection chain scope does not match the ledger")
        return tuple(sorted(
            (
                retry
                for retry in self.retries.values()
                if retry.arena_bracket == arena_bracket
                and retry.chain_scope == chain_scope
            ),
            key=lambda retry: (retry.state, retry.hotkey, retry.bundle_hash),
        ))

    def release_held_retry(
        self,
        hotkey: str,
        bundle_hash: str,
        *,
        arena_bracket: str,
        chain_scope: str,
    ) -> RetryRecord:
        """Trusted reset: remove one held row so the next pass may evaluate it.

        This is deliberately narrower than ``clear_retry``: an operator command
        cannot reset an automatic retry's backoff or mutate another chain/arena.
        """
        if not hotkey or not bundle_hash or not arena_bracket:
            raise ValueError("held retry release requires complete identity")
        if not chain_scope or chain_scope != self.chain_scope:
            raise ValueError("retry release chain scope does not match the ledger")
        key = self._eval_key(hotkey, bundle_hash, arena_bracket)
        retry = self.retries.get(key)
        if retry is None or retry.chain_scope != chain_scope:
            raise KeyError("held retry does not exist in the requested arena/chain")
        if retry.state != RETRY_STATE_HELD:
            raise ValueError("only an operator-held retry may be released")
        del self.retries[key]
        return retry

    def retry_for(
        self, hotkey: str, bundle_hash: str, *, arena_bracket: str
    ) -> Optional[RetryRecord]:
        return self.retries.get(self._eval_key(hotkey, bundle_hash, arena_bracket))

    def clear_retry(
        self, hotkey: str, bundle_hash: str, *, arena_bracket: str
    ) -> None:
        key = self._eval_key(hotkey, bundle_hash, arena_bracket)
        self.retries.pop(key, None)
        self.validator_fault_holds.pop(key, None)

    def is_known(
        self,
        hotkey: str,
        bundle_hash: str,
        *,
        arena_bracket: str = "",
        require_authoritative: bool = False,
        arena=None,
    ) -> bool:
        """True when an exact terminal eval is current enough to suppress replay.

        A pre-v6 registered-arena winner has no retained host sidecar. It cannot
        emit, but treating it as permanently terminal would also make it
        impossible to requalify after the security migration.
        """

        record = self.evals.get(self._eval_key(hotkey, bundle_hash, arena_bracket))
        if record is None:
            return False
        if arena_bracket and (
            not self.chain_scope
            or record.chain_scope != self.chain_scope
            or record.arena_bracket != arena_bracket
        ):
            return False
        if arena is not None and not _record_matches_arena(
            record, _arena_identity(arena)
        ):
            return False
        if (
            arena_bracket
            and require_authoritative
            and (
                record.development_only is not False
                or not self.validator_hotkey
                or record.validator_hotkey != self.validator_hotkey
                or re.fullmatch(r"[0-9a-f]{64}", record.evaluation_id) is None
                or arena is None
                or record.miner_hotkey != hotkey
                or type(record.settlement_round_id) is not int
                or type(record.evaluation_block) is not int
                or record.settlement_round_id != record.round_id
                or record.settlement_round_id
                != record.evaluation_block // arena.settlement.round_blocks
            )
        ):
            return False
        if (
            arena_bracket
            and require_authoritative
            and record.arena_bracket == arena_bracket
            and record.passed
            and math.isfinite(record.score)
            and record.score > 1.0
            and (
                re.fullmatch(
                    r"sha256:[0-9a-f]{64}", record.host_attestation_sha256
                ) is None
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    record.qualification_evidence_sha256,
                ) is None
            )
        ):
            return False
        return True

    def eval_for(self, hotkey: str, bundle_hash: str, *,
                 arena_bracket: str = "") -> Optional[EvalRecord]:
        return self.evals.get(self._eval_key(hotkey, bundle_hash, arena_bracket))

    def _is_original(self, hotkey: str, content_hash: str, round_id: int) -> bool:
        fallback = False
        for r in self.reveals:
            if r.hotkey != hotkey or r.content_hash != content_hash:
                continue
            fallback = fallback or r.original
            if r.round_id == round_id:
                return r.original
        # A chain reveal is global and may be evaluated in several arena daemons or
        # after a settlement-window rollover. Its anti-copy priority does not expire.
        return fallback

    def settle(self, round_id: int, margin: float = 0.02,
               current_sglang_version: str = "") -> SettleResult:
        """Apply king-of-the-hill: a challenger takes the title only if it beats the
        champion by ``margin``. Emission goes to the champion (winner-take-all baseline).
        Copies and non-improvers earn nothing.

        The recorded ``score`` is already a NOISE-CONFIRMED crownable speedup vs the
        round's fresh stock baseline, or 0.0 (see the eval) — so a too-noisy or
        below-bar candidate cannot win here either.

        STALE CHAMPION: a champion's frozen ``score`` is a speedup vs the stock kernels
        of the pin it was crowned under. After a ``PINNED_SGLANG`` bump the stock baseline
        changes, so that frozen number is no longer comparable to a challenger measured
        against the NEW stock. When ``current_sglang_version`` differs from the champion's,
        we refuse to let the stale number gate the round: the best confident challenger
        re-establishes the title by clearing the floor margin over *current* stock, and
        ``champion_stale`` is flagged so the operator re-baselines the old champion.
        """
        rejected_copies: list[str] = []
        candidates: list[Score] = []
        for s in self.scores:
            if s.round_id != round_id or not s.passed or s.arena_bracket:
                continue
            if not self._is_original(s.hotkey, s.content_hash, round_id):
                rejected_copies.append(s.hotkey)
                continue
            candidates.append(s)

        challenger = max(candidates, key=lambda s: s.score, default=None)
        challenger_score = challenger.score if challenger else 0.0

        champion_stale = bool(
            self.champion and current_sglang_version and self.champion.sglang_version
            and self.champion.sglang_version != current_sglang_version
        )
        # A stale champion's frozen ratio isn't comparable to the current pin's baseline,
        # so don't gate on it — require a real win over current fresh stock instead.
        if self.champion and not champion_stale:
            threshold = self.champion.score * (1.0 + margin)
        else:
            threshold = 1.0 + margin

        title_changed = False
        if challenger is not None and challenger_score >= threshold:
            self.champion = Champion(
                content_hash=challenger.content_hash,
                hotkey=challenger.hotkey,
                score=challenger.score,
                round_id=round_id,
                sglang_version=current_sglang_version or challenger.sglang_version,
            )
            title_changed = True
            champion_stale = False  # freshly (re-)crowned under the current pin

        weights = {self.champion.hotkey: 1.0} if self.champion else {}
        return SettleResult(
            champion=self.champion,
            weights=weights,
            title_changed=title_changed,
            challenger_score=challenger_score,
            rejected_copies=sorted(set(rejected_copies)),
            champion_stale=champion_stale,
        )

    def settle_per_target(
        self,
        round_id: int,
        margin: float = 0.02,
        current_sglang_version: str = "",
        *,
        arena=None,
        host_attestation_verifier=None,
        candidate_evidence_sha256: frozenset[str] | None = None,
        validator_hotkey: str | None = None,
    ) -> PerTargetSettleResult:
        """King-of-the-hill independently within each competition target.

        Singleton targets preserve the historical per-slot behavior. Atomic
        targets remain indivisible: a score is bracketed under its target ID once,
        regardless of how many member slots its bundle implements.
        """
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("settlement margin must be finite and non-negative")
        if candidate_evidence_sha256 is not None:
            if arena is None or not candidate_evidence_sha256:
                raise PendingSettlementError(
                    "evidence-filtered settlement requires a registered arena "
                    "and at least one pending score"
                )
            if any(
                re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
                for digest in candidate_evidence_sha256
            ):
                raise PendingSettlementError(
                    "settlement candidate evidence digest is malformed"
                )
        if arena is not None:
            if not self.chain_scope:
                raise ValueError(
                    "registered-arena settlement requires a genesis/netuid-bound ledger"
                )
            if (
                not validator_hotkey
                or validator_hotkey != self.validator_hotkey
            ):
                raise LedgerAttestationError(
                    "registered-arena settlement requires the independently-known "
                    "active validator bound to this ledger"
                )
            if margin != arena.settlement.dethrone_margin:
                raise ValueError(
                    "settlement margin disagrees with immutable arena policy"
                )
        rejected_copies: list[str] = []
        by_target: dict[str, list[Score]] = {}
        matched_evidence: set[str] = set()
        arena_fields = _arena_identity(arena)
        arena_bracket = arena_fields["arena_bracket"]
        for s in self.scores:
            if s.round_id != round_id or not s.passed:
                continue
            if arena is None:
                if s.arena_bracket:
                    continue
            elif not _record_matches_arena(s, arena_fields):
                continue
            elif s.sglang_version != arena.sglang_version:
                continue
            elif (
                s.chain_scope != self.chain_scope
                or not math.isfinite(s.score)
                or s.score <= 1.0
                or type(s.prompt_seed) is not int
                or s.prompt_seed <= 0
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}", s.host_attestation_sha256
                ) is None
                or s.validator_hotkey != validator_hotkey
                or re.fullmatch(r"[0-9a-f]{64}", s.evaluation_id) is None
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    s.qualification_evidence_sha256,
                ) is None
                or s.miner_hotkey != s.hotkey
                or s.settlement_round_id != s.round_id
                or type(s.evaluation_block) is not int
                or s.settlement_round_id
                != s.evaluation_block // arena.settlement.round_blocks
                or s.evaluation_block < s.seed_block
                or s.passed_quality is not True
                or s.passed_timed_quality is not True
                or s.passed_warmup_quality is not True
                or s.passed_speedup is not True
                or s.confident is not True
                or s.crownable is not True
                or not s.quality_evidence
            ):
                continue
            evidence_sha256 = _score_evidence_sha256(s)
            if (
                candidate_evidence_sha256 is not None
                and evidence_sha256 not in candidate_evidence_sha256
            ):
                continue
            if not self._is_original(s.hotkey, s.content_hash, round_id):
                rejected_copies.append(s.hotkey)
                continue
            # Rows loaded from old ledgers were normalized in ``load``; rows
            # created through ``record_score`` are normalized at insertion.
            # An unlabeled winner-take-all score has no target and is not safe to
            # route into target settlement.
            if not s.target:
                continue
            matched_evidence.add(evidence_sha256)
            by_target.setdefault(s.target, []).append(s)

        if (
            candidate_evidence_sha256 is not None
            and matched_evidence != set(candidate_evidence_sha256)
        ):
            raise PendingSettlementError(
                "settlement did not resolve every exact pending score"
            )

        title_changes: dict[str, bool] = {}
        stale_targets: list[str] = []
        if arena is not None:
            stored = self.arena_champions.setdefault(arena_bracket, {})
            champions = {
                target: champion
                for target, champion in stored.items()
                if self._valid_arena_champion(
                    champion, target=target, arena=arena,
                    arena_fields=arena_fields,
                    host_attestation_verifier=host_attestation_verifier,
                    validator_hotkey=validator_hotkey,
                )
            }
            # Quarantine malformed/mis-scoped persisted titles instead of emitting
            # from them; the underlying score/eval audit rows remain available.
            self.arena_champions[arena_bracket] = champions
        else:
            champions = self.champions
        for target, cands in by_target.items():
            champ = champions.get(target)
            stale = bool(champ and current_sglang_version and champ.sglang_version
                         and champ.sglang_version != current_sglang_version)
            threshold = (champ.score * (1.0 + margin)) if (champ and not stale) else (1.0 + margin)
            installed = False
            # Invalid/legacy evidence at the highest numeric score must not block
            # a lower fully-attested challenger (or a same-score requalification).
            for challenger in sorted(cands, key=lambda s: s.score, reverse=True):
                if challenger.score < threshold:
                    break
                proposed = _champion_from_score(
                    challenger,
                    current_sglang_version=current_sglang_version,
                )
                # A Score is not self-authenticating crown authority. Before a
                # newly proposed title can be installed or emitted in THIS settle,
                # require the exact matching retained EvalRecord and every scoped
                # host/source/model/prompt receipt. Waiting for a later
                # ``current_weights`` call to quarantine it would fail open for
                # the settlement response that immediately pushes weights.
                if arena is not None and not self._valid_arena_champion(
                    proposed,
                    target=target,
                    arena=arena,
                    arena_fields=arena_fields,
                    host_attestation_verifier=host_attestation_verifier,
                    validator_hotkey=validator_hotkey,
                ):
                    continue
                champions[target] = proposed
                title_changes[target] = True
                installed = True
                break
            if not installed and stale:
                stale_targets.append(target)

        # A target with no submissions this round still has a standing champion
        # earning emission, so pin staleness must still be surfaced.
        for target, champ in champions.items():
            if target in by_target or not champ:
                continue
            if (current_sglang_version and champ.sglang_version
                    and champ.sglang_version != current_sglang_version):
                stale_targets.append(target)

        # Split emission equally across competition targets with a champion.
        live = {target: c for target, c in champions.items() if c}
        weights: dict[str, float] = {}
        if live:
            share = 1.0 / len(live)
            for c in live.values():
                weights[c.hotkey] = weights.get(c.hotkey, 0.0) + share
        return PerTargetSettleResult(
            champions=dict(champions),
            weights=weights,
            title_changes=title_changes,
            stale_targets=sorted(set(stale_targets)),
            rejected_copies=sorted(set(rejected_copies)),
            arena_bracket=arena_bracket,
        )

    def settle_per_slot(
        self,
        round_id: int,
        margin: float = 0.02,
        current_sglang_version: str = "",
        *,
        arena=None,
        host_attestation_verifier=None,
        validator_hotkey: str | None = None,
    ) -> PerTargetSettleResult:
        """Compatibility alias for singleton-era callers.

        The implementation intentionally delegates to target settlement instead
        of expanding an atomic score into its members.
        """
        return self.settle_per_target(
            round_id,
            margin=margin,
            current_sglang_version=current_sglang_version,
            arena=arena,
            host_attestation_verifier=host_attestation_verifier,
            validator_hotkey=validator_hotkey,
        )
