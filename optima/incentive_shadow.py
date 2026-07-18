"""Signer-free, chain-bound synthetic finite-debt projections.

This module deliberately accepts its three chain reader callables as inputs.  It
therefore has no wallet, database, intake, settlement, or publication dependency
and cannot submit the projection it records.  Its only mutation is one exclusive,
canonical receipt file written after the exact finalized authority reopens.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from optima._strict import require_digest, require_exact_fields, require_int
from optima.finite_debt import (
    PPM,
    DebtClaimState,
    FiniteDebtPolicyManifest,
    project_debt_epoch,
)
from optima.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    sha256_hex,
)


MAX_SHADOW_INPUT_BYTES = 1 << 20
SHADOW_SCHEMA_VERSION = 1
SHADOW_RECEIPT_VERSION = "optima.chain-incentive-shadow.v1"
SYNTHETIC_FIXTURE_KIND = "synthetic"
_BLOCK_HASH = re.compile(r"0x[0-9a-fA-F]{64}\Z")
_FILE_STABILITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


class IncentiveShadowError(ValueError):
    """A shadow input, finalized authority, or receipt is invalid."""


def _strict(value: object, fields: set[str], label: str) -> dict[str, object]:
    return dict(
        require_exact_fields(
            value,
            fields=frozenset(fields),
            label=label,
            error=IncentiveShadowError,
            exact_dict=True,
        )
    )


def _integer(
    value: object,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    return require_int(
        value,
        field=field,
        error=IncentiveShadowError,
        minimum=minimum,
        maximum=maximum,
    )


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=IncentiveShadowError)


def _canonical_block_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _BLOCK_HASH.fullmatch(value) is None:
        raise IncentiveShadowError(f"{field} must be a 0x-prefixed 32-byte hash")
    return value.lower()


def _reject_json_number(value: str) -> object:
    del value
    raise IncentiveShadowError("shadow JSON must not contain floats or non-finite numbers")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise IncentiveShadowError(
                f"shadow JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _stable_file_tuple(info: os.stat_result) -> tuple[object, ...]:
    return tuple(getattr(info, field) for field in _FILE_STABILITY_FIELDS)


def _read_canonical_json(
    path: str | os.PathLike[str], *, label: str
) -> tuple[object, str]:
    """Read one bounded, stable, regular, exactly canonical JSON file."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise IncentiveShadowError("shadow inputs require O_NOFOLLOW support")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise IncentiveShadowError(f"cannot open {label}: {exc}") from None
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= MAX_SHADOW_INPUT_BYTES
        ):
            raise IncentiveShadowError(
                f"{label} must be a nonempty single-link regular file no larger than "
                f"{MAX_SHADOW_INPUT_BYTES} bytes"
            )
        chunks: list[bytes] = []
        observed = 0
        while observed <= MAX_SHADOW_INPUT_BYTES:
            chunk = os.read(fd, min(64 * 1024, MAX_SHADOW_INPUT_BYTES + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        if (
            len(raw) != before.st_size
            or len(raw) > MAX_SHADOW_INPUT_BYTES
            or _stable_file_tuple(after) != _stable_file_tuple(before)
        ):
            raise IncentiveShadowError(f"{label} changed while it was read")
    finally:
        os.close(fd)
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_float=_reject_json_number,
            parse_constant=_reject_json_number,
        )
        encoded = canonical_json_bytes(value)
    except IncentiveShadowError:
        raise
    except (UnicodeError, json.JSONDecodeError, StackIdentityError, ValueError) as exc:
        raise IncentiveShadowError(f"{label} is not strict UTF-8 JSON: {exc}") from None
    if encoded != raw:
        raise IncentiveShadowError(f"{label} is not canonically encoded JSON")
    return value, sha256_hex(raw)


@dataclass(frozen=True)
class SyntheticClaimStateFixture:
    """An explicitly non-authoritative, policy-bound claim-state fixture."""

    policy_digest: str
    claim_states: tuple[DebtClaimState, ...]
    fixture_kind: str = SYNTHETIC_FIXTURE_KIND
    schema_version: int = SHADOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_digest", _digest(self.policy_digest, "fixture policy_digest")
        )
        states = tuple(self.claim_states)
        if any(type(row) is not DebtClaimState for row in states):
            raise IncentiveShadowError("fixture claim states are not exactly typed")
        state_digests = tuple(row.claim.digest for row in states)
        if state_digests != tuple(sorted(set(state_digests))):
            raise IncentiveShadowError(
                "fixture claim states must be uniquely sorted by claim digest"
            )
        object.__setattr__(self, "claim_states", states)
        if self.fixture_kind != SYNTHETIC_FIXTURE_KIND:
            raise IncentiveShadowError("claim-state fixture must be explicitly synthetic")
        if self.schema_version != SHADOW_SCHEMA_VERSION:
            raise IncentiveShadowError("claim-state fixture schema is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_states": [row.to_dict() for row in self.claim_states],
            "fixture_kind": self.fixture_kind,
            "policy_digest": self.policy_digest,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SyntheticClaimStateFixture":
        row = _strict(value, set(cls.__dataclass_fields__), "synthetic claim-state fixture")
        raw_states = row["claim_states"]
        if type(raw_states) is not list:
            raise IncentiveShadowError("fixture claim_states must be an array")
        try:
            states = tuple(DebtClaimState.from_dict(item) for item in raw_states)
        except ValueError as exc:
            raise IncentiveShadowError(f"fixture claim state is invalid: {exc}") from None
        row["claim_states"] = states
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain-incentive-shadow.synthetic-claim-state-fixture",
            self.to_dict(),
        )


@dataclass(frozen=True)
class ShadowRecipient:
    hotkey: str
    uid: int
    ppm: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hotkey, str)
            or not self.hotkey
            or len(self.hotkey) > 256
            or any(char.isspace() for char in self.hotkey)
        ):
            raise IncentiveShadowError("shadow recipient hotkey is malformed")
        _integer(self.uid, "shadow recipient uid")
        _integer(self.ppm, "shadow recipient ppm", maximum=PPM)

    def to_dict(self) -> dict[str, object]:
        return {"hotkey": self.hotkey, "ppm": self.ppm, "uid": self.uid}


@dataclass(frozen=True)
class ShadowChainAuthority:
    genesis_hash: str
    netuid: int
    finalized_block: int
    finalized_block_hash: str
    metagraph_digest: str
    metagraph_size: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "genesis_hash", _canonical_block_hash(self.genesis_hash, "genesis_hash")
        )
        _integer(self.netuid, "netuid")
        _integer(self.finalized_block, "finalized_block")
        object.__setattr__(
            self,
            "finalized_block_hash",
            _canonical_block_hash(self.finalized_block_hash, "finalized_block_hash"),
        )
        object.__setattr__(
            self,
            "metagraph_digest",
            _digest(self.metagraph_digest, "metagraph_digest"),
        )
        _integer(self.metagraph_size, "metagraph_size")

    @property
    def chain_scope_digest(self) -> str:
        return canonical_digest(
            "optima.chain-incentive-shadow.scope",
            {"genesis_hash": self.genesis_hash, "netuid": self.netuid},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "chain_scope_digest": self.chain_scope_digest,
            "finalized_block": self.finalized_block,
            "finalized_block_hash": self.finalized_block_hash,
            "genesis_hash": self.genesis_hash,
            "metagraph_digest": self.metagraph_digest,
            "metagraph_size": self.metagraph_size,
            "netuid": self.netuid,
        }


@dataclass(frozen=True)
class ChainIncentiveShadowReceipt:
    authority: ShadowChainAuthority
    policy_digest: str
    policy_file_sha256: str
    claims_fixture_digest: str
    claims_file_sha256: str
    claims_count: int
    finite_debt_projection_digest: str
    claim_pool_capacity_ppm: int
    total_remaining_units: int
    payout_ppm: int
    miners: tuple[ShadowRecipient, ...]
    reserve: ShadowRecipient
    schema_version: int = SHADOW_SCHEMA_VERSION
    receipt_version: str = SHADOW_RECEIPT_VERSION
    mode: str = SYNTHETIC_FIXTURE_KIND
    submitted: bool = False

    def __post_init__(self) -> None:
        if type(self.authority) is not ShadowChainAuthority:
            raise IncentiveShadowError("shadow authority is not exactly typed")
        for field in (
            "policy_digest",
            "policy_file_sha256",
            "claims_fixture_digest",
            "claims_file_sha256",
            "finite_debt_projection_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        _integer(self.claims_count, "claims_count")
        _integer(
            self.claim_pool_capacity_ppm,
            "claim_pool_capacity_ppm",
            maximum=PPM,
        )
        _integer(self.total_remaining_units, "total_remaining_units")
        _integer(self.payout_ppm, "payout_ppm", maximum=PPM)
        miners = tuple(self.miners)
        if (
            any(type(row) is not ShadowRecipient or row.ppm <= 0 for row in miners)
            or tuple(row.hotkey for row in miners)
            != tuple(sorted({row.hotkey for row in miners}))
        ):
            raise IncentiveShadowError(
                "shadow miner rows must be positive, unique, and hotkey-sorted"
            )
        if type(self.reserve) is not ShadowRecipient:
            raise IncentiveShadowError("shadow reserve row is not exactly typed")
        recipients = (*miners, self.reserve)
        if (
            len({row.hotkey for row in recipients}) != len(recipients)
            or len({row.uid for row in recipients}) != len(recipients)
            or sum(row.ppm for row in miners) != self.payout_ppm
            or self.payout_ppm + self.reserve.ppm != PPM
        ):
            raise IncentiveShadowError(
                "shadow miner/reserve projection is not uniquely mapped or conserved"
            )
        object.__setattr__(self, "miners", miners)
        if self.schema_version != SHADOW_SCHEMA_VERSION:
            raise IncentiveShadowError("shadow receipt schema is unsupported")
        if self.receipt_version != SHADOW_RECEIPT_VERSION:
            raise IncentiveShadowError("shadow receipt version is unsupported")
        if self.mode != SYNTHETIC_FIXTURE_KIND:
            raise IncentiveShadowError("shadow receipt mode is not synthetic")
        if self.submitted is not False:
            raise IncentiveShadowError("shadow receipt submitted must be false")

    def to_dict(self) -> dict[str, object]:
        return {
            "authority": self.authority.to_dict(),
            "inputs": {
                "claims_count": self.claims_count,
                "claims_file_sha256": self.claims_file_sha256,
                "claims_fixture_digest": self.claims_fixture_digest,
                "policy_digest": self.policy_digest,
                "policy_file_sha256": self.policy_file_sha256,
            },
            "mode": self.mode,
            "non_authority": {
                "claims_source": "synthetic_fixture",
                "publication_authority": "none",
                "settlement_authority": "none",
            },
            "projection": {
                "claim_pool_capacity_ppm": self.claim_pool_capacity_ppm,
                "effective_block": self.authority.finalized_block,
                "finite_debt_projection_digest": self.finite_debt_projection_digest,
                "miners": [row.to_dict() for row in self.miners],
                "payout_ppm": self.payout_ppm,
                "reserve": self.reserve.to_dict(),
                "total_ppm": PPM,
                "total_remaining_units": self.total_remaining_units,
            },
            "receipt_version": self.receipt_version,
            "schema_version": self.schema_version,
            "submitted": self.submitted,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain-incentive-shadow.receipt", self.to_dict()
        )

    def envelope(self) -> dict[str, object]:
        return {"receipt": self.to_dict(), "receipt_digest": self.digest}


@dataclass(frozen=True)
class _ShadowInputs:
    policy: FiniteDebtPolicyManifest
    policy_file_sha256: str
    fixture: SyntheticClaimStateFixture
    claims_file_sha256: str


def load_shadow_inputs(
    *,
    policy_path: str | os.PathLike[str],
    claims_fixture_path: str | os.PathLike[str],
    expected_policy_digest: str,
    expected_claims_digest: str,
) -> _ShadowInputs:
    """Reopen and semantically pin both synthetic shadow inputs."""

    expected_policy = _digest(expected_policy_digest, "expected policy digest")
    expected_claims = _digest(expected_claims_digest, "expected claims digest")
    policy_value, policy_file_sha256 = _read_canonical_json(
        policy_path, label="finite-debt policy"
    )
    try:
        policy = FiniteDebtPolicyManifest.from_dict(policy_value)
    except ValueError as exc:
        raise IncentiveShadowError(f"finite-debt policy is invalid: {exc}") from None
    if policy.digest != expected_policy:
        raise IncentiveShadowError("finite-debt policy semantic digest differs")
    fixture_value, claims_file_sha256 = _read_canonical_json(
        claims_fixture_path, label="synthetic claim-state fixture"
    )
    fixture = SyntheticClaimStateFixture.from_dict(fixture_value)
    if fixture.digest != expected_claims:
        raise IncentiveShadowError("synthetic claims semantic digest differs")
    if fixture.policy_digest != policy.digest:
        raise IncentiveShadowError("synthetic claims fixture differs from its policy")
    return _ShadowInputs(
        policy,
        policy_file_sha256,
        fixture,
        claims_file_sha256,
    )


def _assert_output_available(path: str | os.PathLike[str]) -> Path:
    output = Path(path)
    try:
        os.lstat(output)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise IncentiveShadowError(f"cannot inspect shadow output: {exc}") from None
    else:
        raise IncentiveShadowError("shadow output already exists")
    parent = output.parent
    try:
        info = os.stat(parent, follow_symlinks=False)
    except OSError as exc:
        raise IncentiveShadowError(f"cannot inspect shadow output parent: {exc}") from None
    if not stat.S_ISDIR(info.st_mode):
        raise IncentiveShadowError("shadow output parent must be an existing directory")
    return output


def _read_chain_hash(subtensor: object, block: int, field: str) -> str:
    try:
        value = getattr(subtensor, "get_block_hash")(block)
    except Exception as exc:
        raise IncentiveShadowError(f"cannot read {field}: {exc}") from None
    return _canonical_block_hash(str(value), field)


def _read_finalized_point(
    subtensor: object,
    read_finalized_head: Callable[[object], tuple[int, str]],
) -> tuple[int, str]:
    try:
        point = read_finalized_head(subtensor)
    except Exception as exc:
        raise IncentiveShadowError(f"cannot read finalized head: {exc}") from None
    if type(point) is not tuple or len(point) != 2:
        raise IncentiveShadowError("finalized head reader returned a malformed point")
    block = _integer(point[0], "finalized head block")
    block_hash = _canonical_block_hash(point[1], "finalized head hash")
    if _read_chain_hash(subtensor, block, "finalized head hash") != block_hash:
        raise IncentiveShadowError("finalized head height/hash is inconsistent")
    return block, block_hash


def _metagraph_payload(view: object) -> dict[str, object]:
    try:
        netuid = getattr(view, "netuid")
        block = getattr(view, "block")
        block_hash = getattr(view, "block_hash")
        uids = list(getattr(view, "uids"))
        hotkeys = list(getattr(view, "hotkeys"))
        permits = list(getattr(view, "validator_permit"))
        last_update = list(getattr(view, "last_update"))
    except Exception as exc:
        raise IncentiveShadowError(f"metagraph view is malformed: {exc}") from None
    netuid = _integer(netuid, "metagraph netuid")
    block = _integer(block, "metagraph block")
    block_hash = _canonical_block_hash(block_hash, "metagraph block_hash")
    if not len(uids) == len(hotkeys) == len(permits) == len(last_update):
        raise IncentiveShadowError("metagraph columns have different widths")
    canonical_uids = tuple(_integer(value, "metagraph uid") for value in uids)
    canonical_updates = tuple(
        _integer(value, "metagraph last_update") for value in last_update
    )
    if (
        any(type(value) is not bool for value in permits)
        or any(not isinstance(value, str) or not value for value in hotkeys)
        or len(set(canonical_uids)) != len(canonical_uids)
        or len(set(hotkeys)) != len(hotkeys)
    ):
        raise IncentiveShadowError("metagraph membership is invalid or duplicated")
    return {
        "block": block,
        "block_hash": block_hash,
        "hotkeys": hotkeys,
        "last_update": list(canonical_updates),
        "netuid": netuid,
        "uids": list(canonical_uids),
        "validator_permit": permits,
    }


def _fetch_exact_metagraph(
    subtensor: object,
    *,
    netuid: int,
    block: int,
    fetch_metagraph: Callable[..., object],
) -> tuple[object, dict[str, object]]:
    try:
        view = fetch_metagraph(subtensor, netuid, block=block)
    except Exception as exc:
        raise IncentiveShadowError(f"cannot fetch finalized metagraph: {exc}") from None
    return view, _metagraph_payload(view)


def _build_receipt(
    *,
    inputs: _ShadowInputs,
    netuid: int,
    genesis_hash: str,
    finalized_block: int,
    finalized_block_hash: str,
    metagraph: dict[str, object],
) -> ChainIncentiveShadowReceipt:
    if (
        metagraph["netuid"] != netuid
        or metagraph["block"] != finalized_block
        or metagraph["block_hash"] != finalized_block_hash
    ):
        raise IncentiveShadowError(
            "metagraph does not match the exact finalized shadow authority"
        )
    for state in inputs.fixture.claim_states:
        terminal = state.balance.terminal_block
        if (
            state.claim.accepted_crown_block > finalized_block
            or state.claim.settlement_block > finalized_block
            or (terminal is not None and terminal > finalized_block)
        ):
            raise IncentiveShadowError(
                "synthetic claim state contains future chain authority"
            )
    try:
        projection = project_debt_epoch(
            inputs.policy,
            effective_block=finalized_block,
            states=inputs.fixture.claim_states,
        )
    except ValueError as exc:
        raise IncentiveShadowError(f"finite-debt projection failed: {exc}") from None

    hotkeys = metagraph["hotkeys"]
    uids = metagraph["uids"]
    assert isinstance(hotkeys, list) and isinstance(uids, list)
    uid_by_hotkey = dict(zip(hotkeys, uids, strict=True))
    reserve_uid = uid_by_hotkey.get(inputs.policy.reserve_hotkey)
    if type(reserve_uid) is not int:
        raise IncentiveShadowError(
            "finite-debt policy reserve hotkey is absent from the finalized metagraph"
        )
    miner_rows: list[ShadowRecipient] = []
    for weight in projection.weights:
        if weight.hotkey == inputs.policy.reserve_hotkey:
            continue
        uid = uid_by_hotkey.get(weight.hotkey)
        if type(uid) is not int:
            raise IncentiveShadowError(
                "positive finite-debt miner is absent from the finalized metagraph"
            )
        miner_rows.append(ShadowRecipient(weight.hotkey, uid, weight.units))
    miners = tuple(sorted(miner_rows, key=lambda row: row.hotkey))
    reserve = ShadowRecipient(
        inputs.policy.reserve_hotkey,
        reserve_uid,
        projection.reserve_units,
    )
    authority = ShadowChainAuthority(
        genesis_hash,
        netuid,
        finalized_block,
        finalized_block_hash,
        canonical_digest("optima.chain-incentive-shadow.metagraph", metagraph),
        len(hotkeys),
    )
    return ChainIncentiveShadowReceipt(
        authority,
        inputs.policy.digest,
        inputs.policy_file_sha256,
        inputs.fixture.digest,
        inputs.claims_file_sha256,
        len(inputs.fixture.claim_states),
        projection.digest,
        projection.claim_pool_capacity_units,
        projection.total_remaining_units,
        projection.payout_units,
        miners,
        reserve,
    )


def write_shadow_receipt(
    path: str | os.PathLike[str], receipt: ChainIncentiveShadowReceipt
) -> Path:
    """Exclusively write and durably sync one canonical shadow receipt."""

    if type(receipt) is not ChainIncentiveShadowReceipt:
        raise IncentiveShadowError("shadow receipt is not exactly typed")
    output = _assert_output_available(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise IncentiveShadowError("shadow receipt writing requires O_NOFOLLOW support")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
    )
    raw = canonical_json_bytes(receipt.envelope()) + b"\n"
    fd: int | None = None
    created = False
    created_identity: tuple[int, int] | None = None
    write_failure: IncentiveShadowError | None = None
    try:
        fd = os.open(output, flags, 0o600)
        created = True
        info = os.fstat(fd)
        created_identity = (info.st_dev, info.st_ino)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise IncentiveShadowError("shadow receipt write stalled")
            view = view[written:]
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    except FileExistsError:
        raise IncentiveShadowError("shadow output already exists") from None
    except IncentiveShadowError as exc:
        write_failure = exc
    except OSError as exc:
        write_failure = IncentiveShadowError(f"cannot write shadow receipt: {exc}")
    finally:
        if fd is not None:
            os.close(fd)
    if write_failure is not None:
        if created_identity is not None:
            try:
                current = os.lstat(output)
                if (current.st_dev, current.st_ino) == created_identity:
                    os.unlink(output)
            except OSError:
                pass
        raise write_failure from None
    try:
        dir_flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
        dir_flags |= getattr(os, "O_DIRECTORY", 0)
        parent_fd = os.open(output.parent, dir_flags)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError as exc:
        if created and created_identity is not None:
            try:
                current = os.lstat(output)
                if (current.st_dev, current.st_ino) == created_identity:
                    os.unlink(output)
            except OSError:
                pass
        raise IncentiveShadowError(
            f"cannot durably sync shadow receipt directory: {exc}"
        ) from None
    return output


def execute_chain_incentive_shadow(
    *,
    network: str,
    netuid: int,
    policy_path: str | os.PathLike[str],
    claims_fixture_path: str | os.PathLike[str],
    expected_policy_digest: str,
    expected_claims_digest: str,
    output_path: str | os.PathLike[str],
    connect: Callable[[str], object],
    read_finalized_head: Callable[[object], tuple[int, str]],
    fetch_metagraph: Callable[..., object],
) -> ChainIncentiveShadowReceipt:
    """Project synthetic claims against twice-reopened finalized chain authority."""

    if not isinstance(network, str) or not network:
        raise IncentiveShadowError("network selector must be nonempty")
    selected_netuid = _integer(netuid, "netuid")
    inputs = load_shadow_inputs(
        policy_path=policy_path,
        claims_fixture_path=claims_fixture_path,
        expected_policy_digest=expected_policy_digest,
        expected_claims_digest=expected_claims_digest,
    )
    output = _assert_output_available(output_path)
    try:
        subtensor = connect(network)
    except Exception as exc:
        raise IncentiveShadowError(f"cannot connect read-only chain client: {exc}") from None

    genesis_hash = _read_chain_hash(subtensor, 0, "genesis hash")
    finalized_block, finalized_hash = _read_finalized_point(
        subtensor, read_finalized_head
    )
    _view, metagraph = _fetch_exact_metagraph(
        subtensor,
        netuid=selected_netuid,
        block=finalized_block,
        fetch_metagraph=fetch_metagraph,
    )
    receipt = _build_receipt(
        inputs=inputs,
        netuid=selected_netuid,
        genesis_hash=genesis_hash,
        finalized_block=finalized_block,
        finalized_block_hash=finalized_hash,
        metagraph=metagraph,
    )

    reopened_genesis = _read_chain_hash(subtensor, 0, "reopened genesis hash")
    reopened_block, reopened_hash = _read_finalized_point(
        subtensor, read_finalized_head
    )
    if reopened_genesis != genesis_hash or reopened_block < finalized_block:
        raise IncentiveShadowError("finalized chain authority regressed while projected")
    if reopened_block == finalized_block and reopened_hash != finalized_hash:
        raise IncentiveShadowError("finalized head changed at the retained height")
    if _read_chain_hash(
        subtensor, finalized_block, "reopened retained block hash"
    ) != finalized_hash:
        raise IncentiveShadowError("retained finalized block hash changed while projected")
    _reopened_view, reopened_metagraph = _fetch_exact_metagraph(
        subtensor,
        netuid=selected_netuid,
        block=finalized_block,
        fetch_metagraph=fetch_metagraph,
    )
    if reopened_metagraph != metagraph:
        raise IncentiveShadowError(
            "finalized metagraph authority changed while projected"
        )
    write_shadow_receipt(output, receipt)
    return receipt


__all__ = [
    "ChainIncentiveShadowReceipt",
    "IncentiveShadowError",
    "MAX_SHADOW_INPUT_BYTES",
    "SHADOW_RECEIPT_VERSION",
    "SHADOW_SCHEMA_VERSION",
    "SYNTHETIC_FIXTURE_KIND",
    "ShadowChainAuthority",
    "ShadowRecipient",
    "SyntheticClaimStateFixture",
    "execute_chain_incentive_shadow",
    "load_shadow_inputs",
    "write_shadow_receipt",
]
