"""Signer-free, chain-bound synthetic composed incentive projections.

This module is deliberately read-only with respect to chain and settlement
state.  It reopens four canonical, digest-pinned synthetic inputs, projects the
reviewed-discovery and registered-CROWN classes against one exact finalized
metagraph, reopens that same authority, and exclusively writes one receipt.  It
has no wallet, signer, database, intake, or publication dependency.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from optima._strict import require_digest, require_exact_fields, require_int
from optima.finite_debt import PPM, FiniteDebtPolicyManifest
from optima.incentive_composition import (
    DiscoveryClaimState,
    IncentiveCompositionPolicyManifest,
    project_composed_epoch,
)
from optima.incentive_shadow import (
    SYNTHETIC_FIXTURE_KIND,
    ShadowChainAuthority,
    ShadowRecipient,
    SyntheticClaimStateFixture,
    _assert_output_available,
    _fetch_exact_metagraph,
    _read_canonical_json,
    _read_chain_hash,
    _read_finalized_point,
)
from optima.stack_identity import canonical_digest, canonical_json_bytes


COMPOSED_SHADOW_SCHEMA_VERSION = 1
COMPOSED_SHADOW_RECEIPT_VERSION = "optima.chain-incentive-composition-shadow.v1"


class IncentiveCompositionShadowError(ValueError):
    """A composed shadow input, authority, projection, or receipt is invalid."""


def _strict(value: object, fields: set[str], label: str) -> dict[str, object]:
    return dict(
        require_exact_fields(
            value,
            fields=frozenset(fields),
            label=label,
            error=IncentiveCompositionShadowError,
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
        error=IncentiveCompositionShadowError,
        minimum=minimum,
        maximum=maximum,
    )


def _digest(value: object, field: str) -> str:
    return require_digest(
        value,
        field=field,
        error=IncentiveCompositionShadowError,
    )


@dataclass(frozen=True)
class SyntheticDiscoveryStateFixture:
    """An explicitly non-authoritative, policy-bound discovery-state fixture."""

    policy_digest: str
    discovery_states: tuple[DiscoveryClaimState, ...]
    fixture_kind: str = SYNTHETIC_FIXTURE_KIND
    schema_version: int = COMPOSED_SHADOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_digest",
            _digest(self.policy_digest, "discovery fixture policy_digest"),
        )
        states = tuple(self.discovery_states)
        if any(type(row) is not DiscoveryClaimState for row in states):
            raise IncentiveCompositionShadowError(
                "discovery fixture states are not exactly typed"
            )
        state_digests = tuple(row.claim.digest for row in states)
        if state_digests != tuple(sorted(set(state_digests))):
            raise IncentiveCompositionShadowError(
                "discovery fixture states must be uniquely sorted by claim digest"
            )
        object.__setattr__(self, "discovery_states", states)
        if self.fixture_kind != SYNTHETIC_FIXTURE_KIND:
            raise IncentiveCompositionShadowError(
                "discovery fixture must be explicitly synthetic"
            )
        if self.schema_version != COMPOSED_SHADOW_SCHEMA_VERSION:
            raise IncentiveCompositionShadowError(
                "discovery fixture schema is unsupported"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "discovery_states": [row.to_dict() for row in self.discovery_states],
            "fixture_kind": self.fixture_kind,
            "policy_digest": self.policy_digest,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SyntheticDiscoveryStateFixture":
        row = _strict(
            value,
            set(cls.__dataclass_fields__),
            "synthetic discovery-state fixture",
        )
        raw_states = row["discovery_states"]
        if type(raw_states) is not list:
            raise IncentiveCompositionShadowError(
                "discovery fixture states must be an array"
            )
        try:
            row["discovery_states"] = tuple(
                DiscoveryClaimState.from_dict(item) for item in raw_states
            )
        except ValueError as exc:
            raise IncentiveCompositionShadowError(
                f"discovery fixture state is invalid: {exc}"
            ) from None
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain-incentive-composition-shadow.synthetic-discovery-state-fixture",
            self.to_dict(),
        )


@dataclass(frozen=True)
class ShadowClassAllocation:
    claim_digest: str
    hotkey: str
    units: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "claim_digest", _digest(self.claim_digest, "allocation claim_digest")
        )
        if (
            not isinstance(self.hotkey, str)
            or not self.hotkey
            or len(self.hotkey) > 256
            or any(char.isspace() for char in self.hotkey)
        ):
            raise IncentiveCompositionShadowError("allocation hotkey is malformed")
        _integer(self.units, "allocation units", minimum=1, maximum=PPM)

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_digest": self.claim_digest,
            "hotkey": self.hotkey,
            "units": self.units,
        }


@dataclass(frozen=True)
class ShadowClassRecipient:
    hotkey: str
    uid: int
    units: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hotkey, str)
            or not self.hotkey
            or len(self.hotkey) > 256
            or any(char.isspace() for char in self.hotkey)
        ):
            raise IncentiveCompositionShadowError("class recipient hotkey is malformed")
        _integer(self.uid, "class recipient uid")
        _integer(self.units, "class recipient units", minimum=1, maximum=PPM)

    def to_dict(self) -> dict[str, object]:
        return {"hotkey": self.hotkey, "units": self.units, "uid": self.uid}


@dataclass(frozen=True)
class ChainIncentiveCompositionShadowReceipt:
    authority: ShadowChainAuthority
    core_policy_digest: str
    core_policy_file_sha256: str
    core_fixture_digest: str
    core_fixture_file_sha256: str
    core_claims_count: int
    discovery_policy_digest: str
    discovery_policy_file_sha256: str
    discovery_fixture_digest: str
    discovery_fixture_file_sha256: str
    discovery_claims_count: int
    composed_projection_digest: str
    discovery_capacity_units: int
    discovery_remaining_units: int
    discovery_payout_units: int
    core_capacity_units: int
    core_remaining_units: int
    core_payout_units: int
    discovery_allocations: tuple[ShadowClassAllocation, ...]
    core_allocations: tuple[ShadowClassAllocation, ...]
    discovery_recipients: tuple[ShadowClassRecipient, ...]
    core_recipients: tuple[ShadowClassRecipient, ...]
    miners: tuple[ShadowRecipient, ...]
    reserve: ShadowRecipient
    schema_version: int = COMPOSED_SHADOW_SCHEMA_VERSION
    receipt_version: str = COMPOSED_SHADOW_RECEIPT_VERSION
    mode: str = SYNTHETIC_FIXTURE_KIND
    submitted: bool = False

    def __post_init__(self) -> None:
        if type(self.authority) is not ShadowChainAuthority:
            raise IncentiveCompositionShadowError(
                "composed shadow authority is not exactly typed"
            )
        for field in (
            "core_policy_digest",
            "core_policy_file_sha256",
            "core_fixture_digest",
            "core_fixture_file_sha256",
            "discovery_policy_digest",
            "discovery_policy_file_sha256",
            "discovery_fixture_digest",
            "discovery_fixture_file_sha256",
            "composed_projection_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        _integer(self.core_claims_count, "core_claims_count")
        _integer(self.discovery_claims_count, "discovery_claims_count")
        for field in (
            "discovery_capacity_units",
            "discovery_payout_units",
            "core_capacity_units",
            "core_payout_units",
        ):
            _integer(getattr(self, field), field, maximum=PPM)
        _integer(self.discovery_remaining_units, "discovery_remaining_units")
        _integer(self.core_remaining_units, "core_remaining_units")
        if (
            self.discovery_payout_units
            != min(self.discovery_capacity_units, self.discovery_remaining_units)
            or self.core_payout_units
            != min(self.core_capacity_units, self.core_remaining_units)
        ):
            raise IncentiveCompositionShadowError(
                "class payouts differ from capacity and remaining principal"
            )

        discovery_allocations = tuple(self.discovery_allocations)
        core_allocations = tuple(self.core_allocations)
        discovery_recipients = tuple(self.discovery_recipients)
        core_recipients = tuple(self.core_recipients)
        miners = tuple(self.miners)
        for label, rows in (
            ("discovery", discovery_allocations),
            ("core", core_allocations),
        ):
            if (
                any(type(row) is not ShadowClassAllocation for row in rows)
                or tuple(row.claim_digest for row in rows)
                != tuple(sorted({row.claim_digest for row in rows}))
            ):
                raise IncentiveCompositionShadowError(
                    f"{label} allocations are not unique and canonical"
                )
        for label, rows in (
            ("discovery", discovery_recipients),
            ("core", core_recipients),
        ):
            if (
                any(type(row) is not ShadowClassRecipient for row in rows)
                or tuple(row.hotkey for row in rows)
                != tuple(sorted({row.hotkey for row in rows}))
            ):
                raise IncentiveCompositionShadowError(
                    f"{label} recipients are not unique and canonical"
                )
        if (
            any(type(row) is not ShadowRecipient or row.ppm <= 0 for row in miners)
            or tuple(row.hotkey for row in miners)
            != tuple(sorted({row.hotkey for row in miners}))
            or type(self.reserve) is not ShadowRecipient
        ):
            raise IncentiveCompositionShadowError(
                "composed miner/reserve rows are not canonical"
            )

        def allocated_by_hotkey(
            rows: tuple[ShadowClassAllocation, ...],
        ) -> dict[str, int]:
            result: dict[str, int] = {}
            for row in rows:
                result[row.hotkey] = result.get(row.hotkey, 0) + row.units
            return result

        discovery_by_hotkey = allocated_by_hotkey(discovery_allocations)
        core_by_hotkey = allocated_by_hotkey(core_allocations)
        if (
            sum(row.units for row in discovery_allocations)
            != self.discovery_payout_units
            or sum(row.units for row in core_allocations) != self.core_payout_units
            or {row.hotkey: row.units for row in discovery_recipients}
            != discovery_by_hotkey
            or {row.hotkey: row.units for row in core_recipients} != core_by_hotkey
        ):
            raise IncentiveCompositionShadowError(
                "class allocations and recipients do not conserve payouts"
            )
        combined: dict[str, int] = dict(discovery_by_hotkey)
        for hotkey, units in core_by_hotkey.items():
            combined[hotkey] = combined.get(hotkey, 0) + units
        if (
            self.reserve.hotkey in combined
            or {row.hotkey: row.ppm for row in miners} != combined
            or len({row.uid for row in (*miners, self.reserve)}) != len(miners) + 1
            or sum(combined.values()) + self.reserve.ppm != PPM
            or self.discovery_payout_units
            + self.core_payout_units
            + self.reserve.ppm
            != PPM
        ):
            raise IncentiveCompositionShadowError(
                "combined class projection is not uniquely mapped or conserved"
            )

        object.__setattr__(self, "discovery_allocations", discovery_allocations)
        object.__setattr__(self, "core_allocations", core_allocations)
        object.__setattr__(self, "discovery_recipients", discovery_recipients)
        object.__setattr__(self, "core_recipients", core_recipients)
        object.__setattr__(self, "miners", miners)
        if self.schema_version != COMPOSED_SHADOW_SCHEMA_VERSION:
            raise IncentiveCompositionShadowError(
                "composed shadow receipt schema is unsupported"
            )
        if self.receipt_version != COMPOSED_SHADOW_RECEIPT_VERSION:
            raise IncentiveCompositionShadowError(
                "composed shadow receipt version is unsupported"
            )
        if self.mode != SYNTHETIC_FIXTURE_KIND:
            raise IncentiveCompositionShadowError(
                "composed shadow receipt mode is not synthetic"
            )
        if self.submitted is not False:
            raise IncentiveCompositionShadowError(
                "composed shadow receipt submitted must be false"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "authority": self.authority.to_dict(),
            "inputs": {
                "core": {
                    "claims_count": self.core_claims_count,
                    "fixture_digest": self.core_fixture_digest,
                    "fixture_file_sha256": self.core_fixture_file_sha256,
                    "policy_digest": self.core_policy_digest,
                    "policy_file_sha256": self.core_policy_file_sha256,
                },
                "discovery": {
                    "claims_count": self.discovery_claims_count,
                    "fixture_digest": self.discovery_fixture_digest,
                    "fixture_file_sha256": self.discovery_fixture_file_sha256,
                    "policy_digest": self.discovery_policy_digest,
                    "policy_file_sha256": self.discovery_policy_file_sha256,
                },
            },
            "mode": self.mode,
            "non_authority": {
                "core_claims_source": "synthetic_fixture",
                "discovery_claims_source": "synthetic_fixture",
                "publication_authority": "none",
                "review_authority": "none",
                "settlement_authority": "none",
            },
            "projection": {
                "classes": {
                    "registered_crown": {
                        "allocations": [
                            row.to_dict() for row in self.core_allocations
                        ],
                        "capacity_units": self.core_capacity_units,
                        "payout_units": self.core_payout_units,
                        "recipients": [
                            row.to_dict() for row in self.core_recipients
                        ],
                        "remaining_units": self.core_remaining_units,
                    },
                    "reviewed_discovery": {
                        "allocations": [
                            row.to_dict() for row in self.discovery_allocations
                        ],
                        "capacity_units": self.discovery_capacity_units,
                        "payout_units": self.discovery_payout_units,
                        "recipients": [
                            row.to_dict() for row in self.discovery_recipients
                        ],
                        "remaining_units": self.discovery_remaining_units,
                    },
                },
                "composed_projection_digest": self.composed_projection_digest,
                "effective_block": self.authority.finalized_block,
                "miners": [row.to_dict() for row in self.miners],
                "payout_units": self.discovery_payout_units
                + self.core_payout_units,
                "reserve": self.reserve.to_dict(),
                "total_ppm": PPM,
            },
            "receipt_version": self.receipt_version,
            "schema_version": self.schema_version,
            "submitted": self.submitted,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.chain-incentive-composition-shadow.receipt",
            self.to_dict(),
        )

    def envelope(self) -> dict[str, object]:
        return {"receipt": self.to_dict(), "receipt_digest": self.digest}


@dataclass(frozen=True)
class _ComposedShadowInputs:
    core_policy: FiniteDebtPolicyManifest
    core_policy_file_sha256: str
    core_fixture: SyntheticClaimStateFixture
    core_fixture_file_sha256: str
    discovery_policy: IncentiveCompositionPolicyManifest
    discovery_policy_file_sha256: str
    discovery_fixture: SyntheticDiscoveryStateFixture
    discovery_fixture_file_sha256: str


def load_composed_shadow_inputs(
    *,
    core_policy_path: str | os.PathLike[str],
    core_claims_fixture_path: str | os.PathLike[str],
    discovery_policy_path: str | os.PathLike[str],
    discovery_claims_fixture_path: str | os.PathLike[str],
    expected_core_policy_digest: str,
    expected_core_claims_digest: str,
    expected_discovery_policy_digest: str,
    expected_discovery_claims_digest: str,
) -> _ComposedShadowInputs:
    """Reopen and semantically pin all four synthetic composition inputs."""

    expected_core_policy = _digest(
        expected_core_policy_digest, "expected core policy digest"
    )
    expected_core_claims = _digest(
        expected_core_claims_digest, "expected core claims digest"
    )
    expected_discovery_policy = _digest(
        expected_discovery_policy_digest, "expected discovery policy digest"
    )
    expected_discovery_claims = _digest(
        expected_discovery_claims_digest, "expected discovery claims digest"
    )

    core_policy_value, core_policy_file_sha256 = _read_canonical_json(
        core_policy_path, label="core finite-debt policy"
    )
    try:
        core_policy = FiniteDebtPolicyManifest.from_dict(core_policy_value)
    except ValueError as exc:
        raise IncentiveCompositionShadowError(
            f"core finite-debt policy is invalid: {exc}"
        ) from None
    if core_policy.digest != expected_core_policy:
        raise IncentiveCompositionShadowError("core policy semantic digest differs")

    core_fixture_value, core_fixture_file_sha256 = _read_canonical_json(
        core_claims_fixture_path, label="synthetic core claim-state fixture"
    )
    try:
        core_fixture = SyntheticClaimStateFixture.from_dict(core_fixture_value)
    except ValueError as exc:
        raise IncentiveCompositionShadowError(
            f"core claims fixture is invalid: {exc}"
        ) from None
    if core_fixture.digest != expected_core_claims:
        raise IncentiveCompositionShadowError("core claims semantic digest differs")
    if core_fixture.policy_digest != core_policy.digest:
        raise IncentiveCompositionShadowError(
            "synthetic core claims fixture differs from its policy"
        )

    discovery_policy_value, discovery_policy_file_sha256 = _read_canonical_json(
        discovery_policy_path, label="discovery composition policy"
    )
    try:
        discovery_policy = IncentiveCompositionPolicyManifest.from_dict(
            discovery_policy_value
        )
    except ValueError as exc:
        raise IncentiveCompositionShadowError(
            f"discovery composition policy is invalid: {exc}"
        ) from None
    if discovery_policy.digest != expected_discovery_policy:
        raise IncentiveCompositionShadowError(
            "discovery policy semantic digest differs"
        )
    if discovery_policy.innovation_policy_digest != core_policy.digest:
        raise IncentiveCompositionShadowError(
            "discovery composition policy differs from the core policy"
        )
    try:
        discovery_policy.validate_innovation_policy(core_policy)
    except ValueError as exc:
        raise IncentiveCompositionShadowError(
            f"discovery composition policy differs from the core policy: {exc}"
        ) from None

    discovery_fixture_value, discovery_fixture_file_sha256 = _read_canonical_json(
        discovery_claims_fixture_path,
        label="synthetic discovery claim-state fixture",
    )
    discovery_fixture = SyntheticDiscoveryStateFixture.from_dict(
        discovery_fixture_value
    )
    if discovery_fixture.digest != expected_discovery_claims:
        raise IncentiveCompositionShadowError(
            "discovery claims semantic digest differs"
        )
    if discovery_fixture.policy_digest != discovery_policy.digest:
        raise IncentiveCompositionShadowError(
            "synthetic discovery claims fixture differs from its policy"
        )

    return _ComposedShadowInputs(
        core_policy,
        core_policy_file_sha256,
        core_fixture,
        core_fixture_file_sha256,
        discovery_policy,
        discovery_policy_file_sha256,
        discovery_fixture,
        discovery_fixture_file_sha256,
    )


def _aggregate_class_recipients(
    allocations: tuple[ShadowClassAllocation, ...],
    uid_by_hotkey: dict[str, int],
    *,
    label: str,
) -> tuple[ShadowClassRecipient, ...]:
    amounts: dict[str, int] = {}
    for row in allocations:
        uid = uid_by_hotkey.get(row.hotkey)
        if type(uid) is not int:
            raise IncentiveCompositionShadowError(
                f"positive {label} recipient is absent from the finalized metagraph"
            )
        amounts[row.hotkey] = amounts.get(row.hotkey, 0) + row.units
    return tuple(
        ShadowClassRecipient(hotkey, uid_by_hotkey[hotkey], units)
        for hotkey, units in sorted(amounts.items())
    )


def _build_composed_receipt(
    *,
    inputs: _ComposedShadowInputs,
    netuid: int,
    genesis_hash: str,
    finalized_block: int,
    finalized_block_hash: str,
    metagraph: dict[str, object],
) -> ChainIncentiveCompositionShadowReceipt:
    if (
        metagraph["netuid"] != netuid
        or metagraph["block"] != finalized_block
        or metagraph["block_hash"] != finalized_block_hash
    ):
        raise IncentiveCompositionShadowError(
            "metagraph does not match the exact finalized composed-shadow authority"
        )

    for state in inputs.core_fixture.claim_states:
        terminal = state.balance.terminal_block
        if (
            state.claim.accepted_crown_block > finalized_block
            or state.claim.settlement_block > finalized_block
            or (terminal is not None and terminal > finalized_block)
        ):
            raise IncentiveCompositionShadowError(
                "synthetic core state contains future chain authority"
            )
    for state in inputs.discovery_fixture.discovery_states:
        terminal = state.balance.terminal_block
        if state.claim.awarded_block > finalized_block or (
            terminal is not None and terminal > finalized_block
        ):
            raise IncentiveCompositionShadowError(
                "synthetic discovery state contains future chain authority"
            )

    try:
        projection = project_composed_epoch(
            inputs.core_policy,
            inputs.discovery_policy,
            effective_block=finalized_block,
            innovation_states=inputs.core_fixture.claim_states,
            discovery_states=inputs.discovery_fixture.discovery_states,
        )
    except ValueError as exc:
        raise IncentiveCompositionShadowError(
            f"composed incentive projection failed: {exc}"
        ) from None

    hotkeys = metagraph["hotkeys"]
    uids = metagraph["uids"]
    assert isinstance(hotkeys, list) and isinstance(uids, list)
    uid_by_hotkey = dict(zip(hotkeys, uids, strict=True))
    reserve_hotkey = inputs.core_policy.reserve_hotkey
    reserve_uid = uid_by_hotkey.get(reserve_hotkey)
    if type(reserve_uid) is not int:
        raise IncentiveCompositionShadowError(
            "composition reserve hotkey is absent from the finalized metagraph"
        )

    discovery_allocations = tuple(
        ShadowClassAllocation(row.claim_digest, row.hotkey, row.units)
        for row in projection.discovery_allocations
    )
    core_allocations = tuple(
        ShadowClassAllocation(row.claim_digest, row.hotkey, row.units)
        for row in projection.innovation_allocations
    )
    discovery_recipients = _aggregate_class_recipients(
        discovery_allocations,
        uid_by_hotkey,
        label="reviewed-discovery",
    )
    core_recipients = _aggregate_class_recipients(
        core_allocations,
        uid_by_hotkey,
        label="registered-CROWN",
    )

    miner_rows: list[ShadowRecipient] = []
    for weight in projection.weights:
        if weight.hotkey == reserve_hotkey:
            continue
        uid = uid_by_hotkey.get(weight.hotkey)
        if type(uid) is not int:
            raise IncentiveCompositionShadowError(
                "positive composed miner is absent from the finalized metagraph"
            )
        miner_rows.append(ShadowRecipient(weight.hotkey, uid, weight.units))
    miners = tuple(sorted(miner_rows, key=lambda row: row.hotkey))
    reserve = ShadowRecipient(reserve_hotkey, reserve_uid, projection.reserve_units)

    expected_weights = {row.hotkey: row.ppm for row in miners}
    expected_weights[reserve.hotkey] = reserve.ppm
    if {row.hotkey: row.units for row in projection.weights} != expected_weights:
        raise IncentiveCompositionShadowError(
            "projection weights differ from exact finalized recipient mapping"
        )

    authority = ShadowChainAuthority(
        genesis_hash,
        netuid,
        finalized_block,
        finalized_block_hash,
        canonical_digest(
            "optima.chain-incentive-composition-shadow.metagraph", metagraph
        ),
        len(hotkeys),
    )
    return ChainIncentiveCompositionShadowReceipt(
        authority,
        inputs.core_policy.digest,
        inputs.core_policy_file_sha256,
        inputs.core_fixture.digest,
        inputs.core_fixture_file_sha256,
        len(inputs.core_fixture.claim_states),
        inputs.discovery_policy.digest,
        inputs.discovery_policy_file_sha256,
        inputs.discovery_fixture.digest,
        inputs.discovery_fixture_file_sha256,
        len(inputs.discovery_fixture.discovery_states),
        projection.digest,
        projection.discovery_capacity_units,
        projection.discovery_total_remaining_units,
        projection.discovery_payout_units,
        projection.innovation_capacity_units,
        projection.innovation_total_remaining_units,
        projection.innovation_payout_units,
        discovery_allocations,
        core_allocations,
        discovery_recipients,
        core_recipients,
        miners,
        reserve,
    )


def _write_receipt_exclusive(
    path: str | os.PathLike[str], envelope: dict[str, object]
) -> Path:
    """Exclusively write and durably sync one canonical composed receipt."""

    output = _assert_output_available(path)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise IncentiveCompositionShadowError(
            "composed shadow receipt writing requires O_NOFOLLOW support"
        )
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | nofollow
        | getattr(os, "O_CLOEXEC", 0)
    )
    raw = canonical_json_bytes(envelope) + b"\n"
    fd: int | None = None
    created_identity: tuple[int, int] | None = None
    failure: IncentiveCompositionShadowError | None = None
    try:
        fd = os.open(output, flags, 0o600)
        info = os.fstat(fd)
        created_identity = (info.st_dev, info.st_ino)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise IncentiveCompositionShadowError(
                    "composed shadow receipt write stalled"
                )
            view = view[written:]
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    except FileExistsError:
        raise IncentiveCompositionShadowError(
            "composed shadow output already exists"
        ) from None
    except IncentiveCompositionShadowError as exc:
        failure = exc
    except OSError as exc:
        failure = IncentiveCompositionShadowError(
            f"cannot write composed shadow receipt: {exc}"
        )
    finally:
        if fd is not None:
            os.close(fd)
    if failure is not None:
        if created_identity is not None:
            try:
                current = os.lstat(output)
                if (current.st_dev, current.st_ino) == created_identity:
                    os.unlink(output)
            except OSError:
                pass
        raise failure from None
    try:
        dir_flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
        dir_flags |= getattr(os, "O_DIRECTORY", 0)
        parent_fd = os.open(output.parent, dir_flags)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError as exc:
        if created_identity is not None:
            try:
                current = os.lstat(output)
                if (current.st_dev, current.st_ino) == created_identity:
                    os.unlink(output)
            except OSError:
                pass
        raise IncentiveCompositionShadowError(
            f"cannot durably sync composed shadow receipt directory: {exc}"
        ) from None
    return output


def execute_chain_incentive_composition_shadow(
    *,
    network: str,
    netuid: int,
    core_policy_path: str | os.PathLike[str],
    core_claims_fixture_path: str | os.PathLike[str],
    discovery_policy_path: str | os.PathLike[str],
    discovery_claims_fixture_path: str | os.PathLike[str],
    expected_core_policy_digest: str,
    expected_core_claims_digest: str,
    expected_discovery_policy_digest: str,
    expected_discovery_claims_digest: str,
    output_path: str | os.PathLike[str],
    connect: Callable[[str], object],
    read_finalized_head: Callable[[object], tuple[int, str]],
    fetch_metagraph: Callable[..., object],
) -> ChainIncentiveCompositionShadowReceipt:
    """Project both synthetic classes against twice-reopened chain authority."""

    if not isinstance(network, str) or not network:
        raise IncentiveCompositionShadowError("network selector must be nonempty")
    selected_netuid = _integer(netuid, "netuid")
    inputs = load_composed_shadow_inputs(
        core_policy_path=core_policy_path,
        core_claims_fixture_path=core_claims_fixture_path,
        discovery_policy_path=discovery_policy_path,
        discovery_claims_fixture_path=discovery_claims_fixture_path,
        expected_core_policy_digest=expected_core_policy_digest,
        expected_core_claims_digest=expected_core_claims_digest,
        expected_discovery_policy_digest=expected_discovery_policy_digest,
        expected_discovery_claims_digest=expected_discovery_claims_digest,
    )
    output = _assert_output_available(output_path)
    try:
        subtensor = connect(network)
    except Exception as exc:
        raise IncentiveCompositionShadowError(
            f"cannot connect read-only chain client: {exc}"
        ) from None

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
    receipt = _build_composed_receipt(
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
        raise IncentiveCompositionShadowError(
            "finalized chain authority regressed while composed projection ran"
        )
    if reopened_block == finalized_block and reopened_hash != finalized_hash:
        raise IncentiveCompositionShadowError(
            "finalized head changed at the retained composed-shadow height"
        )
    if _read_chain_hash(
        subtensor,
        finalized_block,
        "reopened retained block hash",
    ) != finalized_hash:
        raise IncentiveCompositionShadowError(
            "retained finalized block hash changed while composed projection ran"
        )
    _reopened_view, reopened_metagraph = _fetch_exact_metagraph(
        subtensor,
        netuid=selected_netuid,
        block=finalized_block,
        fetch_metagraph=fetch_metagraph,
    )
    if reopened_metagraph != metagraph:
        raise IncentiveCompositionShadowError(
            "finalized metagraph authority changed while composed projection ran"
        )
    _write_receipt_exclusive(output, receipt.envelope())
    return receipt


__all__ = [
    "COMPOSED_SHADOW_RECEIPT_VERSION",
    "COMPOSED_SHADOW_SCHEMA_VERSION",
    "ChainIncentiveCompositionShadowReceipt",
    "IncentiveCompositionShadowError",
    "ShadowClassAllocation",
    "ShadowClassRecipient",
    "SyntheticDiscoveryStateFixture",
    "execute_chain_incentive_composition_shadow",
    "load_composed_shadow_inputs",
]
