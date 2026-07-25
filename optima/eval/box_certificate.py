"""Sealed instrument certification for one-shot speed verdicts.

The certificate inverts the month's structural mistake: instead of every
verdict re-proving that the box can measure, the box proves it once per
session — stock-vs-stock null brackets at every scored workload shape plus a
known-magnitude honest delta the instrument must resolve — and every
subsequent submission gets exactly one bracket per shape under that sealed
authority. A read that blows its certified scatter bound afterwards
invalidates the CERTIFICATE, never the candidate: the miner does not eat
instrument failures, and nobody holds a retry lever to shop draws with.

Certificates are validator-side operational authority. They gate whether a
program may launch and whether a NO_DECISION names the instrument; they do
not participate in candidate verdict arithmetic, which stays sealed inside
the resident speed policy and its calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

from optima.eval.scoring import relative_spread
from optima.stack_identity import canonical_digest, require_sha256_hex

_LABEL_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_.-")

# A certificate may never promise a floor looser than the referee mandate:
# baseline agreement <= 0.3% under the full protocol, or nothing runs.
MAX_CERTIFIED_FLOOR = 0.003
MAX_CERTIFIED_SCATTER = 0.05
MAX_CERTIFICATE_AGE_SECONDS = 86_400
MAX_CERTIFICATE_VERDICTS = 64


class BoxCertificateError(RuntimeError):
    pass


def _finite_fraction(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) < 1
    ):
        raise BoxCertificateError(f"{field} is not a fraction in [0, 1)")
    return float(value)


def _digest(value: object, *, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise BoxCertificateError(str(exc)) from None


@dataclass(frozen=True)
class NullFloorReceipt:
    """One stock-vs-stock null bracket at one scored workload shape.

    ``scored_rates`` are the per-read scored values (v3: window medians) in
    run order; ``read_scatters`` are each read's own window scatter. The
    floor is the relative spread of the scored rates and must recompute."""

    workload_digest: str
    scored_rates: tuple[float, ...]
    read_scatters: tuple[float, ...]
    floor: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workload_digest",
            _digest(self.workload_digest, field="null workload_digest"),
        )
        rates = tuple(self.scored_rates)
        scatters = tuple(self.read_scatters)
        object.__setattr__(self, "scored_rates", rates)
        object.__setattr__(self, "read_scatters", scatters)
        if (
            len(rates) < 3
            or len(scatters) != len(rates)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
                for value in rates
            )
        ):
            raise BoxCertificateError("null bracket rates are malformed")
        for value in scatters:
            _finite_fraction(value, field="null read scatter")
        if (
            type(self.floor) is not float
            or self.floor != relative_spread(list(rates))
        ):
            raise BoxCertificateError("null floor does not recompute")

    def to_dict(self) -> dict[str, object]:
        return {
            "floor": format(self.floor, ".17g"),
            "read_scatters": [
                format(value, ".17g") for value in self.read_scatters
            ],
            "scored_rates": [
                format(value, ".17g") for value in self.scored_rates
            ],
            "workload_digest": self.workload_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NullFloorReceipt":
        fields = {"floor", "read_scatters", "scored_rates", "workload_digest"}
        if (
            type(value) is not dict
            or set(value) != fields
            or type(value["read_scatters"]) is not list
            or type(value["scored_rates"]) is not list
        ):
            raise BoxCertificateError("null receipt fields differ")
        try:
            result = cls(
                value["workload_digest"],  # type: ignore[arg-type]
                tuple(float(row) for row in value["scored_rates"]),
                tuple(float(row) for row in value["read_scatters"]),
                float(value["floor"]),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise BoxCertificateError("null receipt is malformed") from exc
        if result.to_dict() != value:
            raise BoxCertificateError("null receipt is noncanonical")
        return result


@dataclass(frozen=True)
class KnownEffectReceipt:
    """A known-magnitude honest delta the instrument resolved.

    A quiet floor alone does not prove the instrument measures; it could be
    quietly wrong. The certification bracket must also detect a delta whose
    magnitude is independently established (for example a vendor fusion
    flag), within a declared tolerance that cannot swallow the effect."""

    workload_digest: str
    label: str
    expected_effect: float
    tolerance: float
    measured_effect: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workload_digest",
            _digest(self.workload_digest, field="effect workload_digest"),
        )
        if (
            not isinstance(self.label, str)
            or not 1 <= len(self.label) <= 64
            or not set(self.label) <= _LABEL_CHARS
        ):
            raise BoxCertificateError("known-effect label is malformed")
        expected = _finite_fraction(self.expected_effect, field="expected_effect")
        tolerance = _finite_fraction(self.tolerance, field="tolerance")
        if (
            type(self.measured_effect) is not float
            or not math.isfinite(self.measured_effect)
        ):
            raise BoxCertificateError("measured_effect is not finite")
        if not 0 < tolerance < expected:
            raise BoxCertificateError(
                "known-effect tolerance must be positive and below the effect"
            )
        if abs(self.measured_effect - expected) > tolerance:
            raise BoxCertificateError(
                "known effect was not resolved within its tolerance"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "expected_effect": format(self.expected_effect, ".17g"),
            "label": self.label,
            "measured_effect": format(self.measured_effect, ".17g"),
            "tolerance": format(self.tolerance, ".17g"),
            "workload_digest": self.workload_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "KnownEffectReceipt":
        fields = {
            "expected_effect",
            "label",
            "measured_effect",
            "tolerance",
            "workload_digest",
        }
        if type(value) is not dict or set(value) != fields:
            raise BoxCertificateError("known-effect receipt fields differ")
        try:
            result = cls(
                value["workload_digest"],  # type: ignore[arg-type]
                value["label"],  # type: ignore[arg-type]
                float(value["expected_effect"]),  # type: ignore[arg-type]
                float(value["tolerance"]),  # type: ignore[arg-type]
                float(value["measured_effect"]),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise BoxCertificateError("known-effect receipt is malformed") from exc
        if result.to_dict() != value:
            raise BoxCertificateError("known-effect receipt is noncanonical")
        return result


@dataclass(frozen=True)
class BoxCertificate:
    """The sealed once-per-session instrument authority.

    Expiry is double-bounded: wall-clock age (the floor deteriorated
    monotonically over hours on 2026-07-25) and verdict count (each verdict
    consumes trust in the certification, none refresh it)."""

    logical_hardware_digest: str
    controller_distribution_digest: str
    resident_policy_digest: str
    floor_bound: float
    scatter_bound: float
    issued_unix_s: float
    max_age_seconds: int
    max_verdicts: int
    nulls: tuple[NullFloorReceipt, ...]
    known_effects: tuple[KnownEffectReceipt, ...]

    def __post_init__(self) -> None:
        for field in (
            "logical_hardware_digest",
            "controller_distribution_digest",
            "resident_policy_digest",
        ):
            object.__setattr__(
                self, field, _digest(getattr(self, field), field=field)
            )
        floor_bound = _finite_fraction(self.floor_bound, field="floor_bound")
        scatter_bound = _finite_fraction(self.scatter_bound, field="scatter_bound")
        if not 0 < floor_bound <= MAX_CERTIFIED_FLOOR:
            raise BoxCertificateError(
                "certificate floor bound exceeds the referee mandate"
            )
        if not 0 < scatter_bound <= MAX_CERTIFIED_SCATTER:
            raise BoxCertificateError("certificate scatter bound is unsupported")
        if (
            type(self.issued_unix_s) is not float
            or not math.isfinite(self.issued_unix_s)
            or self.issued_unix_s <= 0
            or type(self.max_age_seconds) is not int
            or not 600 <= self.max_age_seconds <= MAX_CERTIFICATE_AGE_SECONDS
            or type(self.max_verdicts) is not int
            or not 1 <= self.max_verdicts <= MAX_CERTIFICATE_VERDICTS
        ):
            raise BoxCertificateError("certificate lifetime is malformed")
        nulls = tuple(self.nulls)
        effects = tuple(self.known_effects)
        object.__setattr__(self, "nulls", nulls)
        object.__setattr__(self, "known_effects", effects)
        if (
            not nulls
            or any(type(row) is not NullFloorReceipt for row in nulls)
            or len({row.workload_digest for row in nulls}) != len(nulls)
        ):
            raise BoxCertificateError(
                "certificate requires one null bracket per workload shape"
            )
        measured_floors = {row.workload_digest: row.floor for row in nulls}
        if any(
            row.floor > floor_bound
            or any(scatter > scatter_bound for scatter in row.read_scatters)
            for row in nulls
        ):
            raise BoxCertificateError(
                "null evidence does not support the certified bounds"
            )
        if (
            not effects
            or any(type(row) is not KnownEffectReceipt for row in effects)
            or any(row.workload_digest not in measured_floors for row in effects)
        ):
            raise BoxCertificateError(
                "certificate requires known-effect evidence at a certified shape"
            )
        # Detectability binds evidence to evidence: the resolved effect must
        # stand clear of the floor this session actually measured at that
        # shape, not of the acceptance ceiling. A box whose measured floor
        # crowds the effect has not demonstrated it can resolve one.
        if any(
            row.expected_effect <= 2 * measured_floors[row.workload_digest]
            for row in effects
        ):
            raise BoxCertificateError(
                "known effect is not distinguishable from the measured floor"
            )

    @property
    def certified_workloads(self) -> frozenset[str]:
        return frozenset(row.workload_digest for row in self.nulls)

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.qualification.box-certificate", self.to_dict()
        )

    def require_valid(
        self,
        *,
        resident_policy_digest: str,
        workload_digest: str,
        logical_hardware_digest: str,
        controller_distribution_digest: str,
        now_unix_s: float,
        verdicts_used: int,
    ) -> None:
        """Refuse, naming the cause, unless this certificate covers the
        requested measurement right now. Every refusal is infra state — a
        certificate problem never grades a candidate."""

        if (
            self.resident_policy_digest != resident_policy_digest
            or self.logical_hardware_digest != logical_hardware_digest
            or self.controller_distribution_digest
            != controller_distribution_digest
        ):
            raise BoxCertificateError(
                "certificate identity differs from the requested measurement"
            )
        if workload_digest not in self.certified_workloads:
            raise BoxCertificateError(
                "workload shape is not certified by this certificate"
            )
        if (
            type(now_unix_s) is not float
            or not math.isfinite(now_unix_s)
            or now_unix_s < self.issued_unix_s
        ):
            raise BoxCertificateError("certificate clock input is invalid")
        if now_unix_s - self.issued_unix_s > self.max_age_seconds:
            raise BoxCertificateError("certificate has expired by wall clock")
        if type(verdicts_used) is not int or verdicts_used < 0:
            raise BoxCertificateError("certificate verdict accounting is invalid")
        if verdicts_used >= self.max_verdicts:
            raise BoxCertificateError("certificate verdict budget is exhausted")

    def to_dict(self) -> dict[str, object]:
        return {
            "controller_distribution_digest": self.controller_distribution_digest,
            "floor_bound": format(self.floor_bound, ".17g"),
            "issued_unix_s": format(self.issued_unix_s, ".17g"),
            "known_effects": [row.to_dict() for row in self.known_effects],
            "logical_hardware_digest": self.logical_hardware_digest,
            "max_age_seconds": self.max_age_seconds,
            "max_verdicts": self.max_verdicts,
            "nulls": [row.to_dict() for row in self.nulls],
            "resident_policy_digest": self.resident_policy_digest,
            "scatter_bound": format(self.scatter_bound, ".17g"),
        }

    @classmethod
    def from_dict(cls, value: object) -> "BoxCertificate":
        fields = {
            "controller_distribution_digest",
            "floor_bound",
            "issued_unix_s",
            "known_effects",
            "logical_hardware_digest",
            "max_age_seconds",
            "max_verdicts",
            "nulls",
            "resident_policy_digest",
            "scatter_bound",
        }
        if (
            type(value) is not dict
            or set(value) != fields
            or type(value["nulls"]) is not list
            or type(value["known_effects"]) is not list
        ):
            raise BoxCertificateError("certificate fields differ")
        try:
            result = cls(
                value["logical_hardware_digest"],  # type: ignore[arg-type]
                value["controller_distribution_digest"],  # type: ignore[arg-type]
                value["resident_policy_digest"],  # type: ignore[arg-type]
                float(value["floor_bound"]),  # type: ignore[arg-type]
                float(value["scatter_bound"]),  # type: ignore[arg-type]
                float(value["issued_unix_s"]),  # type: ignore[arg-type]
                value["max_age_seconds"],  # type: ignore[arg-type]
                value["max_verdicts"],  # type: ignore[arg-type]
                tuple(NullFloorReceipt.from_dict(row) for row in value["nulls"]),
                tuple(
                    KnownEffectReceipt.from_dict(row)
                    for row in value["known_effects"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise BoxCertificateError("certificate is malformed") from exc
        if result.to_dict() != value:
            raise BoxCertificateError("certificate is noncanonical")
        return result


@dataclass(frozen=True)
class CertificateInvalidation:
    """Typed record of an instrument failure observed under a certificate.

    Sealed alongside the NO_DECISION it explains: the read named here blew
    the certified scatter bound, so the certificate is void from
    ``occurred_unix_s`` and the instrument must re-certify before any
    further verdict. The candidate measured under it is unharmed."""

    certificate_digest: str
    workload_digest: str
    read_role: str
    observed_scatter: float
    scatter_bound: float
    occurred_unix_s: float

    def __post_init__(self) -> None:
        for field in ("certificate_digest", "workload_digest"):
            object.__setattr__(
                self, field, _digest(getattr(self, field), field=field)
            )
        if self.read_role not in {"B", "C", "B_prime", "C_prime", "B_double_prime"}:
            raise BoxCertificateError("invalidation read role is unsupported")
        observed = _finite_fraction(self.observed_scatter, field="observed_scatter")
        bound = _finite_fraction(self.scatter_bound, field="scatter_bound")
        if observed <= bound:
            raise BoxCertificateError(
                "invalidation requires scatter above the certified bound"
            )
        if (
            type(self.occurred_unix_s) is not float
            or not math.isfinite(self.occurred_unix_s)
            or self.occurred_unix_s <= 0
        ):
            raise BoxCertificateError("invalidation clock input is invalid")

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.qualification.box-certificate-invalidation", self.to_dict()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "certificate_digest": self.certificate_digest,
            "observed_scatter": format(self.observed_scatter, ".17g"),
            "occurred_unix_s": format(self.occurred_unix_s, ".17g"),
            "read_role": self.read_role,
            "scatter_bound": format(self.scatter_bound, ".17g"),
            "workload_digest": self.workload_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CertificateInvalidation":
        fields = {
            "certificate_digest",
            "observed_scatter",
            "occurred_unix_s",
            "read_role",
            "scatter_bound",
            "workload_digest",
        }
        if type(value) is not dict or set(value) != fields:
            raise BoxCertificateError("invalidation fields differ")
        try:
            result = cls(
                value["certificate_digest"],  # type: ignore[arg-type]
                value["workload_digest"],  # type: ignore[arg-type]
                value["read_role"],  # type: ignore[arg-type]
                float(value["observed_scatter"]),  # type: ignore[arg-type]
                float(value["scatter_bound"]),  # type: ignore[arg-type]
                float(value["occurred_unix_s"]),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise BoxCertificateError("invalidation is malformed") from exc
        if result.to_dict() != value:
            raise BoxCertificateError("invalidation is noncanonical")
        return result
