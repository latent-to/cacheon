from __future__ import annotations

from dataclasses import replace

import pytest

from cacheon.eval.box_certificate import (
    BoxCertificate,
    BoxCertificateError,
    CertificateInvalidation,
    KnownEffectReceipt,
    NullFloorReceipt,
)
from cacheon.eval.scoring import relative_spread

_S1 = "a" * 64  # decode-heavy scored shape
_S2 = "b" * 64  # long-context scored shape


def _null(workload: str, rates: tuple[float, ...]) -> NullFloorReceipt:
    return NullFloorReceipt(
        workload,
        rates,
        tuple(0.002 for _ in rates),
        relative_spread(list(rates)),
    )


def _effect(workload: str = _S1) -> KnownEffectReceipt:
    return KnownEffectReceipt(workload, "vendor_ar_fusion_flag", 0.0038, 0.002, 0.0041)


def _certificate(**overrides) -> BoxCertificate:
    kwargs = {
        "logical_hardware_digest": "c" * 64,
        "controller_distribution_digest": "d" * 64,
        "resident_policy_digest": "e" * 64,
        "floor_bound": 0.003,
        "scatter_bound": 0.01,
        "issued_unix_s": 1_800_000_000.0,
        "max_age_seconds": 21_600,
        "max_verdicts": 8,
        "nulls": (
            _null(_S1, (5850.0, 5852.0, 5849.0)),
            _null(_S2, (2100.0, 2101.0, 2099.5)),
        ),
        "known_effects": (_effect(),),
    }
    kwargs.update(overrides)
    return BoxCertificate(**kwargs)


def test_certificate_round_trips_and_digest_is_stable() -> None:
    certificate = _certificate()
    assert BoxCertificate.from_dict(certificate.to_dict()) == certificate
    assert certificate.digest == BoxCertificate.from_dict(certificate.to_dict()).digest
    assert certificate.certified_workloads == {_S1, _S2}


def test_certificate_refuses_a_slack_floor_bound() -> None:
    # The <=0.3% mandate is a ceiling on the BOUND: no operator can issue a
    # certificate that promises a mushier instrument.
    with pytest.raises(BoxCertificateError, match="referee mandate"):
        _certificate(floor_bound=0.004)


def test_certificate_requires_null_evidence_under_its_own_bounds() -> None:
    # A null bracket whose spread exceeds the declared floor cannot certify.
    wide = _null(_S1, (5850.0, 5900.0, 5849.0))
    with pytest.raises(BoxCertificateError, match="does not support"):
        _certificate(nulls=(wide, _null(_S2, (2100.0, 2101.0, 2099.5))))
    # A read whose own window scatter exceeds the declared bound cannot either.
    blown = replace(_null(_S1, (5850.0, 5852.0, 5849.0)), read_scatters=(0.002, 0.02, 0.002))
    with pytest.raises(BoxCertificateError, match="does not support"):
        _certificate(nulls=(blown, _null(_S2, (2100.0, 2101.0, 2099.5))))


def test_certificate_requires_a_resolvable_known_effect() -> None:
    # The known effect must be measured within tolerance...
    with pytest.raises(BoxCertificateError, match="not resolved"):
        KnownEffectReceipt(_S1, "vendor_ar_fusion_flag", 0.0038, 0.002, 0.012)
    # ...the tolerance must not swallow the effect...
    with pytest.raises(BoxCertificateError, match="below the effect"):
        KnownEffectReceipt(_S1, "vendor_ar_fusion_flag", 0.0038, 0.0038, 0.0038)
    # ...and the effect must stand clear of the MEASURED floor at its shape:
    # a null spread of ~0.26% cannot certify detection of a 0.38% effect.
    crowded = _null(_S1, (5850.0, 5880.0, 5849.0))
    assert 2 * crowded.floor > 0.0038
    with pytest.raises(BoxCertificateError, match="not distinguishable"):
        _certificate(
            nulls=(crowded, _null(_S2, (2100.0, 2101.0, 2099.5))),
        )
    # A known effect at an uncertified shape proves nothing about the nulls.
    with pytest.raises(BoxCertificateError, match="certified shape"):
        _certificate(known_effects=(_effect("f" * 64),))


def test_certificate_validity_names_each_refusal_cause() -> None:
    certificate = _certificate()
    valid = dict(
        resident_policy_digest="e" * 64,
        workload_digest=_S1,
        logical_hardware_digest="c" * 64,
        controller_distribution_digest="d" * 64,
        now_unix_s=1_800_010_000.0,
        verdicts_used=0,
    )
    certificate.require_valid(**valid)
    with pytest.raises(BoxCertificateError, match="identity differs"):
        certificate.require_valid(**{**valid, "resident_policy_digest": "f" * 64})
    with pytest.raises(BoxCertificateError, match="not certified"):
        certificate.require_valid(**{**valid, "workload_digest": "f" * 64})
    with pytest.raises(BoxCertificateError, match="expired"):
        certificate.require_valid(**{**valid, "now_unix_s": 1_800_000_000.0 + 21_601})
    with pytest.raises(BoxCertificateError, match="exhausted"):
        certificate.require_valid(**{**valid, "verdicts_used": 8})
    with pytest.raises(BoxCertificateError, match="clock input"):
        certificate.require_valid(**{**valid, "now_unix_s": 1.0})


def test_null_floor_must_recompute() -> None:
    rates = (5850.0, 5852.0, 5849.0)
    with pytest.raises(BoxCertificateError, match="does not recompute"):
        NullFloorReceipt(_S1, rates, (0.002, 0.002, 0.002), 0.001)
    receipt = _null(_S1, rates)
    assert NullFloorReceipt.from_dict(receipt.to_dict()) == receipt
    with pytest.raises(BoxCertificateError, match="malformed"):
        NullFloorReceipt(_S1, (5850.0, 5852.0), (0.002, 0.002), 0.0)


def test_invalidation_names_the_instrument_failure() -> None:
    row = CertificateInvalidation(
        "9" * 64, _S1, "B_prime", 0.02, 0.01, 1_800_010_000.0
    )
    assert CertificateInvalidation.from_dict(row.to_dict()) == row
    assert row.digest
    # An in-bounds read is not an invalidation: the record cannot exist.
    with pytest.raises(BoxCertificateError, match="above the certified bound"):
        CertificateInvalidation("9" * 64, _S1, "B_prime", 0.005, 0.01, 1_800_010_000.0)
    with pytest.raises(BoxCertificateError, match="read role"):
        CertificateInvalidation("9" * 64, _S1, "D", 0.02, 0.01, 1_800_010_000.0)
