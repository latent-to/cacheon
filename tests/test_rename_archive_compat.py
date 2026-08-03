"""The immutable validator archive carries the Cacheon identity only."""

from __future__ import annotations

from cacheon.chain.archive import (
    ARCHIVE_SCHEMA,
    DEFAULT_VALIDATOR_ARCHIVE_PREFIX,
)


def test_validator_archive_identity_is_cacheon() -> None:
    assert ARCHIVE_SCHEMA == "cacheon.validator-archive.v1"
    assert DEFAULT_VALIDATOR_ARCHIVE_PREFIX == "cacheon/validator-archive/v1"
