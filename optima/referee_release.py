"""Approved referee release identity.

This file is intentionally excluded from ``arenas.referee_source_digest`` so the
expected digest can attest the rest of the package without a self-referential hash.
Release tooling updates this value after the referee source tree is frozen.  The
serving image must also be pinned by digest; this check catches accidental/mixed
checkouts before an arena can produce a qualification report.
"""

APPROVED_REFEREE_SOURCE_DIGEST = (
    "sha256:0000000000000000000000000000000000000000000000000000000000000000"
)

# Full sanitized release identity: covers optima/, optima_kernels/, approved
# package data, and the canonical release manifest policy. Freeze alongside the
# source digest only after the release tree and outer OCI protocol stop moving.
APPROVED_REFEREE_TREE_DIGEST = (
    "sha256:0000000000000000000000000000000000000000000000000000000000000000"
)
