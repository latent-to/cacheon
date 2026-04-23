"""Fetch miner policy source from HuggingFace and cache it locally.

Pure function — no state mutation, no logging of auth tokens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


class FetchOutcome(str, Enum):
    OK = "ok"
    REJECTED = "rejected"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class FetchResult:
    outcome: FetchOutcome
    path: Path | None = None          # populated when OK
    reason: str | None = None         # populated when REJECTED/DEFERRED


def sanitize_repo(repo: str) -> str:
    """Turn a HF repo id into a safe single path component under ``cache_dir``.

    Replaces ``/`` with ``_`` so ``owner/name`` becomes ``owner_name``. Rejects
    values that could escape the cache layout (e.g. ``..``, ``.``, empty
    segments after normalization) so ``cache_dir / safe / revision`` cannot
    resolve outside ``cache_dir`` even if a caller bypasses on-chain parsing.

    Raises:
        ValueError: If the normalized name is not safe for use as a directory name.
    """
    parts = repo.strip().split("/")
    if not parts or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"invalid repo for cache path: {repo!r}")
    return "_".join(parts)


_HF_HUB_CALL_ERRORS: tuple[type[BaseException], ...] = (
    LocalEntryNotFoundError,
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    HfHubHTTPError,
    HFValidationError,
    OSError,
)


def _fetch_result_from_hf_hub_exc(exc: BaseException) -> FetchResult:
    """Map ``HfApi`` / ``hf_hub_download`` failures to a ``FetchResult``.

    Uses ``isinstance`` in specificity order so subclasses (e.g.
    ``GatedRepoError`` ⊂ ``RepositoryNotFoundError`` ⊂ ``HfHubHTTPError``)
    map correctly regardless of tuple order in ``except``.
    """
    if isinstance(exc, LocalEntryNotFoundError):
        # HF client can raise this on cache edge cases; treat as transient.
        return FetchResult(
            outcome=FetchOutcome.DEFERRED,
            reason=f"local_cache_miss ({exc})",
        )
    if isinstance(exc, GatedRepoError):
        return FetchResult(
            outcome=FetchOutcome.REJECTED,
            reason=f"fetch_forbidden ({exc})",
        )
    if isinstance(exc, (RepositoryNotFoundError, RevisionNotFoundError)):
        return FetchResult(
            outcome=FetchOutcome.REJECTED,
            reason=f"revision_unavailable ({exc})",
        )
    if isinstance(exc, EntryNotFoundError):
        return FetchResult(
            outcome=FetchOutcome.REJECTED,
            reason=f"policy_missing ({exc})",
        )
    if isinstance(exc, HFValidationError):
        return FetchResult(
            outcome=FetchOutcome.REJECTED,
            reason=f"invalid_repo ({exc})",
        )
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is None:
            # No response attached — treat as transient (e.g. connection reset
            # wrapped in HfHubHTTPError without a status).
            return FetchResult(
                outcome=FetchOutcome.DEFERRED,
                reason=f"hf_http_unknown ({exc})",
            )
        if status >= 500:
            return FetchResult(
                outcome=FetchOutcome.DEFERRED,
                reason=f"hf_http_{status} ({exc})",
            )
        if status == 429:
            return FetchResult(
                outcome=FetchOutcome.DEFERRED,
                reason=f"rate_limited ({exc})",
            )
        if status in (408, 425):
            # Request Timeout / Too Early — transient, safe to retry.
            return FetchResult(
                outcome=FetchOutcome.DEFERRED,
                reason=f"hf_http_{status} ({exc})",
            )
        if status in (401, 403):
            return FetchResult(
                outcome=FetchOutcome.REJECTED,
                reason=f"fetch_forbidden ({exc})",
            )
        if 400 <= status < 500:
            # Remaining 4xx (400 Bad Request, 410 Gone, 422, etc.) are the
            # miner's fault — won't get better on retry.
            return FetchResult(
                outcome=FetchOutcome.REJECTED,
                reason=f"hf_http_{status} ({exc})",
            )
        return FetchResult(
            outcome=FetchOutcome.DEFERRED,
            reason=f"hf_http_{status} ({exc})",
        )
    if isinstance(exc, OSError):
        return FetchResult(
            outcome=FetchOutcome.DEFERRED,
            reason=f"fetch_error ({type(exc).__name__}: {exc})",
        )
    raise exc


def fetch_policy_source(
    repo: str,
    revision: str,
    *,
    cache_dir: Path,
    max_bytes: int,
    etag_timeout_s: float,
    hf_token: str | None = None,
) -> FetchResult:
    """Download ``policy.py`` from a HuggingFace repo at a pinned revision.

    Args:
        repo: HF repo id, e.g. ``"hf-user/my-policy"``.
        revision: 40-character hex SHA.
        cache_dir: root directory for the on-disk cache.
        max_bytes: hard cap on file size.
        etag_timeout_s: timeout (seconds) for the HEAD / etag revalidation
            inside ``hf_hub_download``.  Does **not** cap the blob download.
        hf_token: optional HF access token.

    Returns:
        ``FetchResult`` with ``outcome`` set to ``OK``, ``REJECTED``, or
        ``DEFERRED`` and an appropriate ``reason``.
    """
    try:
        safe_repo = sanitize_repo(repo)
    except ValueError as exc:
        return FetchResult(
            outcome=FetchOutcome.REJECTED,
            reason=f"invalid_repo ({exc})",
        )
    revision_dir = cache_dir / safe_repo / revision
    cache_path = revision_dir / "policy.py"
    ok_sentinel = revision_dir / ".ok"

    # Cache hit — only valid if the sentinel exists, meaning the previous
    # download completed cleanly (SIGKILL during download leaves no sentinel).
    if cache_path.exists() and ok_sentinel.exists():
        size = cache_path.stat().st_size
        if size <= max_bytes:
            return FetchResult(outcome=FetchOutcome.OK, path=cache_path)
        # Cached file now exceeds cap (operator may have tightened it).
        # Invalidate and re-fetch rather than permanently rejecting.
        for stale in (cache_path, ok_sentinel):
            try:
                os.unlink(stale)
            except OSError:
                pass

    # Size check via HEAD metadata before downloading
    try:
        paths = HfApi().get_paths_info(
            repo_id=repo,
            paths=["policy.py"],
            revision=revision,
            token=hf_token,
        )
    except _HF_HUB_CALL_ERRORS as exc:
        return _fetch_result_from_hf_hub_exc(exc)

    if not paths:
        return FetchResult(
            outcome=FetchOutcome.REJECTED,
            reason="policy_missing (no paths returned)",
        )

    file_info = paths[0]
    if not hasattr(file_info, "size"):
        # Folder entry (RepoFolder) — policy.py exists as a directory, not a file.
        return FetchResult(
            outcome=FetchOutcome.REJECTED,
            reason="policy_missing (not a regular file)",
        )

    remote_size = file_info.size
    if remote_size is not None and remote_size > max_bytes:
        return FetchResult(
            outcome=FetchOutcome.REJECTED,
            reason=f"too_large ({remote_size} B > {max_bytes} B)",
        )

    # Cold fetch
    try:
        downloaded = hf_hub_download(
            repo_id=repo,
            filename="policy.py",
            revision=revision,
            token=hf_token,
            local_dir=revision_dir,
            etag_timeout=etag_timeout_s,
        )
    except _HF_HUB_CALL_ERRORS as exc:
        return _fetch_result_from_hf_hub_exc(exc)

    downloaded_path = Path(downloaded)

    # Paranoia: stat the on-disk file in case metadata and reality diverged
    size = downloaded_path.stat().st_size
    if size > max_bytes:
        for stale in (downloaded_path, ok_sentinel):
            try:
                os.unlink(stale)
            except OSError:
                pass
        return FetchResult(
            outcome=FetchOutcome.REJECTED,
            reason=f"too_large ({size} B > {max_bytes} B)",
        )

    # Write sentinel only after the file is fully on disk and within limits.
    # If we die between writing policy.py and writing .ok, the next call will
    # re-fetch (the cache-hit branch requires both to exist).
    ok_sentinel.parent.mkdir(parents=True, exist_ok=True)
    ok_sentinel.touch()

    return FetchResult(outcome=FetchOutcome.OK, path=downloaded_path)
