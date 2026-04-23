"""Unit tests for validator.policy_fetch — no network calls."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from validator.policy_fetch import (
    FetchOutcome,
    FetchResult,
    fetch_policy_source,
    sanitize_repo,
)

pytestmark = pytest.mark.unit


def _fake_response(status_code: int | None = 500) -> MagicMock:
    """Build a fake httpx.Response stand-in for HfHubHTTPError(response=...).

    huggingface_hub 1.x requires a keyword-only ``response`` argument typed
    as ``httpx.Response``.  The type annotation is not enforced at runtime,
    so a MagicMock with ``.status_code`` is sufficient for our exception
    handlers, which only read ``exc.response.status_code`` via ``getattr``.
    """
    resp = MagicMock()
    resp.status_code = status_code
    return resp


class TestGatedRepoErrorHierarchy:
    """Regression: GatedRepoError must be caught as fetch_forbidden, not
    revision_unavailable, even though it inherits from RepositoryNotFoundError."""

    def test_gated_repo_is_subclass_of_repository_not_found(self):
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        assert issubclass(GatedRepoError, RepositoryNotFoundError)

    def test_gated_repo_maps_to_fetch_forbidden(self, tmp_path):
        from huggingface_hub.utils import GatedRepoError

        with patch(
            "validator.policy_fetch.HfApi",
            side_effect=GatedRepoError("gated", response=_fake_response(403)),
        ):
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.REJECTED
        assert "fetch_forbidden" in (result.reason or "")
        assert "revision_unavailable" not in (result.reason or "")


class TestHFValidationErrorHandling:
    def test_hf_validation_error_maps_to_rejected(self, tmp_path):
        from huggingface_hub.errors import HFValidationError

        with patch(
            "validator.policy_fetch.HfApi",
            side_effect=HFValidationError("bad repo id"),
        ):
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.REJECTED
        assert "invalid_repo" in (result.reason or "")


class TestUnsafeRepoCachePath:
    def test_fetch_rejects_unsafe_cache_repo_without_hf_call(self, tmp_path):
        with patch("validator.policy_fetch.HfApi") as mock_api:
            result = fetch_policy_source(
                "..",
                "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        mock_api.assert_not_called()
        assert result.outcome is FetchOutcome.REJECTED
        assert "invalid_repo" in (result.reason or "")


class TestSanitizeRepo:
    def test_simple_repo(self):
        assert sanitize_repo("owner/name") == "owner_name"

    def test_no_slash(self):
        assert sanitize_repo("name") == "name"

    def test_dot_dot_segments_rejected(self):
        with pytest.raises(ValueError, match="invalid repo"):
            sanitize_repo("./owner/name")
        with pytest.raises(ValueError, match="invalid repo"):
            sanitize_repo("../owner/name")
        with pytest.raises(ValueError, match="invalid repo"):
            sanitize_repo("..")

    def test_trailing_whitespace_trimmed(self):
        assert sanitize_repo(" owner/name ") == "owner_name"

    def test_leading_dot_legitimate_name(self):
        # Dots in the name itself are preserved.
        assert sanitize_repo(".env/name") == ".env_name"

    def test_double_underscore_empty_segment_rejected(self):
        with pytest.raises(ValueError, match="invalid repo"):
            sanitize_repo("a__b/c")


class _FakeRepoFile:
    """Minimal stand-in for huggingface_hub.hf_api.RepoFile."""
    def __init__(self, size: int):
        self.size = size


class _FakeRepoFolder:
    """Stand-in for huggingface_hub.hf_api.RepoFolder (a directory entry)."""
    pass


def _make_cache_entry(cache_dir: Path, repo: str, revision: str, content: str) -> Path:
    """Create a valid cached policy.py with its .ok sentinel."""
    cache_path = cache_dir / sanitize_repo(repo) / revision / "policy.py"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(content)
    (cache_path.parent / ".ok").touch()
    return cache_path


class TestFetchPolicySourceCacheHit:
    def test_cache_hit_returns_ok(self, tmp_path: Path):
        repo = "owner/name"
        revision = "a" * 40
        cache_path = _make_cache_entry(tmp_path, repo, revision, "pass")

        result = fetch_policy_source(
            repo, revision,
            cache_dir=tmp_path,
            max_bytes=1024,
            etag_timeout_s=30.0,
        )
        assert result.outcome is FetchOutcome.OK
        assert result.path == cache_path

    def test_cache_hit_without_sentinel_refetches(self, tmp_path: Path):
        """A policy.py without a .ok sentinel is treated as a partial download."""
        repo = "owner/name"
        revision = "a" * 40
        cache_path = tmp_path / sanitize_repo(repo) / revision / "policy.py"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("pass")  # no .ok sentinel

        downloaded = tmp_path / "staging" / "policy.py"
        downloaded.parent.mkdir(parents=True)
        downloaded.write_text("pass")

        with patch("validator.policy_fetch.HfApi") as MockApi:
            MockApi.return_value.get_paths_info.return_value = [
                _FakeRepoFile(size=4)
            ]
            with patch(
                "validator.policy_fetch.hf_hub_download",
                return_value=str(downloaded),
            ) as mock_dl:
                fetch_policy_source(
                    repo, revision,
                    cache_dir=tmp_path,
                    max_bytes=1024,
                    etag_timeout_s=30.0,
                )

        mock_dl.assert_called_once()  # must have re-fetched

    def test_cache_hit_too_large_invalidates_and_refetches(self, tmp_path: Path):
        """If the cap shrank, invalidate the cached file and re-fetch."""
        repo = "owner/name"
        revision = "a" * 40
        cache_dir = tmp_path
        cache_path = _make_cache_entry(cache_dir, repo, revision, "x" * 500)

        revision_dir = cache_dir / sanitize_repo(repo) / revision
        downloaded = revision_dir / "policy.py"
        downloaded.write_text("x" * 50)

        with patch("validator.policy_fetch.HfApi") as MockApi:
            MockApi.return_value.get_paths_info.return_value = [
                _FakeRepoFile(size=50)
            ]
            with patch(
                "validator.policy_fetch.hf_hub_download",
                return_value=str(downloaded),
            ):
                result = fetch_policy_source(
                    repo, revision,
                    cache_dir=cache_dir,
                    max_bytes=100,
                    etag_timeout_s=30.0,
                )

        assert result.outcome is FetchOutcome.OK
        assert result.path == cache_path

    def test_cache_hit_is_pure_function(self, tmp_path: Path):
        """Two calls with the same args on a cache hit return identical results."""
        repo = "owner/name"
        revision = "a" * 40
        cache_path = _make_cache_entry(tmp_path, repo, revision, "pass")

        result1 = fetch_policy_source(
            repo, revision,
            cache_dir=tmp_path,
            max_bytes=1024,
            etag_timeout_s=30.0,
        )
        result2 = fetch_policy_source(
            repo, revision,
            cache_dir=tmp_path,
            max_bytes=1024,
            etag_timeout_s=30.0,
        )
        assert result1 == result2


class TestFetchPolicySourceColdFetch:
    def _mock_api(self, size: int):
        mock = MagicMock()
        mock.get_paths_info.return_value = [_FakeRepoFile(size=size)]
        return mock

    def test_cold_fetch_ok(self, tmp_path: Path):
        repo = "owner/name"
        revision = "a" * 40
        cache_dir = tmp_path
        # Place the mock-downloaded file outside the cache path so we don't
        # trigger a cache hit.
        downloaded = tmp_path / "staging" / "policy.py"
        downloaded.parent.mkdir(parents=True)
        downloaded.write_text("class MyPolicy: pass")

        with patch(
            "validator.policy_fetch.HfApi",
            return_value=self._mock_api(size=100),
        ):
            with patch(
                "validator.policy_fetch.hf_hub_download",
                return_value=str(downloaded),
            ) as mock_hf:
                result = fetch_policy_source(
                    repo, revision,
                    cache_dir=cache_dir,
                    max_bytes=1024,
                    etag_timeout_s=30.0,
                )

        mock_hf.assert_called_once()
        call_kwargs = mock_hf.call_args.kwargs
        assert call_kwargs.get("etag_timeout") == 30.0
        assert result.outcome is FetchOutcome.OK
        assert result.path is not None
        assert result.path.name == "policy.py"

    def test_size_checked_before_download(self, tmp_path: Path):
        """If HEAD metadata says the file is too large, hf_hub_download is never called."""
        with patch(
            "validator.policy_fetch.HfApi",
            return_value=self._mock_api(size=10_000_000),
        ):
            with patch(
                "validator.policy_fetch.hf_hub_download",
            ) as mock_hf:
                result = fetch_policy_source(
                    "owner/name", "a" * 40,
                    cache_dir=tmp_path,
                    max_bytes=1024,
                    etag_timeout_s=30.0,
                )

        mock_hf.assert_not_called()
        assert result.outcome is FetchOutcome.REJECTED
        assert "too_large" in (result.reason or "")

    def test_entry_not_found(self, tmp_path: Path):
        from huggingface_hub.utils import EntryNotFoundError

        with patch(
            "validator.policy_fetch.HfApi",
            return_value=self._mock_api(size=100),
        ):
            with patch(
                "validator.policy_fetch.hf_hub_download",
                side_effect=EntryNotFoundError("policy.py not found"),
            ):
                result = fetch_policy_source(
                    "owner/name", "a" * 40,
                    cache_dir=tmp_path,
                    max_bytes=1024,
                    etag_timeout_s=30.0,
                )
        assert result.outcome is FetchOutcome.REJECTED
        assert "policy_missing" in (result.reason or "")

    def test_repository_not_found(self, tmp_path: Path):
        from huggingface_hub.utils import RepositoryNotFoundError

        with patch(
            "validator.policy_fetch.HfApi",
            side_effect=RepositoryNotFoundError(
                "repo not found", response=_fake_response(404)
            ),
        ):
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.REJECTED
        assert "revision_unavailable" in (result.reason or "")

    def test_revision_not_found(self, tmp_path: Path):
        from huggingface_hub.utils import RevisionNotFoundError

        with patch(
            "validator.policy_fetch.HfApi",
            side_effect=RevisionNotFoundError(
                "revision not found", response=_fake_response(404)
            ),
        ):
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.REJECTED
        assert "revision_unavailable" in (result.reason or "")

    def test_gated_repo(self, tmp_path: Path):
        from huggingface_hub.utils import GatedRepoError

        with patch(
            "validator.policy_fetch.HfApi",
            side_effect=GatedRepoError("gated", response=_fake_response(403)),
        ):
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.REJECTED
        assert "fetch_forbidden" in (result.reason or "")

    def test_hf_http_500_deferred(self, tmp_path: Path):
        from huggingface_hub.utils import HfHubHTTPError

        exc = HfHubHTTPError("server error", response=_fake_response(503))

        with patch(
            "validator.policy_fetch.HfApi",
            side_effect=exc,
        ):
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.DEFERRED
        assert "hf_http_503" in (result.reason or "")

    def test_hf_http_401_rejected(self, tmp_path: Path):
        from huggingface_hub.utils import HfHubHTTPError

        exc = HfHubHTTPError("unauthorized", response=_fake_response(401))

        with patch(
            "validator.policy_fetch.HfApi",
            side_effect=exc,
        ):
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.REJECTED
        assert "fetch_forbidden" in (result.reason or "")

    def test_hf_http_no_response_deferred(self, tmp_path: Path):
        """Server disconnects before sending a response — exc.response is None."""
        from huggingface_hub.utils import HfHubHTTPError

        # Construct with a dummy response to satisfy the 1.x constructor,
        # then clear it to simulate the "connection closed" case.
        exc = HfHubHTTPError("connection closed", response=_fake_response())
        exc.response = None

        with patch(
            "validator.policy_fetch.HfApi",
            side_effect=exc,
        ):
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.DEFERRED
        assert "hf_http_unknown" in (result.reason or "")

    def test_generic_network_error_deferred(self, tmp_path: Path):
        with patch(
            "validator.policy_fetch.HfApi",
            side_effect=ConnectionError("network flake"),
        ):
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.DEFERRED
        assert "fetch_error" in (result.reason or "")

    def test_file_exceeding_cap_rejects_and_cleans_up(self, tmp_path: Path):
        repo = "owner/name"
        revision = "a" * 40
        cache_dir = tmp_path
        # Staging path outside cache to avoid cache hit
        downloaded = tmp_path / "staging" / "policy.py"
        downloaded.parent.mkdir(parents=True)
        downloaded.write_text("x" * 200)

        with patch(
            "validator.policy_fetch.HfApi",
            return_value=self._mock_api(size=50),
        ):
            with patch(
                "validator.policy_fetch.hf_hub_download",
                return_value=str(downloaded),
            ):
                result = fetch_policy_source(
                    repo, revision,
                    cache_dir=cache_dir,
                    max_bytes=100,
                    etag_timeout_s=30.0,
                )

        assert result.outcome is FetchOutcome.REJECTED
        assert "too_large" in (result.reason or "")
        # The downloaded file should NOT remain in the cache dir
        cached = cache_dir / sanitize_repo(repo) / revision / "policy.py"
        assert not cached.exists()

    def test_token_passed_through(self, tmp_path: Path):
        # Staging path outside cache to avoid cache hit
        downloaded = tmp_path / "staging" / "policy.py"
        downloaded.parent.mkdir(parents=True)
        downloaded.write_text("pass")

        with patch(
            "validator.policy_fetch.HfApi",
            return_value=self._mock_api(size=50),
        ):
            with patch(
                "validator.policy_fetch.hf_hub_download",
                return_value=str(downloaded),
            ) as mock_hf:
                fetch_policy_source(
                    "owner/name", "a" * 40,
                    cache_dir=tmp_path,
                    max_bytes=1024,
                    etag_timeout_s=30.0,
                    hf_token="my_token",
                )

        call_kwargs = mock_hf.call_args.kwargs
        assert call_kwargs.get("token") == "my_token"
        assert call_kwargs.get("etag_timeout") == 30.0

    def test_cold_fetch_ok_writes_sentinel(self, tmp_path: Path):
        """Successful cold fetch must create a .ok sentinel alongside policy.py."""
        repo = "owner/name"
        revision = "a" * 40
        cache_dir = tmp_path
        downloaded = tmp_path / "staging" / "policy.py"
        downloaded.parent.mkdir(parents=True)
        downloaded.write_text("class MyPolicy: pass")

        with patch(
            "validator.policy_fetch.HfApi",
            return_value=self._mock_api(size=100),
        ):
            with patch(
                "validator.policy_fetch.hf_hub_download",
                return_value=str(downloaded),
            ):
                result = fetch_policy_source(
                    repo, revision,
                    cache_dir=cache_dir,
                    max_bytes=1024,
                    etag_timeout_s=30.0,
                )

        assert result.outcome is FetchOutcome.OK
        revision_dir = cache_dir / sanitize_repo(repo) / revision
        assert (revision_dir / ".ok").exists(), ".ok sentinel must be written on success"

    def test_policy_is_folder_rejects(self, tmp_path: Path):
        """policy.py reported as a directory entry → policy_missing."""
        with patch("validator.policy_fetch.HfApi") as MockApi:
            MockApi.return_value.get_paths_info.return_value = [
                _FakeRepoFolder()  # no `size` attribute
            ]
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.REJECTED
        assert "policy_missing" in (result.reason or "")
        assert "not a regular file" in (result.reason or "")

    def test_rate_limited_429_deferred(self, tmp_path: Path):
        """HTTP 429 (rate limit) must map to DEFERRED with reason rate_limited."""
        from huggingface_hub.utils import HfHubHTTPError

        exc = HfHubHTTPError("rate limited", response=_fake_response(429))

        with patch(
            "validator.policy_fetch.HfApi",
            side_effect=exc,
        ):
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.DEFERRED
        assert "rate_limited" in (result.reason or "")

    def test_local_entry_not_found_deferred(self, tmp_path: Path):
        """LocalEntryNotFoundError must map to DEFERRED, not REJECTED."""
        from huggingface_hub.utils import LocalEntryNotFoundError

        with patch(
            "validator.policy_fetch.HfApi",
            side_effect=LocalEntryNotFoundError("not in local cache"),
        ):
            result = fetch_policy_source(
                "owner/name", "a" * 40,
                cache_dir=tmp_path,
                max_bytes=1024,
                etag_timeout_s=30.0,
            )
        assert result.outcome is FetchOutcome.DEFERRED
        assert "local_cache_miss" in (result.reason or "")
