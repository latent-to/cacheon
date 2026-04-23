"""Unit tests for scripts/remote_validator.py CLI guards."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def _mock_bittensor_module() -> ModuleType:
    """Build a minimal fake ``bittensor`` module for CLI tests."""
    mod = ModuleType("bittensor")

    def _make_wallet(**_kwargs):
        w = MagicMock()
        w.hotkey = MagicMock()
        w.hotkey.ss58_address = "5FakeAddress123"
        return w

    mod.Subtensor = MagicMock
    mod.Wallet = _make_wallet
    return mod


class TestRemoteValidatorCli:
    @patch("scripts.remote_validator.preflight_check")
    def test_fail_fast_without_dry_run_returns_6(self, mock_preflight, tmp_path):
        fake_bt = _mock_bittensor_module()
        with patch.dict("sys.modules", {"bittensor": fake_bt}):
            import importlib
            import scripts.remote_validator as mod
            importlib.reload(mod)

            with patch.object(mod, "run_forever") as mock_loop:
                rc = mod.main([
                    "--network", "test",
                    "--netuid", "460",
                    "--wallet-name", "default",
                    "--wallet-hotkey", "default",
                    "--policy-cache-dir", str(tmp_path / "policy-cache"),
                ])

        assert rc == 6
        mock_loop.assert_not_called()

    @patch("scripts.remote_validator.preflight_check")
    def test_dry_run_does_not_trigger_fail_fast(self, mock_preflight, tmp_path):
        fake_bt = _mock_bittensor_module()
        with patch.dict("sys.modules", {"bittensor": fake_bt}):
            import importlib
            import scripts.remote_validator as mod
            importlib.reload(mod)

            with patch.object(mod, "run_forever") as mock_loop:
                rc = mod.main([
                    "--network", "test",
                    "--netuid", "460",
                    "--wallet-name", "default",
                    "--wallet-hotkey", "default",
                    "--dry-run",
                    "--poll-interval", "1",
                    "--policy-cache-dir", str(tmp_path / "policy-cache"),
                ])

        assert rc == 0
        mock_loop.assert_called_once()
        call_kwargs = mock_loop.call_args.kwargs
        assert call_kwargs["dry_run"] is True

        # Must be the real fetch-backed precheck, not the permissive default
        from validator.challengers import allow_all_precheck
        assert call_kwargs["precheck"] is not allow_all_precheck
        assert "make_fetch_precheck" in call_kwargs["precheck"].__qualname__

    @patch("scripts.remote_validator.preflight_check")
    def test_unwritable_cache_dir_returns_7(self, mock_preflight, tmp_path):
        """Unwritable policy cache dir → exits with code 7 before entering loop."""
        fake_bt = _mock_bittensor_module()
        unwritable = tmp_path / "no-perms"
        unwritable.mkdir()
        unwritable.chmod(0o555)
        try:
            with patch.dict("sys.modules", {"bittensor": fake_bt}):
                import importlib
                import scripts.remote_validator as mod
                importlib.reload(mod)

                with patch.object(mod, "run_forever") as mock_loop:
                    rc = mod.main([
                        "--network", "test",
                        "--netuid", "460",
                        "--wallet-name", "default",
                        "--wallet-hotkey", "default",
                        "--dry-run",
                        "--policy-cache-dir", str(unwritable / "nested"),
                    ])

            assert rc == 7
            mock_loop.assert_not_called()
        finally:
            unwritable.chmod(0o755)
