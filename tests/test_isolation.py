from types import SimpleNamespace
from unittest import TestCase, mock

from optima.eval import _launch


class IsolationTests(TestCase):
    def test_requested_isolation_fails_closed(self):
        cfg = SimpleNamespace(
            isolate=True,
            framework_mode=False,
            allow_unsafe_no_isolation=False,
        )

        with mock.patch.object(_launch, "isolate_network", return_value=False):
            with self.assertRaisesRegex(_launch.IsolationError, "could not be proven"):
                _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)

    def test_framework_mode_requires_isolation_by_default(self):
        cfg = SimpleNamespace(
            isolate=False,
            framework_mode=True,
            allow_unsafe_no_isolation=False,
        )

        with self.assertRaisesRegex(_launch.IsolationError, "framework_mode requires"):
            _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)

    def test_unsafe_dev_override_allows_failed_isolation(self):
        cfg = SimpleNamespace(
            isolate=True,
            framework_mode=True,
            allow_unsafe_no_isolation=True,
        )

        with mock.patch.object(_launch, "isolate_network", return_value=False):
            _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)
