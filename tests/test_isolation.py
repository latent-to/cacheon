import os
from types import SimpleNamespace
from unittest import TestCase, mock
from pathlib import Path

import pytest

from optima.eval import _launch


@pytest.mark.parametrize(
    "checker",
    [
        _launch._path_mount_is_read_only,
        pytest.param(
            lambda path: __import__(
                "optima.eval.oci_worker", fromlist=["_is_read_only"]
            )._is_read_only(path),
            id="oci-worker",
        ),
        pytest.param(
            lambda path: __import__(
                "optima.system_patch", fromlist=["_mount_is_read_only"]
            )._mount_is_read_only(Path(path)),
            id="system-overlay",
        ),
    ],
)
def test_read_only_mount_checks_use_statvfs_flag(monkeypatch, checker):
    monkeypatch.setattr(
        os, "statvfs", lambda _path: SimpleNamespace(f_flag=os.ST_RDONLY)
    )
    assert checker("/candidate-input") is True
    monkeypatch.setattr(os, "statvfs", lambda _path: SimpleNamespace(f_flag=0))
    assert checker("/candidate-input") is False


def _sandbox_proc_reader(*, seccomp: int = 2, filters: int = 1):
    allowed = (1 << 23) | (1 << 24)
    values = {
        "/proc/self/status": (
            f"CapEff:\t{allowed:x}\n"
            f"CapBnd:\t{allowed:x}\n"
            "NoNewPrivs:\t1\n"
            f"Seccomp:\t{seccomp}\n"
            f"Seccomp_filters:\t{filters}\n"
        ),
        "/proc/sys/kernel/yama/ptrace_scope": "1\n",
        "/proc/mounts": "overlay / overlay ro,relatime 0 0\n",
    }

    def read_text(path: Path, *args, **kwargs):
        return values[str(path)]

    return read_text


def _subprocess_value():
    return 17


def _subprocess_hang():
    import time

    time.sleep(60)


def _subprocess_with_grandchild(pid_path):
    import subprocess
    import time
    from pathlib import Path

    child = subprocess.Popen(["sleep", "60"])
    Path(pid_path).write_text(str(child.pid))
    time.sleep(60)


class IsolationTests(TestCase):
    def test_process_hardening_requires_live_seccomp_filter(self):
        with mock.patch.object(
            Path,
            "read_text",
            autospec=True,
            side_effect=_sandbox_proc_reader(),
        ):
            self.assertTrue(_launch._process_sandbox_is_hardened())
        for mode, filters in ((0, 1), (2, 0)):
            with mock.patch.object(
                Path,
                "read_text",
                autospec=True,
                side_effect=_sandbox_proc_reader(seccomp=mode, filters=filters),
            ):
                self.assertFalse(_launch._process_sandbox_is_hardened())

    def test_external_no_egress_is_live_verified(self):
        with mock.patch.dict("os.environ", {"OPTIMA_EXTERNAL_NO_EGRESS": "1"}), \
             mock.patch.object(_launch, "_loopback_is_up", return_value=True), \
             mock.patch.object(_launch, "_network_namespace_is_loopback_only",
                               return_value=True), \
             mock.patch.object(_launch, "_egress_is_blocked", return_value=True), \
             mock.patch.object(_launch, "_process_sandbox_is_hardened",
                               return_value=True):
            self.assertTrue(_launch.isolate_network())

    def test_external_isolation_claim_fails_if_egress_is_reachable(self):
        with mock.patch.dict("os.environ", {"OPTIMA_EXTERNAL_NO_EGRESS": "1"}), \
             mock.patch.object(_launch, "_loopback_is_up", return_value=True), \
             mock.patch.object(_launch, "_network_namespace_is_loopback_only",
                               return_value=True), \
             mock.patch.object(_launch, "_egress_is_blocked", return_value=False), \
             mock.patch.object(_launch, "_process_sandbox_is_hardened",
                               return_value=True):
            self.assertFalse(_launch.isolate_network())

    def test_external_isolation_claim_fails_with_non_loopback_interface(self):
        with mock.patch.dict("os.environ", {"OPTIMA_EXTERNAL_NO_EGRESS": "1"}), \
             mock.patch.object(_launch, "_loopback_is_up", return_value=True), \
             mock.patch.object(_launch, "_network_namespace_is_loopback_only",
                               return_value=False), \
             mock.patch.object(_launch, "_egress_is_blocked", return_value=True), \
             mock.patch.object(_launch, "_process_sandbox_is_hardened",
                               return_value=True):
            self.assertFalse(_launch.isolate_network())

    def test_external_isolation_claim_fails_without_process_hardening(self):
        with mock.patch.dict("os.environ", {"OPTIMA_EXTERNAL_NO_EGRESS": "1"}), \
             mock.patch.object(_launch, "_loopback_is_up", return_value=True), \
             mock.patch.object(_launch, "_network_namespace_is_loopback_only",
                               return_value=True), \
             mock.patch.object(_launch, "_egress_is_blocked", return_value=True), \
             mock.patch.object(_launch, "_process_sandbox_is_hardened",
                               return_value=False):
            self.assertFalse(_launch.isolate_network())

    def test_requested_isolation_fails_closed(self):
        cfg = SimpleNamespace(
            isolate=True,
            framework_mode=False,
            allow_unsafe_no_isolation=False,
        )

        with mock.patch.object(_launch, "isolate_network", return_value=False):
            with self.assertRaisesRegex(_launch.IsolationError, "could not be proven"):
                _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)

    def test_every_candidate_requires_isolation_by_default(self):
        cfg = SimpleNamespace(
            isolate=False,
            framework_mode=True,
            allow_unsafe_no_isolation=False,
        )

        with self.assertRaisesRegex(_launch.IsolationError, "every untrusted candidate"):
            _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)

    def test_unsafe_dev_override_allows_failed_isolation(self):
        cfg = SimpleNamespace(
            isolate=True,
            framework_mode=True,
            allow_unsafe_no_isolation=True,
        )

        with mock.patch.object(_launch, "isolate_network", return_value=False):
            _launch.prepare_candidate_environment(cfg, bundle_path="", active=True)


def _setup_bundle(tmp_path):
    (tmp_path / "k.py").write_text(
        "def entry(x, out):\n    out.copy_(x)\n"
        "def patch_engine():\n    pass\n"
    )
    (tmp_path / "manifest.toml").write_text(
        'bundle_id = "setup-test"\n'
        'abi_version = "optima-op-abi-v0"\n'
        "[[ops]]\n"
        'slot = "activation.silu_and_mul"\n'
        'source = "k.py"\n'
        'entry = "entry"\n'
        'setup = "patch_engine"\n'
        'dtypes = ["float32"]\n'
        'architectures = ["cpu"]\n'
    )
    return tmp_path


def test_setup_bundle_requires_framework_fidelity(tmp_path):
    cfg = SimpleNamespace(
        isolate=True,
        framework_mode=False,
        allow_unsafe_no_isolation=False,
    )
    bundle = _setup_bundle(tmp_path)

    with mock.patch.object(_launch, "isolate_network", return_value=True):
        with pytest.raises(_launch.IsolationError, match="declares setup"):
            _launch.prepare_candidate_environment(cfg, bundle_path=str(bundle), active=True)


def test_external_candidate_rejects_mutable_bundle_mount(tmp_path):
    cfg = SimpleNamespace(
        isolate=True,
        framework_mode=False,
        allow_unsafe_no_isolation=False,
        model_path="",
    )
    bundle = _setup_bundle(tmp_path)
    cfg.framework_mode = True
    with mock.patch.dict("os.environ", {"OPTIMA_EXTERNAL_NO_EGRESS": "1"}), \
         mock.patch.object(_launch, "_path_mount_is_read_only", return_value=False):
        with pytest.raises(_launch.IsolationError, match="mounted read-only"):
            _launch.prepare_candidate_environment(
                cfg, bundle_path=str(bundle), active=True
            )


def test_external_rebuild_bundle_requires_trusted_prebuild(tmp_path):
    bundle = _setup_bundle(tmp_path)
    (bundle / "rebuild.json").write_text('{"steps": []}')
    cfg = SimpleNamespace(
        isolate=True,
        framework_mode=True,
        allow_unsafe_no_isolation=False,
        model_path="",
    )
    with mock.patch.dict("os.environ", {"OPTIMA_EXTERNAL_NO_EGRESS": "1"}), \
         mock.patch.object(_launch, "_path_mount_is_read_only", return_value=True), \
         mock.patch.object(_launch, "isolate_network", return_value=True):
        with pytest.raises(_launch.IsolationError, match="trusted prebuild"):
            _launch.prepare_candidate_environment(
                cfg, bundle_path=str(bundle), active=True
            )


def test_cli_auto_selects_framework_mode_for_setup_bundle(tmp_path):
    from optima.cli import _framework_mode_for_manifest
    from optima.manifest import load_manifest

    manifest = load_manifest(_setup_bundle(tmp_path))
    args = SimpleNamespace(framework_mode=False)

    assert _framework_mode_for_manifest(args, manifest) is True


def test_cli_candidate_isolation_defaults_on_and_requires_explicit_opt_out():
    from optima.cli import build_parser

    parser = build_parser()
    safe = parser.parse_args(["evaluate", "bundle", "--model", "model"])
    unsafe = parser.parse_args(
        ["evaluate", "bundle", "--model", "model", "--no-isolate",
         "--allow-unsafe-no-isolation"]
    )

    assert safe.isolate is True
    assert unsafe.isolate is False
    assert unsafe.allow_unsafe_no_isolation is True


def test_launch_subprocess_watchdog_preserves_normal_result():
    assert _launch.call_in_subprocess(_subprocess_value, timeout_s=10) == 17


def test_launch_subprocess_watchdog_terminates_hung_worker():
    with pytest.raises(RuntimeError, match="timed out"):
        _launch.call_in_subprocess(_subprocess_hang, timeout_s=0.1)


def test_launch_watchdog_terminates_descendant_process_group(tmp_path):
    import os
    import time

    pid_path = tmp_path / "grandchild.pid"
    with pytest.raises(RuntimeError, match="timed out"):
        _launch.call_in_subprocess(
            _subprocess_with_grandchild, str(pid_path), timeout_s=1.0
        )
    grandchild = int(pid_path.read_text())
    for _ in range(50):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"watchdog left grandchild process {grandchild} alive")


def test_prepare_and_entry_share_one_module_instance(tmp_path):
    # A (prepare, forward) op's callables must come from ONE module execution: the
    # seam/verify loaders pull both off a single load_module. Two load_entry calls
    # would re-run the body (side effects twice) and split module globals so state
    # written by prepare would be invisible to entry.
    src = tmp_path / "k.py"
    src.write_text(
        "COUNT = 0\n"
        "_STATE = {}\n"
        "def prepare(w13, w2):\n"
        "    _STATE['p'] = 1\n"
        "    return (w13, w2)\n"
        "def entry(*args):\n"
        "    return _STATE.get('p')\n"
    )
    from optima.sandbox import callable_from, load_module

    module = load_module(src)
    prepare = callable_from(module, "prepare")
    entry = callable_from(module, "entry")
    prepare(None, None)
    assert entry() == 1  # shared globals: entry sees what prepare wrote

    # and the documented hazard is real: a SECOND load is a fresh namespace
    from optima.sandbox import load_entry

    entry2 = load_entry(src, "entry")
    assert entry2() is None
