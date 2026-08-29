import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

from cacheon.eval import engine_worker


def _sandbox_proc_reader(
    *,
    seccomp: int = 2,
    filters: int = 1,
    caps: int = 0,
    root: str = "ro,relatime",
):
    values = {
        "/proc/self/status": (
            f"CapEff:\t{caps:x}\n"
            f"CapBnd:\t{caps:x}\n"
            "NoNewPrivs:\t1\n"
            f"Seccomp:\t{seccomp}\n"
            f"Seccomp_filters:\t{filters}\n"
        ),
        "/proc/sys/kernel/yama/ptrace_scope": "1\n",
        "/proc/mounts": f"overlay / overlay {root} 0 0\n",
    }

    def read_text(path: Path, *args, **kwargs):
        return values[str(path)]

    return read_text


class IsolationTests(TestCase):
    def test_process_hardening_requires_zero_caps_and_live_seccomp(self):
        with mock.patch.object(
            Path, "read_text", autospec=True, side_effect=_sandbox_proc_reader()
        ):
            self.assertTrue(engine_worker._process_sandbox_is_hardened())
        for kwargs in (
            {"seccomp": 0},
            {"filters": 0},
            {"caps": 1 << 23},
        ):
            with mock.patch.object(
                Path,
                "read_text",
                autospec=True,
                side_effect=_sandbox_proc_reader(**kwargs),
            ):
                self.assertFalse(engine_worker._process_sandbox_is_hardened())

    def test_writable_root_requires_sealed_policy_digest(self):
        module_digest = hashlib.sha256(
            Path(engine_worker.__file__).read_bytes()
        ).hexdigest()
        sealed = {
            "CACHEON_DISPOSABLE_WRITABLE_ROOT": "1",
            "CACHEON_CONTROLLER_ENGINE_WORKER_SHA256": module_digest,
        }
        with mock.patch.object(
            Path,
            "read_text",
            autospec=True,
            side_effect=_sandbox_proc_reader(root="rw,relatime"),
        ):
            with mock.patch.dict(os.environ, sealed):
                self.assertTrue(engine_worker._process_sandbox_is_hardened())
            for broken in (
                {},
                {"CACHEON_CONTROLLER_ENGINE_WORKER_SHA256": module_digest},
                {"CACHEON_DISPOSABLE_WRITABLE_ROOT": "1"},
                {**sealed, "CACHEON_DISPOSABLE_WRITABLE_ROOT": "0"},
                {
                    **sealed,
                    "CACHEON_CONTROLLER_ENGINE_WORKER_SHA256": hashlib.sha256(
                        b"different policy bytes"
                    ).hexdigest(),
                },
                {
                    **sealed,
                    "CACHEON_CONTROLLER_ENGINE_WORKER_SHA256": module_digest[:63],
                },
                {
                    **sealed,
                    "CACHEON_CONTROLLER_ENGINE_WORKER_SHA256": (
                        "z" + module_digest[1:]
                    ),
                },
            ):
                with mock.patch.dict(os.environ, broken):
                    for name in sealed:
                        if name not in broken:
                            os.environ.pop(name, None)
                    self.assertFalse(engine_worker._process_sandbox_is_hardened())

    def test_read_only_root_stays_hardened_without_overlay_policy(self):
        with mock.patch.object(
            Path, "read_text", autospec=True, side_effect=_sandbox_proc_reader()
        ):
            with mock.patch.dict(os.environ, {}):
                for name in (
                    "CACHEON_DISPOSABLE_WRITABLE_ROOT",
                    "CACHEON_CONTROLLER_ENGINE_WORKER_SHA256",
                ):
                    os.environ.pop(name, None)
                self.assertTrue(engine_worker._process_sandbox_is_hardened())


def test_engine_kwargs_preserve_candidate_overrides():
    cfg = SimpleNamespace(
        model_path="model",
        dtype="float16",
        mem_fraction_static=0.8,
        seed=7,
        log_level="error",
        attention_backend="baseline-attention",
        candidate_attention_backend="candidate-attention",
        moe_runner_backend="baseline-moe",
        candidate_moe_runner_backend="candidate-moe",
        disable_custom_all_reduce=False,
        extra_engine_kwargs={"page_size": 32},
        candidate_extra_engine_kwargs={"page_size": 64},
    )
    baseline = engine_worker.engine_kwargs(cfg, active=False)
    candidate = engine_worker.engine_kwargs(cfg, active=True)
    assert baseline["attention_backend"] == "baseline-attention"
    assert baseline["moe_runner_backend"] == "baseline-moe"
    assert baseline["page_size"] == 32
    assert "disable_custom_all_reduce" not in baseline
    assert candidate["attention_backend"] == "candidate-attention"
    assert candidate["moe_runner_backend"] == "candidate-moe"
    # No per-arm all-reduce override exists: arms differing on that key are
    # "not a comparison" by the validator's own reporting.
    assert "disable_custom_all_reduce" not in candidate
    assert candidate["page_size"] == 64




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
    from cacheon.sandbox import callable_from, load_module

    module = load_module(src)
    prepare = callable_from(module, "prepare")
    entry = callable_from(module, "entry")
    prepare(None, None)
    assert entry() == 1  # shared globals: entry sees what prepare wrote

    # and the documented hazard is real: a SECOND load is a fresh namespace
    from cacheon.sandbox import load_entry

    entry2 = load_entry(src, "entry")
    assert entry2() is None
