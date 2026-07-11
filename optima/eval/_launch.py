"""Shared engine-launch context manager used by the eval modules.

Centralizes the spawn-safe, tamper-resistant launch: mark this process as the
driver (so it never imports miner code), set the seam env, build the sglang
Engine, and clean it up. Both the KL eval and the benchmark eval use this.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("optima.eval")


class IsolationError(RuntimeError):
    """Raised when candidate isolation was requested but could not be proven."""


@contextmanager
def env(**overrides: str):
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _bring_loopback_up() -> bool:
    """Enable ``lo`` inside the candidate network namespace without shell tools.

    Minimal serving images often omit ``iproute2`` (the B300 arena does). SGLang uses
    localhost IPC, so accepting an isolated namespace with loopback still down merely
    defers failure until engine startup. Linux's interface ioctl is small, deterministic,
    and avoids adding a mutable external command to the trust boundary.
    """
    import fcntl
    import socket
    import struct

    siocgifflags = 0x8913
    siocsifflags = 0x8914
    iff_up = 0x1
    ifreq = struct.Struct("16sH14s")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            request = ifreq.pack(b"lo", 0, b"")
            response = fcntl.ioctl(sock.fileno(), siocgifflags, request)
            _name, flags, _pad = ifreq.unpack(response)
            fcntl.ioctl(
                sock.fileno(), siocsifflags,
                ifreq.pack(b"lo", flags | iff_up, b""),
            )
        return True
    except (OSError, ValueError, struct.error) as exc:
        logger.warning("optima: could not enable isolated loopback (%s)", exc)
        return False


def _loopback_is_up() -> bool:
    import fcntl
    import socket
    import struct

    ifreq = struct.Struct("16sH14s")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            response = fcntl.ioctl(
                sock.fileno(), 0x8913, ifreq.pack(b"lo", 0, b"")
            )
        _name, flags, _pad = ifreq.unpack(response)
        return bool(flags & 0x1)
    except (OSError, ValueError, struct.error):
        return False


def _egress_is_blocked() -> bool:
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
        return False
    except OSError:
        return True


def _network_namespace_is_loopback_only() -> bool:
    """Prove the current network namespace has no non-loopback interface/route.

    A failed connection to one public IP is only a canary: a firewall can reject that
    destination while private or alternate egress remains available.  Production uses
    an OCI ``--network none`` namespace (or an equivalent fresh ``CLONE_NEWNET``), whose
    enforceable topology is stronger and directly inspectable: only ``lo`` exists and
    neither the IPv4 nor IPv6 route table names another interface.

    Fail closed on platforms without Linux proc/sysfs rather than treating a missing
    introspection surface as isolation proof.
    """
    from pathlib import Path

    try:
        # ``/sys/class/net`` can contain kernel bookkeeping files such as
        # ``bonding_masters`` even in a network-none container.  /proc/net/dev is
        # the authoritative list of actual interfaces visible in this namespace.
        netdev_lines = Path("/proc/net/dev").read_text().splitlines()[2:]
        interfaces = {
            line.split(":", 1)[0].strip()
            for line in netdev_lines
            if ":" in line
        }
        if interfaces != {"lo"}:
            return False

        # /proc/net/route has a header followed by tab-separated rows whose first
        # field is the interface.  A loopback-only namespace normally has no rows,
        # but accepting explicit lo routes keeps this check kernel-version agnostic.
        ipv4_lines = Path("/proc/net/route").read_text().splitlines()[1:]
        if any(line.split()[0] != "lo" for line in ipv4_lines if line.split()):
            return False

        # Linux's IPv6 route table stores the device name in the final column.
        ipv6_lines = Path("/proc/net/ipv6_route").read_text().splitlines()
        if any(line.split()[-1] != "lo" for line in ipv6_lines if line.split()):
            return False
    except (OSError, IndexError):
        return False
    return True


def _process_sandbox_is_hardened() -> bool:
    """Verify candidate descendants cannot ptrace/privilege-escalate into the timer.

    The scheduler necessarily shares a PID namespace with the trusted SGLang driver.
    The result HMAC stops file replacement, while this policy stops a native candidate
    from simply attaching to the parent and stealing that key.  Production containers
    use ``--cap-drop ALL --cap-add SYS_NICE --cap-add SYS_RESOURCE`` plus
    ``no-new-privileges``; those two capabilities are the only accepted bounding set.
    """
    from pathlib import Path

    allowed = (1 << 23) | (1 << 24)  # CAP_SYS_NICE, CAP_SYS_RESOURCE
    try:
        status: dict[str, str] = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep:
                status[key] = value.strip()
        effective = int(status["CapEff"], 16)
        bounding = int(status["CapBnd"], 16)
        no_new_privs = int(status["NoNewPrivs"])
        seccomp_mode = int(status["Seccomp"])
        seccomp_filters = int(status["Seccomp_filters"])
        ptrace_scope = int(Path("/proc/sys/kernel/yama/ptrace_scope").read_text())
        root_options = None
        for line in Path("/proc/mounts").read_text().splitlines():
            fields = line.split()
            if len(fields) >= 4 and fields[1] == "/":
                root_options = set(fields[3].split(","))
                break
    except (OSError, KeyError, ValueError):
        return False
    return (
        effective & ~allowed == 0
        and bounding & ~allowed == 0
        and no_new_privs == 1
        and seccomp_mode == 2
        and seccomp_filters >= 1
        and ptrace_scope >= 1
        and root_options is not None
        and "ro" in root_options
    )


def _path_mount_is_read_only(path: str) -> bool:
    """Whether an existing candidate input lives on a read-only mount."""
    try:
        return bool(os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1))
    except OSError:
        return False


def isolate_network() -> bool:
    """Put THIS process (and every child it spawns) into a fresh network namespace with
    NO egress, so untrusted miner code can't reach an external API to fake the output.
    Loopback is brought up so sglang's localhost IPC still works; the model is forced
    offline (it must already be cached). Self-checks that egress is actually gone.

    This is the boundary that makes the framework-mode token-match gate cheat-PROOF: the
    candidate must compute the right tokens — it can't see the trusted reference
    (separate process) and now can't fetch it either. Requires CAP_SYS_ADMIN (run the
    GPU box privileged; chain/cloud secrets live on a separate CPU control box). Returns
    True iff the candidate is confirmed no-egress; logs loudly and returns False if not.
    """
    # Production workers may already be launched by the trusted host with an OCI
    # ``--network none`` policy. Do not create a second namespace there; verify both
    # required properties instead. The env bit is operator-owned and is not accepted
    # without the live self-check.
    if _truthy_env("OPTIMA_EXTERNAL_NO_EGRESS"):
        if not _loopback_is_up():
            logger.error("optima: external isolation claimed but loopback is down")
            return False
        if not _network_namespace_is_loopback_only():
            logger.error(
                "optima: external isolation claimed but namespace is not loopback-only"
            )
            return False
        if not _egress_is_blocked():
            logger.error("optima: external isolation claimed but egress is reachable")
            return False
        if not _process_sandbox_is_hardened():
            logger.error(
                "optima: external isolation lacks cap-drop/no-new-privileges/ptrace guard"
            )
            return False
        _offline_env()
        logger.warning("optima: externally network-isolated (verified no egress; loopback up)")
        return True

    clone_newnet = getattr(os, "CLONE_NEWNET", None)
    if clone_newnet is None or not hasattr(os, "unshare"):
        logger.warning("optima: os.unshare/CLONE_NEWNET unavailable (need py>=3.12); candidate NOT isolated")
        return False
    try:
        os.unshare(clone_newnet)  # fresh netns: only `lo`, which starts DOWN
    except OSError as exc:
        logger.warning("optima: network isolation failed (%s); candidate NOT no-egress", exc)
        return False
    # Bring up loopback (the sglang scheduler<->detokenizer IPC uses localhost); external
    # stays unreachable because the netns has no route off-box.
    if not _bring_loopback_up():
        return False
    if not _network_namespace_is_loopback_only():
        logger.error(
            "optima: ISOLATION FAILED — fresh namespace is not loopback-only"
        )
        return False
    if not _process_sandbox_is_hardened():
        logger.error(
            "optima: ISOLATION FAILED — process capabilities/ptrace policy are unsafe"
        )
        return False
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Self-check: prove egress is actually gone (a fail-closed signal in the log).
    if not _egress_is_blocked():
        logger.error("optima: ISOLATION FAILED — candidate still has network egress!")
        return False
    logger.warning("optima: candidate network-isolated (no egress; loopback only)")
    return True


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def prepare_candidate_environment(cfg, *, bundle_path: str, active: bool) -> None:
    """Apply candidate-only isolation/build work before importing SGLang.

    Every active bundle contains untrusted host Python, even when it implements a
    narrow tensor slot. Production evaluation therefore fails closed unless the
    entire candidate process tree is no-egress. ``allow_unsafe_no_isolation`` remains
    an explicit local-development escape hatch and is never suitable for settlement.
    """
    if not active:
        return
    framework_mode = getattr(cfg, "framework_mode", False)
    isolate = getattr(cfg, "isolate", False)
    allow_unsafe = getattr(cfg, "allow_unsafe_no_isolation", False)
    if bundle_path:
        from optima.manifest import load_manifest

        manifest = load_manifest(bundle_path)
        has_setup = any(op.setup for op in manifest.ops)
        if has_setup and not framework_mode:
            raise IsolationError(
                "bundle declares setup() but framework_mode is not enabled. "
                "Engine-wide mutation requires external token fidelity and isolation."
            )
    if isolate and _truthy_env("OPTIMA_EXTERNAL_NO_EGRESS"):
        immutable_inputs = [bundle_path]
        model_path = str(getattr(cfg, "model_path", "") or "")
        if model_path and os.path.exists(model_path):
            immutable_inputs.append(model_path)
        mutable = [path for path in immutable_inputs
                   if path and not _path_mount_is_read_only(path)]
        if mutable:
            raise IsolationError(
                "candidate bundle/model inputs must be mounted read-only: "
                + ", ".join(mutable)
            )
    if not isolate:
        if not allow_unsafe:
            raise IsolationError(
                "every untrusted candidate requires no-egress isolation. "
                "Use --allow-unsafe-no-isolation only for local throughput debugging."
            )
        logger.error(
            "optima: UNSAFE dev override: candidate is running without requested "
            "network isolation"
        )
        _offline_env()
    if isolate:
        if not isolate_network():
            if not allow_unsafe:
                raise IsolationError(
                    "candidate network isolation was requested but could not be proven. "
                    "Run the eval worker with CAP_SYS_ADMIN/CAP_NET_ADMIN, or inside a "
                    "container/VM whose candidate process has no network egress. "
                    "Use --allow-unsafe-no-isolation only for local throughput debugging."
                )
            logger.error(
                "optima: UNSAFE dev override: candidate network isolation failed; "
                "continuing with egress possible"
            )
            _offline_env()
    if bundle_path:
        from optima.rebuild import apply_rebuild_plan_subprocess

        has_plan = (Path(bundle_path) / "rebuild.json").is_file()
        externally_isolated = _truthy_env("OPTIMA_EXTERNAL_NO_EGRESS")
        if has_plan and externally_isolated:
            if not _truthy_env("OPTIMA_PREBUILT_ARTIFACTS"):
                raise IsolationError(
                    "externally isolated candidate requires a trusted prebuild and "
                    "read-only artifact/overlay mounts (set OPTIMA_PREBUILT_ARTIFACTS=1 "
                    "only after apply_rebuild_plan_subprocess(..., phase='build') succeeds)"
                )
            logger.info(
                "optima: using trusted prebuilt read-only artifacts for %s", bundle_path
            )
        elif apply_rebuild_plan_subprocess(bundle_path, phase="build"):
            logger.warning(
                "optima: built candidate artifacts in subprocess for %s", bundle_path
            )
        _dep_overlay_env(bundle_path)


def _dep_overlay_env(bundle_path: str) -> None:
    """Candidate-local JIT workspace for a dep-patched candidate.

    ``FLASHINFER_WORKSPACE_BASE`` is read ONCE at ``flashinfer.jit.env`` import (a real
    ``os.getenv``, unlike everything else there — verified 2026-07-07), so it must be a
    process env var set BEFORE the engine spawns; the overlay integration cannot rebind
    it later. Without this, a patched JIT build and a stock JIT build of the same
    module name share a cache dir — ninja does invalidate on the changed source path,
    but concurrent candidates would serialize/race on the shared build files.
    """
    import os

    try:
        from optima.dep_policy import overlay_workspace_base
        from optima.manifest import load_manifest

        manifest = load_manifest(bundle_path)
        if manifest.dep_patches:
            ws = overlay_workspace_base(
                bundle_path, tuple(dp.target for dp in manifest.dep_patches)
            )
            ws.mkdir(parents=True, exist_ok=True)
            existing = os.environ.get("FLASHINFER_WORKSPACE_BASE", "").strip()
            if existing and Path(existing).resolve() != ws.resolve():
                logger.warning(
                    "optima: replacing shared/stale FLASHINFER_WORKSPACE_BASE=%s with "
                    "content-addressed candidate workspace %s", existing, ws,
                )
            os.environ["FLASHINFER_WORKSPACE_BASE"] = str(ws)
            logger.info("optima: candidate-local FLASHINFER_WORKSPACE_BASE=%s", ws)
    except Exception:  # noqa: BLE001 - cache identity/setup failures are disqualifying
        logger.exception("optima: dep-overlay env setup failed for %s", bundle_path)
        raise


def engine_kwargs(cfg, *, active: bool = False) -> dict[str, Any]:
    """Translate an ``EvalConfig`` into ``sglang.Engine`` kwargs.

    Shared by both eval paths so multi-GPU knobs (``tp_size`` / ``moe_runner_backend``
    / ``disable_custom_all_reduce``) and deterministic mode apply identically. New
    fields are read with ``getattr`` so an older/duck-typed cfg still works.
    """
    kwargs: dict[str, Any] = dict(
        model_path=cfg.model_path,
        dtype=cfg.dtype,
        mem_fraction_static=cfg.mem_fraction_static,
        random_seed=cfg.seed,
        log_level=cfg.log_level,
    )
    # Only pass these when explicitly set so sglang keeps its strong production
    # defaults otherwise (auto attention backend + CUDA graphs ON). A weak baseline
    # lets miners win against a crippled reference.
    attention_backend = getattr(cfg, "attention_backend", None)
    if active and getattr(cfg, "candidate_attention_backend", None):
        attention_backend = cfg.candidate_attention_backend
    if attention_backend:
        kwargs["attention_backend"] = attention_backend
    if getattr(cfg, "disable_cuda_graph", False):
        kwargs["disable_cuda_graph"] = True
    if getattr(cfg, "deterministic", False):
        kwargs["enable_deterministic_inference"] = True
    if getattr(cfg, "tp_size", None):
        kwargs["tp_size"] = int(cfg.tp_size)
    if getattr(cfg, "max_running_requests", None):
        kwargs["max_running_requests"] = int(cfg.max_running_requests)
    moe_runner_backend = getattr(cfg, "moe_runner_backend", None)
    if active and getattr(cfg, "candidate_moe_runner_backend", None):
        moe_runner_backend = cfg.candidate_moe_runner_backend
    if moe_runner_backend:
        kwargs["moe_runner_backend"] = moe_runner_backend
    disable_custom_all_reduce = getattr(cfg, "disable_custom_all_reduce", False)
    if active and getattr(cfg, "candidate_disable_custom_all_reduce", None) is not None:
        disable_custom_all_reduce = cfg.candidate_disable_custom_all_reduce
    if disable_custom_all_reduce:
        kwargs["disable_custom_all_reduce"] = True
    kwargs.update(getattr(cfg, "extra_engine_kwargs", {}) or {})
    if active:
        kwargs.update(getattr(cfg, "candidate_extra_engine_kwargs", {}) or {})
    return kwargs


def _sweep_gpu_procs() -> int:
    """Kill every OTHER process in this namespace holding an nvidia device fd.

    Failed engine launches can strand scheduler subprocesses that survive both
    the launch child's reap and sglang's own kill cascade (they re-session), each
    pinning the model's full VRAM. Only ever called from a launch subprocess that
    has not created its engine yet, and only when OPTIMA_GPU_SWEEP=1 — a dedicated
    eval box where everything on the visible GPUs belongs to this evaluation.
    """
    import signal

    me = os.getpid()
    killed = 0
    for pid_dir in os.listdir("/proc"):
        if not pid_dir.isdigit() or int(pid_dir) == me:
            continue
        fd_dir = f"/proc/{pid_dir}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    if os.readlink(f"{fd_dir}/{fd}").startswith("/dev/nvidia"):
                        os.kill(int(pid_dir), signal.SIGKILL)
                        killed += 1
                        break
                except OSError:
                    continue
        except OSError:  # process exited, or not ours to inspect
            continue
    if killed:
        logger.warning("optima: GPU sweep killed %d stranded process(es)", killed)
    return killed


def _wait_gpu_drain(threshold_mib: int = 4096, timeout_s: float = 150.0) -> None:
    """Block until every visible GPU is under ``threshold_mib`` used, or timeout.

    Evaluate runs engine launches back-to-back out of subprocesses; the previous
    launch's schedulers release their VRAM a beat after the driver regains control
    (and a wedged shutdown can pin the whole model until the reap in the launch
    finally fires). Sizing the next KV pool against that residue OOMs at startup.
    Polls nvidia-smi (never initializes CUDA in this process); on timeout, warns
    and proceeds — the guard must never fail a run on its own.
    """
    import subprocess
    import time

    deadline = time.monotonic() + timeout_s
    sweep_at = time.monotonic() + 25.0  # give a clean shutdown a fair head start
    swept = os.environ.get("OPTIMA_GPU_SWEEP") != "1"  # disabled -> pretend done
    # Scope the wait to THIS launch's GPUs: on a shared box another lane's engine
    # legitimately holds its own devices for the whole run — without the filter
    # every launch here would stall out the full timeout staring at it.
    query = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        query += ["-i", cvd]
    last = ""
    while time.monotonic() < deadline:
        try:
            out = subprocess.run(query, capture_output=True, text=True, timeout=10).stdout
            used = [int(x) for x in out.split()]
        except Exception:  # noqa: BLE001 — no/odd nvidia-smi: nothing to wait for
            return
        if not used or max(used) < threshold_mib:
            return
        if not swept and time.monotonic() >= sweep_at:
            _sweep_gpu_procs()
            swept = True
        last = ",".join(map(str, used))
        time.sleep(2.0)
    logger.warning("optima: GPUs did not drain below %d MiB within %.0fs (used MiB: %s); "
                   "launching anyway", threshold_mib, timeout_s, last)


@contextmanager
def launched_engine(cfg, *, bundle_path: str, active: bool,
                    audit_rate: float = 0.0, audit_out: Optional[list] = None,
                    member_out: Optional[list] = None):
    """Launch a sglang Engine with the Optima seam configured.

    ``cfg`` is an ``EvalConfig`` (see optima.eval.throughput_kl). The miner
    kernel runs only in the spawned scheduler child; THIS process is marked as
    the driver so it never imports miner code (timing stays tamper-resistant).

    An ACTIVE launch demands seam receipts (see optima/receipts.py): at least one
    scheduler rank must report the bundle loaded+enabled before we hand the engine
    to the caller, and every active scheduler member must report successful
    model-facing output production for every registered slot. Any selected-candidate
    exception fallback disqualifies the run. Without this, a missing bootstrap/env
    or partial multi-slot activation can silently score stock-vs-stock — the
    phantom-pass class hit on 2026-07-07.

    ``audit_rate > 0`` arms the IN-ENGINE AUDIT (optima/audit.py) in the ranks:
    sampled dispatcher calls are re-run through the captured stock baseline and
    compared under the slot's verify tolerances. Only ever set on an UNTIMED
    quality launch — audited calls carry clone+baseline overhead. The rolling
    per-rank audit receipts are appended to ``audit_out`` before cleanup. The
    sampling seed is fixed per launch and shared by all ranks (collective
    baselines need rank-identical sampling; see audit.py).
    """
    import random
    import shutil
    import tempfile

    from optima import receipts, seam

    seam.mark_driver()
    prepare_candidate_environment(cfg, bundle_path=bundle_path, active=active)
    receipt_dir = tempfile.mkdtemp(prefix="optima_receipts_") if active else ""
    extra_env = {"OPTIMA_SEAM_RECEIPT_DIR": receipt_dir} if active else {}
    if active and audit_rate > 0.0:
        extra_env["OPTIMA_SLOT_AUDIT"] = f"{audit_rate:g}"
        extra_env["OPTIMA_SLOT_AUDIT_SEED"] = str(random.SystemRandom().randrange(2**31))
    try:
        with env(
            OPTIMA_BUNDLE_PATH=bundle_path or "",
            OPTIMA_ACTIVE="1" if active else "0",
            OPTIMA_FRAMEWORK_MODE="1" if getattr(cfg, "framework_mode", False) else "0",
            SGLANG_PLUGINS="optima",
            **extra_env,
        ):
            import sglang as sgl

            _wait_gpu_drain()
            resolved_engine_kwargs = engine_kwargs(cfg, active=active)
            engine = sgl.Engine(**resolved_engine_kwargs)
            ok = False
            active_receipts: list[dict] = []
            try:
                if active:
                    active_receipts = receipts.require(
                        receipt_dir, "active", context="candidate engine launch"
                    )
                    expected_members = int(resolved_engine_kwargs.get("tp_size", 1) or 1)
                    observed_pids = {r.get("pid") for r in active_receipts if r.get("pid")}
                    if len(observed_pids) < expected_members:
                        raise RuntimeError(
                            "candidate engine launch: incomplete active-rank coverage "
                            f"({len(observed_pids)}/{expected_members} scheduler members); "
                            "refusing a partially activated TP engine"
                        )
                    logger.info("optima: seam active receipts: %s", active_receipts)
                    if member_out is not None:
                        member_out.extend(active_receipts)
                yield engine
                ok = True
            finally:
                try:
                    engine.shutdown()
                except Exception:  # noqa: BLE001
                    pass
                # Some builds' Engine.shutdown() leaves scheduler subprocesses
                # alive, each pinning the model's whole VRAM — which starves the
                # NEXT launch's pool sizing (measured 2026-07-10: B' startup OOM
                # behind 4x180GB orphaned schedulers). This launch subprocess owns
                # every engine process, so reap the remaining tree before handing
                # control back to the driver.
                try:
                    from sglang.srt.utils import kill_process_tree

                    kill_process_tree(os.getpid(), include_parent=False)
                except Exception:  # noqa: BLE001
                    pass
                if active and ok and audit_out is not None:
                    audit_out.extend(receipts.collect(receipt_dir, "audit"))
                if active and ok:
                    expected_slots = sorted({
                        str(slot)
                        for receipt in active_receipts
                        for slot in receipt.get("slots", ())
                        if str(slot)
                    })
                    completed = receipts.collect(receipt_dir, "completed")
                    fallbacks = receipts.collect(receipt_dir, "fallback")
                    passed, detail = receipts.completed_gate(
                        completed,
                        expected_slots=expected_slots,
                        member_receipts=active_receipts,
                        fallback_receipts=fallbacks,
                    )
                    if not passed:
                        raise RuntimeError(
                            "candidate engine run failed execution coverage: " + detail
                        )
                    logger.info("optima: %s", detail)
    finally:
        if receipt_dir:
            shutil.rmtree(receipt_dir, ignore_errors=True)


def _subprocess_entry(out_path, auth_key, auth_nonce, fn, args, kwargs):
    """Run ``fn(*args, **kwargs)`` and write a bounded JSON-safe outcome."""
    # Give the parent one killable process group for this launch and request kernel
    # cleanup if the parent itself dies. SGLang creates several descendants; an
    # unbounded orphaned launch can pin an entire TP arena indefinitely.
    try:
        import ctypes
        import signal

        os.setsid()
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - non-Linux dev boxes keep basic behavior
        pass
    import traceback

    from optima.ipc import LaunchOutcome, dump_authenticated_file

    try:
        outcome = LaunchOutcome(value=fn(*args, **kwargs), error=None)
    except BaseException:  # noqa: BLE001 - report ANY failure back to the parent
        outcome = LaunchOutcome(value=None, error=traceback.format_exc())
    try:
        dump_authenticated_file(
            out_path, outcome, key=auth_key, nonce=auth_nonce
        )
    except BaseException:  # noqa: BLE001 - unsupported/malicious result type
        # A callable can return an arbitrary miner-defined object.  Never fall back
        # to pickle/repr for it; report the codec failure through the same safe wire.
        serialization_error = traceback.format_exc()
        try:
            dump_authenticated_file(
                out_path,
                LaunchOutcome(
                    value=None,
                    error=("launch result serialization failed:\n"
                           + serialization_error[-64_000:]),
                ),
                key=auth_key,
                nonce=auth_nonce,
            )
        except BaseException:  # noqa: BLE001 - parent treats empty/malformed as fail
            pass


def call_in_subprocess(fn, *args, timeout_s: float | None = None, **kwargs):
    """Run ``fn(*args, **kwargs)`` in a FRESH spawned process; return its result.

    Each model launch must run in its own process. sglang + deterministic mode set
    process-global state (torch deterministic algorithms, the cuBLAS workspace, the
    sampling backend) and hold a CUDA context; a second launch in the same driver
    process inherits that state and — observed on gpt-oss-120b in deterministic mode —
    the candidate launch then produces NaN/garbage. A fresh process makes the baseline
    and candidate launches independent and frees all GPU/host memory between them.

    ``fn`` must be a module-level (spawn-picklable) callable. The result travels back
    through a bounded JSON-safe file (avoids mp.Queue size limits / pipe deadlocks on
    large logprob payloads) whose decoder only reconstructs explicitly allowlisted
    result dataclasses. A hard watchdog owns the whole launch process group; timeout cannot
    leave an engine tree silently pinning the GPUs. Raises ``RuntimeError`` if the
    child crashes, times out, or ``fn`` raises.
    """
    import multiprocessing as mp
    import os
    import secrets
    import signal
    import tempfile

    from optima.ipc import LaunchOutcome, WireError, load_authenticated_file

    if timeout_s is None:
        timeout_s = float(os.environ.get("OPTIMA_LAUNCH_TIMEOUT_S", "7200"))
    if timeout_s <= 0:
        raise ValueError("launch timeout must be positive")
    ctx = mp.get_context("spawn")
    fd, path = tempfile.mkstemp(prefix="optima_launch_", suffix=".json")
    os.close(fd)
    # Kept in parent/driver process memory, never exported through the environment
    # inherited by candidate scheduler ranks.  The nonce binds freshness so a valid
    # prior result file cannot be replayed into this launch.
    auth_key = secrets.token_bytes(32)
    auth_nonce = secrets.token_bytes(16)
    try:
        proc = ctx.Process(
            target=_subprocess_entry,
            args=(path, auth_key, auth_nonce, fn, args, kwargs),
        )
        proc.start()
        proc.join(timeout_s)
        if proc.is_alive():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
            proc.join(10.0)
            if proc.is_alive():
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
                proc.join(10.0)
            raise RuntimeError(
                f"launch subprocess timed out after {timeout_s:g}s and was terminated"
            )
        try:
            outcome = load_authenticated_file(
                path, key=auth_key, nonce=auth_nonce
            )
        except (FileNotFoundError, OSError, WireError) as exc:
            raise RuntimeError(
                f"launch subprocess crashed (exitcode={proc.exitcode}) with no result: {exc}"
            ) from None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if type(outcome) is not LaunchOutcome:
        raise RuntimeError("launch subprocess returned an invalid result envelope")
    if outcome.error:
        raise RuntimeError("launch subprocess failed:\n" + outcome.error)
    return outcome.value
