"""Shared engine-worker policy helpers.

This module is safe for both the legacy development launcher and the isolated
OCI worker to import.  It contains no engine construction, subprocess launch,
or grading authority.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


logger = logging.getLogger("cacheon.eval.engine-worker")


CANDIDATE_NEVER_EXECUTED_MARKER = "cacheon-candidate-never-executed.v1"


class CandidateExecutionCoverageError(RuntimeError):
    """The active candidate failed its positive execution-evidence contract."""


class CandidateNeverExecutedError(CandidateExecutionCoverageError):
    """Every expected rank loaded the slot, but none recorded an execution."""


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _loopback_is_up() -> bool:
    import fcntl
    import socket
    import struct

    request = struct.Struct("16sH14s")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            response = fcntl.ioctl(
                sock.fileno(), 0x8913, request.pack(b"lo", 0, b"")
            )
        _name, flags, _padding = request.unpack(response)
        return bool(flags & 0x1)
    except (OSError, ValueError, struct.error):
        return False


def _network_namespace_is_loopback_only() -> bool:
    """Return whether the current Linux network namespace exposes only loopback."""

    try:
        interfaces = {
            line.split(":", 1)[0].strip()
            for line in Path("/proc/net/dev").read_text().splitlines()[2:]
            if ":" in line
        }
        if interfaces != {"lo"}:
            return False
        ipv4 = Path("/proc/net/route").read_text().splitlines()[1:]
        if any(line.split()[0] != "lo" for line in ipv4 if line.split()):
            return False
        ipv6 = Path("/proc/net/ipv6_route").read_text().splitlines()
        if any(line.split()[-1] != "lo" for line in ipv6 if line.split()):
            return False
    except (OSError, IndexError):
        return False
    return True


def _egress_is_blocked() -> bool:
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
        return False
    except OSError:
        return True


def _process_sandbox_is_hardened() -> bool:
    """Verify privilege isolation and the sealed disposable-root policy."""

    try:
        status: dict[str, str] = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator:
                status[key] = value.strip()
        effective = int(status["CapEff"], 16)
        bounding = int(status["CapBnd"], 16)
        no_new_privileges = int(status["NoNewPrivs"])
        seccomp_mode = int(status["Seccomp"])
        seccomp_filters = int(status["Seccomp_filters"])
        ptrace_scope = int(Path("/proc/sys/kernel/yama/ptrace_scope").read_text())
        root_options = None
        for line in Path("/proc/mounts").read_text().splitlines():
            fields = line.split()
            if len(fields) >= 4 and fields[1] == "/":
                root_options = set(fields[3].split(","))
                break
        policy_digest = os.environ.get(
            "CACHEON_CONTROLLER_ENGINE_WORKER_SHA256", ""
        ).strip()
        writable_overlay = (
            _truthy_env("CACHEON_DISPOSABLE_WRITABLE_ROOT")
            and len(policy_digest) == 64
            and all(character in "0123456789abcdef" for character in policy_digest)
            and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            == policy_digest
        )
    except (OSError, KeyError, ValueError):
        return False
    root_policy = root_options is not None and (
        "ro" in root_options or ("rw" in root_options and writable_overlay)
    )
    return (
        effective == 0
        and bounding == 0
        and no_new_privileges == 1
        and seccomp_mode == 2
        and seccomp_filters >= 1
        and ptrace_scope >= 1
        and root_policy
    )


def _path_mount_is_read_only(path: str) -> bool:
    try:
        return bool(os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1))
    except OSError:
        return False


def engine_kwargs(cfg, *, active: bool = False) -> dict[str, Any]:
    """Translate a development ``EvalConfig`` into ``sglang.Engine`` kwargs."""

    kwargs: dict[str, Any] = dict(
        model_path=cfg.model_path,
        dtype=cfg.dtype,
        mem_fraction_static=cfg.mem_fraction_static,
        random_seed=cfg.seed,
        log_level=cfg.log_level,
    )
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
    if disable_custom_all_reduce:
        kwargs["disable_custom_all_reduce"] = True
    kwargs.update(getattr(cfg, "extra_engine_kwargs", {}) or {})
    if active:
        kwargs.update(getattr(cfg, "candidate_extra_engine_kwargs", {}) or {})
    return kwargs


def _active_execution_members(
    active_receipts: list[dict], *, expected_member_count: int
) -> list[str]:
    """Validate active scheduler membership and return the identical slot set."""

    pids: list[int] = []
    slot_sets: list[tuple[str, ...]] = []
    for receipt in active_receipts:
        pid = receipt.get("pid")
        slots = receipt.get("slots")
        if type(pid) is not int or pid < 1:
            raise RuntimeError("candidate engine launch: malformed active-member PID")
        if (
            not isinstance(slots, list)
            or not slots
            or any(not isinstance(slot, str) or not slot for slot in slots)
            or len(set(slots)) != len(slots)
        ):
            raise RuntimeError("candidate engine launch: malformed active slot set")
        pids.append(pid)
        slot_sets.append(tuple(sorted(slots)))
    if len(set(pids)) != len(pids):
        raise RuntimeError("candidate engine launch: duplicate active-member PID")
    if len(pids) != expected_member_count:
        raise RuntimeError(
            "candidate engine launch: incomplete active-member coverage "
            f"({len(pids)}/{expected_member_count}); refusing a partially active engine"
        )
    if not slot_sets or len(set(slot_sets)) != 1:
        raise RuntimeError(
            "candidate engine launch: scheduler members disagree on registered slots"
        )
    return list(slot_sets[0])


def _routing_reasons(rows: list[dict]) -> str:
    """Render the recorded not-selected reasons for an execution-coverage error."""

    parts: list[str] = []
    for row in rows:
        slot = row.get("slot")
        for reason in row.get("reasons") or ():
            fields = ",".join(reason.get("fields") or ()) or "-"
            parts.append(f"{slot}:{reason.get('outcome')}({fields})")
    return "; not_selected=" + " ".join(sorted(parts)) if parts else ""


#: Prefix marking the one machine-readable execution line in a retained stderr
#: stream. Grepping for it is the whole contract; see ``EXECUTION_SUMMARY_PREFIX``
#: consumers in ``cacheon.eval.explain``.
EXECUTION_SUMMARY_PREFIX = "CACHEON-EXECUTION-SUMMARY: "

#: Same channel, same contract, for the settings the engine was built with.
ENGINE_CONFIG_PREFIX = "CACHEON-ENGINE-CONFIG: "


def build_session_environment(
    *,
    active: bool,
    bundle_path: str,
    framework_mode: bool,
    receipt_dir: str,
    audit_policy: object,
    install_seams: bool,
    gate_environment: dict[str, str],
) -> dict[str, str]:
    """The environment the in-container engine and its TP ranks inherit.

    Pure and separate from the session so it can be asserted directly. The
    previous kernel-trace attempt shipped with a commit message saying the
    audit arm armed it while nothing in production set the variable at all;
    that claim was unfalsifiable because this dict was unreachable from a test.
    """

    audited = audit_policy is not None
    return {
        "CACHEON_ACTIVE": "1" if active else "0",
        "CACHEON_BUNDLE_PATH": bundle_path if active else "",
        "CACHEON_FRAMEWORK_MODE": "1" if framework_mode else "0",
        "CACHEON_SEAM_RECEIPT_DIR": receipt_dir,
        "CACHEON_SLOT_AUDIT": (
            format(audit_policy.sample_rate_ppm / 1_000_000, ".17g") if audited else ""
        ),
        "CACHEON_SLOT_AUDIT_SEED": (
            str(int(audit_policy.validator_seed, 16)) if audited else ""
        ),
        # Names the kernels the candidate actually launched on the device.
        # Bound to the audit role for the same reason ``disable_cuda_graph`` is:
        # arming it replaces every registry entry with a wrapper for the life of
        # the process, so on a timed arm it would tax the hot path of the thing
        # being measured. Empty string reads as disarmed. Baseline arms load no
        # bundle and have nothing to wrap, so it is inert there regardless.
        "CACHEON_KERNEL_TRACE": "1" if audited else "",
        "SGLANG_PLUGINS": "cacheon" if install_seams else "",
        **gate_environment,
    }


def _emit_execution_summary(receipt_dir: str) -> None:
    """Write the execution facts to stderr before the receipt directory is removed.

    The receipt directory is process-local, on a container tmpfs, and deleted at
    teardown, so everything it knows — which slots registered, how many times each
    ran, whether the run was inside a captured graph, and why a call routed to
    stock instead — dies with the worker. Every one of those is what a miner is
    asking for when they ask what happened to their bundle.

    Stderr is the channel that outlives the container: the host drains it, hashes
    every byte, and retains a bounded prefix with its own receipt. Emitted on
    success as well as failure, because "it passed but lost on speed" needs this
    evidence just as much as "it never ran" — and today only the failure path
    says anything at all.

    Never raises. This runs in a teardown path where an exception would mask the
    real outcome of the run.
    """

    try:
        from cacheon import receipts

        summary: dict[str, Any] = {}
        for kind in ("active", "completed", "load_failed", "not_selected"):
            rows = receipts.collect(receipt_dir, kind)
            if rows:
                summary[kind] = rows
        print(
            EXECUTION_SUMMARY_PREFIX + json.dumps(summary, sort_keys=True, default=str),
            file=sys.stderr,
            flush=True,
        )
    except Exception:  # noqa: BLE001 - a diagnostic must not mask the run's outcome
        logger.exception("cacheon: execution summary emit failed")


def _require_execution_completion(
    receipt_dir: str,
    *,
    active_receipts: list[dict],
    expected_slots: list[str],
    expected_member_count: int,
) -> str:
    """Fail closed unless every active member completed every registered slot."""

    from cacheon import receipts

    completed = receipts.collect(receipt_dir, "completed")
    aot_loaded = receipts.collect(receipt_dir, "aot_loaded")
    aot_invoked = receipts.collect(receipt_dir, "aot_invoked")
    passed, detail = receipts.completed_gate(
        completed,
        expected_slots=expected_slots,
        member_receipts=active_receipts,
        expected_member_count=expected_member_count,
    )
    if not passed:
        observed = (
            f"observed_receipts=completed:{len(completed)},"
            f"aot_loaded:{len(aot_loaded)},aot_invoked:{len(aot_invoked)}"
        )
        message = (
            "candidate engine run failed execution coverage: "
            + detail
            + "; "
            + observed
            # The routing reason, when the registry recorded one. Without it this
            # error says only that nothing ran; with it, it says which declared
            # field kept the candidate off every live call.
            + _routing_reasons(receipts.collect(receipt_dir, "not_selected"))
        )
        # Total silence, behind an active-member check that already passed, is
        # the candidate's own defect: the seam wrote ``active`` into this very
        # root, so the path works and nothing dispatched. Anything partial is
        # ambiguous and stays infrastructure.
        if not (completed or aot_loaded or aot_invoked):
            raise CandidateNeverExecutedError(
                message + "; " + CANDIDATE_NEVER_EXECUTED_MARKER
            )
        raise CandidateExecutionCoverageError(message)
    if aot_invoked and not aot_loaded:
        raise CandidateExecutionCoverageError(
            "candidate engine run has sealed CuTe AOT use evidence without "
            "matching load evidence"
        )
    if aot_loaded:
        aot_slots = sorted(
            {
                row.get("slot")
                for row in aot_loaded
                if isinstance(row.get("slot"), str) and row.get("slot")
            }
        )
        if not aot_slots:
            raise CandidateExecutionCoverageError(
                "candidate engine run has malformed CuTe AOT load evidence"
            )
        if not set(aot_slots).issubset(expected_slots):
            raise CandidateExecutionCoverageError(
                "candidate engine run loaded sealed CuTe AOT for an inactive slot"
            )
        loaded_passed, loaded_detail = receipts.completed_gate(
            aot_loaded,
            expected_slots=aot_slots,
            member_receipts=active_receipts,
            expected_member_count=expected_member_count,
        )
        if not loaded_passed:
            raise CandidateExecutionCoverageError(
                "candidate engine run failed sealed CuTe AOT load coverage: "
                + loaded_detail
            )
        aot_passed, aot_detail = receipts.completed_gate(
            aot_invoked,
            expected_slots=aot_slots,
            member_receipts=active_receipts,
            expected_member_count=expected_member_count,
        )
        if not aot_passed:
            raise CandidateExecutionCoverageError(
                "candidate engine run failed sealed CuTe AOT use coverage: "
                + aot_detail
            )
        detail += "; sealed CuTe AOT " + loaded_detail + "; " + aot_detail
    return detail


def _complete_candidate_execution(
    receipt_dir: str,
    *,
    active_receipts: list[dict],
    expected_slots: list[str],
    expected_member_count: int,
    audit_policy: object | None,
) -> None:
    """Keep missing execution terminal only inside the independent audit role."""

    try:
        _require_execution_completion(
            receipt_dir,
            active_receipts=active_receipts,
            expected_slots=expected_slots,
            expected_member_count=expected_member_count,
        )
    except CandidateExecutionCoverageError as exc:
        if audit_policy is None:
            raise
        # The host audit gate grades an empty policy-bound receipt set FAIL.
        # This module is the file bind-mounted into the sealed OCI image, so the
        # conversion must live here rather than in the image-owned session loop.
        print(
            f"CACHEON-AUDIT-CANDIDATE-FAIL: {exc}",
            file=sys.stderr,
            flush=True,
        )


@dataclass(frozen=True)
class EngineWorkerHandle:
    engine: object
    require_completion: Any
    collect_audit_receipts: Any


@contextlib.contextmanager
def _environment(**overrides: str) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def isolated_engine_session(
    cfg: object,
    *,
    bundle_path: str,
    active: bool,
    framework_mode: bool,
    install_seams: bool = True,
    audit_policy: object | None = None,
) -> Iterator[EngineWorkerHandle]:
    """Construct one engine only inside the already-proven OCI worker fence."""

    if (
        not _truthy_env("CACHEON_EXTERNAL_NO_EGRESS")
        or not _truthy_env("CACHEON_ENGINE_WORKER")
        or not all((
            _loopback_is_up(),
            _network_namespace_is_loopback_only(),
            _egress_is_blocked(),
            _process_sandbox_is_hardened(),
        ))
        or (framework_mode and not active)
        or (active and not install_seams)
    ):
        raise RuntimeError("isolated engine session lacks its trusted OCI fence")
    if audit_policy is not None:
        from cacheon.eval.oci_session_protocol import SlotAuditPolicy

        if type(audit_policy) is not SlotAuditPolicy or not active:
            raise RuntimeError(
                "slot audit requires an exact policy and an active candidate engine"
            )
    receipts = None
    if install_seams:
        from cacheon import receipts as receipt_module, seam

        seam.mark_driver()
        receipts = receipt_module
    from cacheon.seams import seam_binding_environment

    gate_environment = seam_binding_environment(
        getattr(cfg, "seam_bindings", ()) if install_seams else ()
    )
    receipt_dir = tempfile.mkdtemp(prefix="cacheon_receipts_") if active else ""
    try:
        session_environment = build_session_environment(
            active=active,
            bundle_path=bundle_path,
            framework_mode=framework_mode,
            receipt_dir=receipt_dir,
            audit_policy=audit_policy,
            install_seams=install_seams,
            gate_environment=gate_environment,
        )
        with _environment(**session_environment):
            import sglang as sgl

            kwargs = engine_kwargs(cfg, active=active)
            if audit_policy is not None:
                # This is a separate, untimed fidelity role.  Timed B/C/B' plans
                # carry no audit policy and therefore retain their sealed graph
                # configuration and zero audit overhead.
                kwargs["disable_cuda_graph"] = True
            # Emitted for the stock arm too, which has no receipt directory, so
            # this cannot ride on the receipts. Without it the settings that
            # decide which backend each arm ran on -- and therefore whether the
            # pair was comparable at all -- are absent from the retained record
            # and unrecoverable afterwards.
            print(
                ENGINE_CONFIG_PREFIX
                + json.dumps(
                    {"arm": "candidate" if active else "stock", "engine": kwargs},
                    sort_keys=True,
                    default=str,
                ),
                file=sys.stderr,
                flush=True,
            )
            engine = sgl.Engine(**kwargs)
            active_receipts: list[dict] = []
            expected_slots: list[str] = []
            expected_members = int(kwargs.get("tp_size", 1) or 1)
            try:
                if active:
                    assert receipts is not None
                    active_receipts = receipts.require(
                        receipt_dir, "active", context="candidate engine launch"
                    )
                    expected_slots = _active_execution_members(
                        active_receipts, expected_member_count=expected_members
                    )
                    if audit_policy is not None and (
                        tuple(expected_slots) != audit_policy.expected_slots
                        or expected_members != audit_policy.expected_member_count
                    ):
                        raise RuntimeError(
                            "slot audit policy differs from active slot/TP membership"
                        )

                def complete() -> None:
                    if active:
                        _complete_candidate_execution(
                            receipt_dir,
                            active_receipts=active_receipts,
                            expected_slots=expected_slots,
                            expected_member_count=expected_members,
                            audit_policy=audit_policy,
                        )

                def collect_audits() -> list[dict]:
                    if not active:
                        return []
                    assert receipts is not None
                    observed = receipts.collect(receipt_dir, "audit")
                    if audit_policy is None and observed:
                        raise RuntimeError(
                            "timed candidate engine unexpectedly emitted audit receipts"
                        )
                    return observed

                yield EngineWorkerHandle(engine, complete, collect_audits)
            finally:
                try:
                    engine.shutdown()
                except Exception:  # noqa: BLE001 - force-reap follows
                    pass
                try:
                    from sglang.srt.utils import kill_process_tree

                    kill_process_tree(os.getpid(), include_parent=False)
                except Exception:  # noqa: BLE001 - outer OCI teardown remains authoritative
                    pass
    finally:
        if receipt_dir:
            _emit_execution_summary(receipt_dir)
            shutil.rmtree(receipt_dir, ignore_errors=True)
