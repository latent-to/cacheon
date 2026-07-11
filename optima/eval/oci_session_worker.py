"""Untrusted in-container engine session worker.

This process deliberately owns no trusted clock, nonce secret, HMAC key, or verdict.
SGLang may deserialize scheduler-controlled pickle into this very process.  Its only
authority is to return bounded primitive evidence to the outer controller.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import math
import os
import struct
from typing import Any

from optima.eval.oci_protocol import CONTAINER_BUNDLE_PATH
from optima.eval.oci_session_protocol import (
    FRAME_HEADER_BYTES,
    FRAME_MAGIC,
    MAX_BATCH_REQUEST_BYTES,
    MAX_CONTROL_BYTES,
    MAX_INIT_BYTES,
    SessionProtocolError,
    batch_response,
    closed_message,
    decode_message,
    error_message,
    evidence_frame,
    frame_message,
    preflight_message,
    ready_message,
    teacher_evidence_frame,
    validate_teacher_request,
    validate_batch_request,
    validate_batch_response,
    validate_close_request,
    validate_init,
)


class CandidateTeacherInputError(SessionProtocolError):
    """Sealed C output cannot be scored by the pinned stock vocabulary/API."""


def _canonical_prompt_ids(engine, prompt: str) -> list[int]:
    manager = getattr(engine, "tokenizer_manager", None)
    tokenizer = getattr(manager, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise SessionProtocolError("stock B' lacks the pinned tokenizer API")
    ids = encode(prompt)
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(token) is not int or not 0 <= token <= 2_147_483_647 for token in ids)
    ):
        raise SessionProtocolError("stock B' canonical tokenization is invalid")
    return ids


def _teacher_forced_traces(engine, prompts, sources, *, top_logprobs_num: int):
    """Use pinned ``Engine.generate(input_ids=...)`` as a prefill-only teacher."""

    from optima.eval.external_quality import TeacherForcedPromptTrace, TeacherForcedTrace

    prompt_ids = [_canonical_prompt_ids(engine, prompt) for prompt in prompts]
    tokenizer = engine.tokenizer_manager.tokenizer
    try:
        vocab_size = len(tokenizer)
    except (TypeError, AttributeError):
        vocab_size = getattr(tokenizer, "vocab_size", None)
    if type(vocab_size) is not int or vocab_size <= 0:
        raise SessionProtocolError("stock B' tokenizer has no bounded vocabulary size")
    for source in ("baseline", "stock_control"):
        if any(token >= vocab_size for ids in sources[source] for token in ids):
            raise SessionProtocolError("stock rollout contains an out-of-vocabulary token")
    if any(token >= vocab_size for ids in sources["candidate"] for token in ids):
        raise CandidateTeacherInputError(
            "candidate rollout contains an out-of-vocabulary token"
        )
    traces_by_source: dict[str, list[TeacherForcedTrace]] = {}
    for source in ("baseline", "candidate", "stock_control"):
        rollout_ids = sources[source]
        full_ids = [prefix + response for prefix, response in zip(prompt_ids, rollout_ids, strict=True)]
        starts = [max(0, len(prefix) - 1) for prefix in prompt_ids]
        sentinels = [[response[0]] for response in rollout_ids]
        outputs = engine.generate(
            input_ids=full_ids,
            sampling_params={"temperature": 0.0, "max_new_tokens": 0, "ignore_eos": True},
            return_logprob=True,
            logprob_start_len=starts,
            top_logprobs_num=int(top_logprobs_num),
            token_ids_logprob=sentinels,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        if not isinstance(outputs, list) or len(outputs) != len(prompts):
            raise SessionProtocolError("stock B' teacher returned wrong prompt coverage")
        source_traces: list[TeacherForcedTrace] = []
        for response_ids, output in zip(rollout_ids, outputs, strict=True):
            if not isinstance(output, dict) or not isinstance(output.get("meta_info"), dict):
                raise SessionProtocolError("stock B' teacher output is malformed")
            meta = output["meta_info"]
            raw_targets = meta.get("input_token_logprobs")
            raw_topk = meta.get("input_top_logprobs")
            raw_sentinel = meta.get("input_token_ids_logprobs")
            if not isinstance(raw_targets, list) or not isinstance(raw_topk, list):
                raise SessionProtocolError("stock B' teacher omitted input logprobs")
            raw_targets = raw_targets[-len(response_ids):]
            raw_topk = raw_topk[-len(response_ids):]
            if len(raw_targets) != len(response_ids) or len(raw_topk) != len(response_ids):
                raise SessionProtocolError("stock B' teacher target coverage mismatch")
            target_logprobs: list[float] = []
            trusted_topk = []
            for expected_id, target, position in zip(
                response_ids, raw_targets, raw_topk, strict=True
            ):
                if not isinstance(target, (list, tuple)) or len(target) < 2:
                    raise SessionProtocolError("stock B' teacher target entry is malformed")
                try:
                    logprob, token_id = float(target[0]), int(target[1])
                except (TypeError, ValueError):
                    raise SessionProtocolError("stock B' teacher target entry is invalid") from None
                if token_id != expected_id or not math.isfinite(logprob):
                    raise SessionProtocolError("stock B' teacher scored the wrong target token")
                target_logprobs.append(max(-1_000.0, min(0.0001, logprob)))
                if not isinstance(position, list) or len(position) != top_logprobs_num:
                    raise SessionProtocolError("stock B' trusted top-k width mismatch")
                clean_position = []
                for entry in position:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                        raise SessionProtocolError("stock B' trusted top-k entry is malformed")
                    clean_position.append((float(entry[0]), int(entry[1]), None))
                trusted_topk.append(tuple(clean_position))
            # ``token_ids_logprob`` is an independent API path.  It is intentionally
            # one sentinel per sequence (not every unique rollout token, which would
            # make decode evidence quadratic).  Cross-check the first target exactly.
            if not isinstance(raw_sentinel, list) or len(raw_sentinel) < len(response_ids):
                raise SessionProtocolError("stock B' teacher omitted sentinel logprobs")
            sentinel_positions = raw_sentinel[-len(response_ids):]
            first = sentinel_positions[0]
            if not isinstance(first, list) or not first:
                raise SessionProtocolError("stock B' teacher sentinel evidence is malformed")
            match = next(
                (
                    entry for entry in first
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2
                    and int(entry[1]) == response_ids[0]
                ),
                None,
            )
            if match is None or not math.isclose(
                float(match[0]), target_logprobs[0], rel_tol=1e-4, abs_tol=1e-4
            ):
                raise SessionProtocolError("stock B' teacher sentinel/target logprobs disagree")
            trace = TeacherForcedTrace(tuple(target_logprobs), tuple(trusted_topk))
            trace.validated(
                expected_tokens=len(response_ids), topk_num=top_logprobs_num
            )
            source_traces.append(trace)
        traces_by_source[source] = source_traces

    result = []
    for index, ids in enumerate(prompt_ids):
        raw = b"".join(int(token).to_bytes(4, "big") for token in ids)
        result.append(TeacherForcedPromptTrace(
            prompt_token_count=len(ids),
            prompt_token_sha256=hashlib.sha256(raw).hexdigest(),
            baseline=traces_by_source["baseline"][index],
            candidate=traces_by_source["candidate"][index],
            stock_control=traces_by_source["stock_control"][index],
        ))
    return tuple(result)


def _read_exact(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
        except InterruptedError:
            continue
        if not chunk:
            raise SessionProtocolError("outer controller closed a partial request")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_json_frame(fd: int, *, max_bytes: int) -> dict[str, Any]:
    header = _read_exact(fd, FRAME_HEADER_BYTES)
    if header[:4] != FRAME_MAGIC:
        raise SessionProtocolError("outer request frame magic/version mismatch")
    size = struct.unpack(">I", header[4:8])[0]
    if size > max_bytes:
        raise SessionProtocolError("outer request frame exceeds its hard bound")
    return decode_message(_read_exact(fd, size), max_bytes=max_bytes)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(fd, view[offset:])
        except InterruptedError:
            continue
        if written <= 0:
            raise SessionProtocolError("could not write outer evidence")
        offset += written


def _protocol_fd() -> int:
    """Reserve original stdout for protocol, then silence the whole engine tree."""

    protocol = os.dup(1)
    os.set_inheritable(protocol, False)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    finally:
        os.close(devnull)
    return protocol


def _control_input_fd(input_fd: int) -> int:
    """Keep control input in one CLOEXEC fd and give engine children /dev/null."""

    control = os.dup(input_fd)
    os.set_inheritable(control, False)
    devnull = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(devnull, input_fd)
    finally:
        os.close(devnull)
    return control


def _is_read_only(path: str) -> bool:
    try:
        return bool(os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1))
    except OSError:
        return False


def _assert_live_session_sandbox(*, active: bool) -> None:
    """Verify the streaming container before any candidate/engine import."""

    from optima.eval._launch import (
        _egress_is_blocked,
        _loopback_is_up,
        _network_namespace_is_loopback_only,
        _process_sandbox_is_hardened,
    )
    from optima.eval.oci_protocol import (
        CONTAINER_ARTIFACT_PATH,
        CONTAINER_JIT_PATH,
        CONTAINER_MODEL_PATH,
        CONTAINER_SOURCE_PATH,
    )

    if not _loopback_is_up():
        raise RuntimeError("isolated OCI loopback is unavailable")
    if not _network_namespace_is_loopback_only() or not _egress_is_blocked():
        raise RuntimeError("streaming OCI session is not loopback-only/no-egress")
    if not _process_sandbox_is_hardened():
        raise RuntimeError("streaming OCI session lacks process sandbox hardening")
    read_only_inputs = [
        CONTAINER_SOURCE_PATH,
        CONTAINER_MODEL_PATH,
        CONTAINER_ARTIFACT_PATH,
    ]
    if active:
        read_only_inputs.append(CONTAINER_BUNDLE_PATH)
    bad = [path for path in read_only_inputs if not _is_read_only(path)]
    if bad:
        raise RuntimeError(
            "streaming OCI input mounts are not read-only: " + ", ".join(bad)
        )
    if (
        not os.path.isdir(CONTAINER_JIT_PATH)
        or _is_read_only(CONTAINER_JIT_PATH)
    ):
        raise RuntimeError("streaming OCI private JIT mount is unavailable")


def _arm_system_overlay(manifest, competition) -> None:
    arena_name = os.environ.get("OPTIMA_OCI_ARENA_NAME", "").strip()
    target = os.environ.get("OPTIMA_OCI_COMPETITION_TARGET", "").strip()
    cache_root = os.environ.get("OPTIMA_OCI_SYSTEM_OVERLAY_ROOT", "").strip()
    if not arena_name or not target or not cache_root or competition.target != target:
        raise RuntimeError("system launch profile target/arena/artifact policy mismatch")
    from pathlib import Path

    from optima.system_patch import system_launch_environment

    os.environ.update(system_launch_environment(
        CONTAINER_BUNDLE_PATH,
        competition_target=target,
        arena_name=arena_name,
        cache_root=Path(cache_root),
    ))


@contextlib.contextmanager
def _launched_host_system_engine(cfg, *, bundle: str):
    """Load an op-packaged whole-host product with no component receipt authority."""

    import shutil
    import tempfile

    from optima import seam
    from optima.eval._launch import (
        _wait_gpu_drain,
        engine_kwargs,
        env,
        prepare_candidate_environment,
    )

    seam.mark_driver()
    prepare_candidate_environment(cfg, bundle_path=bundle, active=True)
    receipt_dir = tempfile.mkdtemp(prefix="optima_system_host_receipts_")
    try:
        with env(
            OPTIMA_BUNDLE_PATH=bundle,
            OPTIMA_ACTIVE="1",
            OPTIMA_FRAMEWORK_MODE="1",
            OPTIMA_SEAM_RECEIPT_DIR=receipt_dir,
            SGLANG_PLUGINS="optima",
        ):
            import sglang as sgl

            _wait_gpu_drain()
            engine = sgl.Engine(**engine_kwargs(cfg, active=True))
            try:
                yield engine
            finally:
                try:
                    engine.shutdown()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    from sglang.srt.utils import kill_process_tree

                    kill_process_tree(os.getpid(), include_parent=False)
                except Exception:  # noqa: BLE001
                    pass
    finally:
        shutil.rmtree(receipt_dir, ignore_errors=True)


@contextlib.contextmanager
def _engine_context(cfg, *, mode: str, audit_out: list, member_out: list):
    active = mode != "baseline"
    bundle = CONTAINER_BUNDLE_PATH if active else ""
    manifest = None
    competition = None
    system_patch = False
    system_product = False
    if active:
        from optima.competition import SYSTEM_MODE, resolve_competition
        from optima.manifest import load_manifest

        manifest = load_manifest(bundle)
        competition = resolve_competition(
            manifest, for_settlement=True, warn_legacy=False
        )
        expected_target = os.environ.get("OPTIMA_OCI_COMPETITION_TARGET", "").strip()
        if competition.target != expected_target:
            raise RuntimeError(
                f"resolved competition {competition.target!r} != profile {expected_target!r}"
            )
        system_product = competition.mode == SYSTEM_MODE
        system_patch = manifest.system is not None
        if system_product and not cfg.framework_mode:
            raise RuntimeError("whole-system candidate requires external framework fidelity")
        if system_product and mode == "candidate_audit":
            raise RuntimeError("whole-system candidate has no component audit session")
        if system_patch:
            _arm_system_overlay(manifest, competition)

    if mode == "candidate_audit":
        cfg = dataclasses.replace(
            cfg,
            timed_iters=1,
            warmup_iters=1,
            conditioning_iters=1,
            disable_cuda_graph=True,
        )
    if system_patch:
        from optima.eval.oci_worker import _launched_system_engine

        with _launched_system_engine(cfg) as (engine, members):
            member_out.extend(members)
            yield engine, cfg
        return
    if system_product:
        with _launched_host_system_engine(cfg, bundle=bundle) as engine:
            # Component/member receipts are intentionally not exported: the product
            # is the externally-qualified whole serving implementation.
            yield engine, cfg
        return

    from optima.eval._launch import launched_engine

    with launched_engine(
        cfg,
        bundle_path=bundle,
        active=active,
        audit_rate=cfg.audit_rate if mode == "candidate_audit" else 0.0,
        audit_out=audit_out,
        member_out=member_out,
    ) as engine:
        yield engine, cfg


def _sampling_params(cfg) -> dict[str, Any]:
    params: dict[str, Any] = {
        "temperature": cfg.temperature,
        "max_new_tokens": cfg.max_new_tokens,
    }
    if cfg.ignore_eos:
        params["ignore_eos"] = True
    return params


def _output_items(outputs: Any, *, require_logprobs: bool) -> list[dict[str, Any]]:
    if isinstance(outputs, dict):
        outputs = [outputs]
    if not isinstance(outputs, list):
        raise SessionProtocolError("engine output is not a list/object")
    items: list[dict[str, Any]] = []
    for output in outputs:
        if not isinstance(output, dict):
            raise SessionProtocolError("engine output item is not an object")
        meta = output.get("meta_info", {})
        if not isinstance(meta, dict):
            meta = {}
        raw_ids = output.get("output_ids") or meta.get("output_ids") or []
        output_ids = [int(token) for token in raw_ids]
        topk: list[list[list[Any]]] = []
        if require_logprobs:
            raw_topk = meta.get("output_top_logprobs") or []
            for raw_position in raw_topk:
                position: list[list[Any]] = []
                for entry in raw_position or []:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                        continue
                    try:
                        logprob = float(entry[0])
                        token_id = int(entry[1])
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(logprob):
                        position.append([logprob, token_id])
                topk.append(position)
        items.append({
            "output_ids": output_ids,
            "top_logprobs": topk,
            # Text is variable-length and therefore excluded from the timed wire.
            "text": "",
        })
    return items


def run_session(*, input_fd: int = 0, output_fd: int | None = None) -> int:
    protocol = _protocol_fd() if output_fd is None else output_fd
    control_input = _control_input_fd(input_fd)
    session_id = ""
    stage = "init"
    try:
        init = _read_json_frame(control_input, max_bytes=MAX_INIT_BYTES)
        session_id, mode, config = validate_init(init)
        active = mode != "baseline"
        _assert_live_session_sandbox(active=active)
        from optima.eval.oci_worker import attest_runtime

        runtime_attestation = attest_runtime()
        _write_all(protocol, frame_message(
            preflight_message(
                session_id=session_id,
                mode=mode,
                runtime_attestation=runtime_attestation,
            ),
            max_bytes=MAX_CONTROL_BYTES,
        ))
        from optima.eval.throughput_kl import EvalConfig

        cfg = EvalConfig(**config)
        audit_out: list = []
        member_out: list = []
        close_binding: tuple[str, str, str] | None = None
        stage = "engine"
        with _engine_context(
            cfg, mode=mode, audit_out=audit_out, member_out=member_out
        ) as (engine, measure_cfg):
            _write_all(protocol, frame_message(
                ready_message(session_id=session_id, mode=mode),
                max_bytes=MAX_CONTROL_BYTES,
            ))
            expected_batches = int(measure_cfg.warmup_iters) + int(measure_cfg.timed_iters)
            seen_request_ids: set[str] = set()
            seen_nonces: set[str] = set()
            seen_prompts: set[str] = set()
            normal_prompt_batches: list[list[str]] = []
            normal_output_ids: list[list[list[int]]] = []
            for expected_index in range(expected_batches):
                stage = f"batch-{expected_index}"
                request = _read_json_frame(
                    control_input, max_bytes=MAX_BATCH_REQUEST_BYTES
                )
                (
                    got_session, request_id, nonce, batch_index, warmup, prompts,
                ) = validate_batch_request(
                    request, expected_count=int(measure_cfg.num_prompts)
                )
                expected_warmup = expected_index < int(measure_cfg.warmup_iters)
                if (
                    got_session != session_id
                    or batch_index != expected_index
                    or warmup != expected_warmup
                    or request_id in seen_request_ids
                    or nonce in seen_nonces
                    or any(prompt in seen_prompts for prompt in prompts)
                ):
                    raise SessionProtocolError(
                        "batch ordering/binding/disjointness invariant failed"
                    )
                seen_request_ids.add(request_id)
                seen_nonces.add(nonce)
                seen_prompts.update(prompts)
                kwargs: dict[str, Any] = {
                    "return_logprob": True,
                    "logprob_start_len": -1,
                    "top_logprobs_num": int(measure_cfg.top_logprobs_num),
                }
                outputs = engine.generate(
                    prompt=list(prompts),
                    sampling_params=_sampling_params(measure_cfg),
                    **kwargs,
                )
                items = _output_items(outputs, require_logprobs=True)
                normal_prompt_batches.append(list(prompts))
                normal_output_ids.append([
                    list(item["output_ids"]) for item in items
                ])
                # Warmup is host-visible conditioning evidence, not a forgeable ack.
                evidence = validate_batch_response(
                    batch_response(
                        session_id=session_id,
                        request_id=request_id,
                        nonce=nonce,
                        batch_index=batch_index,
                        items=items,
                    ),
                    session_id=session_id,
                    request_id=request_id,
                    nonce=nonce,
                    batch_index=batch_index,
                    expected_prompts=int(measure_cfg.num_prompts),
                    max_new_tokens=int(measure_cfg.max_new_tokens),
                    top_logprobs_num=int(measure_cfg.top_logprobs_num),
                    ignore_eos=bool(measure_cfg.ignore_eos),
                    require_logprobs=True,
                    temperature=float(measure_cfg.temperature),
                )
                _write_all(protocol, evidence_frame(
                    evidence,
                    session_id=session_id,
                    request_id=request_id,
                    nonce=nonce,
                    batch_index=batch_index,
                    require_logprobs=True,
                ))
            stage = "posthoc-or-close"
            next_message = _read_json_frame(
                control_input, max_bytes=MAX_BATCH_REQUEST_BYTES
            )
            if next_message.get("type") == "teacher":
                if mode != "baseline":
                    raise SessionProtocolError(
                        "post-hoc teacher requests require a stock B' engine"
                    )
                expected_teacher = [
                    *[("warmup", index) for index in range(int(measure_cfg.warmup_iters))],
                    *[("timed", index) for index in range(int(measure_cfg.timed_iters))],
                ]
                teacher_count: int | None = None
                teacher_seal: str | None = None
                for teacher_index, (expected_phase, expected_phase_index) in enumerate(
                    expected_teacher
                ):
                    stage = f"teacher-{expected_phase}-{expected_phase_index}"
                    request = next_message if teacher_index == 0 else _read_json_frame(
                        control_input, max_bytes=MAX_BATCH_REQUEST_BYTES
                    )
                    raw_prompts = request.get("prompts") if isinstance(request, dict) else None
                    if not isinstance(raw_prompts, list) or not 2 <= len(raw_prompts) <= 64:
                        raise SessionProtocolError("teacher prompt-cluster count is invalid")
                    if teacher_count is None:
                        teacher_count = len(raw_prompts)
                    (
                        got_session, request_id, nonce, phase, batch_index, seal,
                        prompts, sources,
                    ) = validate_teacher_request(
                        request,
                        expected_count=teacher_count,
                        expected_tokens=int(measure_cfg.max_new_tokens),
                    )
                    if (
                        got_session != session_id
                        or phase != expected_phase
                        or batch_index != expected_phase_index
                        or request_id in seen_request_ids
                        or nonce in seen_nonces
                        or (teacher_seal is not None and seal != teacher_seal)
                    ):
                        raise SessionProtocolError(
                            "teacher ordering/binding/disjointness invariant failed"
                        )
                    teacher_seal = seal
                    seen_request_ids.add(request_id)
                    seen_nonces.add(nonce)
                    global_index = (
                        batch_index
                        if phase == "warmup"
                        else int(measure_cfg.warmup_iters) + batch_index
                    )
                    original_prompts = normal_prompt_batches[global_index]
                    original_ids = normal_output_ids[global_index]
                    by_prompt = {
                        prompt: ids for prompt, ids in zip(
                            original_prompts, original_ids, strict=True
                        )
                    }
                    if (
                        len(set(prompts)) != len(prompts)
                        or any(prompt not in by_prompt for prompt in prompts)
                        or sources["stock_control"] != [by_prompt[prompt] for prompt in prompts]
                    ):
                        raise SessionProtocolError(
                            "teacher prompts/B' control do not match measured stock output"
                        )
                    traces = _teacher_forced_traces(
                        engine,
                        prompts,
                        sources,
                        top_logprobs_num=int(measure_cfg.top_logprobs_num),
                    )
                    _write_all(protocol, teacher_evidence_frame(
                        traces,
                        session_id=session_id,
                        request_id=request_id,
                        nonce=nonce,
                        phase=phase,
                        batch_index=batch_index,
                        sealed_rollout_sha256=seal,
                        token_count=int(measure_cfg.max_new_tokens),
                        top_logprobs_num=int(measure_cfg.top_logprobs_num),
                    ))
                stage = "close"
                next_message = _read_json_frame(
                    control_input, max_bytes=MAX_CONTROL_BYTES
                )
            close_binding = validate_close_request(next_message)
            if close_binding[0] != session_id:
                raise SessionProtocolError("close request session mismatch")
        assert close_binding is not None
        _write_all(protocol, frame_message(
            closed_message(
                session_id=session_id,
                request_id=close_binding[1],
                nonce=close_binding[2],
                audit_receipts=audit_out,
                audit_members=member_out,
            ),
            max_bytes=MAX_CONTROL_BYTES,
        ))
        return 0
    except BaseException as exc:  # noqa: BLE001 - untrusted diagnostic only
        try:
            _write_all(protocol, frame_message(
                error_message(
                    session_id=session_id or "0" * 32,
                    stage=stage,
                    error=exc,
                ),
                max_bytes=MAX_CONTROL_BYTES,
            ))
        except BaseException:
            pass
        return 1
    finally:
        os.close(control_input)
        if output_fd is None:
            os.close(protocol)


def main() -> int:
    return run_session()


if __name__ == "__main__":
    raise SystemExit(main())
