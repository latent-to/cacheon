"""Execute one planned OCI batch and timestamp its outputs on the host."""

from cacheon.eval.oci_session_protocol import (
    MAX_BATCH_REQUEST_BYTES,
    AuditReceiptFacts,
    SessionProtocolError,
    batch_request,
    frame_message,
    validate_batch_request,
    validate_audit_evidence,
)


def execute_batch(self):
    """Advance the production outer session by exactly one disclosed batch."""
    from cacheon.eval.oci_outer_session import (
        BatchExecutionEvidence,
        OuterSessionInfrastructureError,
        OuterSessionProtocolError,
        _fresh_id,
        _now,
        _control_or_error,
    )

    if not self.started or self.closed:
        raise OuterSessionInfrastructureError("session is not open")
    index = self.next_batch_index
    if index >= len(self.plan.prompt_batches):
        raise OuterSessionInfrastructureError("session has no remaining planned batch")
    prompts = self.plan.prompt_batches[index]
    max_new_tokens, expected_prompt_tokens = self.plan.request_geometry(index)
    try:
        request_id, nonce = _fresh_id(self.seen), _fresh_id(self.seen)
        request = validate_batch_request(
            batch_request(
                session_id=self.session_id,
                launch_digest=self.plan.launch_digest,
                request_id=request_id,
                nonce=nonce,
                batch_index=index,
                prompts=prompts,
                max_new_tokens=max_new_tokens,
                top_logprobs_num=self.plan.top_logprobs_num,
                temperature=self.plan.temperature,
                expected_prompt_tokens=expected_prompt_tokens,
                measure_phase_latency=self.plan.measure_phase_latency,
            )
        )
        final_warmup = index == self.plan.warmup_count - 1
        first_timed = index == self.plan.warmup_count
        if final_warmup and self.boundary_callback is not None:
            self.boundary_callback("before_final_warmup", index, self.deadline)
        if first_timed and self.boundary_callback is not None:
            self.boundary_callback("before_first_timed", index, self.deadline)
        if self.transport.has_pending_output():
            raise OuterSessionProtocolError("worker emitted early or duplicate output")
        batch_deadline = self._phase_deadline(self.batch_timeout_s)
        request_started = _now(self.clock, previous=self.last_host_time)
        self.transport.write_frame(
            frame_message(request.to_dict(), max_bytes=MAX_BATCH_REQUEST_BYTES),
            deadline=batch_deadline,
        )
        from cacheon.eval.phase_latency import HostTokenClock

        token_clock = HostTokenClock(request, self.clock, request_started)
        progress = (
            {"on_progress": token_clock.observe}
            if request.measure_phase_latency
            else {}
        )
        evidence = self.transport.read_evidence(
            request, deadline=batch_deadline, **progress
        )
        audit_receipts: tuple[AuditReceiptFacts, ...] = ()
        if self.plan.audit_policy is not None:
            try:
                audit_receipts = validate_audit_evidence(
                    _control_or_error(
                        self.transport,
                        session_id=self.session_id,
                        launch_digest=self.plan.launch_digest,
                        deadline=batch_deadline,
                    ),
                    request=request,
                    policy=self.plan.audit_policy,
                )
            except SessionProtocolError as exc:
                raise OuterSessionProtocolError(str(exc)) from None
        completed = _now(self.clock, previous=request_started)
        if completed <= request_started:
            raise OuterSessionInfrastructureError("host batch clock did not advance")
        token_numerator = len(prompts) * max_new_tokens
        if evidence.observed_tokens != token_numerator:
            raise OuterSessionProtocolError("worker evidence token count is not exact")
        row = BatchExecutionEvidence(
            index,
            request_id,
            nonce,
            request_started,
            completed,
            token_numerator,
            evidence,
            audit_receipts,
            token_clock.finish(evidence, completed)
            if request.measure_phase_latency
            else (),
        )
        self.batch_rows.append(row)
        self.last_host_time = completed
        if index + 1 == self.conditioning_start_index:
            self.conditioning_started_at = completed
        if final_warmup and self.boundary_callback is not None:
            self.boundary_callback("after_final_warmup", index, self.deadline)
        if first_timed:
            self.first_timed_completed_at = completed
        if self.transport.has_pending_output():
            raise OuterSessionProtocolError(
                "worker emitted trailing or duplicate output"
            )
        return row
    except BaseException as exc:
        self._fail(exc)
