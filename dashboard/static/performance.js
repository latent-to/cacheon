"use strict";

function measuredValue(tokensPerSecond, meanLatency, replayRate) {
  if (replayRate != null) return `<b>${metricNumber(replayRate, 3)}</b> turns/s (lane-balanced)`;
  if (meanLatency != null) return `<b>${metricNumber(meanLatency, 3)}</b> s/warm turn`;
  return tokensPerSecond != null ? `<b>${metricNumber(tokensPerSecond)}</b> tok/s` : '<span class="muted">Unavailable</span>';
}

function replayPerformance(speed, title) {
  const grade = speed.grading;
  const fastest = speed.score_basis === "fastest_pass_latency";
  const gap = Math.max(0, (grade.required_speedup - speed.speedup) * 100);
  const read = lane => `${lane.role === "B" ? "Incumbent" : "Candidate"} · pass ${metricNumber(lane.window, 0)}`;
  const rows = speed.lanes.map((lane) => `<tr>
    <td>${read(lane)} ${lane.used_for_score ? '<span class="pill info">scored</span>' : ''}</td>
    <td>${metricNumber(lane.warm_turns, 0)}</td><td>${metricNumber(fastest ? lane.mean_warm_latency_s : lane.elapsed_s, 3)}</td>
    ${fastest ? '' : `<td>${metricNumber(lane.turns_per_second, 3)}</td>`}
    <td>${metricNumber(lane.attainment * 100, 2)}%</td></tr>`);
  const diagnostics = speed.lanes.map((lane) => `<tr><td>${read(lane)}</td>
    <td>${metricNumber(lane.mean_ttft_s, 3)}</td><td>${metricNumber(lane.p95_ttft_s, 3)}</td>
    <td>${metricNumber(lane.median_decode_tps, 1)}</td></tr>`);
  return `<section class="performance">${title}
    <h4>Agent replay speedup</h4>
    <p>This attempt replayed a fixed conversation workload at ${metricNumber(speed.load, 0)} concurrent sessions per lane,
      with ${metricNumber(speed.windows, 0)} paired passes. The recorded workload determines the measurement, including for earlier submissions.</p>
    <div class="cards">${card(metricGain(speed.speedup), "Scored improvement")}
      ${card(metricGain(grade.required_speedup), "Required improvement")}
      ${grade.lower_speedup == null ? '' : card(metricGain(grade.lower_speedup), "Lower gain bound (calibrated model)")}
      ${card(metricNumber(gap, 2) + " pp", "Gap to speed threshold")}
      ${card(metricNumber(speed.speed_stage_seconds / 60, 1) + " min", "Speed stage, including startup")}</div>
    <p>${esc(grade.detail)}.</p>
    <div class="metrics-table">${table(["Read", "Warm turns", fastest ? "Mean turn (s)" : "Elapsed serving (s)", ...(fastest ? [] : ["Turns/s"]), "Service attainment"], rows)}</div>
    <p class="metric-note">Timing starts at the first warm release; the opening cold-prefill round is excluded.
      ${fastest ? "The score divides the incumbent’s fastest complete-pass latency by the candidate’s fastest complete-pass latency. The scored passes are marked above."
        : "The score pools elapsed serving time within each lane orientation and combines both orientations geometrically. Mean request latency is diagnostic."}</p>
    <details><summary>Latency and service diagnostics</summary>
    <div class="metrics-table">${table(["Read", "Mean TTFT (s)", "P95 TTFT (s)", "Decode (tok/s)"], diagnostics)}</div>
    <p class="metric-note">TTFT includes queueing. Decode is the median per-user rate after the first token.
      Unsuccessful requests across all passes: ${metricNumber(speed.lanes.reduce((n, row) => n + row.unsuccessful_turns, 0), 0)}.</p>
    <p class="metric-note">Service contract: first token within ${metricNumber(speed.contract.ttft_bound_s, 2)} s,
      decode at least ${metricNumber(speed.contract.decode_floor_tps, 1)} tok/s.
      Attainment includes every attempted turn. Candidate attainment less its noise allowance of
      ${metricNumber(grade.attainment_margin * 100, 2)} percentage points must stay within
      ${metricNumber(grade.attainment_tolerance * 100, 2)} percentage points of the incumbent in each window.</p>
    <p class="metric-note">${fastest ? 'Configured noise allowance' : 'Configured paired window log SD'}: ${metricNumber(grade.null_noise * 100, 3)}% · replay policy ${metricNumber(speed.policy_version, 0)}.
      ${grade.standard_error == null ? '' : `Final log standard error: ${metricNumber(grade.standard_error * 100, 3)}%; configured paired boot log SD: ${metricNumber(grade.boot_noise * 100, 3)}%. The statistical bound depends on the commissioned calibration.`}
      Workload <code style="overflow-wrap:anywhere">${esc(speed.workload_digest)}</code>.</p></details>
  </section>`;
}

const metricNumber = (value, digits = 1) => value != null && Number.isFinite(Number(value))
  ? Number(value).toLocaleString(undefined, {minimumFractionDigits: digits, maximumFractionDigits: digits}) : "—";
const metricMilliseconds = (seconds) => metricNumber(seconds == null ? null : Number(seconds) * 1000, 2);
const metricGain = (ratio) => `${Number(ratio) >= 1 ? "+" : ""}${metricNumber((Number(ratio) - 1) * 100, 2)}%`;
const readLabel = (role) => ({B: "B · baseline before", C: "C · candidate", B_prime: "B′ · baseline after",
  B_double_prime: "B″ · baseline after", B_prefill: "B · baseline before",
  C_prefill: "C · candidate", B_prime_prefill: "B′ · baseline after"})[role] || role;

function performanceMetrics(attempt) {
  const speed = attempt.speed;
  const title = `<h2>Performance · attempt #${esc(attempt.attempt)}</h2>
    <p>${pill(attempt.decision)} <span class="muted small">${esc(attempt.reason || "")}</span></p>`;
  if (speed?.grading_error && !speed.lanes.length)
    return `<section class="performance">${title}<p class="notice">Retained grading evidence could not be read: ${esc(speed.grading_error)}</p></section>`;
  if (["warm_turn_latency", "fixed_work_rate"].includes(speed?.metric)) return replayPerformance(speed, title);
  if (!speed || !speed.lanes.length)
    return `<section class="performance">${title}<p class="metric-note">Retained measurements are unavailable for this attempt.</p></section>`;
  const output = speed.lanes.filter((lane) => lane.tokens_per_second != null);
  const prefill = speed.lanes.filter((lane) => lane.prompts_per_second != null);
  const rateRows = (lanes, field, digits) => lanes.map((lane) => `<tr>
    <td>${esc(readLabel(lane.role))}</td><td class="mono">${metricNumber(lane[field], digits)}</td>
    <td class="mono">${metricNumber(lane.timed_seconds, 3)}</td></tr>`);
  const workloads = new Map();
  for (const lane of output) for (const cell of lane.cells || []) {
    const shape = [cell.input_tokens, cell.output_tokens, cell.concurrency].join(":");
    if (!workloads.has(shape)) workloads.set(shape, {cell, rows: []});
    workloads.get(shape).rows.push(`<tr><td>${esc(readLabel(lane.role))}</td>
      <td class="mono">${metricMilliseconds(cell.mean_ttft_seconds)}</td>
      <td class="mono">${metricMilliseconds(cell.mean_tpot_seconds)}</td>
      <td class="mono">${metricNumber(cell.end_to_end_output_tokens_per_second)}</td>
      <td>${metricNumber(cell.timed_batches, 0)}</td></tr>`);
  }
  const latencyTables = [...workloads.values()].map(({cell, rows}) => `
    <p class="workload-label">${metricNumber(cell.input_tokens, 0)} input tokens · ${metricNumber(cell.output_tokens, 0)} output tokens · concurrency ${metricNumber(cell.concurrency, 0)}</p>
    <div class="metrics-table">${table(["Read", "Mean TTFT (ms)", "Mean TPOT (ms)", "Output tok/s", "Batches"], rows)}</div>`).join("");
  const promptGain = speed.prefill;
  return `<section class="performance">${title}
    <h4>Output throughput</h4>
    <div class="metrics-table">${table(["Read", "Output tok/s", "Timed batches (s)"], rateRows(output, "tokens_per_second", 1))}</div>
    <p class="metric-note">Output throughput includes prompt processing and generation across the measured workload.</p>
    <h4>Prefill</h4>
    ${promptGain ? `<div class="cards">
      ${card(metricGain(promptGain.speedup), "Measured prompt-throughput gain")}
      ${card(promptGain.min_margin != null ? metricNumber(Number(promptGain.min_margin) * 100, 2) + "%" : "Not recorded", "Required prefill margin")}
    </div><p class="metric-note">Compared with the faster baseline read. Qualification also checks decode performance and correctness.</p>` : ""}
    <div class="metrics-table">${table(["Read", "Prompts/s", "Timed batches (s)"], rateRows(prefill, "prompts_per_second", 3), "Prefill was not measured separately in this evaluation.")}</div>
    ${prefill.length ? '<p class="metric-note">Each prompt pass generates one output token. Prompts/s counts completed requests; batch times include every timed prompt batch.</p>' : ""}
    <h4>TTFT / TPOT by workload</h4>
    ${latencyTables ? `${latencyTables}
      <p class="metric-note">TTFT is time to the first delivered token. TPOT is delivery time per subsequent output token. Input length, output length and concurrency are kept separate.</p>`
      : '<div class="empty">Not measured in this evaluation. First-token and subsequent-token delivery timings were not recorded.</div>'}
  </section>`;
}
