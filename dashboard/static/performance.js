"use strict";

// Batch-cell rows only (speed policies 8-15); a replay row reports resultLine and replayLine.
const measuredValue = (tokensPerSecond) => tokensPerSecond != null
  ? `<b>${metricNumber(tokensPerSecond)}</b> tok/s` : '<span class="muted">Unavailable</span>';

// "A → B" for a baseline and a candidate reading, with the candidate's gain when one direction is better.
const sideBySide = (base, cand, digits, better, unit = "") => `${metricNumber(base, digits)} → ${metricNumber(cand, digits)}${unit}${
  better && base && cand ? ` <span class="muted small">${metricGain(better === "more" ? cand / base : base / cand)}</span>` : ""}`;

// A replay result as one line for a table cell: the old turns/s figure told a reader nothing about where a gain was.
const replayLine = (r) => r && r.decode_tps[0] != null
  ? `decode ${sideBySide(...r.decode_tps, 1, "more", " tok/s")}<div>first token ${sideBySide(...r.ttft_s, 2, "less", " s")}</div>` : "";

// What a graded replay row leads with. On 2026-10-01 a +6.13% PASS and a +1.06% PASS both read "PASS / qualified".
const neededLine = (r) => `needed ${metricGain(r.required_speedup)} · ${r.passes}${r.pass_limit ? " of " + r.pass_limit : ""} passes`;
const resultLine = (decision, r) => `<b${{PASS: ' style="color:var(--ok)"', FAIL: ' style="color:var(--bad)"'}[decision] || ""}>${esc(decision)} ${metricGain(r.speedup)}</b>
  <div class="muted small" title="${esc(r.detail)}">${neededLine(r)}</div><div class="muted small">${replayLine(r)}</div>`;

function replayPerformance(speed, title) {
  const grade = speed.grading;
  const fastest = speed.score_basis === "fastest_pass_latency";
  const margin = (speed.speedup - grade.required_speedup) * 100;
  // A pass pairs the two arms that ran at the same moment: same lane orientation, same pass number.
  // Both orientations number their passes from 1; on 2026-10-01 every row of a swapped run read "pass 1".
  const swaps = {B: [], C: []};
  for (const lane of speed.lanes) if (!swaps[lane.role].includes(lane.physical_lane)) swaps[lane.role].push(lane.physical_lane);
  const slot = (lane) => swaps[lane.role].indexOf(lane.physical_lane) + ":" + lane.window;
  const cost = (lane) => fastest ? lane.mean_warm_latency_s : lane.elapsed_s;
  const mark = (lane) => lane.used_for_score ? `<b>${metricNumber(cost(lane), 3)}</b>` : metricNumber(cost(lane), fastest ? 3 : 1);
  const early = speed.window_limit > speed.windows;
  const rows = speed.lanes.filter((lane) => lane.role === "B").map((base, pass) => {
    const cand = speed.lanes.find((lane) => lane.role === "C" && slot(lane) === slot(base));
    return cand ? `<tr><td>Pass ${pass + 1}${swaps.B.indexOf(base.physical_lane) ? ' <span class="muted small">lanes swapped</span>' : ""}</td>
      <td><b>${metricGain(cost(base) / cost(cand))}</b></td><td>${mark(base)} → ${mark(cand)}</td>
      <td>${sideBySide(base.decode_tps, cand.decode_tps, 1, "more")}</td>
      <td>${sideBySide(base.mean_ttft_s, cand.mean_ttft_s, 2, "less")}<div class="muted small">p95 ${sideBySide(base.p95_ttft_s, cand.p95_ttft_s, 2)}</div></td>
      <td>${sideBySide(base.attainment * 100, cand.attainment * 100, 1)}</td></tr>` : "";
  });
  const arm = (role, field) => { const lanes = speed.lanes.filter((lane) => lane.role === role);
    return lanes.reduce((n, lane) => n + lane[field], 0) / lanes.length; };
  return `<section class="performance">${title}
    <h4>Agent replay: measured against the baseline</h4>
    <p>Both sides replayed the same ${metricNumber(speed.lanes[0].warm_turns, 0)} coding-agent turns at ${metricNumber(speed.load, 0)} concurrent sessions per lane,
      ${metricNumber(speed.windows, 0)} time${speed.windows === 1 ? "" : "s"}${early ? ` of a possible ${metricNumber(speed.window_limit, 0)}` : ""}.</p>
    <div class="cards">${card(metricGain(speed.speedup), "Measured gain", grade.decision === "PASS" ? "ok" : "bad")}
      ${card(metricGain(grade.required_speedup), early ? `Needed at pass ${metricNumber(speed.windows, 0)} of ${metricNumber(speed.window_limit, 0)}` : "Needed")}
      ${card(sideBySide(arm("B", "decode_tps"), arm("C", "decode_tps"), 1, "more"), "Decode tok/s, baseline → candidate")}
      ${card(sideBySide(arm("B", "mean_ttft_s"), arm("C", "mean_ttft_s"), 2, "less"), "First token (s), baseline → candidate")}
      ${card((margin >= 0 ? "+" : "") + metricNumber(margin, 2) + " pp", margin >= 0 ? "Above what was needed" : "Short of what was needed")}
      ${card(metricNumber(speed.speed_stage_seconds / 60, 1) + " min", "Speed stage, including startup")}</div>
    <p>${esc(grade.detail[0].toUpperCase() + grade.detail.slice(1))}.${early ? " What is needed shrinks with every pass, so a full run needs less than the figure shown for this pass." : ""}
      ${grade.futility_margin ? `A run whose first lane orientation reads more than ${metricNumber(grade.futility_margin * 100, 2)}% slower than the baseline stops there.` : ""}</p>
    <div class="metrics-table">${table(["Pass", "Gain", fastest ? "Mean turn (s)" : "Same work (s)", "Decode (tok/s)", "First token (s)", "In service level (%)"], rows)}</div>
    <p class="metric-note">Every pair reads baseline → candidate. Each pass runs the two at the same moment on two lanes. Timing starts at the first warm release; the opening cold-prefill round is excluded.
      ${fastest ? "The score divides the baseline’s fastest pass by the candidate’s fastest pass; those two readings are in bold."
        : "The score pools the time within each lane orientation and combines the two orientations geometrically."}
      Decode is every token after the first over every second after it; first-token time includes queueing.</p>
    <details><summary>Service level and noise settings</summary>
    <p class="metric-note">Unsuccessful requests across all passes: ${metricNumber(speed.lanes.reduce((n, row) => n + row.unsuccessful_turns, 0), 0)}.
      ${grade.lower_speedup == null ? "" : `Lower bound of the gain: ${metricGain(grade.lower_speedup)}.`}</p>
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
