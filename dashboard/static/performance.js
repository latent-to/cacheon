"use strict";

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
  const margin = (speed.speedup - grade.required_speedup) * 100;
  const early = speed.window_limit > speed.windows;
  // One engine at a time runs on the whole lane: rows are fresh boots, not simultaneous pairs.
  const mean = (lanes, field) => lanes.reduce((n, lane) => n + lane[field], 0) / lanes.length;
  const bootRows = (role, label) => Object.values(speed.lanes.filter((lane) => lane.role === role)
    .reduce((boots, lane) => ((boots[lane.boot] ||= []).push(lane), boots), {})).map((lanes, boot) => `<tr>
      <td>${label(boot)}</td><td>${metricNumber(mean(lanes, "elapsed_s"), 1)}</td><td>${metricNumber(mean(lanes, "decode_tps"), 1)}</td>
      <td>${metricNumber(mean(lanes, "mean_ttft_s"), 2)}</td><td>${metricNumber(mean(lanes, "attainment") * 100, 1)}</td></tr>`);
  const rows = [...bootRows("B", (boot) => boot < speed.reference_boots ? `Stock reference boot ${boot + 1}` : "Stock bracket after PASS"),
    ...bootRows("C", (boot) => `Candidate boot ${boot + 1}`)];
  const arm = (role, field) => { const lanes = speed.lanes.filter((lane) => lane.role === role);
    return lanes.reduce((n, lane) => n + lane[field], 0) / lanes.length; };
  return `<section class="performance">${title}
    <h4>Agent replay: measured against the baseline</h4>
    <p>Both sides replayed the same ${metricNumber(speed.lanes[0].warm_turns, 0)} coding-agent turns at ${speed.loads.map((load) => metricNumber(load, 0)).join(" / ")} concurrent sessions per lane,
      ${metricNumber(speed.window_limit, 0)} times per fresh engine boot; the candidate ran ${metricNumber(speed.windows / speed.window_limit, 0)} boots against ${metricNumber(speed.reference_boots, 0)} retained stock boots.</p>
    <div class="cards">${card(metricGain(speed.speedup), "Measured gain", grade.decision === "PASS" ? "ok" : "bad")}
      ${card(metricGain(grade.required_speedup), early ? `Needed at pass ${metricNumber(speed.windows, 0)} of ${metricNumber(speed.window_limit, 0)}` : "Needed")}
      ${card(sideBySide(arm("B", "decode_tps"), arm("C", "decode_tps"), 1, "more"), "Decode tok/s, baseline → candidate")}
      ${card(sideBySide(arm("B", "mean_ttft_s"), arm("C", "mean_ttft_s"), 2, "less"), "First token (s), baseline → candidate")}
      ${card((margin >= 0 ? "+" : "") + metricNumber(margin, 2) + " pp", margin >= 0 ? "Above what was needed" : "Short of what was needed")}
      ${card(metricNumber(speed.speed_stage_seconds / 60, 1) + " min", "Speed stage, including startup")}</div>
    <p>${esc(grade.detail[0].toUpperCase() + grade.detail.slice(1))}.${early ? " What is needed shrinks with every pass, so a full run needs less than the figure shown for this pass." : ""}
      ${grade.futility_margin ? `A run whose first boot reads more than ${metricNumber(grade.futility_margin * 100, 2)}% slower than the baseline stops there.` : ""}</p>
    <div class="metrics-table">${table(["Engine boot", "Same work (s)", "Decode (tok/s)", "First token (s)", "In service level (%)"], rows)}</div>
    <p class="metric-note">One engine at a time owns the whole lane. The stock reference is retained across candidates, and its stored answers supply the quality check after a PASS.
      Timing starts at the first warm release; the opening cold-prefill round is excluded. The score compares the mean log same-work time per boot.
      Decode is every token after the first over every second after it; first-token time includes queueing.</p>
    <details><summary>Service level and noise settings</summary>
    <p class="metric-note">Unsuccessful requests across all passes: ${metricNumber(speed.lanes.reduce((n, row) => n + row.unsuccessful_turns, 0), 0)}.
      Lower bound of the gain: ${metricGain(grade.lower_speedup)}.</p>
    <p class="metric-note">Service contract: first token within ${metricNumber(speed.contract.ttft_bound_s, 2)} s,
      decode at least ${metricNumber(speed.contract.decode_floor_tps, 1)} tok/s.
      Attainment includes every attempted turn. ${grade.attainment_tolerance >= 1 ? "It is reported only: this commission seals no attainment bound." : `Candidate attainment less its noise allowance of
      ${metricNumber(grade.attainment_margin * 100, 2)} percentage points must stay within
      ${metricNumber(grade.attainment_tolerance * 100, 2)} percentage points of the incumbent ${grade.attainment_pooled ? "over the pooled windows" : "in each window"}.`}</p>
    <p class="metric-note">Configured window log SD: ${metricNumber(grade.null_noise * 100, 3)}% · replay policy ${metricNumber(speed.policy_version, 0)}.
      Final log standard error: ${metricNumber(grade.standard_error * 100, 3)}%; configured boot log SD: ${metricNumber(grade.boot_noise * 100, 3)}%. The statistical bound depends on the commissioned calibration.
      Workload <code style="overflow-wrap:anywhere">${esc(speed.workload_digest)}</code>.</p></details>
  </section>`;
}

const metricNumber = (value, digits = 1) => value != null && Number.isFinite(Number(value))
  ? Number(value).toLocaleString(undefined, {minimumFractionDigits: digits, maximumFractionDigits: digits}) : "—";
const metricGain = (ratio) => `${Number(ratio) >= 1 ? "+" : ""}${metricNumber((Number(ratio) - 1) * 100, 2)}%`;

function performanceMetrics(attempt) {
  const speed = attempt.speed;
  const title = `<h2>Performance · attempt #${esc(attempt.attempt)}</h2>
    <p>${pill(attempt.decision)} <span class="muted small">${esc(attempt.reason || "")}</span></p>`;
  if (speed?.grading_error && !speed.lanes.length)
    return `<section class="performance">${title}<p class="notice">Retained grading evidence could not be read: ${esc(speed.grading_error)}</p></section>`;
  if (speed?.metric === "fixed_work_rate") return replayPerformance(speed, title);
  // A retained batch-cell attempt (speed policies 8-15) renders no rates: its lane ratio was never the credited gain.
  return `<section class="performance">${title}<p class="metric-note">Retained measurements are unavailable for this attempt.</p></section>`;
}
