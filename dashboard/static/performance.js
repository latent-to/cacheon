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
  // A first lifetime under the continue line ends the stage; its statistical "needed" figure was never the test.
  const stopped = grade.continue_at != null && speed.candidate_boots > 1
    && speed.lanes.filter((lane) => lane.role === "C").every((lane) => lane.boot === 0);
  const needed = stopped ? grade.continue_at : grade.required_speedup;
  const margin = (speed.speedup - needed) * 100;
  const early = speed.window_limit > speed.windows;
  // One engine at a time runs on the whole lane: rows are fresh boots, not simultaneous pairs.
  const mean = (lanes, field) => lanes.reduce((n, lane) => n + lane[field], 0) / lanes.length;
  // Goodput pools served output tokens over the summed timed span; a mean of per-load rates would weight loads equally.
  const goodput = (lanes) => lanes.reduce((n, lane) => n + lane.goodput_tps * lane.elapsed_s, 0)
    / lanes.reduce((n, lane) => n + lane.elapsed_s, 0);
  const bootRows = (role, label) => Object.values(speed.lanes.filter((lane) => lane.role === role)
    .reduce((boots, lane) => ((boots[lane.boot] ||= []).push(lane), boots), {})).map((lanes, boot) => `<tr>
      <td>${label(boot)}</td><td>${metricNumber(mean(lanes, "elapsed_s"), 1)}</td><td>${metricNumber(goodput(lanes), 0)}</td><td>${metricNumber(mean(lanes, "decode_tps"), 1)}</td>
      <td>${metricNumber(mean(lanes, "mean_ttft_s"), 2)}</td><td>${metricNumber(mean(lanes, "attainment") * 100, 1)}</td></tr>`);
  const rows = [...bootRows("B", (boot) => boot < speed.reference_boots ? `Stock reference boot ${boot + 1}` : "Stock bracket after PASS"),
    ...bootRows("C", (boot) => `Candidate boot ${boot + 1}`)];
  const arm = (role, field) => mean(speed.lanes.filter((lane) => lane.role === role), field);
  return `<section class="performance">${title}
    <h4>Agent replay: measured against the baseline</h4>
    <p>Both sides replayed the same ${metricNumber(speed.lanes[0].warm_turns, 0)} coding-agent turns at ${speed.loads.map((load) => metricNumber(load, 0)).join(" / ")} concurrent sessions per lane,
      ${metricNumber(speed.window_limit, 0)} times per fresh engine boot; the candidate ran ${metricNumber(speed.windows / speed.window_limit, 0)} boots against ${metricNumber(speed.reference_boots, 0)} retained stock boots.</p>
    <div class="cards">${card(metricGain(speed.speedup), "Measured gain", grade.decision === "PASS" ? "ok" : "bad")}
      ${card(metricGain(needed), stopped ? "Needed after boot 1 to run boot 2" : early ? `Needed at pass ${metricNumber(speed.windows, 0)} of ${metricNumber(speed.window_limit, 0)}` : "Needed")}
      ${grade.payment_bar == null ? "" : card(metricGain(grade.payment_bar), "Payment bar (best paid PASS + 1.5%)")}
      ${speed.lanes[0].goodput_tps == null ? "" : card(sideBySide(...["B", "C"].map((role) => goodput(speed.lanes.filter((lane) => lane.role === role))), 0, "more"),
        "Warm output tok/s, baseline → candidate")}
      ${card(sideBySide(arm("B", "decode_tps"), arm("C", "decode_tps"), 1, "more"), "Decode tok/s, baseline → candidate")}
      ${card(sideBySide(arm("B", "mean_ttft_s"), arm("C", "mean_ttft_s"), 2, "less"), "First token (s), baseline → candidate")}
      ${card((margin >= 0 ? "+" : "") + metricNumber(margin, 2) + " pp", margin >= 0 ? "Above what was needed" : "Short of what was needed")}
      ${card(metricNumber(speed.speed_stage_seconds / 60, 1) + " min", "Speed stage, including startup")}</div>
    <p>${esc(grade.detail[0].toUpperCase() + grade.detail.slice(1))}.${early ? " What is needed shrinks with every pass, so a full run needs less than the figure shown for this pass." : ""}
      ${grade.continue_at == null ? "" : grade.payment_bar != null && grade.continue_at > 1 - grade.futility_margin
        ? `A run whose first boot reads more than ${metricNumber(grade.futility_margin * 100, 2)}% below the payment bar, under ${metricGain(grade.continue_at)}, stops there: its second boot could not lift it to a paid PASS.${stopped ? " This run stopped after its first boot." : ""}`
        : `A run whose first boot reads more than ${metricNumber(grade.futility_margin * 100, 2)}% slower than the baseline stops there.`}</p>
    <div class="metrics-table">${table(["Engine boot", "Same work (s)", "Warm output tok/s", "Decode (tok/s)", "First token (s)", "In service level (%)"], rows)}</div>
    ${(speed.per_load || []).length ? `<div class="metrics-table">${table(["Load", grade.steady_span ? "Scored s per warm turn" : "Scored same work (s)",
      "Gain", "Warm output tok/s", "Within service limits", "Decode median tok/s (floor)"], speed.per_load.map((row) => `<tr><td>${metricNumber(row.load, 0)} sessions</td>
      <td>${sideBySide(row.stock_cost, row.candidate_cost, grade.steady_span ? 4 : 1)}</td><td><b>${metricGain(row.gain)}</b></td>
      <td>${sideBySide(row.stock_goodput_tps, row.candidate_goodput_tps, 0, "more")}</td>
      <td>${row.stock_service && row.candidate_service ? `${sideBySide(row.stock_service.turns_per_second, row.candidate_service.turns_per_second, 2, "more", " turns/s")}
        <div>${sideBySide(row.stock_service.output_tps, row.candidate_service.output_tps, 1, "more", " output tok/s")}</div>
        <div>${sideBySide(100 * row.stock_service.attainment, 100 * row.candidate_service.attainment, 1, null, "% of turns")}</div>` : "—"}</td>
      <td>${sideBySide(row.stock_decode_tps, row.candidate_decode_tps, 1, "more")}${row.decode_floor_tps == null ? "" : ` (${metricNumber(row.decode_floor_tps, 0)})`}</td></tr>`))}</div>
    <p class="metric-note">The measured gain is the geometric mean of the per-load gains.${grade.steady_span
      ? ` Each load is scored until fewer than ${metricNumber(grade.steady_span * 100, 0)}% of its sessions are still active after the last one starts, so the drain of the longest sessions does not set the time.` : ""}</p>` : ""}
    <p class="metric-note">One engine at a time owns the whole lane. The stock reference is retained across candidates, and its stored answers supply the quality check after a PASS.
      The speed score starts at the first warm release. Rates within service limits count turns meeting both first-token and decode limits over the full replay, including opening requests and drain.
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
