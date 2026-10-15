"use strict";

// Render only observed checkpoints; stages can finish early and have unequal durations.
function runProgress(progress) {
  if (!progress) return '<div id="run-progress"><p class="muted small">Loading run status…</p></div>';
  const completed = progress.completed || [];
  const stages = [["speed", "Speed replay"], ["audit", "Correctness audit"], ["quality", "Pristine check"]];
  return `<section class="callout" aria-label="Run progress">
    <h2>Run progress</h2><strong>${esc(progress.label)}</strong>
    <div class="small">Worker: ${esc(progress.worker_state)}${progress.heartbeat_age_s != null
      ? ` · heartbeat ${dur(progress.heartbeat_age_s)} ago` : ""}</div>
    ${progress.request_id ? `<div class="small">Request <code>${esc(short(progress.request_id, 12))}</code></div>` : ""}
    ${progress.state === "active" ? `<p>${stages.map(([id, label]) =>
      `<span class="pill ${completed.includes(id) ? "ok" : "mute"}">${completed.includes(id) ? "✓" : "○"} ${label}</span>`).join(" ")}</p>
      <div class="small">${completed.length} of 3 completion checkpoints recorded${progress.replay_windows_completed
        ? ` · ${progress.replay_windows_completed} replay windows retained` : ""}${progress.first_orientation_completed
        ? " · first lane orientation retained" : ""}.
      ${progress.checkpoint_age_s != null ? `Checkpoint observed ${dur(progress.checkpoint_age_s)} ago.` : ""}
      Stages take different amounts of time and a run may finish early.</div>` : ""}
  </section>`;
}

async function refreshRunProgress() {
  if (!["overview", "queue"].includes(tab)) return;
  const panel = document.getElementById("run-progress"), requestedArena = arena;
  if (!panel) return;
  const current = () => document.getElementById("run-progress") === panel && requestedArena === arena;
  try {
    const progress = await api("/api/progress");
    if (current()) panel.innerHTML = runProgress(progress);
  } catch (err) {
    if (current()) panel.textContent = `Unable to load run progress: ${err.message}`;
  }
}
