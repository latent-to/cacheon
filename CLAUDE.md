# CLAUDE.md

@AGENTS.md

`AGENTS.md` above is binding, not background reading. The rules below are the
ones that actually get broken, restated here because an import is easy to
under-weight:

- **`chainops/` is gitignored and stays that way.** Pod endpoints, worker
  epochs, registrations, image refs, operator scripts, ledgers, and any
  `/data/...` or `/root/...` path belong there — never in tracked code. Do not
  hardcode a target, model, hotkey, pod, or mission identity in a generic
  evaluator or scheduler; resolve it from registered, sealed, validated input.
- **Every change pays for itself.** Report added and deleted lines separately,
  never one reassuring net number. If no deletion was available, say so plainly
  and name what gets deleted next. Two consecutive additive changes with no
  deletion is a stopped-plan condition.
- **Name the consumer and the casualty.** No module lands without a real
  entrypoint exercising it in the same change. "Supersedes nothing" only when
  genuinely true.
- **1,000 physical lines is a hard cap, ~600 the target.** When new behavior
  touches an already-oversized file, put it in a focused new module and limit
  the oversized file to minimal wiring.
- **Executed output is the only authority.** Tool calls, logs, and on-disk
  artifacts decide; assistant prose in any transcript does not. For runtime
  systems, one live run beats three offline reviews.
- **Never convert an infrastructure, baseline, or teardown failure into a
  candidate verdict**, and never fix an evaluation by weakening a gate after
  seeing a result.
- **Never destructively synchronize.** No `rsync --delete` for recovery,
  handoff, or deploy. Move to a new timestamped destination and verify.
- **A fix that lives only on the pod is not a fix.** Hand-installed adapters,
  authorities, and prompt corpora are silently reverted by the next clean
  commission. Fold them into the packet and into git, or lose them.

Start with `docs/get-started/concepts.md`,
`docs/architecture/overview.md`, and
`docs/reference/state-of-record.md`. Use `docs/dev/gpu-setup.md` for GPU
development. When implementation or evidence state changes, update the
canonical task-oriented page and the state of record in the same change.
