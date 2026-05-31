# Staying current with sglang

"Stay up to date with sglang" is **two** problems, because sglang is both:

- our **baseline** — we score a kernel by its speedup vs sglang's own kernels, so a
  stale baseline means miners optimize against an old frontier (and "wins" may
  already be upstream); and
- our **runtime** — we patch sglang internals (the `SiluAndMul` / `RMSNorm` seams,
  `MultiPlatformOp`, the Engine logprob API, specific `ServerArgs` kwargs), so any
  upgrade can break us.

## The hard constraint: a pinned version (consensus)

You **cannot** have validators on different sglang versions. Different sglang →
different baseline kernels → different scores → Bittensor consensus breaks. So the
sglang version is a **coordinated, pinned subnet parameter**, bumped per "season,"
not "latest on each box." The single source of truth is `PINNED_SGLANG` in
[../optima/compat.py](../optima/compat.py) (currently `0.5.9`).

## Why bump (don't pin forever)

The mission is to push the frontier. Bump on a regular cadence — each sglang minor
release, or monthly — so the competition tracks the real frontier rather than an
old one.

## The bump process (safe + coordinated)

1. **Watch releases.** The clone at `optima/sglang` has the upstream remote;
   `git -C sglang fetch origin --tags` surfaces new tags. (Or watch GitHub releases
   for sgl-project/sglang.)
2. **Static canary.** In a scratch venv, `uv pip install sglang==<new>`, then
   `optima compat`. It introspects the installed sglang (imports + signatures, no
   GPU) and asserts every seam/API we depend on still exists.
3. **Behavioral smoke (on the pod).** If the canary is green, confirm the seam
   still *fires*: `optima bench <broken-bundle>` must still **FAIL** the gate and a
   faithful bundle must behave. A green canary is necessary but not sufficient.
4. **Coordinate + re-baseline.** If both pass: update `PINNED_SGLANG`, announce a
   bump at a block height so **all validators upgrade together**, and
   **re-baseline the champion** — re-score the reigning champion against the *new*
   sglang baseline (the baseline moved, exactly like Affine refreshing its task
   pool; a champion's old speedup isn't comparable to challengers scored on the new
   sglang).
5. **If the canary is RED:** write a small adapter in `optima/integrations/` +
   `optima/seam.py` (the seams are deliberately tiny and isolated for this), then
   re-run from step 3.

## What usually breaks, and where to fix it

| sglang change | canary catches it as | fix in |
|---|---|---|
| seam class renamed/moved (`SiluAndMul`, `RMSNorm`, `MultiPlatformOp`) | `seam: …` FAIL | the import in `integrations/*` + `bootstrap._TARGETS` |
| `forward_cuda` signature change (e.g. residual handling) | `seam: …` detail shows new params | `dispatch.py` dispatcher |
| Engine / `ServerArgs` API change | `Engine.generate …` / `ServerArgs …` FAIL | `eval/_launch.py`, `EvalConfig` |
| a real plugin framework lands (bleeding-edge sglang has one) | (canary still green) | optionally swap the `.pth` for the entry-point plugin — `integrations/sglang_plugin.py` already exists for that |

## Strategic: upstream or moat?

Decide per winning kernel whether it goes **upstream** to sglang (frontier mission;
the baseline rises and the subnet must keep finding new wins) or stays **private**
(a proprietary stack — the managed-service moat). Likely: the subnet's *composed
stack* is the product; you track sglang as the moving base and your stack sits on
top.

## Automation (optional, recommended)

Turn "stay current" into a notification instead of a chore: a weekly scheduled job
(cron / GitHub Action / Claude `schedule`) that checks for a new sglang release,
installs it in a scratch venv, runs `optima compat`, and pings you if it's red or a
new release exists. Then a human decides whether to run the bump process.
