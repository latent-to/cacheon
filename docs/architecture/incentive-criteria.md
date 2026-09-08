# Declared-baseline incentive criteria

This is the requirements reference for the declared-baseline incentive change.
It describes the intended behavior; implementation and deployment status are
tracked separately.

**Terms:** A is the new submission, B is its declared evaluation baseline, and C
is the strongest unfinalized winner for A's target slot, if one exists. A
baseline reference identifies the finalized evaluation HEAD and its exact stack.
**Finalized** means incorporated into the commissioned evaluation baseline,
not merely confirmed on-chain.

1. **Declare and validate the baseline.** Miners include a baseline reference
   when requesting a quote and when submitting. It must match the current top
   finalized submission/HEAD. Check again at queue admission: a quote does not
   preserve eligibility after the finalized HEAD advances. Only matching
   submissions enter the evaluation queue.
2. **Evaluate the declared baseline.** Run each submission against the baseline
   it declared. Submissions behind it in the queue must reference the same
   baseline or a later, stronger HEAD. Do not silently rebind an admitted
   submission to another baseline.
3. **Beat both competitors.** A wins only if it passes the existing correctness
   checks and is faster than B and, when present, C. Use A's evaluation for the
   direct A-versus-B comparison. Compare A with C's already retained measured
   speed; do not run another A-versus-C evaluation. Apply a noise margin to that
   comparison, the same threshold used for A versus B. Ratios
   against different baselines are not directly comparable speeds.
4. **Reward unfinalized rankings immediately.** Use the current unfinalized
   winner rankings to build weight distributions. Winner finalization is not
   a prerequisite for inclusion in those distributions. Evaluation writes the
   scores and ranking decisions to the database; only the standalone weight
   producer service builds and publishes distributions.
5. **Mark stale wins.** At evaluation completion, record whether A's declared
   baseline already has a winner for A's target slot. Show a **stale** marker
   for such a winner in the frontend. The marker does not deny the win if A
   meets criterion 3; retain the completion-time classification.
6. **Show the evaluated baseline.** Submission details display the baseline
   reference that will be used for queued work or was used for completed work.

Finalized baseline selection and unfinalized reward ranking are separate:
miners target the finalized HEAD, while rewards reflect accepted winners as
soon as their evaluations finish.

Development and validation use an isolated branch, temporary databases and
mocked GPU execution. They must not interrupt mainnet evaluations or change
live services, queues, weights or GPU allocations.


Implementation reference: `reservations.baseline_ref` retains the miner's
reference; `reservation_baseline_segments` retains its exact evaluation stack.
`finalized_baselines` publishes the commissioned HEAD. `submission_rankings`
retains the measured candidate rate, comparison context, required ratio,
competitor, winner decision, stale marker and completion block. Existing
qualification evidence retains the direct baseline speedup used by the reward
formula. A losing A-versus-C comparison earns no claim even if A beat B.

Upgrading an existing database reconstructs missing scoring rows from retained
accepted evidence without repeating GPU work. Historical crowns keep their
credit; other historical PASSes are ranked in completion order, and a PASS
that did not beat the strongest same-slot winner earns no claim. New competition uses compatible model, runtime, workload, hardware
and scoring-policy identities across commissioned baseline changes.

A worker must have the exact queued baseline commissioned before dispatch. If
it has another baseline loaded, dispatch reports the required manifest/tree
and stops before creating a lease or GPU request. Draining an older admitted
segment never rolls back the HEAD advertised to new miners.

An operator-approved quality replay correction retains the original measured
speed and noise threshold. Its PASS goes through the same slot competition as
other accepted submissions; correcting a quality HOLD does not itself award
the slot. The correction and original evaluation remain separate evidence.
