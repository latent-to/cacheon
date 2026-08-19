# Submitting a proposal

Production submission is a hotkey-signed, timelock commit-reveal containing a
content hash and an HTTPS fetch URL. The chain carries a reference, not the
archive bytes.

The miner-side implementation is in
[submit.py](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/submit.py),
and the canonical payload is defined by
[payload.py](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/payload.py).

## The identity chain

Submission binds several related but non-interchangeable values:

```text
bundle files --SHA-256--> content_hash --inside canonical payload--> commitment
      |                         |                              |
 exact proposal bytes      fetch authentication          finalized arrival order
```

The URL says where to fetch; it does not define the proposal. The content hash says what
bytes must be recovered; it does not prove the bytes are eligible or correct. The
finalized reveal gives arrival authority; it does not prove that a fetch or qualification
succeeded. Later selected-delta, stack, launch, evidence, and settlement digests bind the
same proposal into progressively narrower decisions.

Keep the printed content hash with your bundle and operator receipts. A bundle name such
as `alice-silu-v1` is for humans and is not enough to diagnose which bytes were evaluated.

## Before you sign

Confirm all of the following against the operator's current announcement:

- network, netuid, active arena, target catalog, and evaluation stack;
- registered target and required target mode;
- submission window, timelock policy, eval-cost quote, and any admission limits;
- the designated version of the submission terms;
- how the operator publishes intake and qualification status.

The repository's
[submission terms](../legal/submission-terms.md)
are currently marked draft and become binding only when an operator announcement
designates a version. Read the designated terms before signing. Ensure you own
or can grant the required rights to every submitted source fragment.

A plain `chain-submit` is hotkey-signed. Paying the eval cost with `--pay` also
needs the coldkey: it transfers TAO to the current subnet owner coldkey and then
commits a v2 payload that points at that transfer.

## Step-by-step commands

Set the operator's announced values once. Do not edit `my_bundle` after you
package or publish it.

```bash
NET="<NETWORK>"
NETUID="<NETUID>"
WALLET="<WALLET>"
HOTKEY="<HOTKEY>"
BUNDLE=my_bundle
URL="https://downloads.example.org/cacheon/my_bundle.tar.gz"
BLOCKS=10
```

Use `python -m cacheon.cli` on GPU hosts. Substitute the real HTTPS URL from
`chain-publish` or from your own host.

### 1. Register the miner hotkey

Skip this if the hotkey is already registered on this netuid. Registration
needs coldkey authorization.

```bash
python -m cacheon.cli chain-register \
  --netuid "$NETUID" --network "$NET" \
  --wallet "$WALLET" --hotkey "$HOTKEY"
```

### 2. Check the frozen bundle

```bash
python -m cacheon.cli scan "$BUNDLE"
python -m cacheon.cli verify "$BUNDLE" --device cuda --dtype bfloat16
```

`scan` and `verify` are diagnostics; they do not pre-approve intake. See
[Bundle checks](#bundle-checks) for what to inspect.

### 3. Host the archive

Either publish through the miner's S3-compatible bucket:

```bash
python -m pip install -e ".[object-store]"
python -m cacheon.cli chain-publish "$BUNDLE" --out dist/my_bundle.tar.gz
```

Or package and upload the archive yourself:

```bash
python -m cacheon.cli chain-package "$BUNDLE" --out dist/my_bundle.tar.gz
```

Copy the printed content hash and the public HTTPS URL. Set `URL` to that
exact URL. Bucket variables, CDN origins, and fetch limits are under
[Publish from the miner's object store](#publish-from-the-miners-object-store).

### 4. Dry-run the unpaid payload

```bash
python -m cacheon.cli chain-submit "$BUNDLE" \
  --url "$URL" \
  --netuid "$NETUID" --network "$NET" \
  --wallet "$WALLET" --hotkey "$HOTKEY" \
  --blocks-until-reveal "$BLOCKS" \
  --dry-run
```

The printed `content_hash` must match the package result. The unpaid payload is
canonical JSON with exactly three fields:

```json
{"v":1,"h":"<64-lowercase-hex>","u":"https://.../my_bundle.tar.gz"}
```

A refused dry-run has not signed or sent anything. The production payload cap
is 1,024 bytes.

### 5. Submit — eval-cost gate off

If the operator's `eval_cost_tao_rao` is `0` (the code default), commit the
unpaid payload. Do not pass `--pay`.

```bash
python -m cacheon.cli chain-submit "$BUNDLE" \
  --url "$URL" \
  --netuid "$NETUID" --network "$NET" \
  --wallet "$WALLET" --hotkey "$HOTKEY" \
  --blocks-until-reveal "$BLOCKS"
```

Then skip to [Inspect public chain state](#7-inspect-public-chain-state).

### 6. Submit — eval-cost gate on

If the operator requires a TAO admission transfer, quote, dry-run `--pay`, then
pay and commit. The destination is the current subnet owner coldkey.

```bash
python -m cacheon.cli chain-eval-cost --netuid "$NETUID" --network "$NET"
python -m cacheon.cli chain-submit "$BUNDLE" \
  --url "$URL" \
  --netuid "$NETUID" --network "$NET" \
  --wallet "$WALLET" --hotkey "$HOTKEY" \
  --blocks-until-reveal "$BLOCKS" \
  --pay \
  --dry-run
```

`--pay --dry-run` does not transfer TAO. The dry-run payload stays v1 because
there is no inclusion pointer yet. Then pay and commit:

```bash
python -m cacheon.cli chain-submit "$BUNDLE" \
  --url "$URL" \
  --netuid "$NETUID" --network "$NET" \
  --wallet "$WALLET" --hotkey "$HOTKEY" \
  --blocks-until-reveal "$BLOCKS" \
  --pay
```

A live `--pay` freezes the quoted amount for 300 blocks (~1 hour) until the
transfer lands, then commits v2. Copy the printed pointer:

```text
eval_cost payment: block=<BLOCK> extrinsic=<INDEX>
```

If the validator operator granted you a fee credit (a make-good after a
validator-side failure of a paid submission), submit as in the gate-off flow:
plain `chain-submit` with no `--pay` and no payment pointer. The credit admits
that one reveal.

```json
{"v":2,"h":"<64-lowercase-hex>","u":"https://.../my_bundle.tar.gz","p":{"b":<block>,"i":<extrinsic_index>}}
```

If the reveal commit fails after the transfer is included, retry **without**
`--pay`. Only this miner hotkey can spend that pointer, and only for this
bundle and netuid. There is no refund.

```bash
python -m cacheon.cli chain-submit "$BUNDLE" \
  --url "$URL" \
  --netuid "$NETUID" --network "$NET" \
  --wallet "$WALLET" --hotkey "$HOTKEY" \
  --blocks-until-reveal "$BLOCKS" \
  --eval-cost-payment-block <BLOCK> \
  --eval-cost-payment-extrinsic-index <INDEX>
```

### 7. Inspect public chain state

```bash
python -m cacheon.cli chain-status \
  --netuid "$NETUID" --network "$NET" \
  --wallet "$WALLET" --hotkey "$HOTKEY"
```

`chain-status` shows subnet and revealed-commitment state. It does not read the
validator's private SQLite intake lifecycle; use the operator's published
status/receipt surface for later stages.

The SDK encrypts the payload for automatic reveal after the requested timelock.
This is not the old local salt/round simulation. The finalized reveal position
provides the consensus arrival order used by intake.

## Bundle checks

Use an explicit contribution target and source-only contents. Then run the
development checks in [step 2](#2-check-the-frozen-bundle). Inspect the tree for
credentials, caches, generated binaries, model data, machine paths, unsupported
licenses, and stale result metadata.

## Publish from the miner's object store

Each miner owns and pays for their own bucket. The validator does not provision
the bucket, receive the miner's credentials, or use authenticated object-store
reads. Install the optional S3 client support:

```bash
python -m pip install -e ".[object-store]"
```

Create S3 credentials and a bucket in the miner's account. Export the
credentials from a private environment file:

```bash
set -a
source .env
set +a
```

The recognized variables are:

```dotenv
CACHEON_OBJECT_STORE_ACCESS_KEY_ID=...
CACHEON_OBJECT_STORE_SECRET_ACCESS_KEY=...
CACHEON_OBJECT_STORE_BUCKET=...
```

`chain-publish` uses the generic S3 backend by default. AWS S3 needs no provider
flag. For another S3-compatible service, its endpoint URL identifies the
service; a custom endpoint defaults to path-style addressing:

```dotenv
CACHEON_OBJECT_STORE_ENDPOINT_URL=https://objects.example
CACHEON_OBJECT_STORE_REGION=us-east-1
```

Use `CACHEON_OBJECT_STORE_ADDRESSING_STYLE=virtual` only when the service expects
virtual-hosted bucket URLs. Known provider names are optional convenience
presets. For example, `CACHEON_OBJECT_STORE_PROVIDER=hippius` supplies Hippius's
endpoint, `decentralized` region, and path-style addressing; the equivalent
fully explicit configuration is:

```dotenv
CACHEON_OBJECT_STORE_ENDPOINT_URL=https://s3.hippius.com
CACHEON_OBJECT_STORE_REGION=decentralized
CACHEON_OBJECT_STORE_ADDRESSING_STYLE=path
```

The command packages the bundle, uploads it under a content-addressed key,
grants anonymous read access, reopens the stored archive, and finally runs the
validator's production HTTPS fetch and hash check without credentials:

```bash
python -m cacheon.cli chain-publish my_bundle \
  --out dist/my_bundle.tar.gz
```

Pass `--create-bucket` only when the named bucket does not exist. The default
key retains the storage-compatibility prefix
`cacheon/miner-bundles/sha256/<content_hash>.tar.gz`. A repeated publication
reuses an existing object only after hardened extraction proves that it has the
committed tree hash; it never replaces a conflicting key. Use
`--object-store-provider hippius|minio` for a known preset, or
`--object-store-endpoint` for any S3-compatible service.
`--public-base-url` handles a separate HTTPS CDN or gateway origin.

The validator sees only the resulting HTTPS URL and content hash; it has no
object-store provider setting or miner credential. Hippius connection details
are maintained in its
[official S3 documentation](https://docs.hippius.com/llms.txt). Do not copy a
miner's credentials onto a validator.

### Manual hosting alternative

For a non-S3 public HTTPS origin, package exactly the identity-bearing files:

```bash
python -m cacheon.cli chain-package my_bundle \
  --out dist/my_bundle.tar.gz
```

The command prints a lowercase SHA-256 content hash. That hash identifies the
canonical extracted bundle tree, not the gzip byte stream. The packager includes
exactly the regular files covered by bundle identity.

Do not edit `my_bundle` after packaging. `chain-submit` re-hashes the directory;
if it changes while the hosted archive does not, the validator will reject the
fetch as a content mismatch.

For extra confidence, extract the hosted object into a clean temporary location and run
`chain-package` against that root, then compare its printed content hash. The wrapper
directory and gzip encoding are not the identity; the sorted relative paths and file bytes
are. Never “refresh” a stable URL with revised content after committing the old hash.

Upload that archive to a stable URL such as:

```text
https://downloads.example.org/cacheon/my_bundle.tar.gz
```

Production URLs must be canonical HTTPS with a public-routable host. Credentials
in the URL, fragments, plaintext HTTP, local files, and private/loopback
destinations are rejected. Fetch retains TLS hostname verification and requires
TLS 1.2 or newer.

Keep the exact object available long enough for reveal, finalized intake, and
configured transport retries. Avoid a short-lived signed URL. The revealed URL
is public chain data, so never embed a secret in it.

The production transport accepts gzip-compressed tar only. Current bounds include
a 64 MiB archive, 256 MiB extracted content, 4,096 logical members, 16 MiB per
regular file, 8 MiB per inspectable source/configuration file, 32 MiB across all
inspectable files, bounded extension metadata, at most five redirects, and one
60-second absolute DNS/TLS/transfer/extraction deadline. The validator re-hashes
the safely extracted identity-bearing tree. See
[fetch.py](https://github.com/latent-to/cacheon/blob/main/cacheon/chain/fetch.py).

## What happens after reveal

The authoritative path is staged:

1. A finalized valid reveal is reserved in durable SQLite intake.
2. The validator fetches the HTTPS archive into private storage, safely
   extracts it, and verifies the committed content hash.
3. It republishes an immutable worker-readable tree and fingerprints the
   selected delta.
4. Target resolution and the `static → build → ABI → graph → abbreviated
   serving` non-crown screens run through a registered arena service.
5. A promoted candidate receives a complete isolated version-3 qualification
   attempt: current v7 resident B/C/[B′] or v8 two-process B/C/B′,
   registered eager audit A, then pristine T.
6. One PASS moves the proposal to `reproduction_pending`. It has **not** crowned.
7. A second independent matching PASS completes qualification; the lower of the
   two reproduced speedups is retained.
8. Transactional settlement may crown, neutralize, or hold the qualified
   candidate according to the frozen target/stack authority and competing
   cohort.
9. Weight projection is a separate audited control-plane action.

!!! info "When a reward begins"
    `qualified` still means no reward. If settlement crowns the proposal, it
    records the reward claim in the same transaction. The validator later combines
    eligible claims into a weight vector and publishes it on-chain. See
    [How miners earn rewards](incentives.md).

There is no universal completion time. Finality, queue bounds, arena capacity,
retry policy, reproduction scheduling, and settlement cadence are operator
configuration.

### Follow one proposal through the states

Suppose the revealed content hash is `H` and its target is
`activation.silu_and_mul`:

1. `reserved` means the finalized arrival has a durable intake row. The proposal has not
   been fetched, so local correctness results are not relevant to its current wait.
2. `fetching` either produces an authenticated private tree or a transport result. A
   timeout may become `transport_retry` for the same `H`; a hash mismatch is a terminal
   candidate problem because the bytes at the URL are not `H`.
3. `published` means an immutable worker tree and selected-delta identity exist. It does
   not mean candidate Python has passed any screen.
4. `screening` records the ordered stage receipts. If ABI fails, changing a local file
   cannot repair `H`; fix the source, package a new hash, and submit it as a new proposal.
   A validator storage fault should instead produce uncertainty for operator retry, not a
   fabricated candidate failure.
5. `promoted` means all five non-crown screens passed and capacity may now be
   spent on the sealed v7 B/C/[B′] or v8 B/C/B′ schedule, registered eager
   audit A, then pristine T. It carries no speed score and no reward.
6. `reproduction_pending` means the first full attempt passed. Continue to describe the
   object as a proposal awaiting independent reproduction.
7. `qualified` means two matching passes exist. Settlement still reopens evidence and
   considers priority/overlap before creating a crown.
8. A settlement crown is an economic record for this target and stack authority. It is
   still not an Engine release.

At each step, ask whether the next action changes proposal identity. Retrying fetch,
reopening retained evidence, or rerunning an independent attempt can preserve `H` under
operator policy. Editing source, metadata, the manifest, or any other identity-bearing
file necessarily creates a new hash and returns to submission.

### What to record from the operator

When the operator exposes receipts, retain at least the content hash, finalized arrival
position, target ID, arena and evaluation-stack digests, selected-delta digest, last
durable status, decision/reason, and evidence or receipt digest. These let both sides
distinguish “wrong bundle,” “same bundle under a different arena generation,” and
“infrastructure could not decide.”

`chain-status` alone cannot supply those lifecycle fields. It observes public subnet and
reveal state, while production intake and qualification state live in validator storage.
Use the operator's designated status surface rather than assuming absence from
`chain-status` output means rejection.

A crown remains separate from source integration and a Cacheon Engine release.
Submitting does not cause the validator to publish miner code as a release.
Reward generation and confirmed publication are described in
[How miners earn rewards](incentives.md).

## Production authority boundary

The supported submission path is `chain-package` followed by `chain-submit`. Finalized
chain intake, SQLite qualification state, transactional settlement, and journaled weight
publication are separate validator authorities. No local ledger or contributor-side
score can create those records.

If the proposal stalls or fails, map its reported lifecycle state through
[Diagnostics](diagnostics.md).
