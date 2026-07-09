# Miner submission terms

> **STATUS: DRAFT.** These terms have not been reviewed by counsel and are not
> yet in effect. They become binding only when a subnet announcement (on-chain
> or in this repository's README) designates a specific version of this
> document as the terms of participation. They are published now so miners can
> see the intended shape before mainnet. **Get legal review before launch.**

These terms exist to make one thing unambiguous: **submitting a kernel to the
subnet grants the subnet operator every right needed to run, improve, and
commercially operate the resulting inference stack — including a hosted
inference service — with subnet emissions as the sole compensation.** The repo
license (Apache-2.0, see [../LICENSE](../LICENSE)) covers the harness code; it
does not and cannot cover what miners submit. This document does.

## 1. Acceptance

Submitting a bundle to the subnet — committing its content hash and fetch URL
on-chain via the commit-reveal mechanism (`optima chain-submit`), signed by
your hotkey — constitutes acceptance of the version of these terms designated
at the time of your commitment. The signed on-chain commitment is the record
of acceptance.

## 2. License grant to the operator

For each submission, you grant the subnet operator a **perpetual, irrevocable,
worldwide, non-exclusive, royalty-free, sublicensable, transferable license**
to use, reproduce, modify, adapt, create derivative works of, distribute,
publicly perform and display, and commercially exploit the submission and
derivatives of it — alone or combined with other software — including, without
limitation, in hosted inference services and other commercial offerings.

## 3. Compensation

Subnet emissions earned under the scoring mechanism are the **sole and
complete compensation** for a submission and for every right granted in these
terms. No royalty, revenue share, or other payment is owed for any use of a
submission, including commercial use in an inference service.

## 4. Your representations

By submitting, you represent that:

1. the submission is your original work, or you have sufficient rights to
   submit it and to make the grants in these terms;
2. the submission does not include third-party code except as permitted by
   that code's license, and you have complied with that license (note: the
   pinned runtime stack — sglang, flashinfer — is Apache-2.0);
3. the submission contains no code intended to subvert the evaluation, exfiltrate
   data, or damage systems (independent of these terms, the harness scans,
   sandboxes, and audits submissions — see the threat model);
4. you have the authority to bind the entity you submit for, if any.

## 5. What you keep

You retain copyright in your submission. The grants above are licenses, not an
assignment. Nothing here restricts your right to use, publish, or license your
own kernel elsewhere. Copy detection and king-of-the-hill settlement govern
what *earns* on the subnet, not what you may do with your own work off-subnet.

## 6. Open questions (deliberately undecided in this draft)

- **Public licensing of revealed bundles.** After the reveal block, a bundle's
  fetch URL is publicly readable on-chain, so revealed source is de facto
  visible. Whether revealed submissions are additionally *licensed* to the
  public (e.g. Apache-2.0 on reveal — regularizing visibility into reuse
  rights) or remain all-rights-reserved-except-the-operator-grant is a
  strategy decision: the first strengthens the open-stack story, the second
  preserves optionality for the operator. Not decided here.
- **Operator identity.** "Subnet operator" must be bound to a concrete legal
  entity before these terms can take effect.

## 7. Changes

Terms may be updated prospectively. A submission is governed by the version
designated when its commitment was signed; updates never apply retroactively.
