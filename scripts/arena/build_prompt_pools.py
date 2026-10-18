"""Build one exact-length GSM8K prompt pool for an arena's prompt authority.

Runs inside the arena's worker image, whose model mount provides the tokenizer. The sealed hidden
judge is gsm8k-numeric-extraction.v1: it decodes one sealed gold sequence per prompt and compares
parsed ``#### N`` numbers, so every prompt is few-shot GSM8K, exemplar Question/Answer pairs as the
long-prefill body and one final ``Question: ...\\nAnswer:`` the engine must continue, and the pool
seals the dataset gold for that final question beside the prompt.

Two contracts hold per prompt: ``len(tokenizer(text).input_ids) == --input-tokens`` exactly, and
``tokenizer(answer, add_special_tokens=False).input_ids`` decodes back to text carrying a
``#### <number>`` cue the judge can parse. A unique header makes every prompt diverge at token 0:
the exemplar stream wraps across prompts, and without unique prefixes radix caching would let the
engine skip real prefill work, silently turning the throughput cell into a cache benchmark. Repair
adjusts only the filler length; header and final block are byte-preserved or the row is skipped.
One pool per workload cell, named ``pool_<input tokens>.json`` beside the corpus for
``build_prompt_authority.py``.
"""

import argparse
import hashlib
import json
import re

GOLD_CUE = re.compile(r"####\s*-?\$?\d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model root mounted in the worker image")
    parser.add_argument("--source-jsonl", required=True, help="the GSM8K training corpus")
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument("--pool-size", type=int, default=2400)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    source = open(args.source_jsonl, "rb").read()
    source_sha = hashlib.sha256(source).hexdigest()
    print("SOURCE_SHA256=%s" % source_sha)
    print("SOURCE_BYTES=%d" % len(source))
    rows = [json.loads(line) for line in source.decode("utf-8").splitlines() if line.strip()]
    print("SOURCE_ROWS=%d" % len(rows))
    candidates = args.pool_size + 600
    if len(rows) < candidates + 500:
        print("ABORT=source_rows_insufficient_%d" % len(rows))
        raise SystemExit(2)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def count(text: str) -> int:
        return len(tokenizer(text).input_ids)

    finals = rows[:candidates]
    stream_text = "".join(
        "Question: %s\nAnswer: %s\n\n" % (row["question"], row["answer"]) for row in rows[candidates:]
    )
    stream = tokenizer(stream_text, add_special_tokens=False).input_ids
    print("STREAM_TOKENS=%d" % len(stream))

    target = args.input_tokens
    prompts: list[str] = []
    golds: list[list[int]] = []
    seen: set[str] = set()
    skipped = 0
    position = 0
    for row in finals:
        if len(prompts) >= args.pool_size:
            break
        question, answer = row["question"], row["answer"]
        qdigest = hashlib.sha256(question.encode()).hexdigest()
        gold_ids = tokenizer(answer, add_special_tokens=False).input_ids
        if not gold_ids or not GOLD_CUE.search(tokenizer.decode(gold_ids, skip_special_tokens=True)):
            skipped += 1
            continue
        header = "Problem set %s.\n\n" % qdigest[:16]
        final_block = "Question: %s\nAnswer:" % question
        take = target - count(header + final_block)
        if take < target // 2:
            skipped += 1
            continue
        if position + take + 256 > len(stream):
            position = 0
        chunk = stream[position:position + take + 256]
        ok = False
        for _ in range(12):
            if take < 1 or take > len(chunk):
                break
            text = header + tokenizer.decode(chunk[:take], skip_special_tokens=True) + final_block
            got = count(text)
            if got == target:
                ok = True
                break
            take -= got - target
        if not ok or not text.startswith(header) or not text.endswith(final_block):
            skipped += 1
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            skipped += 1
            continue
        seen.add(digest)
        position += take
        prompts.append(text)
        golds.append([int(token) for token in gold_ids])
        if len(prompts) % 500 == 0:
            print("POOL_PROGRESS=%d" % len(prompts))

    if len(prompts) < args.pool_size:
        print("ABORT=candidates_exhausted_at_%d" % len(prompts))
        raise SystemExit(2)

    receipt = {
        "schema": "cacheon-c18-prompt-pool-v2",
        "count_contract": "len(AutoTokenizer(model)(text).input_ids)",
        "gold_contract": "AutoTokenizer(model)(answer, add_special_tokens=False).input_ids",
        "input_tokens": target,
        "pool_size": len(prompts),
        "source_sha256": source_sha,
        "skipped_rows": skipped,
        "prompt_sha256s": sorted(seen),
        "prompts": prompts,
        "golds": golds,
    }
    body = json.dumps(receipt, sort_keys=True)
    with open(args.out, "x") as handle:
        handle.write(body)
    print("POOL_OUT=%s" % args.out)
    print("POOL_SHA256=%s" % hashlib.sha256(body.encode()).hexdigest())
    print("POOL_SIZE=%d" % len(prompts))
    print("POOL_SKIPPED=%d" % skipped)
    print("POOL_DONE=1")


if __name__ == "__main__":
    main()
