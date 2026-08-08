from __future__ import annotations

import hashlib

import pytest

from cacheon.eval.tokenizer_identity import (
    SealedTokenizerDecoder,
    TOKENIZER_IDENTITY_DOMAIN,
    TOKENIZER_IDENTITY_PROBES,
    TokenizerIdentityError,
    tokenizer_identity_digest,
)
from cacheon.stack_identity import canonical_digest


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _Tokenizer:
    is_fast = True
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    unk_token_id = 3

    def __init__(self) -> None:
        self.decode_calls: list[tuple[list[int], bool]] = []

    def __len__(self) -> int:
        return 32_000

    def get_added_vocab(self) -> dict[str, int]:
        return {"<extra-b>": 32_001, "<extra-a>": 32_000}

    def encode(self, text: str) -> list[int]:
        return [len(text), sum(text.encode())]

    def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
        self.decode_calls.append((ids, skip_special_tokens))
        return "decoded:" + ",".join(str(value) for value in ids)


def test_identity_reproduces_established_path_free_formula() -> None:
    tokenizer = _Tokenizer()
    model = _h("model-content")
    expected = canonical_digest(
        TOKENIZER_IDENTITY_DOMAIN,
        {
            "added_vocab": [("<extra-a>", 32_000), ("<extra-b>", 32_001)],
            "class": type(tokenizer).__module__
            + "."
            + type(tokenizer).__qualname__,
            "is_fast": True,
            "model_content_digest": model,
            "probe_ids": [tokenizer.encode(value) for value in TOKENIZER_IDENTITY_PROBES],
            "special_token_ids": {
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": None,
                "unk_token_id": 3,
            },
            "vocab_size": 32_000,
        },
    )
    assert tokenizer_identity_digest(tokenizer, model) == expected


def test_decoder_requires_exact_identity_and_preserves_decode_policy() -> None:
    tokenizer = _Tokenizer()
    model = _h("model-content")
    digest = tokenizer_identity_digest(tokenizer, model)
    decoder = SealedTokenizerDecoder(
        tokenizer,
        model_content_digest=model,
        expected_tokenizer_digest=digest,
    )
    assert decoder.tokenizer_digest == digest
    assert decoder((7, 8)) == "decoded:7,8"
    assert tokenizer.decode_calls == [([7, 8], True)]

    with pytest.raises(TokenizerIdentityError, match="differs"):
        SealedTokenizerDecoder(
            tokenizer,
            model_content_digest=model,
            expected_tokenizer_digest=_h("other-tokenizer"),
        )


@pytest.mark.parametrize(
    "field",
    (
        "is_fast",
        "bos_token_id",
        "added_vocab",
        "vocab_size",
        "probe",
        "class",
    ),
)
def test_every_behavior_identity_component_changes_the_digest(field: str) -> None:
    baseline = _Tokenizer()
    model = _h("model-content")
    expected = tokenizer_identity_digest(baseline, model)

    if field == "is_fast":
        baseline.is_fast = False
    elif field == "bos_token_id":
        baseline.bos_token_id = 9
    elif field == "added_vocab":
        baseline.get_added_vocab = lambda: {"<different>": 32_000}  # type: ignore[method-assign]
    elif field == "vocab_size":
        baseline.__class__ = type(
            "VocabTokenizer",
            (_Tokenizer,),
            {"__len__": lambda self: 31_999},
        )
    elif field == "probe":
        baseline.encode = lambda text: [len(text) + 1]  # type: ignore[method-assign]
    else:
        baseline.__class__ = type("OtherTokenizer", (_Tokenizer,), {})

    assert tokenizer_identity_digest(baseline, model) != expected


@pytest.mark.parametrize("value", ((True,), [1], (-1,)))
def test_decoder_rejects_nonexact_or_negative_ids(value: object) -> None:
    tokenizer = _Tokenizer()
    model = _h("model-content")
    decoder = SealedTokenizerDecoder(
        tokenizer,
        model_content_digest=model,
        expected_tokenizer_digest=tokenizer_identity_digest(tokenizer, model),
    )
    with pytest.raises(TokenizerIdentityError, match="input"):
        decoder(value)  # type: ignore[arg-type]


def test_tokenizer_metadata_and_decoder_failures_are_infrastructure_errors() -> None:
    class Broken(_Tokenizer):
        def get_added_vocab(self):
            raise RuntimeError("metadata backend failed")

    with pytest.raises(TokenizerIdentityError, match="metadata") as metadata:
        tokenizer_identity_digest(Broken(), _h("model"))
    assert isinstance(metadata.value.__cause__, RuntimeError)

    tokenizer = _Tokenizer()
    model = _h("model")
    decoder = SealedTokenizerDecoder(
        tokenizer,
        model_content_digest=model,
        expected_tokenizer_digest=tokenizer_identity_digest(tokenizer, model),
    )
    tokenizer.decode = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("decode failed")
    )
    # The decoder bound the original method deliberately; replacing the
    # attribute later cannot swap the commissioned callable.
    assert decoder((1,)) == "decoded:1"

    broken = _Tokenizer()
    broken.decode = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("decode failed")
    )
    broken_decoder = SealedTokenizerDecoder(
        broken,
        model_content_digest=model,
        expected_tokenizer_digest=tokenizer_identity_digest(broken, model),
    )
    with pytest.raises(TokenizerIdentityError, match="decode failed") as decoding:
        broken_decoder((1,))
    assert isinstance(decoding.value.__cause__, RuntimeError)
