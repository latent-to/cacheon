"""Path-free tokenizer identity and decoder binding for sealed B300 prompts."""

from __future__ import annotations

from collections.abc import Mapping

from cacheon.stack_identity import (
    StackIdentityError,
    canonical_digest,
    require_sha256_hex,
)


TOKENIZER_IDENTITY_DOMAIN = "cacheon.private-b300-tokenizer-identity.v1"
TOKENIZER_IDENTITY_PROBES = (
    "Cacheon tokenizer identity probe.",
    " leading-space",
    "newline\nprobe",
    "MiniMax-M3 131072",
)


class TokenizerIdentityError(RuntimeError):
    """A supplied tokenizer differs from its sealed behavioral identity."""


def _digest(value: object, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except StackIdentityError as exc:
        raise TokenizerIdentityError(str(exc)) from None


def _token_id(tokenizer: object, name: str) -> int | None:
    value = getattr(tokenizer, name, None)
    return value if type(value) is int and value >= 0 else None


def _encoded_probe(tokenizer: object, value: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise TokenizerIdentityError("tokenizer has no callable encode method")
    try:
        result = encode(value)
    except Exception as exc:
        raise TokenizerIdentityError("tokenizer identity probe failed") from exc
    if type(result) is not list or any(
        type(token) is not int or token < 0 for token in result
    ):
        raise TokenizerIdentityError(
            "tokenizer identity probe did not return non-negative token IDs"
        )
    return result


def tokenizer_identity_digest(tokenizer: object, model_content_digest: str) -> str:
    """Recompute the established sealed tokenizer behavior identity."""

    model_digest = _digest(model_content_digest, "tokenizer model content")
    added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if not callable(added_vocab):
        raise TokenizerIdentityError("tokenizer has no callable added-vocab method")
    try:
        raw_vocab = added_vocab()
        vocab_size = len(tokenizer)  # type: ignore[arg-type]
    except Exception as exc:
        raise TokenizerIdentityError("tokenizer identity metadata failed") from exc
    if not isinstance(raw_vocab, Mapping):
        raise TokenizerIdentityError("tokenizer added vocabulary is not a mapping")
    rows: list[tuple[str, int]] = []
    for token, index in raw_vocab.items():
        if type(token) is not str or type(index) is not int or index < 0:
            raise TokenizerIdentityError("tokenizer added vocabulary is malformed")
        rows.append((token, index))
    if type(vocab_size) is not int or vocab_size <= 0:
        raise TokenizerIdentityError("tokenizer vocabulary size is malformed")
    return canonical_digest(
        TOKENIZER_IDENTITY_DOMAIN,
        {
            "added_vocab": sorted(rows),
            "class": type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "model_content_digest": model_digest,
            "probe_ids": [
                _encoded_probe(tokenizer, value) for value in TOKENIZER_IDENTITY_PROBES
            ],
            "special_token_ids": {
                name: _token_id(tokenizer, name)
                for name in (
                    "bos_token_id",
                    "eos_token_id",
                    "pad_token_id",
                    "unk_token_id",
                )
            },
            "vocab_size": vocab_size,
        },
    )


class SealedTokenizerDecoder:
    """Decode only after the supplied tokenizer matches one sealed identity."""

    def __init__(
        self,
        tokenizer: object,
        *,
        model_content_digest: str,
        expected_tokenizer_digest: str,
    ) -> None:
        expected = _digest(expected_tokenizer_digest, "expected tokenizer")
        observed = tokenizer_identity_digest(tokenizer, model_content_digest)
        if observed != expected:
            raise TokenizerIdentityError(
                "tokenizer behavior differs from the sealed tokenizer identity"
            )
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise TokenizerIdentityError("tokenizer has no callable decode method")
        self.tokenizer_digest = observed
        self._decode = decode

    def __call__(self, token_ids: tuple[int, ...]) -> str:
        if type(token_ids) is not tuple or any(
            type(token) is not int or token < 0 for token in token_ids
        ):
            raise TokenizerIdentityError(
                "tokenizer decoder input must be non-negative token IDs"
            )
        try:
            result = self._decode(list(token_ids), skip_special_tokens=True)
        except Exception as exc:
            raise TokenizerIdentityError("sealed tokenizer decode failed") from exc
        if type(result) is not str:
            raise TokenizerIdentityError("sealed tokenizer decode did not return text")
        return result


__all__ = [
    "SealedTokenizerDecoder",
    "TOKENIZER_IDENTITY_DOMAIN",
    "TOKENIZER_IDENTITY_PROBES",
    "TokenizerIdentityError",
    "tokenizer_identity_digest",
]
