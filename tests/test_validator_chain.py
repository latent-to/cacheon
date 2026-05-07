"""Unit tests for validator.chain -- pure helpers, no bittensor required."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from validator.chain import (
    CommitmentRecord,
    NotRegisteredError,
    _decode_raw_commitment,
    build_commitments,
    build_winner_take_all_weights,
    parse_commitment_data,
    preflight_check,
    unique_hotkeys,
)

pytestmark = pytest.mark.unit


@dataclass
class FakeMetagraph:
    hotkeys: list[str]


_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64


class TestParseCommitmentData:
    def test_valid_commitment(self):
        raw = json.dumps({"image": "docker.io/user/repo:v1", "digest": _DIGEST_A})
        assert parse_commitment_data(raw) == ("docker.io/user/repo:v1", _DIGEST_A)

    def test_strips_whitespace(self):
        raw = json.dumps(
            {"image": "  docker.io/user/repo:v1  ", "digest": f"  {_DIGEST_A}  "}
        )
        assert parse_commitment_data(raw) == ("docker.io/user/repo:v1", _DIGEST_A)

    def test_extra_fields_ignored(self):
        raw = json.dumps(
            {"image": "docker.io/user/repo:v1", "digest": _DIGEST_A, "note": "hi"}
        )
        assert parse_commitment_data(raw) == ("docker.io/user/repo:v1", _DIGEST_A)

    def test_missing_image(self):
        raw = json.dumps({"digest": _DIGEST_A})
        assert parse_commitment_data(raw) is None

    def test_missing_digest(self):
        raw = json.dumps({"image": "docker.io/user/repo:v1"})
        assert parse_commitment_data(raw) is None

    def test_empty_image(self):
        raw = json.dumps({"image": "", "digest": _DIGEST_A})
        assert parse_commitment_data(raw) is None

    def test_non_json(self):
        assert parse_commitment_data("not json at all") is None

    def test_json_but_not_object(self):
        assert parse_commitment_data(json.dumps([1, 2, 3])) is None

    def test_empty_string(self):
        assert parse_commitment_data("") is None

    def test_none_like_input(self):
        assert parse_commitment_data(None) is None  # type: ignore[arg-type]

    def test_non_string_image(self):
        raw = json.dumps({"image": 123, "digest": _DIGEST_A})
        assert parse_commitment_data(raw) is None

    def test_short_digest_rejected(self):
        raw = json.dumps({"image": "user/repo:v1", "digest": "sha256:abc123"})
        assert parse_commitment_data(raw) is None

    def test_non_sha256_digest_rejected(self):
        raw = json.dumps({"image": "user/repo:v1", "digest": "md5:" + "a" * 64})
        assert parse_commitment_data(raw) is None

    def test_digest_without_prefix_rejected(self):
        raw = json.dumps({"image": "user/repo:v1", "digest": "a" * 64})
        assert parse_commitment_data(raw) is None

    def test_uppercase_hex_in_digest_rejected(self):
        raw = json.dumps({"image": "user/repo:v1", "digest": "sha256:" + "A" * 64})
        assert parse_commitment_data(raw) is None

    def test_image_with_uppercase_rejected(self):
        raw = json.dumps({"image": "Docker.io/User/Repo:v1", "digest": _DIGEST_A})
        assert parse_commitment_data(raw) is None

    def test_image_starting_with_dot_rejected(self):
        raw = json.dumps({"image": ".invalid/repo:v1", "digest": _DIGEST_A})
        assert parse_commitment_data(raw) is None

    def test_simple_image_no_tag(self):
        raw = json.dumps({"image": "myuser/myrepo", "digest": _DIGEST_A})
        assert parse_commitment_data(raw) == ("myuser/myrepo", _DIGEST_A)

    def test_registry_without_port(self):
        raw = json.dumps(
            {"image": "registry.example.com/repo:latest", "digest": _DIGEST_A}
        )
        assert parse_commitment_data(raw) == (
            "registry.example.com/repo:latest",
            _DIGEST_A,
        )

    def test_registry_with_port(self):
        raw = json.dumps(
            {"image": "myregistry.com:5000/myimage:v1", "digest": _DIGEST_A}
        )
        assert parse_commitment_data(raw) == (
            "myregistry.com:5000/myimage:v1",
            _DIGEST_A,
        )

    def test_registry_with_port_no_tag(self):
        raw = json.dumps({"image": "localhost:5000/org/repo", "digest": _DIGEST_A})
        assert parse_commitment_data(raw) == (
            "localhost:5000/org/repo",
            _DIGEST_A,
        )

    def test_old_hf_format_rejected(self):
        """The old repo/revision format must not be accepted."""
        raw = json.dumps({"repo": "hf/repo", "revision": "a" * 40})
        assert parse_commitment_data(raw) is None


class TestDecodeRawCommitment:
    """Test the three on-chain commitment formats we've observed."""

    def test_plain_json(self):
        raw = '{"image": "user/repo:v1", "digest": "sha256:aaa"}'
        assert _decode_raw_commitment(raw) == raw

    def test_hex_encoded_with_0x_prefix(self):
        payload = '{"image": "user/repo:v1"}'
        hex_str = "0x" + payload.encode().hex()
        assert _decode_raw_commitment(hex_str) == payload

    def test_hex_with_scale_prefix(self):
        """Old bittensor SDK stores 0x + SCALE compact length + hex(json).
        The SCALE bytes appear before the '{' in the decoded output."""
        payload = '{"image": "vllm/vllm-openai:latest"}'
        scale_prefix = b"\xe5\x01"
        hex_str = "0x" + (scale_prefix + payload.encode()).hex()
        assert _decode_raw_commitment(hex_str) == payload

    def test_raw_bytes_with_scale_prefix(self):
        """Newer substrate library returns decoded bytes with SCALE prefix
        (e.g. 'E\\x02{"image": ...}')."""
        payload = '{"image": "docker.io/user/repo:v1"}'
        raw = "E\x02" + payload
        assert _decode_raw_commitment(raw) == payload

    def test_bytes_input(self):
        payload = '{"image": "user/repo:v1"}'
        assert _decode_raw_commitment(payload.encode()) == payload

    def test_invalid_hex_returns_as_is(self):
        raw = "0xNOTHEX"
        assert _decode_raw_commitment(raw) == raw

    def test_no_json_brace(self):
        assert _decode_raw_commitment("just plain text") == "just plain text"

    def test_real_hex_commitment_from_chain(self):
        """Exact hex blob observed on testnet 470 for 5GxN97KG (work/work)."""
        raw = (
            "0xe5017b22696d616765223a2022766c6c6d2f766c6c6d2d6f70656e"
            "61693a6c6174657374222c2022646967657374223a202273686132"
            "35363a396566663937333461333062363731336138353636323137"
            "643336663832373736333066643264333163656337663061303239"
            "32383335393031613233616134227d"
        )
        result = _decode_raw_commitment(raw)
        parsed = parse_commitment_data(result)
        assert parsed is not None
        image, digest = parsed
        assert image == "vllm/vllm-openai:latest"
        assert digest == (
            "sha256:9eff9734a30b6713a8566217d36f8277630fd2d31cec7f0a0292835901a23aa4"
        )

    def test_real_scale_prefixed_commitment_from_chain(self):
        """Exact raw blob observed on testnet 470 for 5DJ3Zr (default/two)."""
        raw = (
            'E\x02{"image": "docker.io/xavierlyulatent/cacheon-test-miner:v1",'
            ' "digest": "sha256:6331b7242e44a93868f87049d74f89903d108671'
            'e98abd6fa50c326a722e563a"}'
        )
        result = _decode_raw_commitment(raw)
        parsed = parse_commitment_data(result)
        assert parsed is not None
        image, digest = parsed
        assert image == "docker.io/xavierlyulatent/cacheon-test-miner:v1"
        assert digest == (
            "sha256:6331b7242e44a93868f87049d74f89903d108671e98abd6fa50c326a722e563a"
        )

    def test_double_hex_commitment_from_chain(self):
        """Exact double-hex blob observed on testnet 470 for 5E79pX6k
        (default/three). SDK hex-encoded the JSON, then substrate
        hex-encoded the result again with a SCALE prefix."""
        raw = "0x910430783762323236393664363136373635323233613230323236343666363336623635373232653639366632663738363137363639363537323663373937353663363137343635366537343266363336313633363836353666366532643734363537333734326436643639366536353732336137363332323232633230323236343639363736353733373432323361323032323733363836313332333533363361363333373336363536333333363636313338333233343333363233353335333133373335333836333633333933303330333733333632333333363338333436323339363233333632333333343632363133363338333936363335333436343336363633353333333633313635333033383336333736313330333733383335363432323764"
        result = _decode_raw_commitment(raw)
        parsed = parse_commitment_data(result)
        assert parsed is not None
        image, digest = parsed
        assert image == "docker.io/xavierlyulatent/cacheon-test-miner:v2"
        assert digest == (
            "sha256:c76ec3fa8243b551758cc90073b3684b9b3b34ba689f54d6f5361e0867a0785d"
        )


class TestBuildCommitments:
    def test_single_commitment(self):
        mg = FakeMetagraph(hotkeys=["hk0", "hk1", "hk2"])
        raw = json.dumps({"image": "user/repo:v1", "digest": _DIGEST_A})
        revealed = {"hk1": [(100, raw)]}
        out = build_commitments(mg, revealed)
        assert set(out.keys()) == {1}
        rec = out[1]
        assert rec.uid == 1
        assert rec.hotkey == "hk1"
        assert rec.commit_block == 100
        assert rec.image == "user/repo:v1"
        assert rec.digest == _DIGEST_A
        assert rec.raw == raw

    def test_picks_latest_block_when_multiple_reveals(self):
        mg = FakeMetagraph(hotkeys=["hk0"])
        raw_old = json.dumps({"image": "old/m:v1", "digest": _DIGEST_A})
        raw_new = json.dumps({"image": "new/m:v2", "digest": _DIGEST_B})
        revealed = {"hk0": [(50, raw_old), (200, raw_new), (100, raw_old)]}
        out = build_commitments(mg, revealed)
        assert out[0].commit_block == 200
        assert out[0].image == "new/m:v2"
        assert out[0].digest == _DIGEST_B

    def test_skips_invalid_commitments(self):
        mg = FakeMetagraph(hotkeys=["hk0", "hk1"])
        revealed = {
            "hk0": [(10, "garbage not json")],
            "hk1": [(20, json.dumps({"image": "user/repo:v1", "digest": _DIGEST_A}))],
        }
        out = build_commitments(mg, revealed)
        assert set(out.keys()) == {1}

    def test_hotkey_with_no_commitments_skipped(self):
        mg = FakeMetagraph(hotkeys=["hk0", "hk1"])
        revealed = {"hk1": []}
        out = build_commitments(mg, revealed)
        assert out == {}

    def test_hotkey_not_in_revealed_skipped(self):
        mg = FakeMetagraph(hotkeys=["hk0", "hk1"])
        revealed = {
            "hk_ghost": [(10, json.dumps({"image": "m:v1", "digest": _DIGEST_A}))]
        }
        out = build_commitments(mg, revealed)
        assert out == {}

    def test_uid_ordering_matches_metagraph(self):
        mg = FakeMetagraph(hotkeys=[f"hk{i}" for i in range(5)])
        revealed = {
            f"hk{i}": [
                (
                    100 + i,
                    json.dumps({"image": f"m{i}:latest", "digest": _DIGEST_A}),
                )
            ]
            for i in range(5)
        }
        out = build_commitments(mg, revealed)
        for uid, rec in out.items():
            assert rec.uid == uid
            assert rec.hotkey == f"hk{uid}"

    def test_empty_metagraph(self):
        out = build_commitments(FakeMetagraph(hotkeys=[]), {})
        assert out == {}

    def test_as_eval_key(self):
        rec = CommitmentRecord(
            uid=1,
            hotkey="hk1",
            commit_block=100,
            image="m:v1",
            digest=_DIGEST_A,
            raw="{}",
        )
        assert rec.as_eval_key() == ("hk1", 100)


class TestBuildWinnerTakeAllWeights:
    def test_basic(self):
        w = build_winner_take_all_weights(5, 2)
        assert w == [0.0, 0.0, 1.0, 0.0, 0.0]

    def test_winner_at_edge(self):
        w = build_winner_take_all_weights(3, 0)
        assert w == [1.0, 0.0, 0.0]

    def test_winner_uid_beyond_n_uids(self):
        w = build_winner_take_all_weights(3, 5)
        assert len(w) == 6
        assert w[5] == 1.0
        assert sum(w) == 1.0

    def test_negative_winner_rejected(self):
        with pytest.raises(ValueError):
            build_winner_take_all_weights(5, -1)


class TestUniqueHotkeys:
    def test_empty(self):
        assert unique_hotkeys([]) == set()

    def test_deduplicates(self):
        recs = [
            CommitmentRecord(
                uid=0,
                hotkey="hk1",
                commit_block=1,
                image="m:v1",
                digest=_DIGEST_A,
                raw="",
            ),
            CommitmentRecord(
                uid=1,
                hotkey="hk2",
                commit_block=1,
                image="m:v1",
                digest=_DIGEST_B,
                raw="",
            ),
            CommitmentRecord(
                uid=2,
                hotkey="hk1",
                commit_block=2,
                image="m:v1",
                digest=_DIGEST_C,
                raw="",
            ),
        ]
        assert unique_hotkeys(recs) == {"hk1", "hk2"}


class _FakeHotkey:
    def __init__(self, ss58: str) -> None:
        self.ss58_address = ss58


class _FakeWallet:
    def __init__(self, ss58: str) -> None:
        self.hotkey = _FakeHotkey(ss58)


class _FakePreflightMetagraph:
    def __init__(
        self,
        hotkeys: list[str],
        permits: list[bool],
        stakes: list[float],
    ) -> None:
        self.hotkeys = hotkeys
        self.validator_permit = permits
        self.S = stakes


class _FakePreflightSubtensor:
    def __init__(
        self,
        *,
        registered_hotkeys: set[str],
        hotkeys: list[str],
        permits: list[bool] | None = None,
        stakes: list[float] | None = None,
    ) -> None:
        self._registered = registered_hotkeys
        self._hotkeys = hotkeys
        self._permits = permits if permits is not None else [False] * len(hotkeys)
        self._stakes = stakes if stakes is not None else [0.0] * len(hotkeys)

    def is_hotkey_registered(self, *, netuid: int, hotkey_ss58: str) -> bool:
        return hotkey_ss58 in self._registered

    def metagraph(self, _netuid: int):
        return _FakePreflightMetagraph(
            self._hotkeys,
            self._permits,
            self._stakes,
        )


class TestPreflightCheck:
    def test_unregistered_hotkey_raises(self):
        st = _FakePreflightSubtensor(
            registered_hotkeys=set(),
            hotkeys=["hk_other"],
        )
        wallet = _FakeWallet(ss58="hk_us")
        with pytest.raises(NotRegisteredError) as excinfo:
            preflight_check(st, wallet, netuid=14)
        assert "hk_us" in str(excinfo.value)
        assert "14" in str(excinfo.value)

    def test_registered_with_permit(self):
        st = _FakePreflightSubtensor(
            registered_hotkeys={"hk_us"},
            hotkeys=["hk_other", "hk_us"],
            permits=[False, True],
            stakes=[10.0, 500.0],
        )
        result = preflight_check(st, _FakeWallet("hk_us"), netuid=14)
        assert result.uid == 1
        assert result.has_validator_permit is True
        assert result.stake == 500.0

    def test_registered_without_permit_warns_but_ok(self, caplog):
        st = _FakePreflightSubtensor(
            registered_hotkeys={"hk_us"},
            hotkeys=["hk_us"],
            permits=[False],
            stakes=[0.0],
        )
        with caplog.at_level("WARNING"):
            result = preflight_check(st, _FakeWallet("hk_us"), netuid=14)
        assert result.uid == 0
        assert result.has_validator_permit is False
        assert any("permit" in rec.message.lower() for rec in caplog.records)

    def test_tolerates_missing_validator_permit_attr(self):
        """Older bittensor / mock metagraphs may not expose validator_permit;
        preflight should still succeed with has_validator_permit=False."""

        class SparseMetagraph:
            hotkeys = ["hk_us"]

        class SparseSubtensor:
            def is_hotkey_registered(self, *, netuid, hotkey_ss58):
                return hotkey_ss58 == "hk_us"

            def metagraph(self, _netuid):
                return SparseMetagraph()

        result = preflight_check(SparseSubtensor(), _FakeWallet("hk_us"), netuid=14)
        assert result.uid == 0
        assert result.has_validator_permit is False
        assert result.stake == 0.0
