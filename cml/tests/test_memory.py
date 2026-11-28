from pathlib import Path

import pytest

import memory


def test_chain_roundtrip_and_prefix() -> None:
    versions = [b"a\nb\nc\n", b"a\nB\nc\n", b"a\nB\nc\nd\n"]
    cfg = memory.ChainConfig(window=64, min_match=4, cap=16)
    blob = memory.encode_chain(versions, "x.py", ["c0", "c1", "c2"], cfg)
    header, back = memory.decode_chain(blob)
    assert back == versions
    assert header["path"] == "x.py"
    assert header["sizes"] == [len(v) for v in versions]


def test_chain_rejects_a_corrupt_payload() -> None:
    versions = [b"alpha\n", b"alpha\n" + b"beta\n" * 40]
    cfg = memory.ChainConfig(window=64, min_match=4, cap=16)
    blob = bytearray(memory.encode_chain(versions, "x.py", ["a", "b"], cfg))
    n_head = int.from_bytes(blob[4:8], "little")
    at = 8 + n_head
    size = int.from_bytes(blob[at : at + 4], "little")
    at += 4 + size
    blob[at + 2] ^= 0xFF
    with pytest.raises(ValueError):
        memory.decode_chain(bytes(blob))


def test_unchain_writes_last_version(tmp_path: Path) -> None:
    versions = [b"a\n", b"ab\n"]
    cfg = memory.ChainConfig(window=64, min_match=4, cap=16)
    blob = memory.encode_chain(versions, "x.py", ["a", "b"], cfg)
    path = tmp_path / "x.cmlh"
    path.write_bytes(blob)
    assert memory.run_unchain(path, tmp_path / "out", -1) == 0
    assert (tmp_path / "out" / "v0001.bin").read_bytes() == versions[-1]
