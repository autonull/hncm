import fillcopy

TRAIN = b"def baz(self, x):\n    return self.value + x\n" * 3
TEST = b"def baz(self, z):\n    return self.value + z\n"


def test_long_fill_tokenizer_marks_names_and_groups() -> None:
    ts = fillcopy.tokenize(b"    return self.value(x)\n", min_fill=4)
    kinds = {kind for kind, _ in ts}
    assert b"D" in kinds or b"E" in kinds or b"I" in kinds


def test_encode_decode_roundtrip_with_pack() -> None:
    lib = fillcopy.induce(TRAIN, min_fill=4, min_count=1, max_n=8)
    blob, stats = fillcopy.encode(TEST, lib, TRAIN, min_fill=4, max_n=8)
    order = sorted(lib, key=lambda key: (-lib[key], key))
    assert fillcopy.decode(blob, order, TRAIN) == TEST
    assert stats.roundtrip
    assert stats.fills > 0


def test_local_copy_is_exact() -> None:
    data = b"value = self.alpha + self.beta\nvalue = self.alpha + self.gamma\n"
    lib = fillcopy.induce(data, min_fill=4, min_count=1, max_n=8)
    blob, stats = fillcopy.encode(data, lib, b"", min_fill=4, max_n=8)
    order = sorted(lib, key=lambda key: (-lib[key], key))
    assert fillcopy.decode(blob, order) == data
    assert stats.roundtrip
