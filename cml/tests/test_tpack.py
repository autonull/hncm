import tpack


def test_pack_build_save_search(tmp_path) -> None:
    (tmp_path / "a.py").write_text("def alpha(self, x):\n    return self.value + x\n")
    (tmp_path / "b.py").write_text("def beta(self, y):\n    return self.value + y\n")
    pack = tpack.build(tmp_path, min_fill=4, min_count=1, max_n=8)
    assert pack["templates"]
    assert pack["fills"]
    out = tmp_path / "x.tpack"
    assert tpack.save(pack, out) > 0
    loaded = tpack.load(out)
    hits = tpack.search(loaded, b"def")
    assert hits
