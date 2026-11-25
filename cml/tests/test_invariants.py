from pathlib import Path

import pytest
import torch

import cml


def test_gate_ops() -> None:
    assert cml.GATES["P1a"].check(2.5)
    assert not cml.GATES["P1a"].check(2.7)
    assert cml.GATES["P0a"].check(0.2)
    assert not cml.GATES["P0a"].check(0.1)


def test_manifest_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "x.toml"
    p.write_text(
        '[run]\nname = "x"\nseed = 3\ngates = ["P1a"]\n'
        '[data]\ncorpus = "enwik8_10m"\n'
        '[model]\narm = "concept"\nparams = 1_000_000\n'
    )
    cfg, gates = cml.config_from_manifest(p, None, None)
    assert cfg.seed == 3
    assert gates == ["P1a"]
    assert cml.run_id(p, cfg).endswith("-s3")


def test_space_and_coverage() -> None:
    sp = cml.Space("P1")
    arms = sp.arms()
    assert len(arms) > 100
    assert len(sp.coverage()) < len(arms)


def test_search_and_render() -> None:
    state = cml.run_search("P1", 4.0, 0)
    d = cml.state_dict(state)
    assert d["stage"] == "P1"
    assert d["results"]
    text = cml.render_state(d)
    assert "CML SEARCH" in text
    assert "leaderboard" in text


def test_windows_and_split() -> None:
    ids = torch.arange(20)
    win = cml.windows(ids, 4)
    assert win.shape == (4, 5)
    assert win[0, 0] == 0 and win[0, 1] == 1
    assert win[3, -1] == 16
    assert cml.split_index(torch.ones(10, dtype=torch.long), 5) == 5
    assert cml.split_index(torch.ones(10, dtype=torch.long), 100) == 10


def test_trunk_and_curve() -> None:
    model = cml.Trunk(64, 16, 1, 2, 8)
    x = torch.randint(0, 64, (2, 9))
    logits = model(x[:, :-1])
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(cml.lm_loss(model, x))
    curve = cml.fit_curve([(10_000_000, 2.0), (20_000_000, 1.9)])
    assert curve is not None
    assert cml.curve_bpb(curve, 60_000_000) < 2.0


def test_p1_parse_roundtrip() -> None:
    path = cml.ROOT.parent / "runs/v9/fit-seed-16384-16384-10000000.pkl"
    if not path.exists():
        pytest.skip("the v9 10 MB parse is not present")
    lexicon = cml.read_lexicon(path)
    parse = cml.read_parse(path, lexicon)
    assert b"".join(parse.units) == parse.raw
    assert len(lexicon.strings) > 16_000


def test_byte_pack_roundtrip(tmp_path: Path) -> None:
    assert cml.byte_preset(20_000_000) == (448, 9, 8)
    entries = (
        cml.Entry(b"the ", 5, 0, ("fit:1",)),
        cml.Entry(b"of", 3, 0, ("fit:2",)),
    )
    pack = cml.Pack(entries, cml.Config())
    data = cml.pack_bytes(pack)
    assert data[:4] == cml.PACK_MAGIC
    (tmp_path / "p.bin").write_bytes(data)
    got = cml.read_pack(tmp_path / "p.bin")
    assert [e.data for e in got] == [b"the ", b"of"]
    trie = cml.make_trie(entries)
    assert cml.longest_match(trie, b"xthe y", 1) == (4, 0)


def test_stream_nll_and_span_embeddings() -> None:
    if not torch.cuda.is_available():
        pytest.skip("stream_nll needs a GPU")
    model = cml.Trunk(256, 16, 1, 2, 32).cuda()
    ids = torch.randint(0, 256, (100,))
    nll, hs, n_eval = cml.stream_nll(model, ids, batch=2)
    assert n_eval == 97
    assert torch.isnan(nll[0])
    assert torch.isfinite(nll[1:n_eval]).all()
    assert hs.shape == (100, 16)
    entries = (
        cml.Entry(b"the ", 5, 0, ("fit:1",)),
        cml.Entry(b"of", 3, 0, ("fit:2",)),
        cml.Entry(b"and", 4, 0, ("fit:3",)),
    )
    e_c = cml.span_embeddings(model, entries)
    assert e_c.shape == (3, 16)
    assert torch.allclose(e_c.norm(dim=-1), torch.ones(3), atol=1e-5)
