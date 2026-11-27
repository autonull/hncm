import math
from pathlib import Path

import pytest
import torch

import cml
import codec


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


def test_arith_and_byte_roundtrip() -> None:
    text = (b"alpha = beta + gamma\n" * 80) + bytes(range(256))
    blob = codec.encode_byte(text)
    assert codec.decode_frame(blob) == text
    assert codec.unpack_frame(blob).n == len(text)


def test_parse_reexpands() -> None:
    train = b"def parse(data):\n    return data\n" * 40
    lex = codec.build_lexicon(train, size=64, min_count=4, cap=8)
    assert lex.units
    parse, covered = codec.greedy_parse(train, lex)
    rebuilt = b"".join(
        lex.units[sym] if sym >= 0 else bytes([-1 - sym]) for sym in parse
    )
    assert rebuilt == train
    assert covered > 0


def test_lz_parse_and_copy_roundtrip() -> None:
    data = (b"alpha = beta + gamma\n" * 200) + bytes(range(256))
    tokens = codec.lz_parse(data, window=256, min_match=4, cap=16)
    assert tokens == codec.lz_parse(data, window=256, min_match=4, cap=16)
    arm = codec.CopyArm(window=256, min_match=4, cap=16)
    blob = arm.encode(data, tokens)
    assert (
        codec.CopyArm(window=256, min_match=4, cap=16).decode(blob, len(data)) == data
    )
    assert sum(ell for ell, _ in tokens) > 0


def test_delta_parse_and_roundtrip() -> None:
    prior = (b"def parse(data):\n    return data\n" * 40) + bytes(range(128))
    data = prior[:400] + b"# edit\n" + prior[400:800] + bytes(range(200))
    tokens = codec.delta_parse(data, prior, window=512, min_match=4, cap=16)
    assert tokens == codec.delta_parse(data, prior, window=512, min_match=4, cap=16)
    arm = codec.DeltaArm(
        window=512, min_match=4, cap=16, max_distance=max(len(prior), len(data))
    )
    blob = arm.encode(data, prior, tokens)
    assert (
        codec.DeltaArm(
            window=512, min_match=4, cap=16, max_distance=max(len(prior), len(data))
        ).decode(blob, len(data), prior)
        == data
    )
    assert any(ell for ell, _ in tokens)
    assert any(not ell for ell, _ in tokens)


def test_stage0_coverage_and_charge() -> None:
    import stage0

    train = b"def build(data):\n    return data + data\n" * 40
    test = b"def build(data):\n    return data\n" * 20
    lex = codec.build_lexicon(train, size=32, min_count=4, cap=8)
    rows = stage0.coverage_rows(test, [(32, lex)])
    row = rows[0]
    assert row["pack_bytes"] > 0
    assert 0.0 < row["coverage"] <= 1.0
    assert row["payload_bytes"] > 0
    assert row["static_bpb"] > row["payload_bpb"] / 2
    report = stage0.perfile_rows(test, [(32, lex)], [512], baseline=True)
    assert report["rows"][0]["files"] == 1
    assert report["rows"][0]["shared_mean"] > report["rows"][0]["base_mean"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pointer codec uses cuda")
def test_pointer_roundtrip() -> None:
    import pointer

    data = b"def f(x):\n    return x + 1\n" * 8
    full = codec.build_lexicon(data * 4, size=8, min_count=2, cap=4)
    lex = codec.Lexicon(full.units[:8])
    els = pointer.elements(data, lex)
    cfg = pointer.PointerConfig(
        d=32, layers=1, heads=4, seq=32, active=8, steps=2, batch=2
    )
    model, _ = pointer.train_pointer(els, lex.units, cfg)
    pc = pointer.PointerCodec(model, lex)
    blob = pc.encode(data)
    assert pc.decode(blob, len(data)) == data
    empty = pointer.PointerCodec(model, codec.Lexicon(()))
    blob = empty.encode(data)
    assert empty.decode(blob, len(data)) == data
    raw, tags = pc.continuation(b"def", 8, 0)
    assert len(raw) == 8 and tags
    assert all(t[2].startswith("copy@") or t[2] == "~neural" for t in tags)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="byte codec uses cuda")
def test_static_byte_roundtrip() -> None:
    import bytecodec

    model = bytecodec.Trunk(256, 32, 1, 4, 32).cuda()
    pc = bytecodec.StaticByteCodec(model, half=True)
    data = b"alpha = beta\n" * 16
    blob = pc.encode(data)
    assert pc.decode(blob, len(data)) == data


@pytest.mark.skipif(not torch.cuda.is_available(), reason="byte codec uses cuda")
def test_rep_copy_roundtrip() -> None:
    import bytecodec

    model = bytecodec.Trunk(256, 32, 1, 4, 32).cuda()
    data = (b"alpha = beta + gamma\n" * 8) + bytes(range(40))
    codec = bytecodec.RepCopyCodec(
        bytecodec.StreamCoder(model), window=64, min_match=4, cap=16
    )
    blob = codec.encode(data)
    fresh = bytecodec.RepCopyCodec(
        bytecodec.StreamCoder(model), window=64, min_match=4, cap=16
    )
    assert fresh.decode(blob, len(data)) == data


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


def test_synseg_and_pack_key() -> None:
    assert cml.synseg_bytes(400) == cml.synseg_bytes(400)
    assert cml.synseg_bytes(400) != cml.synseg_bytes(400, seed=1)
    assert b"BEGIN req=" in cml.synseg_bytes(200)
    cfg = cml.Config()
    assert cml.pack_key(cfg) == cml.pack_key(cml.Config())
    assert cml.pack_key(cfg) != cml.pack_key(cml.Config(dict_size=8192))


def test_gate_modes() -> None:
    knn = {"kind": "knn", "theta": 0.5, "tau": 1.0, "alpha": 0.5}
    assert cml.gate_prob(knn, [0.6]) == 1.0
    assert cml.gate_prob(knn, [0.4]) == 0.0
    logistic = {
        "kind": "logistic",
        "w": [1.0, 0.0, 0.0],
        "b": 0.0,
        "mu": [0.0, 0.0, 0.0],
        "sd": [1.0, 1.0, 1.0],
        "tau": 1.0,
        "alpha": 0.5,
    }
    assert cml.gate_prob(logistic, [10.0, 0.0, 0.0]) > 0.9
    assert 0.4 < cml.gate_prob(logistic, [0.0, 0.0, 0.0]) < 0.6


def test_decode_byte_matches_stream() -> None:
    if not torch.cuda.is_available():
        pytest.skip("decode needs a GPU")
    torch.manual_seed(0)
    model = cml.Trunk(256, 32, 1, 2, 16).cuda()
    data = cml.synseg_bytes(600)
    ids = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    nll, _, n = cml.stream_nll(model, ids, batch=4)
    gate = {
        "kind": "logistic",
        "w": [0.0, 0.0, 0.0],
        "b": 0.0,
        "mu": [0.0, 0.0, 0.0],
        "sd": [1.0, 1.0, 1.0],
        "tau": 1e9,
        "alpha": 0.5,
    }
    out = cml.decode_arm(
        model, (), {}, 1, torch.zeros((0, 32)), gate, data, n, mode="byte"
    )
    want = float(nll[1:n].sum()) / math.log(2) / (n - 1)
    assert abs(out["bpb"] - want) < 1e-3
    assert out["secs"] > 0.0


def test_decode_copy_and_fallback() -> None:
    if not torch.cuda.is_available():
        pytest.skip("decode needs a GPU")
    torch.manual_seed(0)
    model = cml.Trunk(256, 32, 1, 2, 32).cuda()
    data = cml.synseg_bytes(4_000)
    entries, total = cml.small_pack(data[:2_000], size=64, min_count=4)
    trie = cml.make_trie(entries)
    e_c = cml.span_embeddings(model, entries)
    ids = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    _, _, n = cml.stream_nll(model, ids, batch=4)
    gate = {
        "kind": "logistic",
        "w": [0.0, 0.0, 0.0],
        "b": 8.0,
        "mu": [0.0, 0.0, 0.0],
        "sd": [1.0, 1.0, 1.0],
        "tau": 1e9,
        "alpha": 0.5,
    }
    out = cml.decode_arm(model, entries, trie, total, e_c, gate, data, n)
    assert out["copies"] > 0 and out["copy_bytes"] > 0
    assert out["coverage"] > 0.2
    strict = cml.decode_arm(
        model, entries, trie, total, e_c, dict(gate, tau=0.0), data, n
    )
    assert strict["copies"] == 0 and strict["verify_fail"] > 0
    assert abs(strict["bpb"] - out["bpb"]) > 0.1


def test_heldout_subset_bookkeeping() -> None:
    entries = tuple(cml.Entry(b"S%02d" % i, 10, 0, ("fit:0",)) for i in range(20))
    data = b"|" + b"|".join(e.data for e in entries)
    trie = cml.make_trie(entries)
    total = 200
    n_eval = len(data)
    nll = torch.zeros(n_eval)
    hs = torch.zeros((n_eval, 4))
    gate = {
        "kind": "logistic",
        "w": [0.0, 0.0, 0.0],
        "b": 8.0,
        "mu": [0.0, 0.0, 0.0],
        "sd": [1.0, 1.0, 1.0],
        "tau": 1.0,
        "alpha": 0.5,
    }
    e_c = torch.zeros((20, 4))
    out = cml.adapter_eval(
        None, entries, trie, total, e_c, gate, data, nll, hs, n_eval, 1000, 10_000
    )
    assert out["held_matches"] == 2
    assert out["held_bytes"] == 6
    assert out["held_accept"] == 1.0
    assert 0.0 <= out["held_bpb"] < 0.01


def test_zeroshot_keys(tmp_path: Path) -> None:
    p = tmp_path / "z.toml"
    p.write_text(
        '[run]\nname = "z"\nseed = 0\ngates = ["P1e"]\n'
        '[data]\ncorpus = "src"\npack_corpus = "enwik8_10m"\n'
        '[model]\narm = "adapter"\nmode = "adapter"\n'
    )
    cfg, gates = cml.config_from_manifest(p, None, None)
    assert cfg.pack_corpus == "enwik8_10m"
    assert cfg.mode == "adapter"
    assert gates == ["P1e"]


def test_src_stream_is_frozen() -> None:
    a = cml.src_bytes()
    b = cml.src_bytes()
    assert a == b
    assert len(a) > 100_000
    assert b"# --- cml/cml.py ---" in a


def test_pack_offsets_and_trace() -> None:
    train = b"def build(data):\n    return data + data\n" * 40
    lex = codec.build_lexicon(train, size=64, min_count=4, cap=8)
    blob = lex.blob()
    for off, unit in zip(lex.offsets(), lex.units):
        assert blob[off : off + len(unit)] == unit
    parse, _ = codec.greedy_parse(train, lex)
    payload = codec.UnitArm(lex).encode(train, parse)
    data, spans = codec.UnitArm(lex).trace(payload, len(train))
    assert data == train
    assert b"".join(b for a, b, _ in spans for b in [data[a:b]]) == data
    for start, end, sym in spans:
        unit = lex.units[sym] if sym >= 0 else bytes([-1 - sym])
        assert data[start:end] == unit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pointer codec uses cuda")
def test_concept_roundtrip_and_stats() -> None:
    data = b"alpha beta alpha"
    lex = codec.Lexicon((b"alpha ", b"beta "))
    ids, lens = cml._concept_stream(data, lex)
    assert cml._concept_roundtrip(ids, lex) == data
    stats = cml._concept_stats(ids, lens, lex)
    assert stats["nbytes"] == len(data)
    assert stats["decisions"] == ids.numel()
    assert stats["bytes_per_decision"] == len(data) / ids.numel()


def test_pointer_design_matrix() -> None:
    import pointer

    data = b"def f(x):\n    return x + 1\n" * 6
    full = codec.build_lexicon(data * 4, size=8, min_count=2, cap=4)
    lex = codec.Lexicon(full.units[:8])
    els = pointer.elements(data, lex)
    for span in ("content", "ngram"):
        for head in ("gate", "joint"):
            cfg = pointer.PointerConfig(
                d=32,
                layers=1,
                heads=4,
                seq=32,
                active=8,
                steps=2,
                batch=2,
                span=span,
                head=head,
            )
            model, _ = pointer.train_pointer(els, lex.units, cfg)
            pc = pointer.PointerCodec(model, lex)
            for parse in ("greedy", "priced"):
                blob = pc.encode(data, parse=parse)
                assert pc.decode(blob, len(data)) == data
            assert len(pc.sample(16, seed=0)) == 16


def test_concept_realize_exact() -> None:
    import concepts

    lib = concepts.arithmetic_library()
    x = concepts.Literal("Int", b"2")
    y = concepts.Literal("Int", b"3")
    mul = concepts.Term("MUL", (x, y))
    root = concepts.Term("ADD", (mul, concepts.Literal("Int", b"4")))
    assert concepts.realize(root, lib) == b"ADD(MUL(2, 3), 4)"
    assert concepts.size(root) == 5
    assert concepts.pairs(root) == {("@root", "ADD"), ("ADD", "MUL")}
    assert concepts.has_ban(root)


def test_concept_typed_holes() -> None:
    import concepts

    lib = concepts.arithmetic_library()
    with pytest.raises(ValueError):
        concepts.realize(concepts.Term("ADD", (concepts.Literal("Int", b"1"),)), lib)
    with pytest.raises(KeyError):
        concepts.realize(concepts.Term("EXP", ()), lib)


def test_concept_library_mdl_gate() -> None:
    import concepts

    lib = concepts.Library()
    template = concepts.arithmetic_library().templates["ADD"]
    cost = 8 * len(template.blob())
    assert not lib.offer(template, cost - 1)
    assert not lib.offer(template, cost)
    assert lib.offer(template, cost + 1)
    assert lib.rejected == 2
    assert "ADD" in lib.templates


def test_concept_composition_generalizes() -> None:
    import concepts

    lib = concepts.arithmetic_library()
    banned = concepts.split(lib, 50, 3, seed=2, ban=True)
    clean = concepts.split(lib, 50, 3, seed=3, ban=False)
    assert all(concepts.has_ban(t) for t in banned)
    assert not any(concepts.has_ban(t) for t in clean)
    assert all(not concepts.has_ban(concepts.twin(t)) for t in banned)
    assert [concepts.size(concepts.twin(t)) for t in banned] == [
        concepts.size(t) for t in banned
    ]
    terms = clean
    blob = concepts.ConceptArm(lib).encode(terms)
    back = concepts.ConceptArm(lib).decode(blob)
    assert [concepts.realize(t, lib) for t in back] == [
        concepts.realize(t, lib) for t in terms
    ]


def test_cortex_stream_roundtrip() -> None:
    import concepts
    import cortex

    lib = cortex.library()
    terms = concepts.split(lib, 20, 3, seed=4, ban=False, banned=cortex.BANNED)
    ids, nbytes = cortex.pack(terms, lib)
    back = cortex.unpacked(ids, lib, len(terms))
    assert [concepts.realize(t, lib) for t in back] == [
        concepts.realize(t, lib) for t in terms
    ]
    assert nbytes == len(b"\n".join(concepts.realize(t, lib) for t in terms))
    assert ids.count(cortex.SEP) == len(terms)


def test_cortex_split_and_twin() -> None:
    import concepts
    import cortex

    lib = cortex.library()
    unseen = concepts.split(lib, 20, 3, seed=5, ban=True, banned=cortex.BANNED)
    assert all(concepts.has_ban(t, cortex.BANNED) for t in unseen)
    twins = [concepts.twin(t, "ADD", "XOR", "AND") for t in unseen]
    assert not any(concepts.has_ban(t, cortex.BANNED) for t in twins)
    assert [concepts.size(t) for t in twins] == [concepts.size(t) for t in unseen]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cortex trains on cuda")
def test_cortex_trains_and_codes() -> None:
    import concepts
    import cortex

    lib = cortex.library()
    terms = concepts.split(lib, 60, 3, seed=6, ban=False, banned=cortex.BANNED)
    ids, _ = cortex.pack(terms, lib)
    model, _ = cortex.train(
        ids, cortex.VOCAB, {"d": 32, "layers": 1, "heads": 4}, 16, 5, 4, 0
    )
    bits, n = cortex.stream_bits(model, ids, 16, batch=4)
    assert n > 0
    assert bits > 0


def test_cortex_sweep_render() -> None:
    import cortex

    def run(depth: int, seen: float, unseen: float, twin: float) -> dict:
        return {
            "depth": depth,
            "rows": [
                {"split": "seen", "cortex_bpb": seen, "byte_bpb": 1.0},
                {"split": "unseen", "cortex_bpb": unseen, "byte_bpb": 1.0},
                {"split": "twin", "cortex_bpb": twin, "byte_bpb": 1.0},
            ],
        }

    report = {
        "runs": [run(3, 1.1, 1.2, 1.0), run(5, 1.0, 1.05, 0.99)],
        "pairs": 100,
        "steps": 10,
        "seq": 64,
        "batch": 8,
        "seed": 0,
        "cfg": {"d": 32, "layers": 1, "heads": 4},
        "params": 1000,
        "seconds": 1.0,
    }
    text = cortex.render_sweep(report)
    assert "| depth |" in text
    assert "| 3 |" in text and "| 5 |" in text
    assert "edge gap" in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="stream codec uses cuda")
def test_stream_and_copy_roundtrip() -> None:
    import bytecodec

    torch.manual_seed(0)
    model = bytecodec.Trunk(256, 32, 1, 2, 16).cuda()
    data = (b"alpha = beta + gamma\n" * 12) + bytes(range(32))
    blob = bytecodec.StreamCoder(model).encode(data)
    assert bytecodec.StreamCoder(model).decode(blob, len(data)) == data
    plain = bytecodec.StreamCoder(model).encode(data)
    copy = bytecodec.CopyCodec(bytecodec.StreamCoder(model), 256, 8, 32)
    blob = copy.encode(data)
    back = bytecodec.CopyCodec(bytecodec.StreamCoder(model), 256, 8, 32).decode(
        blob, len(data)
    )
    assert back == data
    assert len(blob) < len(plain)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fast codec uses cuda")
def test_fast_byte_size_matches() -> None:
    import cml
    import bytecodec

    data = cml.code_corpus().test[:2048]
    model = bytecodec.load_model(cml.RUNS / "codec-byte-model-code-756752/model.pt")
    fast = len(bytecodec.FastByteCodec(model).encode(data))
    slow = len(bytecodec.StaticByteCodec(model).encode(data))
    assert abs(fast - slow) <= 0.01 * slow
