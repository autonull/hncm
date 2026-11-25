from pathlib import Path

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
