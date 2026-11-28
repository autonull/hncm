import templates


def test_tokenize_marks_syntax_and_holes() -> None:
    ts = templates.tokenize(b"def f(x):\n")
    assert (b"F", b"def") in ts
    assert (b"I", b"f") in ts
    assert (b"I", b"x") in ts


def test_induce_and_roundtrip() -> None:
    data = b"def a(x):\n    return x+1\ndef b(y):\n    return y+2\n"
    lines = data.splitlines(keepends=True)
    found, fills = templates.induce(lines, min_count=1, max_n=8)
    assert found
    assert fills
    events = templates.parse(data, found, max_n=8)
    assert any(isinstance(event, templates.Hit) for event in events)
    assert templates.render(events) == data


def test_probe_reports_categories() -> None:
    train = b"def a(x):\n    return x\ndef b(y):\n    return y\n" * 2
    test = b"def c(z):\n    return z\n"
    m = templates.probe(train, test, min_count=1, max_n=8)
    assert m["roundtrip"]
    assert 0.0 < m["fixed"] < 1.0
    assert m["decisions"] > 0.0
    assert m["bytes_per_decision"] > 1.0
