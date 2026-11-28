import codebench


def test_split_and_measure(tmp_path) -> None:
    root = tmp_path
    for i in range(12):
        (root / f"f{i}.py").write_text(
            f"def f{i}(self, x):\n    return self.value + x\n"
        )
    train, test = codebench.split(root, 100_000)
    assert train and test
    stats = codebench.measure(
        train, test, root, min_fill=4, min_count=1, max_n=8, use_trunk=False
    )
    assert stats["train_bytes"] > 0
    assert stats["test_bytes"] > 0
    assert 0.0 <= stats["coverage"] <= 1.0
    assert 0.0 <= stats["exact_lines"] <= 1.0
