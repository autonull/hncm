import os
import random
import subprocess
from pathlib import Path

import pytest

import editdata
import editlm


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )


def test_split_is_deterministic_and_disjoint() -> None:
    repos = ["e/5", "b/2", "d/4", "a/1", "c/3"]
    split = editdata.repo_splits(repos)
    assert split == {
        "a/1": "train",
        "b/2": "train",
        "c/3": "train",
        "d/4": "val",
        "e/5": "test",
    }
    assert editdata.repo_splits(repos) == split
    assert len(set(split.values()) - {"train", "val", "test"}) == 0


def test_split_rejects_too_few() -> None:
    with pytest.raises(ValueError, match="at least three"):
        editdata.repo_splits(["a/1", "b/2"])


def test_accept_blob_filters() -> None:
    old = b"a\nb\nc\nd\ne\nf\ng\nh\ni\nj\n"
    new = b"a\nb\nc\nd\ne\nf\ng\nh\ni\nJ\n"
    assert editdata.accept_blob("pkg/mod.py", old, new, 64 * 1024, 0.2)
    assert editdata.diff_fraction(old, new) == pytest.approx(0.2)
    assert not editdata.accept_blob("pkg/mod.py", old, new, 64 * 1024, 0.1)
    assert not editdata.accept_blob("pkg/vendor/x.py", old, new, 64 * 1024, 0.2)
    assert not editdata.accept_blob("pkg/mod.py", old, new, 10, 0.2)
    assert not editdata.accept_blob("pkg/mod.py", old, b"\x00", 64 * 1024, 0.2)
    assert not editdata.accept_blob("pkg/mod.py", old, old, 64 * 1024, 0.2)


def test_action_roundtrip() -> None:
    old = b"alpha\nbeta\ngamma\n"
    new = b"alpha\nBETA\ngamma\ndelta\n"
    actions = editlm.copy_actions(new, old, min_match=4)
    assert editlm.assemble(actions, old) == new
    assert editlm.assemble(actions, old, len(new)) == new
    assert editlm.load_actions(editlm.dump_actions(actions)) == actions


def test_action_assembler_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        editlm.assemble([editlm.Action("COPY", 0, 1)], b"abc")
    with pytest.raises(ValueError):
        editlm.assemble([editlm.Action("COPY", 9, 1)], b"abc")
    with pytest.raises(ValueError):
        editlm.assemble([editlm.Action("NOPE")], b"abc")
    with pytest.raises(ValueError):
        editlm.assemble([editlm.Action("LIT", data=b"x")], b"abc", expect=2)
    with pytest.raises(ValueError):
        editlm.load_actions(b"\x02\x05x")


def test_spans_roundtrip_and_scores() -> None:
    old = b"alpha\nbeta\ngamma\n"
    new = b"alpha\nBETA\ngamma\ndelta\n"
    spans = editlm.spans_from(old, new)
    assert editlm.assemble_spans(old, spans) == new
    assert all(0 <= s.old_start <= len(old) for s in spans)
    assert all(s.old_start + s.old_len <= len(old) for s in spans)
    assert editlm.span_scores(spans, spans)["location_exact"]
    assert editlm.span_scores(spans, spans)["f1"] == 1.0
    assert editlm.span_scores([], spans)["f1"] == 0.0


def test_transplant_identical_and_shifted() -> None:
    assert editlm.transplant(b"abc", b"abc", b"abd") == b"abd"
    ret_old = b"a\nb\nc\nd\n"
    ret_new = b"a\nB\nc\nd\n"
    assert editlm.transplant(b"x\ny\n" + ret_old, ret_old, ret_new) == (
        b"x\ny\na\nB\nc\nd\n"
    )
    assert editlm.transplant(b"unrelated\n", ret_old, ret_new) is None


def test_locator_forward_and_predict() -> None:
    import torch

    model = editlm.Locator(d=16, layers=1, heads=2, msg=8, win=16)
    msg = torch.zeros(2, 8, dtype=torch.long)
    file = torch.randint(0, 256, (2, 16))
    assert model(msg, file).shape == (2, 16)
    assert model(msg, file, zero_file=True).shape == (2, 16)
    spans = editlm.predict_spans(model, b"x" * 40, "fix", torch.device("cpu"))
    assert isinstance(spans, list)


def test_window_starts_cover_the_tail() -> None:
    starts = editlm._window_starts(100, 16, 8)
    assert starts[0] == 0
    assert starts[-1] == 84
    assert all(b - a <= 8 for a, b in zip(starts, starts[1:]))


def test_span_prf_uses_iou() -> None:
    target = [editlm.Span(10, 4, b"")]
    assert editlm.span_prf([editlm.Span(10, 4, b"")], target)["f1"] == 1.0
    assert editlm.span_prf([editlm.Span(11, 4, b"")], target)["f1"] == 1.0
    assert editlm.span_prf([editlm.Span(30, 4, b"")], target)["f1"] == 0.0
    assert editlm.span_prf([], target)["f1"] == 0.0


def test_sample_window_shapes_and_focus() -> None:
    old = b"".join(f"line {i}\n".encode() for i in range(20))
    start = old.index(b"line 9\n")
    span = editlm.Span(start, 7, b"")
    example = editlm.Labeled(
        {"message": "edit line 9"},
        old,
        [span],
        editlm._mask_from_spans(old, [span]),
        [(start, start + 7)],
    )
    rng = random.Random(0)
    for _ in range(50):
        _, chunk, labels, length = editlm._sample(example, rng, 16, 8)
        assert 0 < length <= 16
        assert labels.shape == (16,)
        assert chunk


def test_locator_trains_on_cpu() -> None:
    import torch

    old = b"alpha\nbeta\ngamma\ndelta\n"
    span = editlm.Span(6, 4, b"")
    example = editlm.Labeled(
        {"message": "edit beta"},
        old,
        [span],
        editlm._mask_from_spans(old, [span]),
        [(span.old_start, span.old_start + span.old_len)],
    )
    model = editlm.Locator(d=16, layers=1, heads=2, msg=8, win=16)
    trained, log = editlm.train_locator(
        [example, example],
        1,
        0,
        "",
        False,
        torch.device("cpu"),
        batch=2,
        model=model,
    )
    assert len(log) == 1
    assert trained is model


def test_mine_local_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    path = repo / "mod.py"
    old = "".join(f"line {i}\n" for i in range(10)).encode()
    new = old.replace(b"line 3\n", b"LINE 3\n")
    path.write_bytes(old)
    git(repo, "add", "mod.py")
    git(repo, "commit", "-q", "-m", "add mod")
    path.write_bytes(new)
    git(repo, "add", "mod.py")
    git(repo, "commit", "-q", "-m", "edit line 3")

    records = editdata.mine_repo(
        repo, editdata.MineConfig("local/test", max_file=64 * 1024, max_diff=0.2)
    )
    assert len(records) == 1
    rec = records[0]
    assert rec["path"] == "mod.py"
    assert rec["message"] == "edit line 3"
    assert rec["old_size"] == len(old)
    assert rec["new_size"] == len(new)
    assert rec["diff_frac"] == pytest.approx(0.2)
    assert rec["old"] != rec["new"]
