import difflib
from pathlib import Path

import pytest

import memory
import tools


def test_patch_guard_exact_and_reject(tmp_path: Path) -> None:
    base = tmp_path / "old.py"
    new = tmp_path / "new.py"
    base.write_text("alpha\nbeta\n")
    new.write_text("alpha\nBETA\ngamma\n")
    diff = tmp_path / "p.diff"
    diff.write_text(
        "\n".join(
            difflib.unified_diff(
                base.read_text().splitlines(),
                new.read_text().splitlines(),
                fromfile=str(base),
                tofile=str(new),
                lineterm="",
            )
        )
        + "\n"
    )
    assert tools.run_patch(base, diff, new, 0) == 0
    with pytest.raises(SystemExit):
        tools.run_patch(base, diff, base, 0)


def test_logdelta_roundtrip(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    versions = [f"run {i}\ncommon line\n".encode() * 20 for i in range(3)]
    for i, v in enumerate(versions):
        (logs / f"run{i}.log").write_bytes(v)
    out = tmp_path / "logs.cmlh"
    assert tools.run_logdelta(logs, out, tmp_path / "r.md", tmp_path / "j.json") == 0
    header, back = memory.load_chain(out)
    assert back == versions
