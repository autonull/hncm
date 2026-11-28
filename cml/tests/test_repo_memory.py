import json
import subprocess
from pathlib import Path

import memory


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=a@b", "-c", "user.name=a", *args],
        check=True,
        capture_output=True,
    )


def test_repo_index_roundtrip_and_search(tmp_path: Path, capsys) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    path = repo / "x.py"
    path.write_text("alpha\nneedle\n")
    _git(repo, "add", "x.py")
    _git(repo, "commit", "-qm", "v1")
    path.write_text("alpha\nneedle\nbeta\n" * 20)
    _git(repo, "commit", "-qam", "v2")

    out = tmp_path / "idx"
    memory.run_repo(
        repo,
        out,
        tmp_path / "report.md",
        tmp_path / "index.json",
        "",
        None,
        "",
        0,
        65536,
        1 << 20,
        8,
        256,
        16,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    row = manifest["files"][0]
    header, versions = memory.load_chain(out / row["file"])
    assert versions[-1] == path.read_bytes()
    assert manifest["stats"]["exact"]

    memory.run_repo(
        None,
        out,
        tmp_path / "report.md",
        tmp_path / "index.json",
        "needle",
        out,
        "",
        0,
        65536,
        1 << 20,
        8,
        256,
        16,
    )
    text = capsys.readouterr().out
    assert "x.py:2" in text and "needle" in text
