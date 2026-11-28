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


def _build(
    repo: Path,
    out: Path,
    *,
    global_mode: bool = False,
    update: bool = False,
    limit: int = 0,
) -> None:
    memory.run_repo(
        repo,
        out,
        out.parent / "report.md",
        out.parent / "index.json",
        "",
        None,
        "",
        limit,
        65536,
        1 << 20,
        8,
        256,
        16,
        global_mode,
        update,
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
    _build(repo, out)
    manifest = json.loads((out / "manifest.json").read_text())
    row = manifest["files"][0]
    header, versions = memory.load_chain(out / row["file"])
    assert versions[-1] == path.read_bytes()
    assert manifest["stats"]["exact"]
    assert manifest["stats"]["head"]

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

    out2 = tmp_path / "gidx"
    _build(repo, out2, global_mode=True)
    gman = json.loads((out2 / "manifest.json").read_text())
    assert gman["stats"]["global"]
    _, gversions = memory.load_chain(out2 / "repo.cmlh")
    assert gversions[-1] == path.read_bytes()


def test_repo_update_appends_without_rebuild(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("alpha\n" * 100)
    (repo / "b.py").write_text("beta\n" * 100)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "v1")
    out = tmp_path / "idx"
    _build(repo, out)
    first = (out / "manifest.json").read_bytes()

    (repo / "a.py").write_text("alpha\n" * 100 + "new\n")
    (repo / "c.py").write_text("gamma\n" * 50)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "v2")

    encoded: list[str] = []
    real = memory.encode_chain

    def spy(versions, path, commits, cfg):
        encoded.append(path)
        return real(versions, path, commits, cfg)

    monkeypatch.setattr(memory, "encode_chain", spy)
    _build(repo, out, update=True)
    assert encoded == ["c.py"]

    updated = json.loads((out / "manifest.json").read_text())
    assert updated["stats"]["head"] != json.loads(first)["stats"]["head"]
    out2 = tmp_path / "full"
    _build(repo, out2, limit=0)
    full = json.loads((out2 / "manifest.json").read_text())
    assert {k: v for k, v in updated["stats"].items() if k != "seconds"} == {
        k: v for k, v in full["stats"].items() if k != "seconds"
    }
    for row in updated["files"]:
        assert (out / row["file"]).read_bytes() == (out2 / row["file"]).read_bytes()
        header, versions = memory.load_chain(out / row["file"])
        assert versions[-1] == (repo / row["path"]).read_bytes()

    before = (out / "manifest.json").read_bytes()
    encoded.clear()
    _build(repo, out, update=True)
    assert encoded == []
    assert (out / "manifest.json").read_bytes() == before


def test_repo_global_update_appends_exact(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("alpha\n" * 100)
    (repo / "b.py").write_text("beta\n" * 100)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "v1")
    out = tmp_path / "idx"
    _build(repo, out, global_mode=True)

    (repo / "a.py").write_text("alpha\n" * 100 + "new\n")
    (repo / "c.py").write_text("gamma\n" * 50)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "v2")
    _build(repo, out, global_mode=True, update=True)

    manifest = json.loads((out / "manifest.json").read_text())
    rows = {row["path"]: row for row in manifest["files"]}
    shared = memory.load_chain(out / "repo.cmlh")[1]
    for path, row in rows.items():
        versions = memory._row_versions(shared, row)
        assert versions[-1] == (repo / path).read_bytes()
    assert len(rows["a.py"]["ranges"]) == 2
    assert rows["c.py"]["ranges"] == [[3, 4]]
