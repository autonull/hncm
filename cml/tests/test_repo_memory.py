import gzip
import json
import subprocess
from pathlib import Path

import mcp_server
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
    checkpoint: int = 0,
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
        checkpoint,
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


def test_repo_global_checkpoints(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("a line\n" * 40)
    (repo / "b.py").write_text("b line\n" * 40)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "v1")
    for i in range(12):
        (repo / "a.py").write_text("a line\n" * 40 + f"v{i}\n")
        _git(repo, "commit", "-qam", f"v{i + 2}")

    out = tmp_path / "idx"
    _build(repo, out, global_mode=True, checkpoint=4)
    manifest = json.loads((out / "manifest.json").read_text())
    stats = manifest["stats"]
    assert stats["checkpoint"] == 4
    assert [c["index"] for c in manifest["checkpoints"]] == list(range(4, 14, 4))
    checkpoint_bytes = sum(c["bytes"] for c in manifest["checkpoints"])
    assert stats["checkpoint_bytes"] == checkpoint_bytes
    assert stats["chain_bytes"] == (
        (out / "repo.cmlh").stat().st_size + checkpoint_bytes
    )

    blob = (out / "repo.cmlh").read_bytes()
    raw = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
    _, versions = memory.decode_chain(raw)

    def no_full(path: Path) -> tuple[dict, list[bytes]]:
        raise AssertionError("checkpoint access decoded the whole chain")

    monkeypatch.setattr(memory, "load_chain", no_full)
    index = mcp_server.Index(out)
    for row in manifest["files"]:
        indices = [i for start, end in row["ranges"] for i in range(start, end)]
        for i in range(row["versions"]):
            assert index.version(row["path"], i) == versions[indices[i]]
    assert index._shared is None


def test_repo_global_checkpoints_from_update(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("a\n" * 40)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "v1")
    for i in range(6):
        (repo / "a.py").write_text("a\n" * 40 + f"v{i}\n")
        _git(repo, "commit", "-qam", f"v{i + 2}")
    out = tmp_path / "idx"
    _build(repo, out, global_mode=True)
    before = json.loads((out / "manifest.json").read_text())["stats"]["chain_bytes"]
    _build(repo, out, global_mode=True, update=True, checkpoint=3)
    manifest = json.loads((out / "manifest.json").read_text())
    assert [c["index"] for c in manifest["checkpoints"]] == [3, 6]
    assert manifest["stats"]["chain_bytes"] > before
    assert (out / "ckpt").is_dir()
