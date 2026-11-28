import json
from pathlib import Path

import mcp_server
import memory


def test_mcp_tools(tmp_path: Path) -> None:
    versions = [b"a\nneedle\n", b"a\nneedle\nb\n"]
    blob = memory.encode_chain(versions, "x.py", ["c0", "c1"], memory.ChainConfig())
    (tmp_path / "x.py.cmlh").write_bytes(blob)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "stats": {"exact": True},
                "files": [{"path": "x.py", "file": "x.py.cmlh"}],
            }
        )
    )
    index = mcp_server.Index(tmp_path)
    init = mcp_server.handle(index, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init["result"]["serverInfo"]["name"] == "repo-memory"
    tools = mcp_server.handle(
        index, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    )
    names = {t["name"] for t in tools["result"]["tools"]}
    assert {"repo_search", "repo_read", "repo_versions", "repo_diff"} <= names
    search = mcp_server.handle(
        index,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "repo_search", "arguments": {"query": "needle"}},
        },
    )
    assert "x.py:2" in search["result"]["content"][0]["text"]
    read = mcp_server.handle(
        index,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "repo_read", "arguments": {"path": "x.py"}},
        },
    )
    assert read["result"]["content"][0]["text"] == versions[-1].decode()


def test_mcp_resources(tmp_path: Path) -> None:
    versions = [b"a\nneedle\n", b"a\nneedle\nb\n"]
    blob = memory.encode_chain(versions, "x.py", ["c0", "c1"], memory.ChainConfig())
    (tmp_path / "x.py.cmlh").write_bytes(blob)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "stats": {"exact": True},
                "files": [{"path": "x.py", "file": "x.py.cmlh", "versions": 2}],
            }
        )
    )
    index = mcp_server.Index(tmp_path)
    init = mcp_server.handle(index, {"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert "resources" in init["result"]["capabilities"]

    listed = mcp_server.handle(
        index, {"jsonrpc": "2.0", "id": 2, "method": "resources/list"}
    )
    assert listed["result"]["resources"][0]["uri"] == "repo://x.py"

    latest = mcp_server.handle(
        index,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/read",
            "params": {"uri": "repo://x.py"},
        },
    )
    assert latest["result"]["contents"][0]["text"] == versions[-1].decode()

    first = mcp_server.handle(
        index,
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/read",
            "params": {"uri": "repo://x.py@0"},
        },
    )
    assert first["result"]["contents"][0]["text"] == versions[0].decode()

    templates = mcp_server.handle(
        index, {"jsonrpc": "2.0", "id": 5, "method": "resources/templates/list"}
    )
    assert templates["result"]["resourceTemplates"][0]["uriTemplate"] == (
        "repo://{path}@{version}"
    )
    bad = mcp_server.handle(
        index,
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "resources/read",
            "params": {"uri": "repo://x.py@9"},
        },
    )
    assert bad["error"]["code"] == -32602


def test_mcp_global_ranges(tmp_path: Path) -> None:
    versions = [b"a\n", b"b\n", b"c\n"]
    blob = memory.encode_chain(versions, "repo", [], memory.ChainConfig())
    (tmp_path / "repo.cmlh").write_bytes(blob)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "stats": {"exact": True, "global": True},
                "files": [
                    {
                        "path": "a.py",
                        "file": "repo.cmlh",
                        "ranges": [[0, 1]],
                        "versions": 1,
                    },
                    {
                        "path": "b.py",
                        "file": "repo.cmlh",
                        "ranges": [[1, 2], [2, 3]],
                        "versions": 2,
                    },
                ],
            }
        )
    )
    index = mcp_server.Index(tmp_path)
    header, got = index.versions("b.py")
    assert got == versions[1:]
    assert header["sizes"] == [2, 2]
    assert index.latest("b.py") == b"c\n"
