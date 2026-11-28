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
