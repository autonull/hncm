"""Minimal MCP stdio server for an exact repo-memory index.

Tools: repo_files, repo_search, repo_read, repo_versions, repo_diff.
Every read decodes the exact chain and checks its SHA; search is exact text.
Run: `uv run cml mcp --index <repo-index>`.
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

import memory

PROTOCOL = "2024-11-05"
TOOLS = [
    {
        "name": "repo_files",
        "description": "List the files in the exact repo-memory index.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "repo_search",
        "description": "Exact text search over the latest version; returns path:line hits.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "repo_read",
        "description": "Read one file version exactly (default: latest).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "version": {"type": "integer", "default": -1},
            },
            "required": ["path"],
        },
    },
    {
        "name": "repo_versions",
        "description": "List the versions of one file with sizes and blob SHAs.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "repo_diff",
        "description": "Exact unified diff between two versions of one file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "before": {"type": "integer", "default": 0},
                "after": {"type": "integer", "default": -1},
            },
            "required": ["path"],
        },
    },
]


class Index:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.rows = {row["path"]: row for row in self.manifest["files"]}

    def versions(self, path: str) -> tuple[dict, list[bytes]]:
        row = self.rows.get(path)
        if row is None:
            raise KeyError(path)
        return memory.load_chain(self.root / row["file"])

    def latest(self, path: str) -> bytes:
        return self.versions(path)[1][-1]


def _text(content: str, error: bool = False) -> dict:
    return {
        "content": [{"type": "text", "text": content}],
        "isError": error,
    }


def call(index: Index, name: str, args: dict) -> dict:
    if name == "repo_files":
        return _text("\n".join(sorted(index.rows)))
    if name == "repo_search":
        query = str(args.get("query", ""))
        if not query:
            return _text("repo_search: empty query", True)
        limit = max(1, min(int(args.get("limit", 20)), 200))
        needle = query.encode()
        hits = []
        for path in sorted(index.rows):
            if len(hits) >= limit:
                break
            last = index.latest(path)
            at = last.find(needle)
            while at >= 0 and len(hits) < limit:
                line = last.count(b"\n", 0, at) + 1
                start = last.rfind(b"\n", 0, at) + 1
                end = last.find(b"\n", at)
                text = last[start : end if end >= 0 else len(last)]
                hits.append(f"{path}:{line}: {text.decode('utf-8', 'replace')[:160]}")
                at = last.find(needle, at + 1)
        return _text("\n".join(hits) if hits else f"no exact hit for {query!r}")
    if name == "repo_read":
        path = str(args.get("path", ""))
        version = int(args.get("version", -1))
        header, versions = index.versions(path)
        if version < 0:
            version = len(versions) - 1
        if not 0 <= version < len(versions):
            return _text(f"repo_read: version {version} out of range", True)
        return _text(versions[version].decode("utf-8", "replace"))
    if name == "repo_versions":
        path = str(args.get("path", ""))
        header, versions = index.versions(path)
        rows = [
            {"index": i, "bytes": len(v), "sha": header["shas"][i]}
            for i, v in enumerate(versions)
        ]
        return _text(json.dumps(rows, indent=1))
    if name == "repo_diff":
        path = str(args.get("path", ""))
        before = int(args.get("before", 0))
        after = int(args.get("after", -1))
        header, versions = index.versions(path)
        if after < 0:
            after = len(versions) - 1
        if not (0 <= before < len(versions) and 0 <= after < len(versions)):
            return _text("repo_diff: version out of range", True)
        diff = difflib.unified_diff(
            versions[before].decode("utf-8", "replace").splitlines(),
            versions[after].decode("utf-8", "replace").splitlines(),
            fromfile=f"{path}@{before}",
            tofile=f"{path}@{after}",
            lineterm="",
        )
        return _text("\n".join(diff))
    return _text(f"unknown tool: {name}", True)


def handle(index: Index, req: dict) -> dict | None:
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "repo-memory", "version": "0.1"},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = req.get("params") or {}
        try:
            result = call(index, params.get("name", ""), params.get("arguments") or {})
        except (KeyError, ValueError) as exc:
            result = _text(f"{type(exc).__name__}: {exc}", True)
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if rid is None:
        return None
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def run(index_dir: Path) -> int:
    index = Index(index_dir)
    print(f"repo-memory MCP: {len(index.rows)} files", file=sys.stderr, flush=True)
    out = sys.stdout
    for line in sys.stdin:
        body = line
        if line.lower().startswith("content-length:"):
            n = int(line.split(":", 1)[1])
            body = sys.stdin.read(n)
        body = body.strip()
        if not body:
            continue
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            continue
        resp = handle(index, req)
        if resp is not None:
            out.write(json.dumps(resp) + "\n")
            out.flush()
    return 0
