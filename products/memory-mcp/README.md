# `memory-mcp` — exact repo memory for an agent

An MCP stdio server over the `repo-memory` index. The agent lists files,
searches exact text, reads any version, lists versions, and diffs two
versions. Every read is SHA-verified.

```bash
uv run cml mcp --index /path/to/repo-index
```

OpenCode config:

```json
{
  "mcp": {
    "repo-memory": {
      "type": "local",
      "command": [
        "uv", "run", "--directory", "/path/to/cml",
        "cml", "mcp", "--index", "/path/to/repo-index"
      ],
      "enabled": true
    }
  }
}
```

## Tools

| tool | arguments | returns |
|---|---|---|
| `repo_files` | none | every indexed path |
| `repo_search` | `query`, `limit` | exact `path:line: text` hits |
| `repo_read` | `path`, `version=-1` | exact file bytes |
| `repo_versions` | `path` | index, size, blob SHA per version |
| `repo_diff` | `path`, `before`, `after` | exact unified diff |

## Why it matters

The agent gets exact repository context without retraining and without a
flat span-id codec. The memory is queried as bytes; the model conditions
on the result. This is the product path from the exact codecs to the
memory-augmented LM.
