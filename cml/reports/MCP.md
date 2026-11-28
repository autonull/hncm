# Repo-memory MCP server

The index is usable from an MCP client with no model and no network. Every
read decodes the exact chain and checks the SHA-256; every search is exact
text, not a fuzzy match.

## Tools

| tool | arguments | result |
|---|---|---|
| `repo_files` | none | newline-separated paths |
| `repo_search` | `query`, `limit` | `path:line: text` hits |
| `repo_read` | `path`, `version` | exact file text (default latest) |
| `repo_versions` | `path` | version sizes and blob SHAs |
| `repo_diff` | `path`, `before`, `after` | exact unified diff |

## Resources

| URI | meaning |
|---|---|
| `repo://<path>` | latest version of one file |
| `repo://<path>@<n>` | version `n` of one file |

`resources/list` returns one concrete resource per file; the
`repo://{path}@{version}` template covers older versions. `repo://` paths
are percent-encoded, so a path with spaces still round-trips.

## OpenCode config

```json
{
  "mcp": {
    "repo-memory": {
      "type": "local",
      "command": [
        "uv", "--directory", "/home/me/hncm/cml",
        "run", "cml", "mcp", "--index", "/tmp/fastapi-index"
      ],
      "enabled": true
    }
  }
}
```

Build the index first:

```bash
cd cml
uv run cml repo --repo /path/to/repo.git --out /tmp/fastapi-index --limit 80
uv run cml repo --repo /path/to/repo.git --out /tmp/fastapi-index --update
```

## Test

`tests/test_mcp.py` runs `initialize`, both list methods, and reads the
latest and a pinned version. A manual stdio run against the 80-file FastAPI
index returned one resource per file and exact bytes for `@0`.

## Limit

A read replays the prefix of that file's chain, so the latest version costs
the whole chain. Random access is bounded by one file, not the repo.
