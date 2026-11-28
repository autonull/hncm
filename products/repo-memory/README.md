# `repo-memory` — exact, searchable repository history

One installable index of a repository's files and versions. Byte-exact,
SHA-verified, per-file random access, and exact span search with
`path:line` provenance. CPU-only; no model, no training.

```bash
uv run cml repo --repo /path/to/repo.git --out repo-index --limit 80
uv run cml repo --repo /path/to/repo.git --out repo-archive --global
uv run cml repo --index repo-index --search "def routing"
```

## What it is

Each file's history is an exact delta chain (`CMLH`): the first version is
raw, every later version is a delta against the previous one. The chain is
stored gzip-wrapped when that is smaller, so the first version also pays
once. Decode replays the prefix and checks every SHA-256. The index adds a
manifest with each file's path, version count, raw/chain bytes, and HEAD
blob SHA.

`--search` decodes the last version of every indexed file and returns exact
`path:line` hits. It is exact text search over the stored memory, not a
fuzzy or vector search.

## Numbers

Sample: `tiangolo/fastapi` bare repo, 80 path-sorted files, 1,320 versions.

| unit | bytes | ratio |
|---|---:|---:|
| raw versions | 9,839,584 | 1.0x |
| per-file gzip | 3,413,837 | 2.9x |
| **repo index** | **476,080** | **20.7x** |
| one global gzip | 384,624 | 25.6x |

The index is **7.2x smaller than per-file gzip** and keeps per-file,
per-version access. A single gzip over the whole concatenation is smaller
still, but it has no file/version boundary, no digest, and no `path:line`
search. The index is the access layer; global gzip is a different unit.

`--global` stores all versions in one exact chain: **390,024 B = 25.2x**,
8.8x smaller than per-file gzip. It is the archive mode; random access is
a prefix replay, so use the per-file mode for a live index.

Build and verify: 34 s on CPU for the 80-file per-file sample; 48 s for
the global chain.

## Use it when

- an agent or reviewer needs the exact current file and its history;
- you want exact search with provenance instead of fuzzy retrieval;
- you need install/detach/delete at file granularity without retraining.

Do not use it as a single-file compressor; the first version is raw. Random
access is a prefix replay within that file's chain.

## Reproduce

```bash
cd cml
uv run cml repo --repo ~/.cache/hncm/edit/repos/tiangolo__fastapi.git \
  --out /tmp/fastapi-index --limit 80 \
  --report reports/REPO-INDEX.md --json reports/repo-index.json
uv run cml repo --index /tmp/fastapi-index --search "FastAPI"
uv run pytest tests/test_repo_memory.py
```
