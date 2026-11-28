# Repo memory: exact history compression

Five real files, 1,823 versions, **90,124,650 B → 2,255,519 B = 40.0x**,
byte-exact. CPU-only, no training. gzip of the same concatenation is
`6.1x`, so the chain is **6.5x better than gzip**.

Each chain stores the first version raw and every later version as a delta
against the previous one. Decode replays the prefix and checks every
SHA-256. This is version storage, not a fuzzy patch.

## Results

| file | versions | raw B | chain B | ratio | gzip B | gzip ratio |
|---|---:|---:|---:|---:|---:|---:|
| `fastapi/dependencies/utils.py` | 183 | 6,040,346 | 175,919 | 34.3x | 752,115 | 8.0x |
| `fastapi/routing.py` | 194 | 19,116,395 | 503,005 | 38.0x | 1,594,051 | 12.0x |
| `requests/models.py` | 703 | 16,984,381 | 425,780 | 39.9x | 702,670 | 24.2x |
| `src/_pytest/fixtures.py` | 368 | 24,779,717 | 609,550 | 40.7x | 6,107,033 | 4.1x |
| `src/_pytest/python.py` | 375 | 23,203,811 | 541,265 | 42.9x | 5,652,677 | 4.1x |
| **total** | **1,823** | **90,124,650** | **2,255,519** | **40.0x** | **14,808,546** | **6.1x** |

Encode + full decode + digest check: about 2 minutes total on one CPU.
Per-file JSON: `cml/reports/repomemory/*.json`.

## Container `CMLH`

| bytes | field |
|---|---|
| 4 | magic `CMLH` |
| 4 | header length, little-endian u32 |
| n | JSON: `path`, `commits`, `sizes`, `shas`, window/min-match/cap/depth |
| rest | u32-length-prefixed payloads: raw v0, then one delta per version |

`unchain` replays the prefix and verifies every size and SHA-256. `--index
i` writes an earlier version; a wrong or truncated chain is refused.

## Why this matters

- **Agent context**: keep a file's full history at 2.3% of its raw size and
  materialize any version exactly when the agent asks.
- **Review and audit**: version history is reproducible from the chain plus
  the commit list; no lossy summary.
- **Cross-repo**: same code, no model, no GPU. A chain is seconds to build.

## Limits

- One path per chain; a repo is a set of chains, not one container yet.
- Random access is a prefix replay; the last version costs the whole chain.
- The first version is raw; very short files pay the header.
- The chain needs the same commit list for provenance. It does not replace
  git, which stores the objects it reads.

## Reproduce

```bash
cd cml
uv run cml chain --repo /path/to/repo.git --path fastapi/routing.py \
  --out routing.cmlh --json reports/repomemory/routing.json
uv run cml unchain routing.cmlh --out-dir out/
uv run pytest tests/test_memory.py
```

Code: `cml/memory.py`. Tests: `cml/tests/test_memory.py`. The delta arm is
documented in `products/delta/README.md` and `cml/reports/DELTA.md`.
