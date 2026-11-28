# `memory` — exact history compression

Store a file's full git history at **40x** on real code, byte-exact.
CPU-only, no model, no training.

```bash
uv run cml chain --repo /path/to/repo.git --path fastapi/routing.py \
  --out routing.cmlh
uv run cml unchain routing.cmlh --out-dir out/       # last version
uv run cml unchain routing.cmlh --out-dir out/ --index 12
```

## What it is

The first version is stored raw. Every later version is coded as a delta
against the previous one with the shipped delta arm. `unchain` replays the
prefix and checks every SHA-256, so a wrong or truncated chain is refused.
This is exact version storage, not a fuzzy patch or a summary.

## Numbers

Five files, 1,823 versions, **90,124,650 B → 2,255,519 B = 40.0x**;
gzip of the same concatenation is `6.1x`, so the chain is **6.5x better
than gzip**. Encode + full decode + digest check: about 2 minutes on CPU.

| file | versions | raw B | chain B | ratio |
|---|---:|---:|---:|---:|
| `fastapi/dependencies/utils.py` | 183 | 6,040,346 | 175,919 | 34.3x |
| `fastapi/routing.py` | 194 | 19,116,395 | 503,005 | 38.0x |
| `requests/models.py` | 703 | 16,984,381 | 425,780 | 39.9x |
| `src/_pytest/fixtures.py` | 368 | 24,779,717 | 609,550 | 40.7x |
| `src/_pytest/python.py` | 375 | 23,203,811 | 541,265 | 42.9x |

Details and the gzip comparison: `cml/reports/REPO-MEMORY.md`.

## Use it when

- an agent or reviewer needs earlier versions of a file;
- you want exact history without storing every blob;
- the file has a text history and less than a few hundred versions.

Do not use it as a single-file compressor; the first version is raw.
Random access is a prefix replay, so the latest version costs the chain.

## Reproduce

```bash
cd cml
uv run cml chain --repo ~/.cache/hncm/edit/repos/psf__requests.git \
  --path requests/models.py --out models.cmlh \
  --json reports/repomemory/models.json
uv run cml unchain models.cmlh --out-dir out/
uv run pytest tests/test_memory.py
```
