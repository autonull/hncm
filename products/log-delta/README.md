# `log-delta` — exact CI log history

Store a sequence of log runs at **27.3x** on a 5-run sample, byte-exact,
1.8x better than a single gzip over the same bytes. CPU-only.

```bash
uv run cml logdelta --dir runs/logs --out logs.cmlh
uv run cml unchain logs.cmlh --out-dir out/ --index 4
```

## What it is

The first run is stored raw; every later run is a delta against the
previous one. The chain is gzip-wrapped, so the seed also pays once.
Decode replays the chain and checks every SHA-256.

## Numbers

Synthetic near-identical pytest-style logs, 5 runs, 504,465 B:

| unit | bytes | ratio |
|---|---:|---:|
| raw logs | 504,465 | 1.0x |
| one gzip | 32,827 | 15.4x |
| **log chain** | **18,445** | **27.3x** |

The chain is 1.8x smaller than gzip and keeps run granularity. Encode,
decode, and digest check: 0.7 s CPU.

## Use it when

- CI keeps every run but only a few lines change;
- you need the exact run that failed, not a summary;
- storage or transfer of logs matters.

Do not use it on unrelated logs; the delta arm needs a shared history.
