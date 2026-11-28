# `log-delta` — exact CI log history

Store a sequence of log runs byte-exact with version boundaries and SHA-256.
The synthetic 5-run sample reached **27.3x** and beat gzip, but real pytest
logs do not: gzip over the concatenation is 2.6x–3.1x smaller. Use this
only when exact per-run replay matters, not as a compression product.

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

The synthetic chain is 1.8x smaller than gzip. Real pytest logs reverse
that result:

| real set | raw B | chain B | chain / raw | gzip / raw | gzip / chain |
|---|---:|---:|---:|---:|---:|
| near-identical, 30 runs | 32,190 | 1,773 | 18.2x | 46.6x | 0.39 |
| changed, 16 runs | 8,548 | 3,268 | 2.6x | 8.1x | 0.32 |

The ratio gates pass, but one gzip over the concatenated logs is 2.6x and
3.1x smaller. The first run is raw and every version pays a payload
header; gzip shares one stream. See `cml/reports/LOG-DELTA-REAL.md`.

## Use it when

- you need the exact run that failed, not a summary;
- you need per-run SHA-256 and random access;
- the logs are long and change slowly enough that the synthetic shape
  holds; check the gzip baseline on your own logs first.

Do not use it on unrelated logs, and do not claim compression over gzip
without measuring the concatenation.
