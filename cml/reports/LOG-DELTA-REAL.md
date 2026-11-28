# Log delta on real pytest logs

Status: 2026-11-28. Gate **L1** from `TODO10.md`. The raw-ratio gate
passes, but gzip wins the fair baseline. Do not ship logdelta as a
compression product.

## Setup

Real pytest output from this repository, captured with
`uv run pytest -v -p no:cacheprovider ... > runNN.log`.

| set | runs | how the logs differ |
|---|---:|---|
| near-identical | 30 | same two test files, `-v`, repeated |
| changed | 16 | eight file sets, each once `-v` and once `-q` |

The near set varies only timings and cache/collection noise. The changed
set varies selected files and verbosity, so test names, order, and summary
lines differ.

## Result

| set | raw B | chain B | chain / raw | gzip B | gzip / raw | gzip / chain |
|---|---:|---:|---:|---:|---:|---:|
| near-identical | 32,190 | 1,773 | **18.2x** | 691 | **46.6x** | 0.39 |
| changed | 8,548 | 3,268 | **2.6x** | 1,053 | **8.1x** | 0.32 |

Both raw-ratio gates pass (≥10x near, ≥2x changed). Both fair-baseline
comparisons fail: a single gzip over the concatenated logs is **2.6x** and
**3.1x** smaller than the chain. The first log is stored raw, and each
version pays a payload header; gzip shares one stream across all versions.

## Verdict

- Record L1 as a **pass on the stated ratio** and a **stop on the product
  headline**. gzip is the fair unit for a log stream.
- Keep the chain only for exact versioned replay, per-version SHA-256, and
  random access; do not claim compression over gzip.
- No delta-arm tuning: the gap is structural for short, highly similar
  logs. The synthetic 27.3x sample did not predict the real baseline.

## Artifacts

- `logs-near.cmlh`: 30 exact versions, 1,773 B.
- `logs-changed.cmlh`: 16 exact versions, 3,268 B.
- `log-delta-real.json`: both stat blocks and the commands.

Reproduce the last version from each chain:

```bash
cd cml
uv run cml unchain reports/logs-near.cmlh --out-dir /tmp/logs-near --index -1
uv run cml unchain reports/logs-changed.cmlh --out-dir /tmp/logs-changed --index -1
```
