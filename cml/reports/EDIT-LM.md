# Exact Edit Bench: retrieval baseline (G0)

- arm **R retrieval**; deterministic; one job; no GPU
- split `test`; n = 1648
- manifest repos: encode/httpx, pallets/flask, psf/requests, pytest-dev/pytest, tiangolo/fastapi
- cache `/home/me/.cache/hncm/edit`
- manifest sha256 `02c8c56542235773`

## Task

Given `old`, the commit message, and the file path, predict `new`
exactly. Retrieval picks the nearest train `(old, new)` pair by
message Jaccard and path match, then replays its line changes on
`old` where the context aligns. A wrong result scores zero.

## Data

| split | repos | commits |
|---|---|---:|
| train | encode/httpx, pallets/flask, psf/requests | 5672 |
| val | pytest-dev/pytest | 5491 |
| test | tiangolo/fastapi | 1648 |

- filter: one text file, ≤ 65536 B, diff ≤ 20% (changed lines / new lines)
- split by repository, deduplicated by `(old, new)` within a split
- `encode/httpx` head `e347fb5e93b9`
- `pallets/flask` head `49943b5a9441`
- `psf/requests` head `3c7baaa05ab4`
- `pytest-dev/pytest` head `71e8b57bcd1c`
- `tiangolo/fastapi` head `05e6df7e906c`

## G0

| test | threshold | observed | result |
|---|---:|---:|---|
| retrieval exact | < 20% | 0.0% | pass |

## Results

| arm | n | exact | valid | edit distance | output bytes |
|---|---:|---:|---:|---:|---:|
| R retrieval | 1648 | 0.000 | 0.027 | 0.020 | 0.991 |

## Reading

Retrieval only sees the train repositories, so it cannot replay
the query repository's own history. Its exact rate is a lower
bound, not an upper bound. G0 passes: the benchmark is hard
enough to train against. Proceed to E1 (locator) and C1
(copy-SFT) on the same split. G0 is recorded.

## Next

| gate | arm | needs |
|---|---|---|
| GE1 | E1 locator | span F1 +20 points over message-only |
| GC1 | C1 copy-SFT | +10 exact or +20 test pass, ≤5% output bytes |
| GE2 | E2 compiler | +10 exact or +20 test pass, ≤20% output bytes |

Val is `pytest-dev/pytest`; test is the last repository by name.
Do not tune on test. One job at a time.
