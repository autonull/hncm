# Exact Edit Bench: E1 locator (GE1)

- arm **E1 locator**; split `val`; n = 500
- seed `0`; steps `1200`; pretrain `synthetic`
- manifest sha256 `02c8c56542235773`
- model `d256 l6 h8`; 5,176,065 params; window 1024 B

## Task

Given `old` and the commit message, predict the changed byte spans
in `old`. The gate metric is byte-overlap F1 over changed bytes,
the natural segmentation score. Span-match F1 (IoU ≥ 0.5) is a
stricter secondary label. Insertions (`old_len == 0`) are excluded
because a byte classifier cannot localize an empty span; the writer
and Track C handle them.

## Gate

| test | bar | locator | message-only | delta | result |
|---|---:|---:|---:|---:|---|
| byte F1 | +20 points | 0.000 | 0.000 | +0.000 | fail |

## Results

| arm | byte F1 | byte P | byte R | match F1 | location exact |
|---|---:|---:|---:|---:|---:|
| E1 locator | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| message-only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Data

| split | repos | files |
|---|---|---:|
| train | encode/httpx, pallets/flask, psf/requests | 5672 |
| val | pytest-dev/pytest | 5491 |
| test | tiangolo/fastapi | 1648 |

The message-only baseline is the same model and schedule with the
file embedding zeroed; it sees the message and positions but not
the old bytes. The delta isolates the value of file content.

Usable files with a non-empty changed span: train 3723, val/test 500.

## Reading

GE1 fails: file content does not add 20 byte-F1 points. The
next lever is the locator input, not a longer schedule: try
message-to-file cross-attention, line-level labels, or a
retrieval-located prior. Record the failure; do not widen the
benchmark.

## Next

E1b locator: change the input, not the budget. Try line-level
labels or explicit message-to-file cross-attention; rerun GE1
before E2. C1 copy-SFT is independent and can run in parallel.
Val is `pytest-dev/pytest`; test is `tiangolo/fastapi`. Do not
tune on test.
