# `delta` — exact version compression

Compress a new version of a file as a set of copies from the old version
plus literals, then restore it **bit-for-bit**. CPU-only, no model, no
training.

```bash
uv run cml delta old.py new.py            # -> new.py.cmld
uv run cml undelta new.py.cmld --ref old.py
```

## What it is

Given `old` and `new`, the encoder finds LZ77 matches in `old + output`,
codes them with an adaptive arithmetic coder, and stores only the payload
that is not a copy. The decoder holds `old` and replays the same actions;
it never searches. Unchanged bytes cost almost nothing, so the ratio grows
as the edit shrinks.

The container is `CMLD`: magic, JSON header, coded payload. The header
records the length and SHA-256 of the new file and the SHA-256 of the old
file. `undelta` refuses a mismatched reference or output, so the wrong
version cannot silently corrupt the result.

## Numbers

Pinned 756,752 B code test, sha `1ccd9ffc7f68` (`cml/reports/DELTA.md`):

| arm | payload B | bpb | ratio | exact |
|---|---:|---:|---:|---|
| raw new | 756,752 | 8.000 | 1.0x | — |
| delta, 2% in 20 hunks | 13,380 | 0.141 | **56.6x** | yes |
| delta, scattered 2% | 27,176 | 0.287 | **27.8x** | yes |

The exact decoder reproduces every one of the 756,752 B. The payload is
computed from the parse, not a model estimate.

## Use it when

- you already have the previous version of the file;
- the file is larger than about 1 KB (the header is ~180 B);
- you need exact restoration, not a fuzzy patch.

Do not use it as a standalone compressor: the old file is the dictionary.
Do not delete the old version: the decoder needs it.

## Why this ships

It is exact and measured, and the alternative learned products did not
clear their first gates:

| track | gate | result |
|---|---|---|
| R retrieval | G0 | exact **0.000** on 1,648 test commits (`reports/EDIT-LM.md`) |
| E1 locator | GE1 | byte F1 **0.000** vs message-only **0.000** (`reports/EDIT-LOCATOR.md`) |
| C1 copy-SFT | GC1 | exact **0.000**, output **0.044x** (`reports/EDIT-COPY.md`) |

The delta codec asks a narrower question: the edit is already known, can
we store it compactly? It answers yes. Predicting the edit from a commit
message remains open; the Exact Edit Bench is the foundation for that work.

## Reproduce

```bash
cd cml
uv run cml delta --bench --bytes 756752 --depth 128 --hunks 20
uv run cml delta old.py new.py -o new.py.cmld
uv run cml undelta new.py.cmld --ref old.py -o restored.py
uv run pytest tests
```

Reports: `cml/reports/DELTA.md`. Code: `cml/codec.py`
(`DeltaArm`, `delta_parse`). Tests: `cml/tests/test_invariants.py`
(`test_delta_parse_and_roundtrip`).
