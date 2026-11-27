# `codec` — a lossless byte codec for code

Compress source files and restore them bit-for-bit. The codec is a trained
byte trunk (a causal transformer over bytes) driving a 32-bit arithmetic
coder, wrapped in a self-describing container with a SHA-256 check.

```bash
uv run cml compress NOTES.md            # -> NOTES.md.cmlc
uv run cml decompress NOTES.md.cmlc      # -> NOTES.md, digest checked
uv run cml delta NOTES.v1.md NOTES.v2.md    # -> NOTES.v2.md.cmld
uv run cml undelta NOTES.v2.md.cmld --ref NOTES.v1.md
```

## Numbers (pinned 756,752 B code test, sha `1ccd9ffc7f68`)

| arm | payload | ratio | status |
|---|---:|---:|---|
| 20M byte trunk (exact codec) | **1.266 bpb** | **6.3x** | shipped |
| same + priced LZ parse | 0.852 bpb | 9.4x | frontier, size instrument |
| exact priced codec, 32 KB prefix | 1.076 bpb | 7.4x | measured; 0.852 not yet reproduced |
| per 100 KB file, model installed once | 1.696 bpb | 5.9x | measured |
| 20M pointer/concept codec | 1.518 bpb | 5.3x | loses to bytes at equal params |
| **delta, 2% edits in 20 hunks** | **0.141 bpb** | **56.6x** | **shipped, exact** |
| delta, 2% scattered edits | 0.287 bpb | 27.8x | measured, exact |

## How it works

1. **Trunk.** A causal transformer over bytes: `d448`, 8 layers, 8 heads,
   256-token context, 19.53M parameters, trained 8,000 steps on 48 MB of
   repo code. One model, installed once; a second file does not retrain.
2. **Arithmetic coder.** The WNC 32-bit range coder codes each byte against
   the trunk's quantized distribution, so the stream is exact and needs no
   model at decode beyond the same weights.
3. **Chunked contexts.** Each 256-byte chunk restarts the context, which is
   how the trunk was trained; `FastByteCodec` prices a chunk in one forward,
   and `StaticByteCodec` is the exact codec and the round-trip proof.
4. **Copy arm (frontier).** An LZ parse finds long matches; a backward DP
   prices every candidate match against the trunk's own literal cost and
   keeps only matches that beat literals (`priced_parse`), with four repeat
   distances. It is measured at 0.852 bpb on the size instrument. The exact
   priced decoder (`cml copycodec --priced`) reproduces a 32 KB prefix at
   1.076 bpb, so the 0.852 rung is not promoted yet: the gap is the exact
   token and literal charge, not the parse.
5. **Delta arm (shipped).** Code a new file as LZ matches against the old
   file plus adaptive literals. The decoder holds the old file; it never
   searches. A 2% edit in 20 hunks gives 0.141 bpb (56.6x), and 2%
   scattered 8-byte edits give 0.287 bpb (27.8x). The parse and decode are
   CPU-only and exact.

## Container: `.cmlc` / `.cmld`

| bytes | field |
|---|---|
| 4 | magic `CMLC` (single file) or `CMLD` (delta) |
| 4 | header length, little-endian u32 |
| n | UTF-8 JSON: `run`, `params`, `seq`, `n`, `sha256`, `proof`, `git`; delta adds `ref_n`, `ref_sha`, `window`, `min_match`, `cap`, `depth` |
| rest | arithmetic-coded payload |

`decompress` checks the SHA-256 against the header and refuses a mismatch.
`undelta` also checks the reference SHA-256, so the wrong old file cannot
silently corrupt the output.

## Roadmap to 10x+

The product goal is **10x+ on real code**, reached on two tracks:

1. **Delta arm (v1, shipped).** LZ against the old file plus literals is
   exact and measured at **56.6x** on 2% edits in hunks and **27.8x** on
   scattered edits, well over the 10x product target. No training.
2. **Single file (v2).** The priced parse is measured at `0.852 (9.4x)` on
   the size instrument; the exact 32 KB prefix is `1.076`. The rungs from
   there are costed in the plan (`TODO8.md` §0.1): horizon 2-4 KB and a
   2 MB window (`~0.80`), two-pass literals (`~0.75`), code-aware context
   (`~0.71`), a 30M trunk (`~0.68`), then 1K context (`0.633`). Promote a
   rung only when its own decoder reproduces it.

## Known limits

- **Header ~180 B**: files below ~1 KB do not pay. Use it for files of
  kilobytes and up.
- **Model is 9.7 MB** (fp16): installed once per machine, never charged per
  file. Per-file accounting fails (`19,002` bpb at 1 KB).
- **Decode is sequential**: one forward per byte, minutes per MB. Encode is
  fast (one forward per 256-byte chunk, 290x the sequential path).
- **Delta needs the old file**: the reference is not stored, only its
  SHA-256. Keep the old version; the decoder refuses a mismatched one.
- **No provenance beyond the header.** The format records the training
  corpus and git sha; it does not embed a per-file audit trail.

## Reproduce

```bash
uv run cml codec --arm byte-model --d 448 --layers 8 --steps 8000 \
  --scale-cap 48000000 --bytes 756752 --proof 4096     # train + cache
uv run cml compress products/codec/README.md
uv run cml decompress products/codec/README.md.cmlc
uv run cml perfile-model --run codec-byte-model-code-756752-d448-l8
uv run cml copycodec --run codec-byte-model-code-756752 \
  --bytes 756752 --window 65536 --min-match 8 --cap 256
uv run cml copycodec --run codec-byte-model-code-756752-d448-l8 \
  --bytes 32768 --window 65536 --priced --proof 32768
uv run cml delta --bench --bytes 756752 --depth 128 --hunks 20
```

Reports: `cml/reports/COPY-MODEL.md`, `PER-FILE-MODEL.md`,
`cml/reports/COVERAGE.md`, `cml/reports/COPY.md`, `cml/reports/DELTA.md`.

Tests: `uv run pytest cml/tests` (the exact codec, the proof, the
container, the copy parse, the delta parse and the renderers).
