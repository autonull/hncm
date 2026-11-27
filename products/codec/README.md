# `codec` — a lossless byte codec for code

Compress source files and restore them bit-for-bit. The codec is a trained
byte trunk (a causal transformer over bytes) driving a 32-bit arithmetic
coder, wrapped in a self-describing container with a SHA-256 check.

```bash
uv run cml compress NOTES.md            # -> NOTES.md.cmlc
uv run cml decompress NOTES.md.cmlc      # -> NOTES.md, digest checked
```

## Numbers (pinned 756,752 B code test, sha `1ccd9ffc7f68`)

| arm | payload | ratio | status |
|---|---:|---:|---|
| 20M byte trunk (exact codec) | **1.266 bpb** | **6.3x** | shipped |
| same + priced LZ parse | 0.852 bpb | 9.4x | frontier, size instrument |
| per 100 KB file, model installed once | 1.696 bpb | 5.9x | measured |
| 20M pointer/concept codec | 1.518 bpb | 5.3x | loses to bytes at equal params |

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
   distances. It is measured at 0.852 bpb but is a *size instrument*, not
   yet a codec: its decoder must reproduce the encoder's rounding exactly.

## Container: `.cmlc`

| bytes | field |
|---|---|
| 4 | magic `CMLC` |
| 4 | header length, little-endian u32 |
| n | UTF-8 JSON: `run`, `params`, `seq`, `n`, `sha256`, `proof`, `git` |
| rest | arithmetic-coded payload |

`decompress` checks the SHA-256 against the header and refuses a mismatch.

## Roadmap to 10x+

The product goal is **10x+ on real code**, reached on two tracks:

1. **Delta arm (v1, no training).** Code a new version as LZ matches
   against the old file plus literals: a 2% edit copies ~98% of the file,
   so expect **50-300x on small edits**. Budget 1 h; ships first.
2. **Single file (v2).** The priced parse is measured at `0.852 (9.4x)` on
   the size instrument. The rungs from there are costed in the plan
   (`TODO8.md` §0.1): horizon 2-4 KB and a 2 MB window (`~0.80`), two-pass
   literals (`~0.75`), code-aware context (`~0.71`), a 30M trunk (`~0.68`),
   then 1K context (`0.633`). Each rung ships only when its decoder
   reproduces it; the product above stays exact throughout.

## Known limits

- **Header ~180 B**: files below ~1 KB do not pay. Use it for files of
  kilobytes and up.
- **Model is 9.7 MB** (fp16): installed once per machine, never charged per
  file. Per-file accounting fails (`19,002` bpb at 1 KB).
- **Decode is sequential**: one forward per byte, minutes per MB. Encode is
  fast (one forward per 256-byte chunk, 290x the sequential path).
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
```

Reports: `cml/reports/COPY-MODEL.md`, `PER-FILE-MODEL.md`,
`cml/reports/COVERAGE.md`, `cml/reports/COPY.md`.

Tests: `uv run pytest cml/tests` (the exact codec, the proof, the
container, the copy parse and the renderers).
