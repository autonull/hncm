# `repo-delta` — exact delta between two commits

Compress one file from commit `a` to commit `b` with the shipped delta
arm, byte-exact. Useful when you need a patch, not the whole history.

```bash
uv run cml repo-delta --repo /path/to/repo.git --path README.md \
  --before <sha-a> --after <sha-b> --out readme.delta
```

## Numbers

`tiangolo/fastapi` `README.md`, two commits: 22,830 B → **266 B (85.8x)**,
decode verified byte-exact.

## What it is

Both blobs are read from git; `delta_parse` builds the copy/literal
program; the shipped `DeltaArm` entropy-codes it. The command decodes its
own payload before writing, so a non-exact result is refused.

## Use it when

- you need a reviewable patch between two commits;
- the full history index is overkill;
- the result must be exact and verifiable.
