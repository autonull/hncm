# `patch-guard` — apply only exact patches

Apply a unified diff and refuse the result unless it equals the expected
file byte-for-byte. A guard, not a compressor.

```bash
uv run cml patch --base old.py --diff change.patch --expect new.py
```

Exit 0 means the applied bytes equal `new.py` exactly. A mismatch or a
failed apply is a hard error. Use it before an agent writes a patch, in CI,
or as the final step of an edit pipeline.

## What it is

`patch` applies the diff to a temporary copy of `base`; the guard compares
the bytes with `--expect` and never touches the original. `--strip N`
controls the path prefix, as in `patch -pN`.

## Use it when

- an agent proposes a diff and you need an exactness check;
- a patch came from a model or a fuzzy merge;
- the repo must not accept a non-exact result.
