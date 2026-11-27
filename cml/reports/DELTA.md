# Delta arm: old file as dictionary

- pair `code sha 1ccd9ffc7f68`, 756,752 B old, 756,752 B new, 2.0% replacement edits in 20 spans, seed 0
- window 1,048,576 B, min match 8, cap 256, depth 128
- the decoder holds the old file and copies from it; it never searches

| arm | payload B | bpb | ratio |
|---|---:|---:|---:|
| raw new | 756,752 | 8.000 | 1.0x |
| delta | 13,380 | 0.141 | 56.6x |

## Read

- 4,530 matches copy 99.9% of the new file; the rest is literals.
- the exact decoder reproduces all 756,752 B in 1.96 s.
- 2% replacement edits leave long runs intact, so the delta arm turns them into a handful of literal spans.
