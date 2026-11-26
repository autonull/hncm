# Install transfer (P0b, no oracle)

- corpus `code` sha `1ccd9ffc7f68`, train 7,243,247 B, test 756,752 B
- package `networkx`: 54 held-out files split at 27; A 400,094 B, B 356,442 B
- 8000 steps, 256 active units, ngram-joint, seq 256

`(b)` is induced on one half and encodes the other, so the pack never
touches the bytes in its payload.  `(a)` is the train pack; `(c)` is
all literals.

| direction | pack | units | pack B | payload B | payload bpb | coverage | proof |
|---|---|---:|---:|---:|---:|---:|:---:|
| A->B | (a) train | 256 | 2771 | 69449 | 1.559 | 0.456 | OK |
| A->B | (b) half | 256 | 2586 | 81906 | 1.838 | 0.540 | OK |
| A->B | (c) none | 0 | 12 | 237473 | 5.330 | 0.000 | OK |
| B->A | (a) train | 256 | 2771 | 90560 | 1.811 | 0.451 | OK |
| B->A | (b) half | 256 | 2404 | 100999 | 2.020 | 0.522 | OK |
| B->A | (c) none | 0 | 12 | 269815 | 5.395 | 0.000 | OK |

| direction | `(b)-(a)` | `(c)-(a)` | gate |
|---|---:|---:|---|
| A->B | +0.280 | +3.771 | FAIL |
| B->A | +0.209 | +3.584 | FAIL |

Bars: `|(b)-(a)| <= 0.15` and `(c)-(a) >= 0.20`, in both directions.
Overall: **FAIL**.
