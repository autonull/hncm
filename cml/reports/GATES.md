# CML gates

| run | train MB | params | steps | payload | bigram | gates |
|---|---:|---:|---:|---:|---:|---|
| `p0-adapter-75cb215f-s0` | 9.0 | 21.9M | 0 | 0.997 | — | P0a:✓ P0b:✓ P0c:✓ P0d:— P0e:✓ P0f:✓ P0g:✓ |
| `p1-check-8m-9237d73f-s0` | 8.0 | 20.5M | 6000 | 1.992 | 2.709 | P1a:✓ P1b:✓ |
| `p1-class-10m-289cc414-s0` | 9.0 | 39.9M | 2000 | 2.096 | 2.667 | P1a:✓ P1b:✓ P1c:— P1d:— P1e:— |
| `p1-class-10m-3084fd95-s0` | 9.0 | 20.5M | 6000 | 1.920 | 2.667 | P1a:✓ P1b:✓ P1c:— P1d:— P1e:— |
| `p1-class-10m-6fcc581a-s0` | 9.0 | 20.5M | 2000 | 2.149 | 2.667 | P1a:✓ P1b:✓ P1c:— P1d:— P1e:— |
| `p1-class-10m-7005ccd6-s1` | 9.0 | 20.5M | 6000 | 1.912 | 2.667 | P1a:✓ P1b:✓ P1c:— P1d:— P1e:— |
| `p1-class-10m-abaa4d7e-s0` | 9.0 | 8.9M | 6000 | 1.962 | 2.667 | P1a:✓ P1b:✓ P1c:— P1d:— P1e:— |
| `p1-class-10m-d71928a1-s0` | 9.0 | 8.9M | 2000 | 2.246 | 2.667 | P1a:✓ P1b:✓ P1c:— P1d:— P1e:— |
| `p1-class-10m-e41685bb-s0` | 9.0 | 39.9M | 6000 | 1.926 | 2.667 | P1a:✓ P1b:✓ P1c:✓ P1d:— P1e:— |
| `p1-class-10m-f7ae3fd7-s2` | 9.0 | 20.5M | 6000 | 1.904 | 2.667 | P1a:✓ P1b:✓ P1c:— P1d:— P1e:— |

## Verdict

The seed-0 P1 payload is **1.920 bpb** at 20.5M parameters and 6000 steps (`p1-class-10m-3084fd95-s0`); seeds 0-2 at that point are **1.912 ± 0.008**. The same-slice count bigram is 2.667; the v9 reference on the 10 MB block is 2.696 (2.729 is the 10 MB-prefix rate).

P0: base 1.827 → adapter 0.997 bpb; rate saved +0.830, pack charge 0.072, net **+0.758**; accept 79.4%, coverage 79.6%, pack 90.1 kB (`p0-adapter-75cb215f-s0`).

The second split is the `p1-check-8m` row: train 8 MB and test the 8-9 MB slice, so the win is not one easy slice.

Provenance: lexicon 300.9 kB `4032d5e3d1f4`; corpus `10c136c4dc9d`.

The 10 MB parse is the 90-100 MB enwik8 val block; the lexicon is the v9 shared pack. The first 9 MB is the train and the last 1 MB is the test.

P1a ≤2.60, P1b ≤2.293 and P1c ≤2.29 at 60M are measured; P1d and P1e need P0 and are pending.

The 40M point is 0.006 bpb above the 20.5M point; the curve is flat, so the 10 MB pilot is data-bound and P1c passes by saturation, not by a capacity slope.
