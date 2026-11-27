# Coverage: the pack lever, priced before training

- corpus `code` sha `1ccd9ffc7f68`, held-out 756,752 B
- packs induce from the frozen train split, min count 16, unit cap 16
- `static` is a fixed unit table: zero-order symbol entropy plus 8
  bits per literal. It is the cost without context, not a floor.
- `payload` and `shared` are the measured unit arm, the second one
  charging the pack once. `delta` is against the byte arm.

| units | pack B | coverage | literals | lit frac | mean unit | decisions | static | payload | shared | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 2,771 | 0.453 | 413,909 | 0.809 | 1.48 | 511,933 | 5.523 | 3.078 | 3.107 | +0.556 |
| 512 | 5,443 | 0.620 | 287,616 | 0.659 | 1.73 | 436,560 | 4.825 | 3.197 | 3.255 | +0.675 |
| 1024 | 10,942 | 0.721 | 211,412 | 0.555 | 1.99 | 380,708 | 4.365 | 3.038 | 3.154 | +0.516 |
| 2048 | 22,079 | 0.817 | 138,709 | 0.434 | 2.37 | 319,336 | 3.844 | 2.773 | 3.006 | +0.250 |

## Read

- byte arm: 2.522 bpb on the same bytes.
- coverage moves +0.364 from the smallest to the largest pack; the Stage 0 gate is +0.10.
- the measured payload gains 0.305 bpb across the ladder; the
  pack charge must stay below that gain.
- the unit arm is +0.250 bpb against the byte arm at its best: a static
  pack is not a win. The pack pays only through the pointer, which
  predicts a span from context (`reports/SCALE.md`: 1.518 shared).
