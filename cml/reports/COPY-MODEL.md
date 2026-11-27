# Copy arm in front of the trained trunk

- run `codec-byte-model-code-756752`: `d256 l6 h8 seq256`, 4.86M params, trained on the code train split
- corpus `code` sha `1ccd9ffc7f68`, held-out 756,752 B, same bytes as the recorded run
- LZ window 65,536 B, min match 8, cap 256
- matches cost a token bit, a distance slot, the slot's raw bits and a length
  symbol; literals and copied bytes advance the same trunk cache, which resets
  every 256 output bytes to stay in the distribution the trunk trained on
- exact round-trip verified at 32,768 B; the 756,752 B decode was stopped after
  the payload, so exactness rests on the smaller stream

| arm | payload B | bpb |
|---|---:|---:|
| recorded byte model | 160,092 | 1.693 |
| LZ copy + trunk | 131,757 | **1.393** |
| delta | -28,335 | **-17.7%** |

## Read

- The copy arm saves **17.7% payload** against the same trained model on the
  same bytes: above the Stage 3 gate (5%) and at the top of the plan's
  expected `-0.10..-0.25`.
- The copy arm's payload also beats the 20M pointer codec's `1.488` on the same
  test (`SCALE.md`): a CPU preprocessor plus a 4.86M byte model beats a
  20M neural codec on the payload. Copy is the lever, not capacity.
- Two defects produced the earlier losses: copied bytes were coded as literals,
  and a rolling position table put the trunk out of distribution (6.94 bpb).
- The cost is the problem: 2,431 s to encode 756 KB, one pass on one GPU. A
  product path needs a faster literal step (batched windows or a smaller
  trunk), not a bigger window.
- This is not a concept win: it is a copy lever on the byte model, orthogonal
  to the negative Cortex result.

Reproduce with:

```bash
uv run cml copycodec --run codec-byte-model-code-756752 --bytes 756752 \
  --window 65536 --min-match 8 --cap 256
```
