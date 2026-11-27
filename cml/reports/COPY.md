# Copy: the LZ77 preprocessor, charged

- corpus `code` sha `1ccd9ffc7f68`, held-out 756,752 B
- window 4,096 B, min match 4, match cap 32, depth 8
- charge: token bit, distance slot, raw slot bits, length symbol;
  literals escape to the order-2 byte arm.

| token | value |
|---|---:|
| tokens | 167,274 |
| matches | 65,038 |
| copy fraction | 0.865 |
| mean match B | 10.06 |
| copy payload | 202,561 B (2.141 bpb) |
| byte payload | 238,574 B (2.522 bpb) |
| saving | 15.1% |

## Read

- copy saves 15.1% at this window; the Stage 0 gate is 5%. The lever holds.
- The preprocessor is exact and costs one parse pass; a Stage 3 run
  can reuse it in front of the trained trunk without new training.

Parse took 0.6 s; codec took 20.4 s.
