# Install delta (P0b)

- corpus `code` sha `1ccd9ffc7f68`, train 7,243,247 B, test 756,752 B
- package `networkx`: 54 held-out files / 756,536 B
- 3000 steps, 256 active units, ngram-joint, seq 256

| pack | units | pack B | payload B | payload bpb | coverage | proof |
|---|---:|---:|---:|---:|---:|:---:|
| (a) train | 256 | 2771 | 183352 | 1.939 | 0.453 | OK |
| (b) package | 256 | 2555 | 191700 | 2.027 | 0.544 | OK |
| (c) none | 0 | 12 | 463148 | 4.898 | 0.000 | OK |

- `(b) - (a)` = **+0.088 bpb** (bar `|d| <= 0.15`): PASS
- `(c) - (a)` = **+2.959 bpb** (bar `>= +0.20`): PASS

The install is causal when the same trained model reads a pack induced
on the target package nearly as well as its train pack, and both beat
all literals.  `(b)` is an oracle pack (induced on the encoded bytes);
it measures transfer of the encoder/head, not held-out pack induction.

| gate | result |
|---|---|
| package pack within 0.15 bpb of train | PASS |
| train pack beats no pack by 0.20 bpb | PASS |
