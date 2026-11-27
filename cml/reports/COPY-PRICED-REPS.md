# Copy arm in front of the trained trunk

- run `codec-byte-model-code-756752-d448-l8` on `code` sha `1ccd9ffc7f68`, held-out 32,768 B
- LZ window 65,536 B, min match 8, cap 256
- parse: priced DP against literal bits
- distance code: four repeat distances
- a repeat distance pays a 2-bit index; a new distance pays a
  token bit, a slot and the slot's raw bits; a length symbol follows;
  literals and copied bytes advance the same trunk cache, so the
  decoder rebuilds the literal context

| arm | payload B | bpb |
|---|---:|---:|
| stream only | 5,880 | 1.436 |
| priced parse, size instrument | 4,188 | 1.022 |
| LZ copy + stream | 4,401 | 1.074 |
| recorded run | 119,691 | 1.265 |

- exact round-trip on 4,096 of 32,768 B; copy changes the payload by +25.2% against the stream arm on the same bytes.
- exact vs its own size instrument: +0.052 bpb.

Took 422 s on one GPU.
