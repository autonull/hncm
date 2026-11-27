# Copy arm in front of the trained trunk

- run `codec-byte-model-code-756752-d448-l8` on `code` sha `1ccd9ffc7f68`, held-out 32,768 B
- LZ window 65,536 B, min match 8, cap 256
- parse: priced DP against literal bits
- matches cost a token bit, a distance slot, raw slot bits and a
  length symbol; literals and copied bytes advance the same trunk
  cache, so the decoder rebuilds the literal context

| arm | payload B | bpb |
|---|---:|---:|
| stream only | 5,880 | 1.436 |
| LZ copy + stream | 4,409 | 1.076 |
| recorded run | 119,691 | 1.265 |

- exact round-trip on the 32,768 B prefix (`--proof 32768`); a full
  756,752 B exact run was not attempted.
- copy changes the payload by +25.0% against the stream arm on the same bytes.

Took 489 s on one GPU.
