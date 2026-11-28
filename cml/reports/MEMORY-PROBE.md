# A0.0: exact-span oracle on seq 256

- run `codec-byte-model-code-756752-d448-l8` on `code` sha `7ae9de7b4a15`, held-out 756,752 B
- base: 19,530,112 params, seq 256, 1.265 bpb stream-only
- pack ids are free; pack bytes are charged once per file
- literal arm: the frozen trunk's own static CDF
- reference: LZ copy arm `0.852` bpb (TODO8, 756,752 B)
- both sides are size instruments; the exact copy arm adds ~5%

| pack | coverage | payload bpb | pack B | pack bpb | total bpb | delta | calls/B | speedup | round-trip | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 256 | 0.410 | 1.022 | 2,895 | 0.031 | 1.052 | +0.200 | 0.590 | 1.69x | OK | fail |
| 1024 | 0.686 | 0.684 | 11,333 | 0.120 | 0.804 | -0.048 | 0.314 | 3.19x | OK | pass |
| 4096 | 0.869 | 0.364 | 45,551 | 0.482 | 0.845 | -0.007 | 0.131 | 7.64x | OK | pass |

## Read

- best pack **1024**: total 0.804 bpb vs copy 0.852 (-0.048).
- coverage 0.686; one model call per literal (0.314 calls/B, 3.19x fewer calls than the stream arm).
- provenance: every emitted byte is a tagged pack span or a neural literal; parse round-trip OK.

**A0.0 pass.** The oracle beats the LZ copy arm on bits, so the memory integration holds on this base and is worth the 2K base (A0.1). This is an upper bound with free ids, not a trained model.

Took 15 s on one GPU.
