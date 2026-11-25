# H-NCM v8: N1b, enwik8 at 10 MB

The model is fitted on the 9 MB train (counts cached), the 1 MB val is tiled
with the first-order DP (the fit uses the chunked DP), the table is chosen on
the held-out 10% of the train (`sel`), and the model is charged over the
10 MB stream. `tail` is the hard split; `concept` is the one normalized
model (the H-Net rows are the tail's context).

| dict | table | sel | concept rate | model kB | concept charged | tail charged |
|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 4096 | 3.029 | 2.991 | 133.8 | 3.098 | 3.134 |
| 8192 | 8192 | 2.981 | 2.927 | 165.0 | 3.059 | 3.122 |
| 8192 | 16384 | 2.948 | 2.857 | 222.5 | 3.035 | 3.126 |
| 16384 | 4096 | 3.001 | 2.873 | 248.8 | 3.072 | 3.094 |

The codecs on the same 1 MB val: PPMd **2.160**, brotli 2.344, bz2 2.393,
lzma 2.427, zstd 2.511, gzip 2.969, zlib 2.998.

The selection rule costs nothing here: `sel` picks 8192/16384 (2.948) and the
val agrees (3.035). The concept arm beats the hard split by 0.09 bpb at half
the model, so the one normalized model is the better architecture.

## The finding

At 10 MB the concept arm is **3.035** charged, **+0.875** behind PPMd. The
covered part is not the problem: at the 16k dict the coverage is 0.756 and the
covered tokens are ~2.0 per covered byte (below PPMd's 2.16 overall). The
tail is: 24% of the bytes at 5.6 bpb (the SVD cold start). The real context
fixes that part: order-2 is 3.83 vs 4.48 on the 1 MB tail, and the order-3
ceiling is 1.62. With the tail at ~3.0 the 16k/16k total is ~2.6; at 100 MB
the charge falls to 0.02, the coverage rises, and the same tail gives ~1.8-2.0
against PPMd's ~1.8. The 10 MB target of 2.50 is not reachable without the
context; the 100 MB is the scale claim, not the 10 MB.
