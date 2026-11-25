# H-NCM v8: N1a, the coverage probe

The dictionary is fitted on the 9 MB train and the 1 MB val is tiled with
the unigram cover (a lower bound on the first-order DP's coverage). The
shared tail is the rank-r factorization of the byte-context table, fitted on
the train tail, r chosen on the held-out 10% of the train. `predicted =
cov * covered + (1-cov) * tail + 8 * model / stream`; the concept table is
not in the charge, so the total is a lower bound.

| corpus | vocab | dict | coverage | covered bpb | tail bpb | tail bytes | r | model kB | charge | predicted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enwik8_10m | substr | 2048 | 0.545 | 1.997 | 5.321 | 455458 | 32 | 57.8 | 0.046 | 3.557 |
| enwik8_10m | substr | 4096 | 0.629 | 2.030 | 5.453 | 371195 | 32 | 83.8 | 0.067 | 3.367 |
| enwik8_10m | substr | 8192 | 0.698 | 2.032 | 5.606 | 302283 | 32 | 138.1 | 0.111 | 3.223 |
| enwik8_10m | substr | 16384 | 0.756 | 2.039 | 5.717 | 244471 | 32 | 251.4 | 0.201 | 3.139 |
| enwik8_10m | onsets | 2048 | 0.414 | 1.793 | 4.740 | 585879 | 64 | 87.0 | 0.070 | 3.589 |
| enwik8_10m | onsets | 4096 | 0.477 | 1.845 | 4.798 | 522735 | 64 | 109.7 | 0.088 | 3.477 |
| enwik8_10m | onsets | 8192 | 0.536 | 1.889 | 4.840 | 463648 | 64 | 156.3 | 0.125 | 3.382 |

## The defect the probe exposes

The tail is 24-46% of the val at 10 MB and 4.7-5.7 bpb: the SVD restarts at
the start symbol (256) for every span, but the decoder holds the *real*
preceding bytes. Fitted on the train tail of enwik8_1m, the real context is:

| context | fitted bpb | val ceiling bpb |
|---|---:|---:|
| order 1 (the SVD, cold start 4.48) | 4.17 | 3.80 |
| order 2 | **3.83** | 2.75 |
| order 3 | 3.73 | 1.62 |
| order 4 | 3.71 | 0.93 |

The order-2 table is 23,851 entries / 95 kB and pays at 10 MB (0.088 bpb of
charge for ~0.35 bpb of tail gain) but not at 1 MB (0.877). The SVD was a
1 MB size choice; the tail's context is the scale lever.

## The NCM at the same budget loses to the counts

A 15.0 kB GRU (the previous byte, the concept class, the position) trained
with the GPU for 10 epochs reaches 4.608 bpb on the 1 MB tail
against the rank-16 SVD's 4.483: the learned tail learns the
bigram from scratch and is capacity-starved; the SVD is a more efficient
parameterization of the same counts. The count order-1/2 tables are the
right tail model at scale, not a 15 kB GRU.

## The codec at 1 MB enwik8

The complete model is the *concept* arm (one normalized model: every token
is coded by the row of the previous token, so the tail's bytes get their
context from the H-Net rows), not the hard split. At dict 1024/table 2048:

| arm | rate | model kB | charged |
|---|---:|---:|---:|
| concept | 3.452 | 27.3 | **3.671** |
| tail (hard split) | 3.457 | 60.2 | 3.939 |
| PPMd on the same val | 2.476 | in-stream | 2.476 |

So at 1 MB the model is 1.2 bpb behind PPMd: the H-Net rows are order 1,
and the covered token unigram is 2.0 bpb. The gap is the context order, not
the representation. Next: the concept arm at 10 MB with the order-2 tail and
the first-order concept table, in the codec; the chunked DP is required
before that run.
