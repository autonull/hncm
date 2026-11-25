# H-NCM v9: the coverage probe (N8, N9)

The dictionary is fitted on the train and the val is tiled with the unigram
cover (a lower bound on the first-order DP's coverage and a higher bound on
its covered rate). Two shared tails are fitted on the train's uncovered
spans and chosen on the held-out 10% of the train: the rank-r
factorization of the byte table, and the pruned order-2 counts with the
order-1 backoff. `predicted = cov * covered + (1-cov) * tail +
8 * model / stream`; `predicted ord2` repeats it with the count tail. The
concept table's charge is not included, so both totals are lower bounds.

| corpus | dict | coverage | covered bpb | tail bpb | ord2 bpb | ord2 top | tail bytes | r | model kB | charge | predicted | predicted ord2 | calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enwik8_10m | 8192 | 0.699 | 2.162 | 5.602 | 5.236 | 1024 | 300782 | 32 | 124.5 | 0.100 | 3.296 | 3.226 | 49671 |
| enwik8_10m | 16384 | 0.766 | 2.146 | 5.758 | 5.464 | 1024 | 234287 | 32 | 224.3 | 0.179 | 3.171 | 3.139 | 44691 |
| enwik8_10m | 32768 | 0.815 | 2.129 | 5.922 | 5.693 | 1024 | 184655 | 32 | 433.2 | 0.347 | 3.176 | 3.167 | 40287 |

## Gates

| gate | corpus | value | status |
|---|---|---|---|
| C1 | enwik8_10m | dict=16384 cov=0.766 pred=3.171 model=224.3 kB | **pass** |
| C2 | enwik8_10m | ord2=5.464 svd=5.758 pred2=3.139 top=1024 | **pass** |

C1 is the probe's own gate: the best budget covers more than half of the
val. The decision it feeds: if coverage rises with the stream, the
dictionary and the concept LM are the story and N1b runs them at 10 MB; if
coverage stalls, the tail is the story and N2/M5 (the learned tail) runs
first.

