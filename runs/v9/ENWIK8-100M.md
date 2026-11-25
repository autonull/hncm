# H-NCM v9: N10, enwik8 at 100 MB

The vocabulary is the union seed (64k, selected on the 10 MB stream)
plus the 90 MB heavy hitters (count >= 32, per-length top 8k), one
EM unigram on the 81 MB fit. The table is fitted on the first 90% of
the train and chosen on the held-out 10%; the charge is over 100 MB;
the val is 10 MB. The `ord2` column is the order-2 tail on the DP
path, charged; `projected` adds it. The codec record at this size is
not run: the 10 MB val encode is out of Python's throughput.

| dict | table | coverage | rate | model kB | charge | charged | sel charged | tail bytes | ord2 bpb | row bpb | ord2 projected | ppmd |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 16384 | 0.912 | 2.484 | 451.9 | 0.036 | 2.520 | 2.540 | 884920 | 5.043 | 5.781 | 2.475 | 1.834 |

The 10 MB targets were 2.94 (recorded); the 100 MB gate is 1.90. PPMd
on the same 10 MB val is measured by `codec._ppmd`.

