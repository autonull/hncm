# H-NCM v7: one model, K files (the amortization)

The val is split into K contiguous shards. Every codec compresses each shard on its own; the concept LM tiles each shard with the same dictionary and table. `archive` charges the model over train + the K shards (the v7 convention); `shared` over the K shards only (a deploy); `per_file` makes each shard pay it alone (what a generic codec cannot avoid).

## shakespeare (dict 1024, table 2048, model 23.1 kB)

| K | shard B | zlib | gzip | bz2 | lzma | zstd | brotli | ppmd | rate | archive | shared | per-file |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 111536 | 3.218 | 3.199 | 2.635 | 2.947 | 2.984 | 2.825 | 2.504 | 2.921 | 3.087 | 4.575 | 4.575 |
| 2 | 55768 | 3.300 | 3.283 | 2.776 | 3.091 | 3.120 | 2.905 | 2.660 | 2.921 | 3.087 | 4.575 | 6.229 |
| 4 | 27884 | 3.414 | 3.410 | 2.942 | 3.268 | 3.263 | 2.995 | 2.857 | 2.921 | 3.087 | 4.575 | 9.537 |
| 8 | 13942 | 3.551 | 3.567 | 3.158 | 3.503 | 3.447 | 3.108 | 3.113 | 2.921 | 3.087 | 4.575 | 16.153 |
| 16 | 6971 | 3.715 | 3.748 | 3.443 | 3.808 | 3.663 | 3.260 | 3.429 | 2.922 | 3.087 | 4.576 | 29.385 |

## synseg (dict 512, table 256, model 7.2 kB)

| K | shard B | zlib | gzip | bz2 | lzma | zstd | brotli | ppmd | rate | archive | shared | per-file |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 120000 | 2.225 | 2.220 | 1.513 | 1.951 | 1.941 | 1.954 | 1.500 | 1.275 | 1.323 | 1.757 | 1.757 |
| 2 | 60000 | 2.326 | 2.331 | 1.718 | 2.138 | 2.119 | 2.124 | 1.725 | 1.275 | 1.323 | 1.757 | 2.239 |
| 4 | 30000 | 2.492 | 2.520 | 2.026 | 2.368 | 2.312 | 2.321 | 2.064 | 1.275 | 1.324 | 1.757 | 3.204 |
| 8 | 15000 | 2.731 | 2.774 | 2.497 | 2.678 | 2.578 | 2.581 | 2.578 | 1.276 | 1.324 | 1.758 | 5.133 |
| 16 | 7500 | 3.072 | 3.153 | 3.130 | 3.151 | 2.974 | 2.955 | 3.296 | 1.278 | 1.326 | 1.760 | 8.991 |

## syngrid (dict 302, table 8192, model 51.8 kB)

| K | shard B | zlib | gzip | bz2 | lzma | zstd | brotli | ppmd | rate | archive | shared | per-file |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 120016 | 2.886 | 2.866 | 2.129 | 2.005 | 2.453 | 2.295 | 2.063 | 1.594 | 1.939 | 5.045 | 5.045 |
| 2 | 60008 | 2.972 | 2.946 | 2.304 | 2.121 | 2.665 | 2.448 | 2.232 | 1.594 | 1.939 | 5.045 | 8.497 |
| 4 | 30004 | 3.098 | 3.069 | 2.476 | 2.275 | 2.827 | 2.565 | 2.411 | 1.594 | 1.939 | 5.045 | 15.399 |
| 8 | 15002 | 3.180 | 3.171 | 2.686 | 2.481 | 2.959 | 2.663 | 2.630 | 1.594 | 1.939 | 5.045 | 29.205 |
| 16 | 7501 | 3.244 | 3.266 | 2.956 | 2.780 | 3.099 | 2.769 | 2.940 | 1.595 | 1.940 | 5.046 | 56.817 |

## Gates

| gate | corpus | value | status |
|---|---|---|---|
| G6 | shakespeare | shared 4.575 -> 4.576 (K 1->16); per-file 4.575 -> 29.385 | **pass** |
| G7 | shakespeare | bz2 2.635 -> 3.443 (K 1->16) | **pass** |
| G8 | shakespeare | archive 3.087; bz2 crosses at K=8 | **pass** |
| G6 | synseg | shared 1.757 -> 1.760 (K 1->16); per-file 1.757 -> 8.991 | **pass** |
| G7 | synseg | bz2 1.513 -> 3.130 (K 1->16) | **pass** |
| G8 | synseg | archive 1.323; bz2 crosses at K=1 | **pass** |
| G6 | syngrid | shared 5.045 -> 5.046 (K 1->16); per-file 5.045 -> 56.817 | **pass** |
| G7 | syngrid | bz2 2.129 -> 2.956 (K 1->16) | **pass** |
| G8 | syngrid | archive 1.939; bz2 crosses at K=1 | **pass** |

