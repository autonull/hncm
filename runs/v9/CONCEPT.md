# H-NCM v7: the concept LM, charged (M1-M3)

The vocabulary (word onsets or text substrings) and the budget (dict size x table size) are chosen by the *rule*: the charged total on the last 10% of the train, which the model never fitted, never on the val. `oracle` is the best grid point on the val plus the header that transmits the choice (a lower bound for any budget rule; the gap to `rule` is the cost of the rule). The table is refitted on the path the DP searches (2 EM rounds, chosen on the same 10%). The model is charged 6 bytes per bigram entry and amortised over train+val. The rate is the exact first-order Viterbi.

| corpus | vocab | stream | zlib | gzip | bz2 | lzma | zstd | brotli | ppmd | rule | oracle (vocab, dict, table) | oracle | vs zlib | vs best | x zlib |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| enwik8_10m | union | 10.00 MB | 2.998 | 2.969 | 2.393 | 2.427 | 2.511 | 2.344 | 2.160 | 2.751 | union, 16384, 16384 | 2.784 | -0.214 | +0.625 | 5.20 MB |

## enwik8_10m (vocab union, dict 16384, 16384 bigram entries)

| dict | table | model kB | unigram | rate | charged | sel rate | sel charged | zlib crossing | tokens | covered | calls |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 8192 | 140.8 | 3.171 | 2.712 | 2.825 | 2.631 | 2.744 | 3.94 MB | 446305 | 0.686 | 314437 |
| 8192 | 16384 | 189.9 | 3.171 | 2.655 | 2.807 | 2.563 | 2.715 | 4.43 MB | 444805 | 0.687 | 312549 |
| 8192 | 32768 | 288.3 | 3.171 | 2.607 | 2.838 | 2.510 | 2.741 | 5.90 MB | 443364 | 0.689 | 310667 |
| 16384 | 8192 | 240.5 | 2.952 | 2.609 | 2.801 | 2.533 | 2.725 | 4.94 MB | 384289 | 0.750 | 250090 |
| 16384 | 16384 | 289.7 | 2.952 | 2.553 | 2.784 | 2.464 | 2.695 | 5.20 MB | 382737 | 0.752 | 247688 |
| 16384 | 32768 | 388.0 | 2.952 | 2.499 | 2.809 | 2.402 | 2.713 | 6.21 MB | 381662 | 0.754 | 246177 |
| 32768 | 8192 | 449.5 | 2.782 | 2.525 | 2.885 | 2.450 | 2.810 | 7.60 MB | 335907 | 0.798 | 201665 |
| 32768 | 16384 | 498.7 | 2.782 | 2.475 | 2.874 | 2.392 | 2.791 | 7.62 MB | 334703 | 0.801 | 199319 |
| 32768 | 32768 | 597.0 | 2.782 | 2.425 | 2.902 | 2.334 | 2.811 | 8.32 MB | 333860 | 0.802 | 197788 |

| vocab | dict | table | model kB | rate | charged | sel charged | oracle charged |
|---|---:|---:|---:|---:|---:|---:|---:|
| union | 16384 | 16384 | 289.7 | 2.519 | 2.751 | 2.695 | 2.784 |

### arms at dict 16384, 16384 entries

| arm | rate | charged | tokens | bytes/token | covered | calls |
|---|---:|---:|---:|---:|---:|---:|
| DP | 2.553 | 2.784 | 382737 | 2.61 | 0.752 | 247688 |
| greedy | 2.756 | 2.988 | 370889 | 2.70 | 0.766 | 233913 |
| fixed4 | 4.950 | 5.182 | 250000 | 4.00 | 1.000 | 1 |
| unigram | 2.952 | 3.105 | 370214 | 2.70 | 0.766 | 234288 |
| DP-em | 2.519 | 2.751 | 390767 | 2.56 | 0.742 | 257832 |
| greedy-em | 2.758 | 2.990 | 370889 | 2.70 | 0.766 | 233913 |

crossings of the envelope (MB of stream, versus each codec):

- zlib: 5.20 MB
- gzip: 5.57 MB
- bz2: never
- lzma: never
- zstd: never
- brotli: never
- ppmd: never

## Gates

| gate | corpus | value | status |
|---|---|---|---|
| G1 | enwik8_10m | rule=2.751 oracle=2.784 zlib=2.998 | **pass** |
| G1a | enwik8_10m | rule=2.751 oracle=2.784 ppmd=2.160 | **fail** |
| G2 | enwik8_10m | DP=2.519 greedy=2.758 gain=+0.239 | **pass** |
| G3 | enwik8_10m | DP=2.519 fixed4=4.950 unigram=2.952 | **pass** |

