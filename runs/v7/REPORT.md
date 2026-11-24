# H-NCM v7 - solved tiling over a pruned span dictionary (substr)

## synseg

stream 1.20 MB; chosen min_count 128; dictionary 8361 spans, 62.9 kB (0.420 bpb); byte unigram 4.640 bpb; gzip 2.225 bpb; crossover unigram 0.17 MB, gzip 0.93 MB

| min_count | dict | model kB | rate | charged |
|---:|---:|---:|---:|---:|
| 1 | 1095207 | 15516.0 | 1.305 | 104.743 |
| 4 | 508451 | 6485.5 | 1.318 | 44.554 |
| 16 | 110351 | 1233.0 | 1.555 | 9.775 |
| 64 | 9369 | 70.9 | 1.679 | 2.152 |
| 128 | 8361 | 62.9 | 1.682 | 2.101 |
| 512 | 2702 | 18.0 | 3.261 | 3.380 |
| 2048 | 381 | 1.1 | 4.392 | 4.399 |

| arm | lambda | rate | charged | tokens | bytes/token | align |
|---|---:|---:|---:|---:|---:|---:|
| DP | 0.0 | 1.682 | 2.101 | 19512 | 6.15 | 0.850 |
| DP | 1.0 | 1.682 | 2.101 | 19512 | 6.15 | 0.850 |
| DP | 2.0 | 1.682 | 2.101 | 19512 | 6.15 | 0.850 |
| DP | 4.0 | 1.682 | 2.101 | 19512 | 6.15 | 0.850 |
| fixed16 | 0.0 | 4.636 | 5.056 | 7500 | 16.00 | 0.090 |
| fixed4 | 0.0 | 3.306 | 3.726 | 30000 | 4.00 | 0.189 |
| fixed8 | 0.0 | 3.939 | 4.358 | 15000 | 8.00 | 0.137 |
| greedy | 0.0 | 1.751 | 2.171 | 19004 | 6.31 | 0.886 |
| greedy | 1.0 | 1.751 | 2.171 | 19004 | 6.31 | 0.886 |
| greedy | 2.0 | 1.751 | 2.171 | 19004 | 6.31 | 0.886 |
| greedy | 4.0 | 1.751 | 2.171 | 19004 | 6.31 | 0.886 |
| oracle | 0.0 | 1.913 | 2.332 | 17737 | 6.77 | 1.000 |
| random | 0.0 | 4.285 | 4.705 | 19275 | 6.23 | 0.150 |
| random | 1.0 | 4.285 | 4.705 | 19275 | 6.23 | 0.150 |
| random | 2.0 | 4.285 | 4.705 | 19275 | 6.23 | 0.150 |
| random | 4.0 | 4.285 | 4.705 | 19275 | 6.23 | 0.150 |

## syngrid

stream 1.20 MB; chosen min_count 512; dictionary 2478 spans, 15.0 kB (0.100 bpb); byte unigram 3.911 bpb; gzip 2.886 bpb; crossover unigram 0.09 MB, gzip 0.34 MB

| min_count | dict | model kB | rate | charged |
|---:|---:|---:|---:|---:|
| 1 | 1088267 | 13901.4 | 1.882 | 94.548 |
| 4 | 252735 | 2680.7 | 1.956 | 19.825 |
| 16 | 132420 | 1269.4 | 1.985 | 10.447 |
| 64 | 9101 | 67.4 | 2.504 | 2.953 |
| 128 | 4380 | 28.9 | 2.525 | 2.717 |
| 512 | 2478 | 15.0 | 2.530 | 2.630 |
| 2048 | 410 | 1.4 | 3.599 | 3.608 |

| arm | lambda | rate | charged | tokens | bytes/token | align |
|---|---:|---:|---:|---:|---:|---:|
| DP | 0.0 | 2.530 | 2.630 | 30219 | 3.97 | 0.076 |
| DP | 1.0 | 2.530 | 2.630 | 30219 | 3.97 | 0.076 |
| DP | 2.0 | 2.530 | 2.630 | 30219 | 3.97 | 0.076 |
| DP | 4.0 | 2.530 | 2.630 | 30219 | 3.97 | 0.076 |
| fixed16 | 0.0 | 3.906 | 4.006 | 7501 | 16.00 | 0.400 |
| fixed4 | 0.0 | 2.551 | 2.651 | 30004 | 4.00 | 1.000 |
| fixed8 | 0.0 | 3.906 | 4.006 | 15002 | 8.00 | 0.667 |
| greedy | 0.0 | 2.551 | 2.651 | 30004 | 4.00 | 1.000 |
| greedy | 1.0 | 2.551 | 2.651 | 30004 | 4.00 | 1.000 |
| greedy | 2.0 | 2.551 | 2.651 | 30004 | 4.00 | 1.000 |
| greedy | 4.0 | 2.551 | 2.651 | 30004 | 4.00 | 1.000 |
| oracle | 0.0 | 2.551 | 2.651 | 30004 | 4.00 | 1.000 |
| random | 0.0 | 3.801 | 3.901 | 19276 | 6.23 | 0.190 |
| random | 1.0 | 3.801 | 3.901 | 19276 | 6.23 | 0.190 |
| random | 2.0 | 3.801 | 3.901 | 19276 | 6.23 | 0.190 |
| random | 4.0 | 3.801 | 3.901 | 19276 | 6.23 | 0.190 |

## shakespeare

stream 1.12 MB; chosen min_count 128; dictionary 5792 spans, 40.7 kB (0.292 bpb); byte unigram 4.777 bpb; gzip 3.218 bpb; crossover unigram 0.20 MB, gzip 3.85 MB

| min_count | dict | model kB | rate | charged |
|---:|---:|---:|---:|---:|
| 1 | 934991 | 10907.0 | 2.257 | 80.486 |
| 4 | 292870 | 3008.2 | 2.355 | 23.931 |
| 16 | 55846 | 491.0 | 2.636 | 6.158 |
| 64 | 13070 | 101.7 | 2.974 | 3.703 |
| 128 | 5792 | 40.7 | 3.133 | 3.426 |
| 512 | 1209 | 6.2 | 3.518 | 3.563 |
| 2048 | 452 | 1.6 | 3.932 | 3.944 |

| arm | lambda | rate | charged | tokens | bytes/token | align |
|---|---:|---:|---:|---:|---:|---:|
| DP | 0.0 | 3.133 | 3.426 | 36954 | 3.02 | 0.239 |
| DP | 1.0 | 3.133 | 3.426 | 36954 | 3.02 | 0.239 |
| DP | 2.0 | 3.133 | 3.426 | 36954 | 3.02 | 0.239 |
| DP | 4.0 | 3.133 | 3.426 | 36954 | 3.02 | 0.239 |
| fixed16 | 0.0 | 4.829 | 5.121 | 6971 | 16.00 | 0.092 |
| fixed4 | 0.0 | 4.106 | 4.398 | 27884 | 4.00 | 0.210 |
| fixed8 | 0.0 | 4.813 | 5.105 | 13942 | 8.00 | 0.148 |
| greedy | 0.0 | 3.352 | 3.645 | 34634 | 3.22 | 0.398 |
| greedy | 1.0 | 3.352 | 3.645 | 34634 | 3.22 | 0.398 |
| greedy | 2.0 | 3.352 | 3.645 | 34634 | 3.22 | 0.398 |
| greedy | 4.0 | 3.352 | 3.645 | 34634 | 3.22 | 0.398 |
| oracle | 0.0 | 4.262 | 4.554 | 20153 | 5.53 | 1.000 |
| random | 0.0 | 4.693 | 4.985 | 17935 | 6.22 | 0.169 |
| random | 1.0 | 4.693 | 4.985 | 17935 | 6.22 | 0.169 |
| random | 2.0 | 4.693 | 4.985 | 17935 | 6.22 | 0.169 |
| random | 4.0 | 4.693 | 4.985 | 17935 | 6.22 | 0.169 |

## Gates

| gate | corpus | value | status |
|---|---|---|---|
| G0 | synseg | DP charged=2.101 gzip=2.225 (model=0.420 bpb) | **pass** |
| G1 | synseg | DP=1.682 gzip=2.225 lam=0.0 | **pass** |
| G2 | synseg | DP charged=2.101 unigram=4.640 | **pass** |
| G5 | synseg | DP=1.682 fixed=3.306 lam=0.0 | **pass** |
| G5c | synseg | DP=1.682 greedy=1.751 gain=+0.070 lam=0.0 | **pass** |
| G0 | synseg | DP charged=2.101 gzip=2.225 (model=0.420 bpb) | **pass** |
| G1 | synseg | DP=1.682 gzip=2.225 lam=1.0 | **pass** |
| G2 | synseg | DP charged=2.101 unigram=4.640 | **pass** |
| G5 | synseg | DP=1.682 fixed=3.306 lam=1.0 | **pass** |
| G5c | synseg | DP=1.682 greedy=1.751 gain=+0.070 lam=1.0 | **pass** |
| G0 | synseg | DP charged=2.101 gzip=2.225 (model=0.420 bpb) | **pass** |
| G1 | synseg | DP=1.682 gzip=2.225 lam=2.0 | **pass** |
| G2 | synseg | DP charged=2.101 unigram=4.640 | **pass** |
| G5 | synseg | DP=1.682 fixed=3.306 lam=2.0 | **pass** |
| G5c | synseg | DP=1.682 greedy=1.751 gain=+0.070 lam=2.0 | **pass** |
| G0 | synseg | DP charged=2.101 gzip=2.225 (model=0.420 bpb) | **pass** |
| G1 | synseg | DP=1.682 gzip=2.225 lam=4.0 | **pass** |
| G2 | synseg | DP charged=2.101 unigram=4.640 | **pass** |
| G5 | synseg | DP=1.682 fixed=3.306 lam=4.0 | **pass** |
| G5c | synseg | DP=1.682 greedy=1.751 gain=+0.070 lam=4.0 | **pass** |
| G0 | syngrid | DP charged=2.630 gzip=2.886 (model=0.100 bpb) | **pass** |
| G1 | syngrid | DP=2.530 gzip=2.886 lam=0.0 | **pass** |
| G2 | syngrid | DP charged=2.630 unigram=3.911 | **pass** |
| G5 | syngrid | DP=2.530 fixed=2.551 lam=0.0 | **pass** |
| G5c | syngrid | DP=2.530 greedy=2.551 gain=+0.021 lam=0.0 | **pass** |
| G0 | syngrid | DP charged=2.630 gzip=2.886 (model=0.100 bpb) | **pass** |
| G1 | syngrid | DP=2.530 gzip=2.886 lam=1.0 | **pass** |
| G2 | syngrid | DP charged=2.630 unigram=3.911 | **pass** |
| G5 | syngrid | DP=2.530 fixed=2.551 lam=1.0 | **pass** |
| G5c | syngrid | DP=2.530 greedy=2.551 gain=+0.021 lam=1.0 | **pass** |
| G0 | syngrid | DP charged=2.630 gzip=2.886 (model=0.100 bpb) | **pass** |
| G1 | syngrid | DP=2.530 gzip=2.886 lam=2.0 | **pass** |
| G2 | syngrid | DP charged=2.630 unigram=3.911 | **pass** |
| G5 | syngrid | DP=2.530 fixed=2.551 lam=2.0 | **pass** |
| G5c | syngrid | DP=2.530 greedy=2.551 gain=+0.021 lam=2.0 | **pass** |
| G0 | syngrid | DP charged=2.630 gzip=2.886 (model=0.100 bpb) | **pass** |
| G1 | syngrid | DP=2.530 gzip=2.886 lam=4.0 | **pass** |
| G2 | syngrid | DP charged=2.630 unigram=3.911 | **pass** |
| G5 | syngrid | DP=2.530 fixed=2.551 lam=4.0 | **pass** |
| G5c | syngrid | DP=2.530 greedy=2.551 gain=+0.021 lam=4.0 | **pass** |
| G0 | shakespeare | DP charged=3.426 gzip=3.218 (model=0.292 bpb) | **fail** |
| G1 | shakespeare | DP=3.133 gzip=3.218 lam=0.0 | **pass** |
| G2 | shakespeare | DP charged=3.426 unigram=4.777 | **pass** |
| G5 | shakespeare | DP=3.133 fixed=4.106 lam=0.0 | **pass** |
| G5c | shakespeare | DP=3.133 greedy=3.352 gain=+0.219 lam=0.0 | **pass** |
| G0 | shakespeare | DP charged=3.426 gzip=3.218 (model=0.292 bpb) | **fail** |
| G1 | shakespeare | DP=3.133 gzip=3.218 lam=1.0 | **pass** |
| G2 | shakespeare | DP charged=3.426 unigram=4.777 | **pass** |
| G5 | shakespeare | DP=3.133 fixed=4.106 lam=1.0 | **pass** |
| G5c | shakespeare | DP=3.133 greedy=3.352 gain=+0.219 lam=1.0 | **pass** |
| G0 | shakespeare | DP charged=3.426 gzip=3.218 (model=0.292 bpb) | **fail** |
| G1 | shakespeare | DP=3.133 gzip=3.218 lam=2.0 | **pass** |
| G2 | shakespeare | DP charged=3.426 unigram=4.777 | **pass** |
| G5 | shakespeare | DP=3.133 fixed=4.106 lam=2.0 | **pass** |
| G5c | shakespeare | DP=3.133 greedy=3.352 gain=+0.219 lam=2.0 | **pass** |
| G0 | shakespeare | DP charged=3.426 gzip=3.218 (model=0.292 bpb) | **fail** |
| G1 | shakespeare | DP=3.133 gzip=3.218 lam=4.0 | **pass** |
| G2 | shakespeare | DP charged=3.426 unigram=4.777 | **pass** |
| G5 | shakespeare | DP=3.133 fixed=4.106 lam=4.0 | **pass** |
| G5c | shakespeare | DP=3.133 greedy=3.352 gain=+0.219 lam=4.0 | **pass** |

