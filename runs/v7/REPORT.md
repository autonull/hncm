# H-NCM v7 - solved tiling over a pruned span dictionary (onsets)

## synseg

stream 1.20 MB; chosen vocab 512; dictionary 768 spans, 5.7 kB (0.038 bpb); byte unigram 4.640 bpb; gzip 2.225 bpb; crossover unigram 0.01 MB, gzip 0.05 MB

| vocab | dict | model kB | rate | charged |
|---:|---:|---:|---:|---:|
| 256 | 512 | 3.0 | 1.916 | 1.936 |
| 512 | 768 | 5.7 | 1.290 | 1.328 |
| 1024 | 894 | 7.0 | 1.285 | 1.332 |
| 2048 | 894 | 7.0 | 1.285 | 1.332 |
| 4096 | 894 | 7.0 | 1.285 | 1.332 |
| 8192 | 894 | 7.0 | 1.285 | 1.332 |
| 16384 | 894 | 7.0 | 1.285 | 1.332 |
| 32768 | 894 | 7.0 | 1.285 | 1.332 |

| arm | lambda | rate | charged | tokens | bytes/token | covered | calls | align |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DP | 0.0 | 1.290 | 1.328 | 18219 | 6.59 | 0.996 | 485 | 0.987 |
| DP | 1.0 | 1.290 | 1.328 | 18219 | 6.59 | 0.996 | 485 | 0.987 |
| DP | 2.0 | 1.290 | 1.328 | 18219 | 6.59 | 0.996 | 485 | 0.987 |
| DP | 4.0 | 1.290 | 1.328 | 18219 | 6.59 | 0.996 | 485 | 0.987 |
| fixed16 | 0.0 | 4.636 | 4.674 | 7500 | 16.00 | 1.000 | 1 | 0.090 |
| fixed4 | 0.0 | 4.592 | 4.630 | 30000 | 4.00 | 1.000 | 1 | 0.189 |
| fixed8 | 0.0 | 4.557 | 4.595 | 15000 | 8.00 | 1.000 | 1 | 0.137 |
| greedy | 0.0 | 1.290 | 1.328 | 18219 | 6.59 | 0.996 | 485 | 0.987 |
| greedy | 1.0 | 1.290 | 1.328 | 18219 | 6.59 | 0.996 | 485 | 0.987 |
| greedy | 2.0 | 1.290 | 1.328 | 18219 | 6.59 | 0.996 | 485 | 0.987 |
| greedy | 4.0 | 1.290 | 1.328 | 18219 | 6.59 | 0.996 | 485 | 0.987 |
| oracle | 0.0 | 1.376 | 1.414 | 17737 | 6.77 | 1.000 | 1 | 1.000 |
| random | 0.0 | 4.615 | 4.653 | 19275 | 6.23 | 0.968 | 3831 | 0.150 |
| random | 1.0 | 4.615 | 4.653 | 19275 | 6.23 | 0.968 | 3831 | 0.150 |
| random | 2.0 | 4.615 | 4.653 | 19275 | 6.23 | 0.968 | 3831 | 0.150 |
| random | 4.0 | 4.615 | 4.653 | 19275 | 6.23 | 0.968 | 3831 | 0.150 |

## syngrid

stream 1.20 MB; chosen vocab 512; dictionary 558 spans, 2.6 kB (0.018 bpb); byte unigram 3.911 bpb; gzip 2.886 bpb; crossover unigram 0.01 MB, gzip 0.03 MB

| vocab | dict | model kB | rate | charged |
|---:|---:|---:|---:|---:|
| 256 | 512 | 2.3 | 2.279 | 2.294 |
| 512 | 558 | 2.6 | 2.051 | 2.068 |
| 1024 | 558 | 2.6 | 2.051 | 2.068 |
| 2048 | 558 | 2.6 | 2.051 | 2.068 |
| 4096 | 558 | 2.6 | 2.051 | 2.068 |
| 8192 | 558 | 2.6 | 2.051 | 2.068 |
| 16384 | 558 | 2.6 | 2.051 | 2.068 |
| 32768 | 558 | 2.6 | 2.051 | 2.068 |

| arm | lambda | rate | charged | tokens | bytes/token | covered | calls | align |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DP | 0.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 | 1 | 1.000 |
| DP | 1.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 | 1 | 1.000 |
| DP | 2.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 | 1 | 1.000 |
| DP | 4.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 | 1 | 1.000 |
| fixed16 | 0.0 | 3.906 | 3.923 | 7501 | 16.00 | 1.000 | 1 | 0.400 |
| fixed4 | 0.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 | 1 | 1.000 |
| fixed8 | 0.0 | 3.906 | 3.923 | 15002 | 8.00 | 1.000 | 1 | 0.667 |
| greedy | 0.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 | 1 | 1.000 |
| greedy | 1.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 | 1 | 1.000 |
| greedy | 2.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 | 1 | 1.000 |
| greedy | 4.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 | 1 | 1.000 |
| oracle | 0.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 | 1 | 1.000 |
| random | 0.0 | 3.847 | 3.864 | 19276 | 6.23 | 0.968 | 3829 | 0.190 |
| random | 1.0 | 3.847 | 3.864 | 19276 | 6.23 | 0.968 | 3829 | 0.190 |
| random | 2.0 | 3.847 | 3.864 | 19276 | 6.23 | 0.968 | 3829 | 0.190 |
| random | 4.0 | 3.847 | 3.864 | 19276 | 6.23 | 0.968 | 3829 | 0.190 |

## shakespeare

stream 1.12 MB; chosen vocab 4096; dictionary 4352 spans, 39.7 kB (0.285 bpb); byte unigram 4.777 bpb; gzip 3.218 bpb; crossover unigram 0.19 MB, gzip 2.03 MB

| vocab | dict | model kB | rate | charged |
|---:|---:|---:|---:|---:|
| 256 | 512 | 2.6 | 3.877 | 3.896 |
| 512 | 768 | 4.8 | 3.756 | 3.791 |
| 1024 | 1280 | 9.5 | 3.435 | 3.504 |
| 2048 | 2304 | 19.3 | 3.237 | 3.375 |
| 4096 | 4352 | 39.7 | 3.062 | 3.347 |
| 8192 | 8448 | 82.5 | 2.905 | 3.497 |
| 16384 | 16640 | 175.0 | 2.816 | 4.071 |
| 32768 | 28949 | 315.6 | 2.709 | 4.972 |

| arm | lambda | rate | charged | tokens | bytes/token | covered | calls | align |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DP | 0.0 | 3.062 | 3.347 | 52978 | 2.11 | 0.667 | 37168 | 0.551 |
| DP | 1.0 | 3.062 | 3.347 | 52978 | 2.11 | 0.667 | 37168 | 0.551 |
| DP | 2.0 | 3.062 | 3.347 | 52978 | 2.11 | 0.667 | 37168 | 0.551 |
| DP | 4.0 | 3.062 | 3.347 | 52978 | 2.11 | 0.667 | 37168 | 0.551 |
| fixed16 | 0.0 | 4.829 | 5.114 | 6971 | 16.00 | 1.000 | 1 | 0.092 |
| fixed4 | 0.0 | 4.725 | 5.010 | 27884 | 4.00 | 1.000 | 1 | 0.210 |
| fixed8 | 0.0 | 4.814 | 5.099 | 13942 | 8.00 | 1.000 | 1 | 0.148 |
| greedy | 0.0 | 3.062 | 3.347 | 52978 | 2.11 | 0.667 | 37168 | 0.551 |
| greedy | 1.0 | 3.062 | 3.347 | 52978 | 2.11 | 0.667 | 37168 | 0.551 |
| greedy | 2.0 | 3.062 | 3.347 | 52978 | 2.11 | 0.667 | 37168 | 0.551 |
| greedy | 4.0 | 3.062 | 3.347 | 52978 | 2.11 | 0.667 | 37168 | 0.551 |
| oracle | 0.0 | 3.126 | 3.411 | 20153 | 5.53 | 1.000 | 1 | 1.000 |
| random | 0.0 | 4.812 | 5.097 | 17935 | 6.22 | 0.968 | 3561 | 0.169 |
| random | 1.0 | 4.812 | 5.097 | 17935 | 6.22 | 0.968 | 3561 | 0.169 |
| random | 2.0 | 4.812 | 5.097 | 17935 | 6.22 | 0.968 | 3561 | 0.169 |
| random | 4.0 | 4.812 | 5.097 | 17935 | 6.22 | 0.968 | 3561 | 0.169 |

## Gates

| gate | corpus | value | status |
|---|---|---|---|
| G0 | synseg | DP charged=1.328 gzip=2.225 (model=0.038 bpb) | **pass** |
| G1 | synseg | DP=1.290 gzip=2.225 lam=0.0 | **pass** |
| G2 | synseg | DP charged=1.328 unigram=4.640 | **pass** |
| G5 | synseg | DP=1.290 fixed=4.592 lam=0.0 | **pass** |
| G5c | synseg | DP=1.290 greedy=1.290 gain=+0.000 lam=0.0 | **pass** |
| G0 | synseg | DP charged=1.328 gzip=2.225 (model=0.038 bpb) | **pass** |
| G1 | synseg | DP=1.290 gzip=2.225 lam=1.0 | **pass** |
| G2 | synseg | DP charged=1.328 unigram=4.640 | **pass** |
| G5 | synseg | DP=1.290 fixed=4.592 lam=1.0 | **pass** |
| G5c | synseg | DP=1.290 greedy=1.290 gain=+0.000 lam=1.0 | **pass** |
| G0 | synseg | DP charged=1.328 gzip=2.225 (model=0.038 bpb) | **pass** |
| G1 | synseg | DP=1.290 gzip=2.225 lam=2.0 | **pass** |
| G2 | synseg | DP charged=1.328 unigram=4.640 | **pass** |
| G5 | synseg | DP=1.290 fixed=4.592 lam=2.0 | **pass** |
| G5c | synseg | DP=1.290 greedy=1.290 gain=+0.000 lam=2.0 | **pass** |
| G0 | synseg | DP charged=1.328 gzip=2.225 (model=0.038 bpb) | **pass** |
| G1 | synseg | DP=1.290 gzip=2.225 lam=4.0 | **pass** |
| G2 | synseg | DP charged=1.328 unigram=4.640 | **pass** |
| G5 | synseg | DP=1.290 fixed=4.592 lam=4.0 | **pass** |
| G5c | synseg | DP=1.290 greedy=1.290 gain=+0.000 lam=4.0 | **pass** |
| G0 | syngrid | DP charged=2.068 gzip=2.886 (model=0.018 bpb) | **pass** |
| G1 | syngrid | DP=2.051 gzip=2.886 lam=0.0 | **pass** |
| G2 | syngrid | DP charged=2.068 unigram=3.911 | **pass** |
| G5 | syngrid | DP=2.051 fixed=2.051 lam=0.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 gain=+0.000 lam=0.0 | **pass** |
| G0 | syngrid | DP charged=2.068 gzip=2.886 (model=0.018 bpb) | **pass** |
| G1 | syngrid | DP=2.051 gzip=2.886 lam=1.0 | **pass** |
| G2 | syngrid | DP charged=2.068 unigram=3.911 | **pass** |
| G5 | syngrid | DP=2.051 fixed=2.051 lam=1.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 gain=+0.000 lam=1.0 | **pass** |
| G0 | syngrid | DP charged=2.068 gzip=2.886 (model=0.018 bpb) | **pass** |
| G1 | syngrid | DP=2.051 gzip=2.886 lam=2.0 | **pass** |
| G2 | syngrid | DP charged=2.068 unigram=3.911 | **pass** |
| G5 | syngrid | DP=2.051 fixed=2.051 lam=2.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 gain=+0.000 lam=2.0 | **pass** |
| G0 | syngrid | DP charged=2.068 gzip=2.886 (model=0.018 bpb) | **pass** |
| G1 | syngrid | DP=2.051 gzip=2.886 lam=4.0 | **pass** |
| G2 | syngrid | DP charged=2.068 unigram=3.911 | **pass** |
| G5 | syngrid | DP=2.051 fixed=2.051 lam=4.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 gain=+0.000 lam=4.0 | **pass** |
| G0 | shakespeare | DP charged=3.347 gzip=3.218 (model=0.285 bpb) | **fail** |
| G1 | shakespeare | DP=3.062 gzip=3.218 lam=0.0 | **pass** |
| G2 | shakespeare | DP charged=3.347 unigram=4.777 | **pass** |
| G5 | shakespeare | DP=3.062 fixed=4.725 lam=0.0 | **pass** |
| G5c | shakespeare | DP=3.062 greedy=3.062 gain=+0.000 lam=0.0 | **pass** |
| G0 | shakespeare | DP charged=3.347 gzip=3.218 (model=0.285 bpb) | **fail** |
| G1 | shakespeare | DP=3.062 gzip=3.218 lam=1.0 | **pass** |
| G2 | shakespeare | DP charged=3.347 unigram=4.777 | **pass** |
| G5 | shakespeare | DP=3.062 fixed=4.725 lam=1.0 | **pass** |
| G5c | shakespeare | DP=3.062 greedy=3.062 gain=+0.000 lam=1.0 | **pass** |
| G0 | shakespeare | DP charged=3.347 gzip=3.218 (model=0.285 bpb) | **fail** |
| G1 | shakespeare | DP=3.062 gzip=3.218 lam=2.0 | **pass** |
| G2 | shakespeare | DP charged=3.347 unigram=4.777 | **pass** |
| G5 | shakespeare | DP=3.062 fixed=4.725 lam=2.0 | **pass** |
| G5c | shakespeare | DP=3.062 greedy=3.062 gain=+0.000 lam=2.0 | **pass** |
| G0 | shakespeare | DP charged=3.347 gzip=3.218 (model=0.285 bpb) | **fail** |
| G1 | shakespeare | DP=3.062 gzip=3.218 lam=4.0 | **pass** |
| G2 | shakespeare | DP charged=3.347 unigram=4.777 | **pass** |
| G5 | shakespeare | DP=3.062 fixed=4.725 lam=4.0 | **pass** |
| G5c | shakespeare | DP=3.062 greedy=3.062 gain=+0.000 lam=4.0 | **pass** |

