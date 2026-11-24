# H-NCM v7 - solved tiling over a pruned span dictionary

## synseg

stream 1.20 MB; chosen min_count 32; dictionary 812 spans, 6.1 kB (0.041 bpb); byte unigram 4.640 bpb; gzip 2.225 bpb; crossover unigram 0.01 MB, gzip 0.05 MB

| min_count | dict | model kB | rate | charged |
|---:|---:|---:|---:|---:|
| 1 | 895 | 7.0 | 1.285 | 1.332 |
| 2 | 894 | 7.0 | 1.285 | 1.332 |
| 4 | 894 | 7.0 | 1.285 | 1.332 |
| 8 | 894 | 7.0 | 1.285 | 1.332 |
| 16 | 894 | 7.0 | 1.285 | 1.332 |
| 32 | 812 | 6.1 | 1.288 | 1.329 |

| arm | lambda | rate | charged | tokens | bytes/token | align |
|---|---:|---:|---:|---:|---:|---:|
| DP | 0.0 | 1.288 | 1.329 | 18055 | 6.65 | 0.991 |
| DP | 1.0 | 1.288 | 1.329 | 18055 | 6.65 | 0.991 |
| DP | 2.0 | 1.288 | 1.329 | 18055 | 6.65 | 0.991 |
| DP | 4.0 | 1.288 | 1.329 | 18055 | 6.65 | 0.991 |
| fixed16 | 0.0 | 4.636 | 4.677 | 7500 | 16.00 | 0.090 |
| fixed4 | 0.0 | 4.592 | 4.633 | 30000 | 4.00 | 0.189 |
| fixed8 | 0.0 | 4.557 | 4.598 | 15000 | 8.00 | 0.137 |
| greedy | 0.0 | 1.288 | 1.329 | 18055 | 6.65 | 0.991 |
| greedy | 1.0 | 1.288 | 1.329 | 18055 | 6.65 | 0.991 |
| greedy | 2.0 | 1.288 | 1.329 | 18055 | 6.65 | 0.991 |
| greedy | 4.0 | 1.288 | 1.329 | 18055 | 6.65 | 0.991 |
| oracle | 0.0 | 1.347 | 1.388 | 17737 | 6.77 | 1.000 |
| random | 0.0 | 4.615 | 4.656 | 19275 | 6.23 | 0.150 |
| random | 1.0 | 4.615 | 4.656 | 19275 | 6.23 | 0.150 |
| random | 2.0 | 4.615 | 4.656 | 19275 | 6.23 | 0.150 |
| random | 4.0 | 4.615 | 4.656 | 19275 | 6.23 | 0.150 |

## syngrid

stream 1.20 MB; chosen min_count 1; dictionary 558 spans, 2.6 kB (0.018 bpb); byte unigram 3.911 bpb; gzip 2.886 bpb; crossover unigram 0.01 MB, gzip 0.03 MB

| min_count | dict | model kB | rate | charged |
|---:|---:|---:|---:|---:|
| 1 | 558 | 2.6 | 2.051 | 2.068 |
| 2 | 558 | 2.6 | 2.051 | 2.068 |
| 4 | 558 | 2.6 | 2.051 | 2.068 |
| 8 | 558 | 2.6 | 2.051 | 2.068 |
| 16 | 558 | 2.6 | 2.051 | 2.068 |
| 32 | 558 | 2.6 | 2.051 | 2.068 |

| arm | lambda | rate | charged | tokens | bytes/token | align |
|---|---:|---:|---:|---:|---:|---:|
| DP | 0.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 |
| DP | 1.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 |
| DP | 2.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 |
| DP | 4.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 |
| fixed16 | 0.0 | 3.906 | 3.923 | 7501 | 16.00 | 0.400 |
| fixed4 | 0.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 |
| fixed8 | 0.0 | 3.906 | 3.923 | 15002 | 8.00 | 0.667 |
| greedy | 0.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 |
| greedy | 1.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 |
| greedy | 2.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 |
| greedy | 4.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 |
| oracle | 0.0 | 2.051 | 2.068 | 30004 | 4.00 | 1.000 |
| random | 0.0 | 3.847 | 3.864 | 19276 | 6.23 | 0.190 |
| random | 1.0 | 3.847 | 3.864 | 19276 | 6.23 | 0.190 |
| random | 2.0 | 3.847 | 3.864 | 19276 | 6.23 | 0.190 |
| random | 4.0 | 3.847 | 3.864 | 19276 | 6.23 | 0.190 |

## shakespeare

stream 1.12 MB; chosen min_count 4; dictionary 5127 spans, 47.6 kB (0.341 bpb); byte unigram 4.777 bpb; gzip 3.218 bpb; crossover unigram 0.21 MB, gzip 1.78 MB

| min_count | dict | model kB | rate | charged |
|---:|---:|---:|---:|---:|
| 1 | 29148 | 317.2 | 2.709 | 4.985 |
| 2 | 10961 | 109.1 | 2.852 | 3.635 |
| 4 | 5127 | 47.6 | 3.004 | 3.345 |
| 8 | 2630 | 22.5 | 3.207 | 3.368 |
| 16 | 1431 | 10.9 | 3.391 | 3.469 |
| 32 | 873 | 5.8 | 3.715 | 3.756 |

| arm | lambda | rate | charged | tokens | bytes/token | align |
|---|---:|---:|---:|---:|---:|---:|
| DP | 0.0 | 3.004 | 3.345 | 51124 | 2.18 | 0.565 |
| DP | 1.0 | 3.004 | 3.345 | 51124 | 2.18 | 0.565 |
| DP | 2.0 | 3.004 | 3.345 | 51124 | 2.18 | 0.565 |
| DP | 4.0 | 3.004 | 3.345 | 51124 | 2.18 | 0.565 |
| fixed16 | 0.0 | 4.829 | 5.171 | 6971 | 16.00 | 0.092 |
| fixed4 | 0.0 | 4.725 | 5.067 | 27884 | 4.00 | 0.210 |
| fixed8 | 0.0 | 4.813 | 5.155 | 13942 | 8.00 | 0.148 |
| greedy | 0.0 | 3.005 | 3.346 | 51040 | 2.19 | 0.566 |
| greedy | 1.0 | 3.005 | 3.346 | 51040 | 2.19 | 0.566 |
| greedy | 2.0 | 3.005 | 3.346 | 51040 | 2.19 | 0.566 |
| greedy | 4.0 | 3.005 | 3.346 | 51040 | 2.19 | 0.566 |
| oracle | 0.0 | 3.069 | 3.411 | 20153 | 5.53 | 1.000 |
| random | 0.0 | 4.812 | 5.153 | 17935 | 6.22 | 0.169 |
| random | 1.0 | 4.812 | 5.153 | 17935 | 6.22 | 0.169 |
| random | 2.0 | 4.812 | 5.153 | 17935 | 6.22 | 0.169 |
| random | 4.0 | 4.812 | 5.153 | 17935 | 6.22 | 0.169 |

## Gates

| gate | corpus | value | status |
|---|---|---|---|
| G0 | synseg | DP=1.288 unigram=4.640 lam=0.0 | **pass** |
| G1 | synseg | DP=1.288 gzip=2.225 lam=0.0 | **pass** |
| G0b | synseg | charged DP=1.329 gzip=2.225 (model=0.041 bpb at 1.20 MB) | **pass** |
| G4 | synseg | oracle=1.347 fixed=4.592 | **pass** |
| G5 | synseg | DP=1.288 fixed=4.592 align=0.991/0.189 lam=0.0 | **pass** |
| G5c | synseg | DP=1.288 greedy=1.288 lam=0.0 | **pass** |
| G0 | synseg | DP=1.288 unigram=4.640 lam=1.0 | **pass** |
| G1 | synseg | DP=1.288 gzip=2.225 lam=1.0 | **pass** |
| G0b | synseg | charged DP=1.329 gzip=2.225 (model=0.041 bpb at 1.20 MB) | **pass** |
| G4 | synseg | oracle=1.347 fixed=4.592 | **pass** |
| G5 | synseg | DP=1.288 fixed=4.592 align=0.991/0.189 lam=1.0 | **pass** |
| G5c | synseg | DP=1.288 greedy=1.288 lam=1.0 | **pass** |
| G0 | synseg | DP=1.288 unigram=4.640 lam=2.0 | **pass** |
| G1 | synseg | DP=1.288 gzip=2.225 lam=2.0 | **pass** |
| G0b | synseg | charged DP=1.329 gzip=2.225 (model=0.041 bpb at 1.20 MB) | **pass** |
| G4 | synseg | oracle=1.347 fixed=4.592 | **pass** |
| G5 | synseg | DP=1.288 fixed=4.592 align=0.991/0.189 lam=2.0 | **pass** |
| G5c | synseg | DP=1.288 greedy=1.288 lam=2.0 | **pass** |
| G0 | synseg | DP=1.288 unigram=4.640 lam=4.0 | **pass** |
| G1 | synseg | DP=1.288 gzip=2.225 lam=4.0 | **pass** |
| G0b | synseg | charged DP=1.329 gzip=2.225 (model=0.041 bpb at 1.20 MB) | **pass** |
| G4 | synseg | oracle=1.347 fixed=4.592 | **pass** |
| G5 | synseg | DP=1.288 fixed=4.592 align=0.991/0.189 lam=4.0 | **pass** |
| G5c | synseg | DP=1.288 greedy=1.288 lam=4.0 | **pass** |
| G0 | syngrid | DP=2.051 unigram=3.911 lam=0.0 | **pass** |
| G1 | syngrid | DP=2.051 gzip=2.886 lam=0.0 | **pass** |
| G0b | syngrid | charged DP=2.068 gzip=2.886 (model=0.018 bpb at 1.20 MB) | **pass** |
| G4 | syngrid | oracle=2.051 fixed=2.051 | **fail** |
| G5 | syngrid | DP=2.051 fixed=2.051 align=1.000/1.000 lam=0.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 lam=0.0 | **pass** |
| G0 | syngrid | DP=2.051 unigram=3.911 lam=1.0 | **pass** |
| G1 | syngrid | DP=2.051 gzip=2.886 lam=1.0 | **pass** |
| G0b | syngrid | charged DP=2.068 gzip=2.886 (model=0.018 bpb at 1.20 MB) | **pass** |
| G4 | syngrid | oracle=2.051 fixed=2.051 | **fail** |
| G5 | syngrid | DP=2.051 fixed=2.051 align=1.000/1.000 lam=1.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 lam=1.0 | **pass** |
| G0 | syngrid | DP=2.051 unigram=3.911 lam=2.0 | **pass** |
| G1 | syngrid | DP=2.051 gzip=2.886 lam=2.0 | **pass** |
| G0b | syngrid | charged DP=2.068 gzip=2.886 (model=0.018 bpb at 1.20 MB) | **pass** |
| G4 | syngrid | oracle=2.051 fixed=2.051 | **fail** |
| G5 | syngrid | DP=2.051 fixed=2.051 align=1.000/1.000 lam=2.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 lam=2.0 | **pass** |
| G0 | syngrid | DP=2.051 unigram=3.911 lam=4.0 | **pass** |
| G1 | syngrid | DP=2.051 gzip=2.886 lam=4.0 | **pass** |
| G0b | syngrid | charged DP=2.068 gzip=2.886 (model=0.018 bpb at 1.20 MB) | **pass** |
| G4 | syngrid | oracle=2.051 fixed=2.051 | **fail** |
| G5 | syngrid | DP=2.051 fixed=2.051 align=1.000/1.000 lam=4.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 lam=4.0 | **pass** |
| G0 | shakespeare | DP=3.004 unigram=4.777 lam=0.0 | **pass** |
| G1 | shakespeare | DP=3.004 gzip=3.218 lam=0.0 | **pass** |
| G0b | shakespeare | charged DP=3.345 gzip=3.218 (model=0.341 bpb at 1.12 MB) | **fail** |
| G4 | shakespeare | oracle=3.069 fixed=4.725 | **pass** |
| G5 | shakespeare | DP=3.004 fixed=4.725 align=0.565/0.210 lam=0.0 | **pass** |
| G5c | shakespeare | DP=3.004 greedy=3.005 lam=0.0 | **pass** |
| G0 | shakespeare | DP=3.004 unigram=4.777 lam=1.0 | **pass** |
| G1 | shakespeare | DP=3.004 gzip=3.218 lam=1.0 | **pass** |
| G0b | shakespeare | charged DP=3.345 gzip=3.218 (model=0.341 bpb at 1.12 MB) | **fail** |
| G4 | shakespeare | oracle=3.069 fixed=4.725 | **pass** |
| G5 | shakespeare | DP=3.004 fixed=4.725 align=0.565/0.210 lam=1.0 | **pass** |
| G5c | shakespeare | DP=3.004 greedy=3.005 lam=1.0 | **pass** |
| G0 | shakespeare | DP=3.004 unigram=4.777 lam=2.0 | **pass** |
| G1 | shakespeare | DP=3.004 gzip=3.218 lam=2.0 | **pass** |
| G0b | shakespeare | charged DP=3.345 gzip=3.218 (model=0.341 bpb at 1.12 MB) | **fail** |
| G4 | shakespeare | oracle=3.069 fixed=4.725 | **pass** |
| G5 | shakespeare | DP=3.004 fixed=4.725 align=0.565/0.210 lam=2.0 | **pass** |
| G5c | shakespeare | DP=3.004 greedy=3.005 lam=2.0 | **pass** |
| G0 | shakespeare | DP=3.004 unigram=4.777 lam=4.0 | **pass** |
| G1 | shakespeare | DP=3.004 gzip=3.218 lam=4.0 | **pass** |
| G0b | shakespeare | charged DP=3.345 gzip=3.218 (model=0.341 bpb at 1.12 MB) | **fail** |
| G4 | shakespeare | oracle=3.069 fixed=4.725 | **pass** |
| G5 | shakespeare | DP=3.004 fixed=4.725 align=0.565/0.210 lam=4.0 | **pass** |
| G5c | shakespeare | DP=3.004 greedy=3.005 lam=4.0 | **pass** |

