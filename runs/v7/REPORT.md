# H-NCM v7 - solved tiling over a free span dictionary

## synseg

dictionary: 895 spans (<= 16 bytes), free; byte unigram: 4.640 bpb; gzip: 2.225 bpb

| arm | lambda | total | tokens | bytes/token | align |
|---|---:|---:|---:|---:|---:|
| DP | 0.0 | 1.285 | 17741 | 6.76 | 1.000 |
| DP | 1.0 | 1.285 | 17741 | 6.76 | 1.000 |
| DP | 2.0 | 1.285 | 17741 | 6.76 | 1.000 |
| DP | 4.0 | 1.285 | 17741 | 6.76 | 1.000 |
| fixed16 | 0.0 | 4.636 | 7500 | 16.00 | 0.090 |
| fixed4 | 0.0 | 4.591 | 30000 | 4.00 | 0.189 |
| fixed8 | 0.0 | 4.557 | 15000 | 8.00 | 0.137 |
| greedy | 0.0 | 1.285 | 17741 | 6.76 | 1.000 |
| greedy | 1.0 | 1.285 | 17741 | 6.76 | 1.000 |
| greedy | 2.0 | 1.285 | 17741 | 6.76 | 1.000 |
| greedy | 4.0 | 1.285 | 17741 | 6.76 | 1.000 |
| oracle | 0.0 | 1.285 | 17737 | 6.77 | 1.000 |
| random | 0.0 | 4.616 | 19275 | 6.23 | 0.150 |
| random | 1.0 | 4.616 | 19275 | 6.23 | 0.150 |
| random | 2.0 | 4.616 | 19275 | 6.23 | 0.150 |
| random | 4.0 | 4.616 | 19275 | 6.23 | 0.150 |

## syngrid

dictionary: 558 spans (<= 16 bytes), free; byte unigram: 3.911 bpb; gzip: 2.886 bpb

| arm | lambda | total | tokens | bytes/token | align |
|---|---:|---:|---:|---:|---:|
| DP | 0.0 | 2.051 | 30004 | 4.00 | 1.000 |
| DP | 1.0 | 2.051 | 30004 | 4.00 | 1.000 |
| DP | 2.0 | 2.051 | 30004 | 4.00 | 1.000 |
| DP | 4.0 | 2.051 | 30004 | 4.00 | 1.000 |
| fixed16 | 0.0 | 3.906 | 7501 | 16.00 | 0.400 |
| fixed4 | 0.0 | 2.051 | 30004 | 4.00 | 1.000 |
| fixed8 | 0.0 | 3.906 | 15002 | 8.00 | 0.667 |
| greedy | 0.0 | 2.051 | 30004 | 4.00 | 1.000 |
| greedy | 1.0 | 2.051 | 30004 | 4.00 | 1.000 |
| greedy | 2.0 | 2.051 | 30004 | 4.00 | 1.000 |
| greedy | 4.0 | 2.051 | 30004 | 4.00 | 1.000 |
| oracle | 0.0 | 2.051 | 30004 | 4.00 | 1.000 |
| random | 0.0 | 3.847 | 19276 | 6.23 | 0.190 |
| random | 1.0 | 3.847 | 19276 | 6.23 | 0.190 |
| random | 2.0 | 3.847 | 19276 | 6.23 | 0.190 |
| random | 4.0 | 3.847 | 19276 | 6.23 | 0.190 |

## shakespeare

dictionary: 29148 spans (<= 16 bytes), free; byte unigram: 4.777 bpb; gzip: 3.218 bpb

| arm | lambda | total | tokens | bytes/token | align |
|---|---:|---:|---:|---:|---:|
| DP | 0.0 | 2.709 | 38496 | 2.90 | 0.687 |
| DP | 1.0 | 2.709 | 38496 | 2.90 | 0.687 |
| DP | 2.0 | 2.709 | 38496 | 2.90 | 0.687 |
| DP | 4.0 | 2.709 | 38496 | 2.90 | 0.687 |
| fixed16 | 0.0 | 4.829 | 6971 | 16.00 | 0.092 |
| fixed4 | 0.0 | 4.728 | 27884 | 4.00 | 0.210 |
| fixed8 | 0.0 | 4.805 | 13942 | 8.00 | 0.148 |
| greedy | 0.0 | 2.738 | 37803 | 2.95 | 0.695 |
| greedy | 1.0 | 2.738 | 37803 | 2.95 | 0.695 |
| greedy | 2.0 | 2.738 | 37803 | 2.95 | 0.695 |
| greedy | 4.0 | 2.738 | 37803 | 2.95 | 0.695 |
| oracle | 0.0 | 2.776 | 20153 | 5.53 | 1.000 |
| random | 0.0 | 4.817 | 17935 | 6.22 | 0.169 |
| random | 1.0 | 4.817 | 17935 | 6.22 | 0.169 |
| random | 2.0 | 4.817 | 17935 | 6.22 | 0.169 |
| random | 4.0 | 4.817 | 17935 | 6.22 | 0.169 |

## Gates

| gate | corpus | value | status |
|---|---|---|---|
| G0 | synseg | DP=1.285 unigram=4.640 lam=0.0 | **pass** |
| G4 | synseg | oracle=1.285 fixed=4.591 | **pass** |
| G5 | synseg | DP=1.285 fixed=4.591 align=1.000/0.189 lam=0.0 | **pass** |
| G5c | synseg | DP=1.285 greedy=1.285 lam=0.0 | **pass** |
| G0 | synseg | DP=1.285 unigram=4.640 lam=1.0 | **pass** |
| G4 | synseg | oracle=1.285 fixed=4.591 | **pass** |
| G5 | synseg | DP=1.285 fixed=4.591 align=1.000/0.189 lam=1.0 | **pass** |
| G5c | synseg | DP=1.285 greedy=1.285 lam=1.0 | **pass** |
| G0 | synseg | DP=1.285 unigram=4.640 lam=2.0 | **pass** |
| G4 | synseg | oracle=1.285 fixed=4.591 | **pass** |
| G5 | synseg | DP=1.285 fixed=4.591 align=1.000/0.189 lam=2.0 | **pass** |
| G5c | synseg | DP=1.285 greedy=1.285 lam=2.0 | **pass** |
| G0 | synseg | DP=1.285 unigram=4.640 lam=4.0 | **pass** |
| G4 | synseg | oracle=1.285 fixed=4.591 | **pass** |
| G5 | synseg | DP=1.285 fixed=4.591 align=1.000/0.189 lam=4.0 | **pass** |
| G5c | synseg | DP=1.285 greedy=1.285 lam=4.0 | **pass** |
| G0 | syngrid | DP=2.051 unigram=3.911 lam=0.0 | **pass** |
| G4 | syngrid | oracle=2.051 fixed=2.051 | **fail** |
| G5 | syngrid | DP=2.051 fixed=2.051 align=1.000/1.000 lam=0.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 lam=0.0 | **pass** |
| G0 | syngrid | DP=2.051 unigram=3.911 lam=1.0 | **pass** |
| G4 | syngrid | oracle=2.051 fixed=2.051 | **fail** |
| G5 | syngrid | DP=2.051 fixed=2.051 align=1.000/1.000 lam=1.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 lam=1.0 | **pass** |
| G0 | syngrid | DP=2.051 unigram=3.911 lam=2.0 | **pass** |
| G4 | syngrid | oracle=2.051 fixed=2.051 | **fail** |
| G5 | syngrid | DP=2.051 fixed=2.051 align=1.000/1.000 lam=2.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 lam=2.0 | **pass** |
| G0 | syngrid | DP=2.051 unigram=3.911 lam=4.0 | **pass** |
| G4 | syngrid | oracle=2.051 fixed=2.051 | **fail** |
| G5 | syngrid | DP=2.051 fixed=2.051 align=1.000/1.000 lam=4.0 | **fail** |
| G5c | syngrid | DP=2.051 greedy=2.051 lam=4.0 | **pass** |
| G0 | shakespeare | DP=2.709 unigram=4.777 lam=0.0 | **pass** |
| G4 | shakespeare | oracle=2.776 fixed=4.728 | **pass** |
| G5 | shakespeare | DP=2.709 fixed=4.728 align=0.687/0.210 lam=0.0 | **pass** |
| G5c | shakespeare | DP=2.709 greedy=2.738 lam=0.0 | **pass** |
| G0 | shakespeare | DP=2.709 unigram=4.777 lam=1.0 | **pass** |
| G4 | shakespeare | oracle=2.776 fixed=4.728 | **pass** |
| G5 | shakespeare | DP=2.709 fixed=4.728 align=0.687/0.210 lam=1.0 | **pass** |
| G5c | shakespeare | DP=2.709 greedy=2.738 lam=1.0 | **pass** |
| G0 | shakespeare | DP=2.709 unigram=4.777 lam=2.0 | **pass** |
| G4 | shakespeare | oracle=2.776 fixed=4.728 | **pass** |
| G5 | shakespeare | DP=2.709 fixed=4.728 align=0.687/0.210 lam=2.0 | **pass** |
| G5c | shakespeare | DP=2.709 greedy=2.738 lam=2.0 | **pass** |
| G0 | shakespeare | DP=2.709 unigram=4.777 lam=4.0 | **pass** |
| G4 | shakespeare | oracle=2.776 fixed=4.728 | **pass** |
| G5 | shakespeare | DP=2.709 fixed=4.728 align=0.687/0.210 lam=4.0 | **pass** |
| G5c | shakespeare | DP=2.709 greedy=2.738 lam=4.0 | **pass** |

