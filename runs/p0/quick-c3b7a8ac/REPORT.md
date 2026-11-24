# H-NCM v6 - report

format: 7  cfg: `c3b7a8ac1371b40304e9fec0b721d41055b83d0dd7d4a702f23b94229041881d`  device: cuda  torch: 2.15.0
corpora: {'synseg': 'effd9e48ece4f16c5a3532812f6cbe77aec43709b585107f0886ca92300fbe79', 'syngrid': '24ae10a789f2360b0d1ddfcf7bc2b148016035679f78ff2d94a9ac73221259b1'}

## Runs

| claim | corpus | seed | arm | lambda | total | recon | code | boundary | align | util | H(code) | calls | wall |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| health | syngrid | 0 | abs-pos | 0.0 | 5.983 | 2.869 | 3.114 | 0.000 | 1.000 | 1.000 | 379.525 | 2 | 0.0002 |
| health | syngrid | 1 | abs-pos | 0.0 | 5.943 | 2.808 | 3.135 | 0.000 | 1.000 | 1.000 | 378.603 | 2 | 0.0002 |
| health | syngrid | 2 | abs-pos | 0.0 | 5.947 | 2.840 | 3.107 | 0.000 | 1.000 | 1.000 | 377.803 | 2 | 0.0002 |
| health | syngrid | 0 | constant-code | 0.0 | 3.907 | 3.907 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0001 |
| health | syngrid | 1 | constant-code | 0.0 | 3.910 | 3.910 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0001 |
| health | syngrid | 2 | constant-code | 0.0 | 3.911 | 3.911 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0001 |
| health | syngrid | 0 | fixed16 | 0.0 | 4.452 | 3.639 | 0.813 | 0.000 | 0.400 | 1.000 | 85.785 | 2 | 0.0002 |
| health | syngrid | 1 | fixed16 | 0.0 | 4.505 | 3.680 | 0.825 | 0.000 | 0.400 | 1.000 | 86.167 | 2 | 0.0002 |
| health | syngrid | 2 | fixed16 | 0.0 | 4.474 | 3.658 | 0.816 | 0.000 | 0.400 | 1.000 | 86.248 | 2 | 0.0002 |
| health | syngrid | 0 | fixed4 | 0.0 | 5.970 | 2.846 | 3.125 | 0.000 | 1.000 | 1.000 | 378.523 | 2 | 0.0002 |
| health | syngrid | 1 | fixed4 | 0.0 | 6.003 | 2.922 | 3.081 | 0.000 | 1.000 | 1.000 | 371.900 | 2 | 0.0002 |
| health | syngrid | 2 | fixed4 | 0.0 | 5.951 | 2.820 | 3.130 | 0.000 | 1.000 | 1.000 | 380.575 | 2 | 0.0002 |
| health | syngrid | 0 | fixed8 | 0.0 | 4.990 | 3.390 | 1.600 | 0.000 | 0.667 | 1.000 | 185.201 | 2 | 0.0002 |
| health | syngrid | 1 | fixed8 | 0.0 | 5.042 | 3.437 | 1.604 | 0.000 | 0.667 | 1.000 | 184.459 | 2 | 0.0002 |
| health | syngrid | 2 | fixed8 | 0.0 | 5.018 | 3.406 | 1.613 | 0.000 | 0.667 | 1.000 | 186.652 | 2 | 0.0002 |
| health | syngrid | 0 | no-quant | 0.0 | 2.785 | 2.637 | 0.148 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0001 |
| health | syngrid | 1 | no-quant | 0.0 | 2.841 | 2.694 | 0.147 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0001 |
| health | syngrid | 2 | no-quant | 0.0 | 2.811 | 2.662 | 0.149 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0001 |
| health | syngrid | 0 | rel-pos | 0.0 | 5.970 | 2.846 | 3.125 | 0.000 | 1.000 | 1.000 | 378.523 | 2 | 0.0002 |
| health | syngrid | 1 | rel-pos | 0.0 | 6.003 | 2.922 | 3.081 | 0.000 | 1.000 | 1.000 | 371.900 | 2 | 0.0002 |
| health | syngrid | 2 | rel-pos | 0.0 | 5.951 | 2.820 | 3.130 | 0.000 | 1.000 | 1.000 | 380.575 | 2 | 0.0002 |
| compression | syngrid | None | unigram | None | 3.908 | 3.908 | 0.000 | 0.000 |  |  |  | 1 | 0.0000 |
| health | synseg | 0 | abs-pos | 0.0 | 6.849 | 3.758 | 3.090 | 0.000 | 0.216 | 1.000 | 378.014 | 2 | 0.0002 |
| health | synseg | 1 | abs-pos | 0.0 | 6.832 | 3.741 | 3.091 | 0.000 | 0.216 | 1.000 | 377.463 | 2 | 0.0002 |
| health | synseg | 2 | abs-pos | 0.0 | 6.851 | 3.776 | 3.075 | 0.000 | 0.216 | 1.000 | 375.127 | 2 | 0.0002 |
| health | synseg | 0 | constant-code | 0.0 | 4.641 | 4.641 | 0.000 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0001 |
| health | synseg | 1 | constant-code | 0.0 | 4.642 | 4.642 | 0.000 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0001 |
| health | synseg | 2 | constant-code | 0.0 | 4.644 | 4.644 | 0.000 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0001 |
| health | synseg | 0 | fixed16 | 0.0 | 5.201 | 4.415 | 0.786 | 0.000 | 0.149 | 1.000 | 83.633 | 2 | 0.0002 |
| health | synseg | 1 | fixed16 | 0.0 | 5.240 | 4.432 | 0.808 | 0.000 | 0.149 | 0.969 | 83.573 | 2 | 0.0002 |
| health | synseg | 2 | fixed16 | 0.0 | 5.222 | 4.428 | 0.794 | 0.000 | 0.149 | 1.000 | 84.420 | 2 | 0.0002 |
| health | synseg | 0 | fixed4 | 0.0 | 6.864 | 3.723 | 3.141 | 0.000 | 0.216 | 1.000 | 381.135 | 2 | 0.0002 |
| health | synseg | 1 | fixed4 | 0.0 | 6.911 | 3.779 | 3.132 | 0.000 | 0.216 | 1.000 | 378.813 | 2 | 0.0002 |
| health | synseg | 2 | fixed4 | 0.0 | 6.897 | 3.754 | 3.142 | 0.000 | 0.216 | 1.000 | 382.904 | 2 | 0.0002 |
| health | synseg | 0 | fixed8 | 0.0 | 5.761 | 4.188 | 1.573 | 0.000 | 0.182 | 1.000 | 183.606 | 2 | 0.0002 |
| health | synseg | 1 | fixed8 | 0.0 | 5.808 | 4.216 | 1.592 | 0.000 | 0.182 | 1.000 | 182.771 | 2 | 0.0002 |
| health | synseg | 2 | fixed8 | 0.0 | 5.783 | 4.195 | 1.588 | 0.000 | 0.182 | 1.000 | 185.217 | 2 | 0.0002 |
| health | synseg | 0 | no-quant | 0.0 | 3.736 | 3.587 | 0.149 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0001 |
| health | synseg | 1 | no-quant | 0.0 | 3.782 | 3.634 | 0.148 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0001 |
| health | synseg | 2 | no-quant | 0.0 | 3.756 | 3.607 | 0.149 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0001 |
| health | synseg | 0 | rel-pos | 0.0 | 6.864 | 3.723 | 3.141 | 0.000 | 0.216 | 1.000 | 381.135 | 2 | 0.0002 |
| health | synseg | 1 | rel-pos | 0.0 | 6.911 | 3.779 | 3.132 | 0.000 | 0.216 | 1.000 | 378.813 | 2 | 0.0002 |
| health | synseg | 2 | rel-pos | 0.0 | 6.897 | 3.754 | 3.142 | 0.000 | 0.216 | 1.000 | 382.904 | 2 | 0.0002 |
| compression | synseg | None | unigram | None | 4.637 | 4.637 | 0.000 | 0.000 |  |  |  | 1 | 0.0000 |

## Gates

| gate | claim | value | threshold | status |
|---|---|---|---|---|
| G0 | codec health synseg | util=1.000 delta=0.890 H=380.951 total=6.891 uni=4.637 | util>=0.2, delta>=0.1, H>=2.0, total<unigram | **fail** |
| G6 | leak synseg | delta_recon(rel-abs)=-0.006 | <=0.05 | **pass** |
| G9 | statistics synseg | seeds=3 | 3 seeds | **pass** |
| G0 | codec health syngrid | util=1.000 delta=1.046 H=376.999 total=5.975 uni=3.908 | util>=0.2, delta>=0.1, H>=2.0, total<unigram | **fail** |
| G6 | leak syngrid | delta_recon(rel-abs)=0.024 | <=0.05 | **pass** |
| G9 | statistics syngrid | seeds=3 | 3 seeds | **pass** |

## Verdicts

- **P1: not supported**
- **P2: not supported**
- **P3: not supported**

## Decision

G0 fails: the codec is unhealthy. Fix the quantizer/prior/decoder before the DP. Do not interpret structure results.
A negative result with clean controls is publishable.
