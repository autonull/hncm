# H-NCM v6 - report

format: 6  cfg: `dbe8d3867219d926e4fbbc06f6eed225da7466838a784d9897d2aae3e486d201`  device: cuda  torch: 2.15.0
corpora: {'synseg': 'effd9e48ece4f16c5a3532812f6cbe77aec43709b585107f0886ca92300fbe79', 'syngrid': '24ae10a789f2360b0d1ddfcf7bc2b148016035679f78ff2d94a9ac73221259b1'}

## Runs

| claim | corpus | seed | arm | lambda | total | recon | code | boundary | align | util | H(code) | calls | wall |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| health | syngrid | 0 | abs-pos | 0.0 | 7.170 | 3.488 | 3.683 | 0.000 | 1.000 | 1.000 | 671.012 | 2 | 0.0005 |
| health | syngrid | 1 | abs-pos | 0.0 | 6.060 | 3.537 | 2.523 | 0.000 | 1.000 | 0.969 | 664.174 | 2 | 0.0005 |
| health | syngrid | 2 | abs-pos | 0.0 | 6.028 | 3.490 | 2.538 | 0.000 | 1.000 | 0.969 | 674.061 | 2 | 0.0005 |
| health | syngrid | 0 | constant-code | 0.0 | 3.910 | 3.910 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0005 |
| health | syngrid | 1 | constant-code | 0.0 | 3.913 | 3.913 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0005 |
| health | syngrid | 2 | constant-code | 0.0 | 3.919 | 3.919 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0005 |
| health | syngrid | 0 | rel-pos | 0.0 | 6.297 | 2.976 | 3.321 | 0.000 | 1.000 | 1.000 | 693.941 | 2 | 0.0006 |
| health | syngrid | 1 | rel-pos | 0.0 | 5.564 | 2.895 | 2.669 | 0.000 | 1.000 | 1.000 | 697.838 | 2 | 0.0005 |
| health | syngrid | 2 | rel-pos | 0.0 | 5.739 | 2.893 | 2.846 | 0.000 | 1.000 | 1.000 | 689.004 | 2 | 0.0005 |
| compression | syngrid | None | unigram | None | 3.908 | 3.908 | 0.000 | 0.000 |  |  |  | 1 | 0.0000 |
| health | synseg | 0 | abs-pos | 0.0 | 6.848 | 3.793 | 3.055 | 0.000 | 0.216 | 1.000 | 693.347 | 2 | 0.0004 |
| health | synseg | 1 | abs-pos | 0.0 | 6.747 | 4.240 | 2.507 | 0.000 | 0.216 | 1.000 | 654.634 | 2 | 0.0004 |
| health | synseg | 2 | abs-pos | 0.0 | 7.287 | 4.251 | 3.036 | 0.000 | 0.216 | 1.000 | 682.504 | 2 | 0.0006 |
| health | synseg | 0 | constant-code | 0.0 | 4.644 | 4.644 | 0.000 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0006 |
| health | synseg | 1 | constant-code | 0.0 | 4.643 | 4.643 | 0.000 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0005 |
| health | synseg | 2 | constant-code | 0.0 | 4.646 | 4.646 | 0.000 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0005 |
| health | synseg | 0 | rel-pos | 0.0 | 6.855 | 3.754 | 3.101 | 0.000 | 0.216 | 1.000 | 689.843 | 2 | 0.0008 |
| health | synseg | 1 | rel-pos | 0.0 | 6.001 | 3.748 | 2.253 | 0.000 | 0.216 | 1.000 | 678.930 | 2 | 0.0004 |
| health | synseg | 2 | rel-pos | 0.0 | 6.499 | 3.747 | 2.751 | 0.000 | 0.216 | 1.000 | 687.284 | 2 | 0.0006 |
| compression | synseg | None | unigram | None | 4.637 | 4.637 | 0.000 | 0.000 |  |  |  | 1 | 0.0000 |

## Gates

| gate | claim | value | threshold | status |
|---|---|---|---|---|
| G0 | codec health synseg | util=1.000 delta=0.894 H=685.352 total=6.452 uni=4.637 | util>=0.2, delta>=0.1, H>=2.0, total<unigram | **fail** |
| G6 | leak synseg | delta_recon(rel-abs)=-0.345 | <=0.05 | **pass** |
| G9 | statistics synseg | seeds=3 | 3 seeds | **pass** |
| G0 | codec health syngrid | util=1.000 delta=0.993 H=693.594 total=5.867 uni=3.908 | util>=0.2, delta>=0.1, H>=2.0, total<unigram | **fail** |
| G6 | leak syngrid | delta_recon(rel-abs)=-0.584 | <=0.05 | **pass** |
| G9 | statistics syngrid | seeds=3 | 3 seeds | **pass** |

## Verdicts

- **P1: not supported**
- **P2: not supported**
- **P3: not supported**

## Decision

G0 fails: the codec is unhealthy. Fix the quantizer/prior/decoder before the DP. Do not interpret structure results.
A negative result with clean controls is publishable.
