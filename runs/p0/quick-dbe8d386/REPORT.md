# H-NCM v6 - report

format: 6  cfg: `dbe8d3867219d926e4fbbc06f6eed225da7466838a784d9897d2aae3e486d201`  device: cuda  torch: 2.15.0
corpora: {'synseg': 'effd9e48ece4f16c5a3532812f6cbe77aec43709b585107f0886ca92300fbe79', 'syngrid': '24ae10a789f2360b0d1ddfcf7bc2b148016035679f78ff2d94a9ac73221259b1'}

## Runs

| claim | corpus | seed | arm | lambda | total | recon | code | boundary | align | util | H(code) | calls | wall |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| health | syngrid | 0 | abs-pos | 0.0 | 7.145 | 3.526 | 3.618 | 0.000 | 1.000 | 1.000 | 1.300 | 2 | 0.0003 |
| health | syngrid | 1 | abs-pos | 0.0 | 6.065 | 3.539 | 2.526 | 0.000 | 1.000 | 1.000 | 1.292 | 2 | 0.0002 |
| health | syngrid | 2 | abs-pos | 0.0 | 7.065 | 3.622 | 3.443 | 0.000 | 1.000 | 0.938 | 1.228 | 2 | 0.0002 |
| health | syngrid | 0 | constant-code | 0.0 | 3.910 | 3.910 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0005 |
| health | syngrid | 1 | constant-code | 0.0 | 3.913 | 3.913 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0003 |
| health | syngrid | 2 | constant-code | 0.0 | 3.919 | 3.919 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 2 | 0.0002 |
| health | syngrid | 0 | rel-pos | 0.0 | 6.315 | 3.010 | 3.305 | 0.000 | 1.000 | 1.000 | 1.351 | 2 | 0.0004 |
| health | syngrid | 1 | rel-pos | 0.0 | 5.558 | 2.896 | 2.662 | 0.000 | 1.000 | 1.000 | 1.362 | 2 | 0.0003 |
| health | syngrid | 2 | rel-pos | 0.0 | 5.750 | 2.889 | 2.861 | 0.000 | 1.000 | 1.000 | 1.343 | 2 | 0.0003 |
| compression | syngrid | None | unigram | None | 3.908 | 3.908 | 0.000 | 0.000 |  |  |  | 1 | 0.0000 |
| health | synseg | 0 | abs-pos | 0.0 | 6.929 | 3.798 | 3.131 | 0.000 | 0.216 | 1.000 | 1.343 | 2 | 0.0004 |
| health | synseg | 1 | abs-pos | 0.0 | 7.013 | 4.282 | 2.730 | 0.000 | 0.216 | 1.000 | 1.303 | 2 | 0.0006 |
| health | synseg | 2 | abs-pos | 0.0 | 7.024 | 4.162 | 2.862 | 0.000 | 0.216 | 1.000 | 1.351 | 2 | 0.0002 |
| health | synseg | 0 | constant-code | 0.0 | 4.644 | 4.644 | 0.000 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0006 |
| health | synseg | 1 | constant-code | 0.0 | 4.643 | 4.643 | 0.000 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0004 |
| health | synseg | 2 | constant-code | 0.0 | 4.646 | 4.646 | 0.000 | 0.000 | 0.216 | 0.000 | 0.000 | 2 | 0.0003 |
| health | synseg | 0 | rel-pos | 0.0 | 6.834 | 3.755 | 3.080 | 0.000 | 0.216 | 1.000 | 1.351 | 2 | 0.0004 |
| health | synseg | 1 | rel-pos | 0.0 | 5.998 | 3.749 | 2.249 | 0.000 | 0.216 | 1.000 | 1.328 | 2 | 0.0005 |
| health | synseg | 2 | rel-pos | 0.0 | 6.498 | 3.747 | 2.751 | 0.000 | 0.216 | 1.000 | 1.341 | 2 | 0.0002 |
| compression | synseg | None | unigram | None | 4.637 | 4.637 | 0.000 | 0.000 |  |  |  | 1 | 0.0000 |

## Gates

| gate | claim | value | threshold | status |
|---|---|---|---|---|
| G0 | codec health synseg | util=1.000 delta=0.894 H=1.340 total=6.443 uni=4.637 | util>=0.2, delta>=0.1, H>=2.0, total<unigram | **fail** |
| G6 | leak synseg | delta_recon(rel-abs)=-0.331 | <=0.05 | **pass** |
| G9 | statistics synseg | seeds=3 | 3 seeds | **pass** |
| G0 | codec health syngrid | util=1.000 delta=0.982 H=1.352 total=5.874 uni=3.908 | util>=0.2, delta>=0.1, H>=2.0, total<unigram | **fail** |
| G6 | leak syngrid | delta_recon(rel-abs)=-0.631 | <=0.05 | **pass** |
| G9 | statistics syngrid | seeds=3 | 3 seeds | **pass** |

## Verdicts

- **P1: not supported**
- **P2: not supported**
- **P3: not supported**

## Decision

G0 fails: the codec is unhealthy. Fix the quantizer/prior/decoder before the DP. Do not interpret structure results.
A negative result with clean controls is publishable.
