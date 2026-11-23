# H-NCM v6 - report

format: 6  cfg: `cd8d9e84512ec6e6a50e0f4812602320608b733089b126de43b9a2be3acd42d1`  device: cuda  torch: 2.15.0
corpora: {'synseg': 'effd9e48ece4f16c5a3532812f6cbe77aec43709b585107f0886ca92300fbe79'}

## Runs

| claim | corpus | seed | arm | lambda | total | recon | code | boundary | align | util | H(code) | calls | wall |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| health | synseg | 0 | abs-pos | 0.0 | 7.245 | 3.882 | 3.364 | 0.000 | 0.221 | 1.000 | 1.365 | 2 | 0.0002 |
| health | synseg | 0 | constant-code | 0.0 | 4.641 | 4.641 | 0.000 | 0.000 | 0.221 | 0.000 | 0.000 | 2 | 0.0004 |
| health | synseg | 0 | no-quant | 0.0 | 3.783 | 3.655 | 0.128 | 0.000 | 0.221 | 0.000 | 0.000 | 2 | 0.0002 |
| health | synseg | 0 | rel-pos | 0.0 | 7.632 | 4.206 | 3.426 | 0.000 | 0.221 | 1.000 | 1.362 | 2 | 0.0003 |
| compression | synseg | None | unigram | None | 4.637 | 4.637 | 0.000 | 0.000 |  |  |  | 1 | 0.0000 |

## Gates

| gate | claim | value | threshold | status |
|---|---|---|---|---|
| G0 | codec health synseg | util=1.000 delta=0.435 H=1.362 total=7.632 uni=4.637 | util>=0.2, delta>=0.1, H>=2.0, total<unigram | **fail** |
| G6 | leak synseg | delta_recon(abs-rel)=-0.325 | <=0.05 | **pass** |
| G9 | statistics synseg | seeds=1 | 3 seeds | **warn** |

## Verdicts

- **P1: not supported**
- **P2: not supported**
- **P3: not supported**

## Decision

G0 fails: the codec is unhealthy. Fix the quantizer/prior/decoder before the DP. Do not interpret structure results.
A negative result with clean controls is publishable.
