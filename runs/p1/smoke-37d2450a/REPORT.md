# H-NCM v6 - report

format: 6  cfg: `37d2450a2c0e04b0ae3fc96abef1f6d49b6e3fccc85a6f581863777fe73dafa8`  device: cuda  torch: 2.15.0
corpora: {'synseg': 'effd9e48ece4f16c5a3532812f6cbe77aec43709b585107f0886ca92300fbe79'}

## Runs

| claim | corpus | seed | arm | lambda | total | recon | code | boundary | align | util | H(code) | calls | wall |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| structure | synseg | 0 | BPE | 0.0 | 7.764 | 4.170 | 2.827 | 0.767 | 0.322 | 1.000 | 239.917 | 2 | 0.0008 |
| structure | synseg | 0 | BPE | 2.0 | 7.764 | 4.170 | 2.827 | 0.767 | 0.322 | 1.000 | 239.917 | 2 | 0.0008 |
| structure | synseg | 0 | DP | 0.0 | 6.244 | 4.639 | 1.601 | 0.003 | 0.244 | 1.000 | 67.227 | 2 | 0.0052 |
| structure | synseg | 0 | DP | 2.0 | 6.244 | 4.639 | 1.601 | 0.003 | 0.244 | 1.000 | 67.227 | 2 | 0.0056 |
| structure | synseg | 0 | IsoWarp | 0.0 | 7.287 | 4.047 | 3.240 | 0.000 | 0.270 | 1.000 | 209.759 | 2 | 0.0014 |
| structure | synseg | 0 | IsoWarp | 2.0 | 7.287 | 4.047 | 3.240 | 0.000 | 0.270 | 1.000 | 209.759 | 2 | 0.0007 |
| structure | synseg | 0 | fixed | 0.0 | 6.397 | 4.544 | 1.853 | 0.000 | 0.236 | 1.000 | 123.170 | 2 | 0.0018 |
| structure | synseg | 0 | fixed | 2.0 | 6.397 | 4.544 | 1.853 | 0.000 | 0.236 | 1.000 | 123.170 | 2 | 0.0006 |
| structure | synseg | 0 | oracle | 0.0 | 6.275 | 4.612 | 1.663 | 0.000 | 1.000 | 1.000 | 85.570 | 2 | 0.0009 |
| structure | synseg | 0 | oracle | 2.0 | 6.275 | 4.612 | 1.663 | 0.000 | 1.000 | 1.000 | 85.570 | 2 | 0.0009 |
| structure | synseg | 0 | oracle-costed | 0.0 | 6.899 | 4.612 | 1.663 | 0.624 | 1.000 | 1.000 | 85.570 | 2 | 0.0009 |
| structure | synseg | 0 | oracle-costed | 2.0 | 6.899 | 4.612 | 1.663 | 0.624 | 1.000 | 1.000 | 85.570 | 2 | 0.0009 |
| structure | synseg | 0 | random | 0.0 | 7.269 | 4.502 | 1.903 | 0.865 | 0.269 | 1.000 | 144.948 | 2 | 0.0009 |
| structure | synseg | 0 | random | 2.0 | 7.287 | 4.526 | 1.905 | 0.856 | 0.213 | 1.000 | 144.086 | 2 | 0.0011 |
| compression | synseg | None | unigram | None | 4.648 | 4.648 | 0.000 | 0.000 |  |  |  | 1 | 0.0000 |

## Gates

| gate | claim | value | threshold | status |
|---|---|---|---|---|
| G4 | oracle ceiling synseg lam=0.0 | I_oracle=0.122 CI=[0.085,0.158] | I>0, CI low>0 | **pass** |
| G5 | DP deployable synseg lam=0.0 | I_DP=0.153 I_oracle=0.122 align=0.244/0.236 | I_DP>=max(.5 I_oracle,2SEM); align+0.05 | **fail** |
| G5b | DP between synseg lam=0.0 | oracle=6.275 DP=6.244 fixed=6.397 | oracle<DP<fixed | **fail** |
| G5c | DP vs BPE synseg lam=0.0 | DP=6.244 BPE=7.764 | DP<=BPE | **pass** |
| G5d | DP vs IsoWarp synseg lam=0.0 | DP=6.244 IsoWarp=7.287 | DP<=IsoWarp | **pass** |
| G4 | oracle ceiling synseg lam=2.0 | I_oracle=0.122 CI=[0.085,0.158] | I>0, CI low>0 | **pass** |
| G5 | DP deployable synseg lam=2.0 | I_DP=0.153 I_oracle=0.122 align=0.244/0.236 | I_DP>=max(.5 I_oracle,2SEM); align+0.05 | **fail** |
| G5b | DP between synseg lam=2.0 | oracle=6.275 DP=6.244 fixed=6.397 | oracle<DP<fixed | **fail** |
| G5c | DP vs BPE synseg lam=2.0 | DP=6.244 BPE=7.764 | DP<=BPE | **pass** |
| G5d | DP vs IsoWarp synseg lam=2.0 | DP=6.244 IsoWarp=7.287 | DP<=IsoWarp | **pass** |
| G9 | statistics synseg | seeds=1 | 3 seeds | **warn** |

## Verdicts

- **P1: not supported**
- **P2: not supported**
- **P3: not supported**

## Decision

G4 passes but G5 fails: the oracle helps, the paid DP does not. Report the ceiling and the negative.
A negative result with clean controls is publishable.
