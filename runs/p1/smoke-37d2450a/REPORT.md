# H-NCM v6 - report

format: 6  cfg: `37d2450a2c0e04b0ae3fc96abef1f6d49b6e3fccc85a6f581863777fe73dafa8`  device: cuda  torch: 2.15.0
corpora: {'synseg': 'effd9e48ece4f16c5a3532812f6cbe77aec43709b585107f0886ca92300fbe79'}

## Runs

| claim | corpus | seed | arm | lambda | total | recon | code | boundary | align | util | H(code) | calls | wall |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| structure | synseg | 0 | BPE | 0.0 | 8.919 | 4.719 | 3.433 | 0.767 | 0.322 | 1.000 | 1.843 | 2 | 0.0004 |
| structure | synseg | 0 | BPE | 2.0 | 8.919 | 4.719 | 3.433 | 0.767 | 0.322 | 1.000 | 1.843 | 2 | 0.0005 |
| structure | synseg | 0 | DP | 0.0 | 6.384 | 4.763 | 1.620 | 0.001 | 0.244 | 1.000 | 0.486 | 2 | 0.0035 |
| structure | synseg | 0 | DP | 2.0 | 6.384 | 4.763 | 1.620 | 0.001 | 0.244 | 1.000 | 0.486 | 2 | 0.0034 |
| structure | synseg | 0 | IsoWarp | 0.0 | 8.786 | 4.696 | 4.089 | 0.000 | 0.270 | 1.000 | 1.780 | 2 | 0.0007 |
| structure | synseg | 0 | IsoWarp | 2.0 | 8.786 | 4.696 | 4.089 | 0.000 | 0.270 | 1.000 | 1.780 | 2 | 0.0004 |
| structure | synseg | 0 | fixed | 0.0 | 6.728 | 4.754 | 1.974 | 0.000 | 0.236 | 1.000 | 0.871 | 2 | 0.0009 |
| structure | synseg | 0 | fixed | 2.0 | 6.728 | 4.754 | 1.974 | 0.000 | 0.236 | 1.000 | 0.871 | 2 | 0.0003 |
| structure | synseg | 0 | oracle | 0.0 | 6.494 | 4.792 | 1.702 | 0.000 | 1.000 | 1.000 | 0.607 | 2 | 0.0004 |
| structure | synseg | 0 | oracle | 2.0 | 6.494 | 4.792 | 1.702 | 0.000 | 1.000 | 1.000 | 0.607 | 2 | 0.0004 |
| structure | synseg | 0 | oracle-costed | 0.0 | 7.117 | 4.792 | 1.702 | 0.624 | 1.000 | 1.000 | 0.607 | 2 | 0.0004 |
| structure | synseg | 0 | oracle-costed | 2.0 | 7.117 | 4.792 | 1.702 | 0.624 | 1.000 | 1.000 | 0.607 | 2 | 0.0004 |
| structure | synseg | 0 | random | 0.0 | 7.704 | 4.743 | 2.096 | 0.865 | 0.269 | 1.000 | 1.035 | 2 | 0.0006 |
| structure | synseg | 0 | random | 2.0 | 7.660 | 4.745 | 2.059 | 0.856 | 0.213 | 1.000 | 1.055 | 2 | 0.0006 |
| compression | synseg | None | unigram | None | 4.648 | 4.648 | 0.000 | 0.000 |  |  |  | 1 | 0.0000 |

## Gates

| gate | claim | value | threshold | status |
|---|---|---|---|---|
| G4 | oracle ceiling synseg lam=0.0 | I_oracle=0.234 CI=[0.198,0.276] | I>0, CI low>0 | **pass** |
| G5 | DP deployable synseg lam=0.0 | I_DP=0.344 I_oracle=0.234 align=0.244/0.236 | I_DP>=max(.5 I_oracle,2SEM); align+0.05 | **fail** |
| G5b | DP between synseg lam=0.0 | oracle=6.494 DP=6.384 fixed=6.728 | oracle<DP<fixed | **fail** |
| G5c | DP vs BPE synseg lam=0.0 | DP=6.384 BPE=8.919 | DP<=BPE | **pass** |
| G5d | DP vs IsoWarp synseg lam=0.0 | DP=6.384 IsoWarp=8.786 | DP<=IsoWarp | **pass** |
| G4 | oracle ceiling synseg lam=2.0 | I_oracle=0.234 CI=[0.198,0.276] | I>0, CI low>0 | **pass** |
| G5 | DP deployable synseg lam=2.0 | I_DP=0.344 I_oracle=0.234 align=0.244/0.236 | I_DP>=max(.5 I_oracle,2SEM); align+0.05 | **fail** |
| G5b | DP between synseg lam=2.0 | oracle=6.494 DP=6.384 fixed=6.728 | oracle<DP<fixed | **fail** |
| G5c | DP vs BPE synseg lam=2.0 | DP=6.384 BPE=8.919 | DP<=BPE | **pass** |
| G5d | DP vs IsoWarp synseg lam=2.0 | DP=6.384 IsoWarp=8.786 | DP<=IsoWarp | **pass** |
| G9 | statistics synseg | seeds=1 | 3 seeds | **warn** |

## Verdicts

- **P1: not supported**
- **P2: not supported**
- **P3: not supported**

## Decision

G4 passes but G5 fails: the oracle helps, the paid DP does not. Report the ceiling and the negative.
A negative result with clean controls is publishable.
