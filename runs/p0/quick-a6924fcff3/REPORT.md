# H-NCM v5 report

preset `quick` phase `0` device `cuda` torch `2.15.0`

started `2026-11-23T19:57:08.168653Z` finished `2026-11-23T20:00:26.215456Z`

cfg fingerprint `a6924fcff327c672`


## Claims

P1 absolute compression; P2 parallelism dividend; P3 boundary mechanism.
All three are reported, including negative results. Thresholds come from `Cfg`.


> Diagnostic only: preset != full. Controls are decisive only on the full preset.


## synseg (sha effd9e48ece4, 1,200,028 bytes)

unigram `4.635` bpb

byte GPT `4.295` bpb


| cell | bound bpb | recon | c1 | bnd | codes | rate | align |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed+ar | 4.637 | 4.637 | 0.000 | 0.000 | 0.000 | 0.000 | 0.145 |
| fixed+cf | 4.638 | 4.637 | 0.000 | 0.000 | 0.001 | 0.000 | 0.145 |
| oracle+ar | 4.594 | 4.202 | 0.000 | 0.392 | 0.000 | 0.000 | 0.579 |
| oracle+cf | 4.607 | 4.202 | 0.000 | 0.403 | 0.002 | 0.000 | 0.579 |
| rate-only+ar | 4.638 | 4.638 | 0.000 | 0.000 | 0.000 | 0.000 | 0.150 |
| rate-only+cf | 4.640 | 4.638 | 0.000 | 0.000 | 0.002 | 0.000 | 0.150 |

Gates: E1 `False`, E2 `False`, E3 `True`, P2b `True` (mean -0.023), G3 `False`, G4 `{'oracle': False, 'rate-only': False}`

total(fixed+CF) `4.638` bpb; unigram `4.635`; byte GPT `4.295`; AR sample `5.125`


### Interaction I (primary k = 1)

| arm | I mean | SEM | CI low | CI high | n |
|---|---:|---:|---:|---:|---:|
| oracle | -0.0001 | nan | -0.0001 | -0.0001 | 1 |
| rate-only | -0.0000 | nan | -0.0000 | -0.0000 | 1 |

### Mechanism verdicts

E4 oracle: `False` (I=-0.0001, CI [-0.0001, -0.0001]).
E4n rate-only null: `False` (I=-0.0000, CI [-0.0000, -0.0000]).
E5 boundary: missing (run phase 2).

### Utility Pareto (sample quality, sequential calls)

| arm | calls | ref bpb | ratio |
|---|---:|---:|---:|
| fixed+CF s0 | 7 | 5.151 | 14.86 |
| fixed+CF s0 CI | | | [8.00, 26.00] |
| oracle+CF s0 | 5 | 5.072 | 27.00 |
| oracle+CF s0 CI | | | [9.64, 27.00] |
| rate-only+CF s0 | 5 | 5.129 | 27.00 |
| rate-only+CF s0 CI | | | [27.00, 27.00] |

### Decision

Oracle ceiling is zero (E4 fails; I=-0.0001). No no-oracle arm can pass P3. Report P1/P2 if E1/E3 hold; stop router work. P2a calls ratio `True`; P2b dynamic vs fixed at equal calls `True` (mean -0.023).

## syngrid (sha 24ae10a789f2, 1,200,128 bytes)

unigram `3.911` bpb

byte GPT `2.785` bpb


| cell | bound bpb | recon | c1 | bnd | codes | rate | align |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed+ar | 4.088 | 3.908 | 0.000 | 0.000 | 0.180 | 0.000 | 1.000 |
| fixed+cf | 4.123 | 3.908 | 0.000 | 0.000 | 0.215 | 0.000 | 1.000 |
| oracle+ar | 4.062 | 3.907 | 0.000 | 0.000 | 0.155 | 0.000 | 1.000 |
| oracle+cf | 4.142 | 3.907 | 0.000 | 0.000 | 0.234 | 0.000 | 1.000 |
| rate-only+ar | 3.914 | 3.913 | 0.000 | 0.000 | 0.001 | 0.000 | 0.258 |
| rate-only+cf | 3.951 | 3.913 | 0.000 | 0.001 | 0.037 | 0.000 | 0.258 |

Gates: E1 `False`, E2 `False`, E3 `False`, P2b `False` (mean +0.022), G3 `False`, G4 `{'oracle': False, 'rate-only': False}`

total(fixed+CF) `4.123` bpb; unigram `3.911`; byte GPT `2.785`; AR sample `5.398`


### Interaction I (primary k = 1)

| arm | I mean | SEM | CI low | CI high | n |
|---|---:|---:|---:|---:|---:|
| oracle | -0.0004 | nan | -0.0004 | -0.0003 | 1 |
| rate-only | -0.0000 | nan | -0.0000 | +0.0000 | 1 |

### Mechanism verdicts

NEGATIVE CONTROL FAILED for ['oracle']: the I CI excludes 0 on syngrid. The boundary effect is a rate-distortion shift or noise.

### Utility Pareto (sample quality, sequential calls)

| arm | calls | ref bpb | ratio |
|---|---:|---:|---:|
| fixed+CF s0 | None | nan | nan |
| fixed+CF s0 CI | | | [8.00, 26.00] |
| oracle+CF s0 | None | nan | nan |
| oracle+CF s0 CI | | | [9.64, 27.00] |
| rate-only+CF s0 | None | nan | nan |
| rate-only+CF s0 CI | | | [9.64, 27.00] |

### Decision

NEGATIVE CONTROL FAILED for ['oracle']: the I CI excludes 0 on syngrid. The boundary effect is a rate-distortion shift or noise.

## Provenance

```json
{
  "format": 5,
  "preset": "quick",
  "phase": 0,
  "device": "cuda",
  "torch": "2.15.0",
  "gpu": "NVIDIA GeForce RTX 3080",
  "started": "2026-11-23T19:57:08.168653Z",
  "finished": "2026-11-23T20:00:26.215456Z"
}
```

