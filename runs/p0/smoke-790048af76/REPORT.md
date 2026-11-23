# H-NCM v5 report

preset `smoke` phase `0` device `cuda` torch `2.15.0`

started `2026-11-23T19:26:11.656559Z` finished `2026-11-23T19:27:01.913826Z`

cfg fingerprint `92dac90e93deb9af`


## Claims

P1 absolute compression; P2 parallelism dividend; P3 boundary mechanism.
All three are reported, including negative results. Thresholds come from `Cfg`.


> Diagnostic only: preset != full. Controls are decisive only on the full preset.


## synseg (sha effd9e48ece4, 1,200,028 bytes)

unigram `4.638` bpb

byte GPT `4.820` bpb


| cell | bound bpb | recon | c1 | bnd | codes | rate | align |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed+ar | 6.235 | 4.849 | 0.112 | 0.000 | 1.273 | 0.000 | 0.146 |
| fixed+cf | 6.741 | 4.849 | 0.133 | 0.000 | 1.758 | 0.000 | 0.146 |
| oracle+ar | 6.111 | 4.861 | 0.064 | 0.388 | 0.799 | 0.000 | 0.579 |
| oracle+cf | 6.754 | 4.861 | 0.142 | 0.392 | 1.358 | 0.000 | 0.579 |
| rate-only+ar | 6.201 | 4.938 | 0.028 | 0.407 | 0.828 | 0.000 | 0.126 |
| rate-only+cf | 7.053 | 4.938 | 0.061 | 0.424 | 1.630 | 0.000 | 0.126 |

Gates: E1 `False`, E2 `False`, E3 `True`, P2b `False` (mean +0.179), G3 `False`, G4 `{'oracle': False, 'rate-only': False}`

total(fixed+CF) `6.741` bpb; unigram `4.638`; byte GPT `4.820`; AR sample `5.486`


### Interaction I (primary k = 1)

| arm | I mean | SEM | CI low | CI high | n |
|---|---:|---:|---:|---:|---:|
| oracle | -0.0021 | nan | -0.0024 | -0.0019 | 1 |
| rate-only | -0.0054 | nan | -0.0057 | -0.0051 | 1 |

### Mechanism verdicts

E4 oracle: `False` (I=-0.0021, CI [-0.0024, -0.0019]).
E4n rate-only null: `False` (I=-0.0054, CI [-0.0057, -0.0051]).
E5 boundary: missing (run phase 2).

### Utility Pareto (sample quality, sequential calls)

| arm | calls | ref bpb | ratio |
|---|---:|---:|---:|
| fixed+CF s0 | None | nan | nan |
| fixed+CF s0 CI | | | [7.20, 12.00] |
| oracle+CF s0 | 4 | 5.813 | 12.75 |
| oracle+CF s0 CI | | | [8.50, 12.75] |
| rate-only+CF s0 | 4 | 5.876 | 12.75 |
| rate-only+CF s0 CI | | | [12.75, 12.75] |

### Decision

Oracle ceiling is zero (E4 fails; I=-0.0021). No no-oracle arm can pass P3. Report P1/P2 if E1/E3 hold; stop router work. P2a calls ratio `True`; P2b dynamic vs fixed at equal calls `False` (mean +0.179).

## syngrid (sha 24ae10a789f2, 1,200,128 bytes)

unigram `3.926` bpb

byte GPT `4.091` bpb


| cell | bound bpb | recon | c1 | bnd | codes | rate | align |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed+ar | 5.497 | 4.114 | 0.094 | 0.000 | 1.289 | 0.000 | 1.000 |
| fixed+cf | 5.988 | 4.114 | 0.112 | 0.000 | 1.762 | 0.000 | 1.000 |
| oracle+ar | 4.788 | 4.141 | 0.059 | 0.009 | 0.579 | 0.000 | 1.000 |
| oracle+cf | 5.520 | 4.141 | 0.095 | 0.018 | 1.267 | 0.000 | 1.000 |
| rate-only+ar | 4.850 | 4.183 | 0.064 | 0.423 | 0.180 | 0.000 | 0.226 |
| rate-only+cf | 5.736 | 4.183 | 0.093 | 0.432 | 1.028 | 0.000 | 0.226 |

Gates: E1 `False`, E2 `False`, E3 `True`, P2b `False` (mean +0.572), G3 `False`, G4 `{'oracle': False, 'rate-only': False}`

total(fixed+CF) `5.988` bpb; unigram `3.926`; byte GPT `4.091`; AR sample `4.718`


### Interaction I (primary k = 1)

| arm | I mean | SEM | CI low | CI high | n |
|---|---:|---:|---:|---:|---:|
| oracle | -0.0038 | nan | -0.0040 | -0.0035 | 1 |
| rate-only | -0.0062 | nan | -0.0066 | -0.0058 | 1 |

### Mechanism verdicts

NEGATIVE CONTROL FAILED for ['oracle', 'rate-only']: the I CI excludes 0 on syngrid. The boundary effect is a rate-distortion shift or noise.

### Utility Pareto (sample quality, sequential calls)

| arm | calls | ref bpb | ratio |
|---|---:|---:|---:|
| fixed+CF s0 | 3 | 4.437 | 12.00 |
| fixed+CF s0 CI | | | [12.00, 12.00] |
| oracle+CF s0 | 4 | 5.176 | 12.75 |
| oracle+CF s0 CI | | | [12.64, 12.75] |
| rate-only+CF s0 | 4 | 5.181 | 12.75 |
| rate-only+CF s0 CI | | | [12.75, 12.75] |

### Decision

NEGATIVE CONTROL FAILED for ['oracle', 'rate-only']: the I CI excludes 0 on syngrid. The boundary effect is a rate-distortion shift or noise.

## Provenance

```json
{
  "format": 5,
  "preset": "smoke",
  "phase": 0,
  "device": "cuda",
  "torch": "2.15.0",
  "gpu": "NVIDIA GeForce RTX 3080",
  "started": "2026-11-23T19:26:11.656559Z",
  "finished": "2026-11-23T19:27:01.913826Z"
}
```

