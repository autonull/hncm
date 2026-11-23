# H-NCM v5 report

preset `smoke` phase `2` device `cuda` torch `2.15.0`

started `2026-11-23T19:41:06.077047Z` finished `2026-11-23T19:41:36.513750Z`

cfg fingerprint `0b4cf7846ea2a0ad`


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
| surprise+ar | 6.308 | 4.961 | 0.068 | 0.412 | 0.867 | 0.000 | 0.178 |
| surprise+cf | 7.149 | 4.961 | 0.116 | 0.425 | 1.646 | 0.000 | 0.178 |
| oracle-sup+ar | 6.209 | 4.929 | 0.057 | 0.423 | 0.801 | 0.000 | 0.186 |
| oracle-sup+cf | 7.066 | 4.929 | 0.103 | 0.434 | 1.600 | 0.000 | 0.186 |

Gates: E1 `False`, E2 `False`, E3 `True`, P2b `False` (mean +0.208), G3 `False`, G4 `{'oracle': False, 'rate-only': False, 'surprise': False, 'oracle-sup': False}`

total(fixed+CF) `6.741` bpb; unigram `4.638`; byte GPT `4.820`; AR sample `5.486`


### Interaction I (primary k = 1)

| arm | I mean | SEM | CI low | CI high | n |
|---|---:|---:|---:|---:|---:|
| oracle | -0.0021 | nan | -0.0024 | -0.0019 | 1 |
| rate-only | -0.0054 | nan | -0.0057 | -0.0051 | 1 |
| surprise | -0.0052 | nan | -0.0055 | -0.0050 | 1 |
| oracle-sup | -0.0055 | nan | -0.0057 | -0.0052 | 1 |

E5 no-oracle >= max(0.5*I_oracle, 2*SEM): `False`

align(surprise) `0.178` vs align(grid) `0.146`: `False`


### Utility Pareto (sample quality, sequential calls)

| arm | calls | ref bpb | ratio |
|---|---:|---:|---:|
| fixed+CF s0 | None | nan | nan |
| fixed+CF s0 CI | | | [7.20, 12.00] |
| oracle+CF s0 | 4 | 5.813 | 12.75 |
| oracle+CF s0 CI | | | [8.50, 12.75] |
| rate-only+CF s0 | 4 | 5.876 | 12.75 |
| rate-only+CF s0 CI | | | [12.75, 12.75] |
| surprise+CF s0 | 4 | 5.758 | 12.75 |
| surprise+CF s0 CI | | | [12.75, 12.75] |
| oracle-sup+CF s0 | 4 | 6.010 | 12.75 |
| oracle-sup+CF s0 CI | | | [8.50, 12.75] |

### Decision

Oracle ceiling is zero (E4 fails; I=-0.0021). No no-oracle arm can pass P3. Report P1/P2 if E1/E3 hold; stop router work. P2a calls ratio `True`; P2b dynamic vs fixed at equal calls `False` (mean +0.208).

## Provenance

```json
{
  "format": 5,
  "preset": "smoke",
  "phase": 2,
  "device": "cuda",
  "torch": "2.15.0",
  "gpu": "NVIDIA GeForce RTX 3080",
  "started": "2026-11-23T19:41:06.077047Z",
  "finished": "2026-11-23T19:41:36.513750Z"
}
```

