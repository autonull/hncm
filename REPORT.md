# H-NCM v5 report

preset `smoke` phase `0` device `cuda` torch `2.15.0`

started `2026-11-23T20:05:31.555152Z` finished `2026-11-23T20:05:46.835003Z`

cfg fingerprint `5ce866f3f296346b`


## Claims

P1 absolute compression; P2 parallelism dividend; P3 boundary mechanism.
All three are reported, including negative results. Thresholds come from `Cfg`.


> Diagnostic only: preset != full. Controls are decisive only on the full preset.


## synseg (sha effd9e48ece4, 1,200,028 bytes)

unigram `4.638` bpb

byte GPT `4.820` bpb


| cell | bound bpb | recon | c1 | bnd | codes | rate | align |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed+ar | incomplete | | | | | | |
| fixed+cf | incomplete | | | | | | |
| oracle+ar | incomplete | | | | | | |
| oracle+cf | incomplete | | | | | | |
| rate-only+ar | incomplete | | | | | | |
| rate-only+cf | incomplete | | | | | | |

### Codec health (arms stopped before priors)

| chunker | util | RD | achieved | align |
|---|---:|---|---:|---:|
| fixed | [0.53, 0.31] | [4.848, 4.851, 4.85] | 1.075 | 0.146 |
| oracle | [0.59, 0.12] | [4.854, 4.861, 4.862] | 1.574 | 0.579 |
| rate-only | [0.53, 0.03] | [4.936, 4.934, 4.931] | 1.409 | 0.126 |

Gates: E1 `False`, E2 `False`, E3 `False`, P2b `False` (mean +nan), G3 `False`, G4 `{'oracle': False, 'rate-only': False}`

total(fixed+CF) `nan` bpb; unigram `4.638`; byte GPT `4.820`; AR sample `nan`


### Interaction I (primary k = 1)

| arm | I mean | SEM | CI low | CI high | n |
|---|---:|---:|---:|---:|---:|

### Mechanism verdicts

E4 oracle: missing; mechanism cannot be evaluated.
E4n rate-only: missing.
E5 boundary: missing (run phase 2).

### Utility Pareto (sample quality, sequential calls)

| arm | calls | ref bpb | ratio |
|---|---:|---:|---:|

### Decision

Phase 0 oracle arm missing; mechanism claim cannot be evaluated. P2a calls ratio `False`; P2b dynamic vs fixed at equal calls `False` (mean +nan).

## syngrid (sha 24ae10a789f2, 1,200,128 bytes)

unigram `3.926` bpb

byte GPT `4.091` bpb


| cell | bound bpb | recon | c1 | bnd | codes | rate | align |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed+ar | incomplete | | | | | | |
| fixed+cf | incomplete | | | | | | |
| oracle+ar | incomplete | | | | | | |
| oracle+cf | incomplete | | | | | | |
| rate-only+ar | incomplete | | | | | | |
| rate-only+cf | incomplete | | | | | | |

### Codec health (arms stopped before priors)

| chunker | util | RD | achieved | align |
|---|---:|---|---:|---:|
| fixed | [0.34, 0.09] | [4.124, 4.124, 4.124] | 0.984 | 1.000 |
| oracle | [0.47, 0.09] | [4.128, 4.131, 4.13] | 0.960 | 1.000 |
| rate-only | [0.03, 0.03] | [4.2, 4.18, 4.178] | 1.078 | 0.226 |

Gates: E1 `False`, E2 `False`, E3 `False`, P2b `False` (mean +nan), G3 `False`, G4 `{'oracle': False, 'rate-only': False}`

total(fixed+CF) `nan` bpb; unigram `3.926`; byte GPT `4.091`; AR sample `nan`


### Interaction I (primary k = 1)

| arm | I mean | SEM | CI low | CI high | n |
|---|---:|---:|---:|---:|---:|

### Mechanism verdicts

Negative control has no arm.

### Utility Pareto (sample quality, sequential calls)

| arm | calls | ref bpb | ratio |
|---|---:|---:|---:|

### Decision

Negative control has no arm. P2a calls ratio `False`; P2b dynamic vs fixed at equal calls `False` (mean +nan).

## Provenance

```json
{
  "format": 5,
  "preset": "smoke",
  "phase": 0,
  "device": "cuda",
  "torch": "2.15.0",
  "gpu": "NVIDIA GeForce RTX 3080",
  "started": "2026-11-23T20:05:31.555152Z",
  "finished": "2026-11-23T20:05:46.835003Z"
}
```

