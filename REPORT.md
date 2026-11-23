# H-NCM v6 report

Status: `hncm6.py` implements the v6 plan, but every run below is a smoke or quick
diagnostic. **No P1/P2/P3 claim is established.** The controls are not decisive at
these presets.

## Phase 0 - codec health (quick, synseg, seed 0, 300 steps)

| arm | total | recon | code | util | H(code) |
|---|---:|---:|---:|---:|---:|
| `rel-pos` | 7.632 | 4.206 | 3.426 | 1.000 | 1.362 |
| `constant-code` | 4.641 | 4.641 | 0.000 | 0.000 | 0.000 |
| `abs-pos` | 7.245 | 3.882 | 3.364 | 1.000 | 1.365 |
| `no-quant` | 3.995 | 3.868 | 0.128 | - | - |
| `unigram` | 4.637 | 4.637 | - | - | - |

- **G0 fails**: `util=1.000` and `delta_recon=0.435` pass, but `H(code)=1.362 < 2`
  and `total=7.632 > unigram=4.637`. The code carries information, but its entropy
  model is worse than the empirical code entropy and its rate is too high.
- **G6 passes**: `recon(abs-pos) - recon(rel-pos) = -0.325 <= 0.05`.
- One seed only; G9 is a warning.

## Phase 1 - structure (smoke, synseg, seed 0, 40 steps)

| arm | total | boundary | align |
|---|---:|---:|---:|
| `fixed` | 6.728 | 0.000 | 0.236 |
| `DP` | 6.384 | 0.001 | 0.244 |
| `oracle` | 6.494 | 0.000 | 1.000 |
| `oracle-costed` | 7.117 | 0.624 | 1.000 |
| `BPE` | 8.919 | 0.767 | 0.322 |
| `IsoWarp` | 8.786 | 0.000 | 0.270 |
| `random` | 7.682 | 0.861 | 0.241 |

- **G4 passes**: `I_oracle=0.234`, bootstrap CI `[0.198, 0.276]`.
- **G5 fails**: `I_DP=0.344` passes the effect size, but `align(DP)=0.244` is below
  `align(fixed)+0.05=0.286`.
- **G5b fails**: the oracle `6.494` is worse than the DP `6.384`, so
  `oracle < DP < fixed` does not hold. The oracle tiling is not a ceiling here.
- **G5c/G5d pass**: DP beats BPE and IsoWarp.
- Smoke only; one seed; no P3 claim.

## Phase 2 - compression and parallelism (smoke, shakespeare, seed 0, 40 steps)

| arm | total bpb |
|---|---:|
| `DP` | 6.713 |
| `fixed` | 7.149 |
| `byte GPT` | 5.009 |
| `byte AR` | 5.584 |
| `unigram` | 4.995 |
| `gzip` | 3.218 |

- **G1 fails**: `DP=6.713` is not below `unigram-0.10=4.895`.
- **G2 fails**: `DP=6.713` is not below `byte GPT+0.05=5.059`.
- **G2b fails**: `DP=6.713` is not below `gzip+0.10=3.318`.
- **G3 fails as unmatched**: the closest DP/AR pair differs by `1.129 > 0.05` bpb.
- P1 and P2 are negative. P3 is independent and is also negative at the smoke gate.

## Decision

G0 fails first. Per the decision tree, fix the codec before interpreting the DP:
the conditional entropy model must beat the empirical code entropy, and the code rate
must fall below the gain in reconstruction. Do not report the Phase 1 effects as P3
evidence, and do not run the full Phase 1 before G0 passes.

The result is a negative diagnostic with clean accounting: the tiling is paid for
DP/BPE/IsoWarp, the fixed grid and oracle are free, the codec is shared, and the
decoder is span-relative. That is the intended v6 protocol; the implementation is
not yet healthy enough for a claim.

## Artifacts

- `runs/p0/quick-cd8d9e84/results.json` and `REPORT.md`
- `runs/p1/smoke-37d2450a/results.json` and `REPORT.md`
- `runs/p2/smoke-bad03cb9/results.json` and `REPORT.md`
- `hncm6.py`, `_selftest` passes; `ruff format`/`ruff check` pass.
- The v5 negative report remains at `runs/p0/quick-a6924fcff3/REPORT.md`.

## Provenance

```json
{
  "format": 6,
  "torch": "2.15.0",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 3080",
  "seeds": [0],
  "presets": ["smoke", "quick"],
  "note": "diagnostic only; not P1/P2/P3 evidence"
}
```
