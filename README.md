# H-NCM: Hierarchical Neural Concept Modeling

Status: `hncm6.py` implements the v6 plan. Phase 0 is measured at budget (3 seeds,
512 windows) with honest accounting. **No P1/P2/P3 claim is established: G0
fails.**

## Result

The codec is healthy but not competitive. Honest Phase 0, preset `quick`, 3 seeds,
512 windows, 400 steps (`runs/p0/quick-c3b7a8ac/`), mean of 3 seeds:

| corpus | arm | total | recon | code | H(code) | util |
|---|---|---:|---:|---:|---:|---:|
| synseg | rel-pos | 6.891 | 3.752 | 3.138 | 381.0 | 1.000 |
| synseg | constant-code | 4.642 | 4.642 | 0.000 | 0.0 | 0.0 |
| synseg | no-quant | 3.758 | 3.609 | 0.149 | 0.0 | 0.0 |
| synseg | unigram | 4.637 | 4.637 | 0.000 | - | - |
| syngrid | rel-pos | 5.975 | 2.863 | 3.112 | 377.0 | 1.000 |
| syngrid | constant-code | 3.909 | 3.909 | 0.000 | 0.0 | 0.0 |
| syngrid | no-quant | 2.812 | 2.664 | 0.148 | 0.0 | 0.0 |
| syngrid | unigram | 3.908 | 3.908 | 0.000 | - | - |

- G0 **fail**: `util=1.000`, `delta_recon=0.890/1.046`, `H=381/377` bits/window
  pass, `total` does not. G6 **pass** (`rel-abs = -0.006/+0.024`). G9 **pass**.
- The code is fully used and carries information, but it costs 3.14/3.11 bpb and
  buys 0.89/1.05 bpb of reconstruction: 0.28-0.34 bpb per bpb of code. `total` is monotone
  in the code budget at every swept operating point (levels 4/8, `d_c` 1/2/4, stride
  4/8/16) and at 4x the training steps. No `lambda` and no tiling can pass G0.
- Per the decision tree, G0 fails, so Phase 1 is not run: the DP cannot be
  interpreted on an uncompetitive codec. The negative is publishable because the
  controls are clean.

## Hypothesis

Replacing the router with an exact DP over a learned cost field, and RVQ with scalar
quantization + a low-rank prior, yields (i) `total < unigram` on a real corpus,
(ii) `calls_AR/calls_DP >= 8` at matched total bits, and (iii) `I_structure > 0` at
matched lambda. The v5 interaction `I` is a ceiling, not the deployable claim.

(i) fails at the codec, before the DP: the code does not pay for itself. The next
decisive step is code transfer efficiency (the reconstruction must gain > 1 bpb per
code bit), not the tiling.

## Implementation

- `hncm6.py` is the complete self-contained script (one file): shared codec, scalar
  quantization, low-rank prior, exact DP cost field, span-relative decoder,
  closed-loop generation, controls, baselines, staged training, resume, gates, and
  report.
- The entropy model is decodable: it conditions only on the span start, span length,
  the transmitted global factor, and the already-sent codes. The rate is charged once
  per transmitted span code; the padded tail is neither charged, decoded, nor counted.
- Run the smallest decisive phase first; scale only after G0 and G6 pass.
- Commands:
  ```bash
  python hncm6.py selftest
  python hncm6.py run --preset quick --phase 0 --seeds 0,1,2
  python hncm6.py report --results runs/latest/results.json
  ```

## Artifacts

- `runs/p0/quick-c3b7a8ac/results.json` and `REPORT.md` (honest Phase 0,
  `format: 7`).
- `TODO1.md` (plan and progress); `REPORT.md` (headline).
- `hncm6.py`, `_selftest` passes; `ruff format`/`ruff check` pass.
- Superseded and misleading: `runs/p0/quick-dbe8d386/` (`format: 6`: rate charged
  per position, entropy model conditioned on the latent), `runs/p1/smoke-37d2450a/`,
  `runs/p2/smoke-bad03cb9/`. Do not cite them.
- The v5 negative report remains at `runs/p0/quick-a6924fcff3/REPORT.md`.
