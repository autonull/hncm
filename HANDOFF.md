# H-NCM v5 — session handoff

Read `TODO1.md` first. It is the research plan and the gate definitions.
Implementation: `hncm5.py`, one file, no backwards compatibility.
Run with `python hncm5.py ...` or `uv run python hncm5.py ...`.

## Status at handoff

- v5 implemented: frozen `Cfg`/presets/JSON override; fixed, oracle, rate-only,
  byte-surprise boundary, oracle-supervised boundary and BPE chunkers; shared
  codec; AR/CF priors; valid-token masking; oracle snapping; gate registry;
  atomic `results.json`; fingerprint-checked resume; HPO stage 1; report/plots.
- `_selftest()` passes. `ruff format`/`ruff check` pass.
- Phase 0 smoke, Phase 0 quick, Phase 2 smoke and HPO stage 1 have run.
- **No claim is established.** Smoke/quick codecs collapsed:
  `min(util) < 0.20` and/or RD not monotone. The new Phase 0 rule records
  `kind="codec"` health and stops the arm before priors. The latest `REPORT.md`
  shows the stopped arms; that is the intended behavior, not evidence.
- `runs/latest` -> `p0/smoke-5ce...` (codec-health report).
- Quick diagnostic: `runs/p0/quick-a6924fcff3/REPORT.md`.
- Phase 2 smoke: `runs/p2/smoke-0b4cf7846e/REPORT.md`.
- HPO stage 1: `runs/hpo/stage1-5ce866f3f2/hpo.json`, best score 5.711.

Quick numbers (diagnostic only, codec collapsed):

- synseg: unigram 4.635, byte GPT 4.295, fixed+CF 4.638; oracle `I=-0.0001`;
  rate-only `I=-0.0000`; `util=0.02`.
- syngrid: unigram 3.911, byte GPT 2.785, fixed+CF 4.123; oracle
  `I=-0.0004`; negative control fails; `util` 0.09-0.19.
- P2a/P2b pass numerically (`14.9-27x`; P2b `-0.023` bpb) but the codec is
  collapsed; do not report them as P2 evidence.
- Phase 2 smoke: surprise alignment 0.178 vs grid 0.146; below the `+0.05`
  P3d gate; E5 fails.

## Next decisive step

1. Full Phase 0, two seeds:

   ```bash
   python hncm5.py run --preset full --phase 0 --seeds 0,1 --resume 0
   python hncm5.py report --results runs/latest/results.json
   ```

   Budget ~40 min on the 3080 per the plan; verify with the clock.

2. Read codec health before priors. If `min(util) < 0.20` or RD is not
   monotone, the arm is stopped and there is no interaction to interpret.
   Sweep `K/Q/stride/lam_fixed/beta/eh_layers` with HPO stage 1 and retry:

   ```bash
   python hncm5.py hpo --stage 1 --trials 100
   ```

3. Decision (from `REPORT.md`, all CIs paired over windows):

   - E4 oracle CI lower <= 0 -> ceiling is zero. Report P1/P2 if E1/E3 hold;
     stop router work. Do not claim P3.
   - E4 > 0 and E4n CI covers 0 -> expected. Continue Phase 1 and Phase 2.
   - E4 > 0 and E4n CI excludes 0 -> surprise. Verify E5 and alignment
     before claiming P3.
   - E4 > 0 but E1 fails -> P1 negative; P3 is still viable.
   - E1/E2 and E4/E5 and E3 pass -> all three claims hold.

4. Phase 1: `hpo --stage 1 --trials 100`; promote top 20 percent. Use the
   fixed codec's achieved rate as `rho_dynamic` for Phase 2. `match_dynamic`
   only applies at phase >= 1; phase 0 uses one `rho`.

5. Phase 2:

   ```bash
   python hncm5.py run --preset quick --phase 2 --seeds 0,1,2 --resume 0
   python hncm5.py report --results runs/latest/results.json
   ```

   Arms: fixed, oracle, rate-only, surprise, oracle-sup. Never use oracle or
   oracle-sup on shakespeare.

6. Phase 3: `run --phase 3`; report P1/P2 only. P3 is not claimed on
   shakespeare.

## Gates (one source: `Cfg`)

- E1 `total < unigram - 0.10`.
- E2 `total <= byte GPT + 0.05`.
- E3 `calls_AR/calls_CF >= 8` at `|quality| <= 0.05`; P2b dynamic not worse
  than fixed by more than 0.02 bpb at equal calls.
- E4 oracle `I > 0` on synseg; `I ~ 0` on syngrid.
- E4n rate-only `I ~ 0`; CI covers 0.
- E5 no-oracle `I >= max(0.5*I_oracle, 2*SEM)` and
  `align(no-oracle) >= align(grid) + 0.05`; `I ~ 0` on syngrid.
- G3 RD monotone and `min(util) >= 0.20`.
- G4 `|achieved[f] - achieved[d]| <= 0.05`.

## Commands / env

```bash
python hncm5.py run --preset smoke --phase 0
python hncm5.py hpo --stage 1 --trials 20
python hncm5.py report --results runs/latest/results.json
HNCM_CFG='{"K":64,"Q":3}' HNCM_RESUME=1 python hncm5.py run --preset quick
```

A changed config gets a new `p<phase>/<preset>-<fingerprint>/` directory.
`runs/latest` is a relative symlink. `HNCM_OUT` sets the base directory.
`.pt` checkpoints are ignored by git; results/reports/plots are not.

## Pitfalls

- Smoke and quick are diagnostic only. Do not use them as P1/P2/P3 evidence.
- Codec collapse is the main failure. Check `util`/RD before priors; the code
  stops the arm.
- `rho` is a cap: the natural rate must be above the target. Do not use 3.4
  with small `K/Q`.
- Oracle onsets are forbidden on shakespeare; `Cfg` asserts it.
- The dynamic layout's `bnd` level makes the dynamic codec not identical to
  fixed; syngrid `I` is a control, not exact zero.
- Stale `prior_*.pt` may remain after a stopped arm; `--resume 0` overwrites.
- `runs/latest` currently points to the smoke codec-health run; the quick
  report is a separate path above.
