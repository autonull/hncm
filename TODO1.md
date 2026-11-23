# TODO1 — H-NCM v5: prove the parallelism dividend, not just the codec

Status: `hncm4.py` is the fixed v3 (runs smoke in ~30 s, resumes in ~5 s). It establishes
plumbing only. It does **not** establish the thesis: the learned router is at chance
(`align 0.146–0.149` vs grid `0.145`), `rho=1.0` collapses the RVQ (`util 0.03–0.06`), and
healthy codecs lose to unigram (`6.0–6.2` vs `4.64` bpb). v3 results are invalid: its corpora
were 85 % NUL bytes.

Decision: **rewrite as `hncm5.py`**, one file first, no backwards compatibility. Keep
`hncm4.py` as the reference implementation. This file is the research plan and design
record; the implementation target is `hncm5.py`. Split into a package only if a second
caller appears.

---

## 1. Claims and falsification

Co-primary claims. Report all three, including negative ones. Do not select one post hoc.

P1 — absolute compression. The two-part concept code must beat the unigram baseline and
approach or beat the byte GPT at the same data.

```
P1a: total < unigram − 0.10
P1b: total ≤ byte GPT + 0.05        (parity; a win is total < byte GPT)
```

P2 — parallelism dividend. At matched sample quality, CF must decode with far fewer
sequential calls than AR, and the dynamic chunker must beat the fixed chunker at equal calls.

```
P2a: calls_AR / calls_CF ≥ 8        at |quality_CF − quality_AR| ≤ 0.05
P2b: quality_CF(dynamic) ≤ quality_CF(fixed) + 0.02 at equal calls
```

P3 — mechanism. The gain comes from content boundaries, not a rate-distortion shift.
The oracle arm is a ceiling and a control; the **deployable** claim is the no-oracle arm.

```
P3a: I_oracle > 0 on synseg, CI lower > 0        (ceiling; controls only)
P3b: I_no-oracle > 0 and ≥ max(0.5 × I_oracle, 2·SEM)   (deployable)
P3c: I ≈ 0 on syngrid for both arms, CI covers 0
P3d: align(no-oracle) ≥ align(grid) + 0.05
```

Quality is byte-GPT bpb of generated samples; ε = 0.05 bpb. A P1 failure does not
invalidate P2/P3, and a P3 failure does not invalidate P1/P2. They are separate claims
with separate controls.

Caveat: a smaller AR–CF gap can also come from a **weaker AR model** (lower code
dependence), not from better chunking. `I > 0` is necessary, not sufficient. `P2b` (absolute
utility) and `P3d` (alignment) are mandatory.

Endpoints and gates (one source of truth; do not duplicate a threshold elsewhere):

| Endpoint | Gate | Claim | Test | Threshold |
|---|---|---|---|---|
| E1 | G1 | P1a | concept total beats unigram | `total < unigram − 0.10` |
| E2 | G2 | P1b | concept total near byte GPT | `total ≤ byte GPT + 0.05` |
| E3 | G7 | P2 | depth at matched quality | `calls_AR/calls_CF ≥ 8`; `quality_CF ≤ quality_AR + 0.05`; `quality_CF(d) ≤ quality_CF(f) + 0.02` |
| E4 | G5 | P3a | oracle `I` (ceiling) | `I_oracle > 0` on synseg; `I_oracle ≈ 0` on syngrid |
| E4n | G5n | P3-null | rate-only no-oracle `I` (null) | `|I_no-oracle| ≤ 2·SEM`; `align ≈ grid` |
| E5 | G6 | P3b/c/d | boundary no-oracle `I` and alignment | `I_no-oracle ≥ max(0.5 I_oracle, 2·SEM)`; `align ≥ grid + 0.05`; `I ≈ 0` on syngrid |
| — | G3 | health | RD monotone + codebook used | `RD[k] ≥ RD[k+1]`, `min(util) ≥ 0.20` |
| — | G4 | matching | achieved rate matched | `|achieved[f] − achieved[d]| ≤ 0.05` |

Decision logic:

- `E4` fails (oracle `I ≈ 0`) → the ceiling is zero; no no-oracle arm can pass P3. Report
  P1/P2 if `E1`/`E3` hold; stop router work. Do not claim the mechanism.
- `E4` passes and `E4n ≈ 0` → expected: the oracle helps, the rate-only router does not.
  Continue to Phase 2 with the no-oracle boundary arms. This is the main path.
- `E4` passes and `E4n > 0` → the rate-only router already helps; verify `E5` and alignment
  before claiming P3. Treat `E4n` as a surprise, not as the expected result.
- `E4` passes but `E1` fails → P1 is negative; P3 is viable. Continue to Phase 2. Report P1
  negative.
- `E4`/`E5` pass but `E3` fails → P3 holds, P2 does not; report P1/P3.
- `E1`/`E2` and `E4`/`E5` and `E3` pass → all three claims hold.

Primary endpoint for P2: `calls_AR / calls_CF` at matched quality (`ε = 0.05` bpb), 3
seeds, paired windows. Report the median and a paired bootstrap CI. The current measured
upper bound is 17× on synseg at `rho=3.4` (103 calls vs 6), but the router was at chance;
the fixed codec reached 24× at `rho=2.0` (72 calls vs 3).

---

## 2. Why v4 is not evidence

| rho | codec | total fixed/dyn | I | control | failure |
|---|---|---|---|---|---|
| 1.0 | collapsed (`util 0.03–0.06`, RD flat) | 4.97 / 5.17 | −0.013 | fail | codes ignored (P1 fail) |
| 2.0 | healthy | 6.18 / 6.16 | +0.012 | pass | P1 fail |
| 3.4 | healthy | 6.00 / 6.19 | +0.006 | pass | P1 fail; align at chance |

The +0.006…+0.016 `I` is a rate-distortion shift. The dynamic codec is trading
reconstruction for code rate, not finding word boundaries. `I > 0` alone is not mechanism
support; E3 and E5 are mandatory.

At `rho=2.0`, the utility Pareto is stronger than the compression table suggests:
`fixed+CF(k=1)` reaches `4.722` at 3 calls vs fixed AR `4.721` at 72 calls — 24× fewer
sequential calls at equal sample quality. `dynamic+CF(k=2)` reaches `4.773` at 6 calls vs
dynamic AR `4.742` at 103 calls — 17× fewer, within `0.031` bpb. But fixed CF is
**better in absolute terms** than dynamic CF at low depth, and dynamic AR is worse than
fixed AR. The dynamic gap is smaller partly because its AR model is weaker. That is the
confound G7 must block; `I > 0` alone is not enough. At 8 samples, the 0.002 dynamic-vs-fixed
difference is noise; Phase 3 uses 128 samples and 3 seeds, and the paired CI must exclude 0.

Three structural causes:

1. **No boundary signal.** The router's only supervision is a rate term that a constant
   offset minimizes. It converges to grid. No HPO over `K/Q/stride` can fix this.
2. **Rate target on the wrong model.** The codec's rate uses a shallow entropy head
   (3.0 bpb), weaker than the AR prior (2.1–2.7). It is a pessimistic proxy. A target
   below the natural rate (v4 used `rho=1.0`) collapses the codebook to satisfy it.
3. **No bottleneck.** The decoder can reconstruct any chunking, so reconstruction does not
   prefer content chunks. The rate term dominates and collapses the code.

---

## 3. Phase 0 — ceiling + null (do this first, ~40 min)

Two arms, same codec config, same windows:

- **oracle**: true onsets on the control corpora. Ceiling; controls only; never shakespeare.
- **no-oracle**: the v4 rate-only router. Null; expected `I ≈ 0`; the deployable baseline.

The oracle is a diagnostic, not a dependency. The deployable claim is the no-oracle arm.

```python
class Chunker(Protocol):
    dynamic: bool
    def boundaries(self, h: Tensor, x: Tensor) -> Boundary: ...

FixedChunker(stride)          # every stride bytes
OracleChunker(onsets)          # true word onsets (controls only)
RateOnlyChunker(...)           # v4 router: rate term only
BoundaryChunker(...)           # no-oracle byte-surprise / CF-in-the-loop (Phase 2)
BpeChunker(tokenizer)          # optional reference, no labels
```

The layout allows **one boundary per stride group** (`M = L/stride` chunks). The oracle
snaps each true onset to the nearest allowed offset in its group; the residual misalignment
is ≤ `stride/2` bytes. This keeps the layout unchanged and the learned-vs-oracle comparison
fair. If the oracle is strong but the snapped boundary loses the effect, upgrade to a per-byte
boundary head in Phase 2. Do not change the layout in Phase 0.

Run:

- `synseg`, `syngrid`; chunker ∈ {fixed, oracle, rate-only}; codec 1500 steps; 2 seeds.
- Measure E1, E4 (oracle), E4n (rate-only), plus G3/G4, total bpb, `calls`, sample quality.
- Train priors only if `util ≥ 0.20`; otherwise report the codec health and stop the arm.

Decision:

- `E4` fails (oracle `I ≈ 0`) → the ceiling is zero; no no-oracle arm can pass P3. Report
  P1/P2 if `E1`/`E3` hold; stop router work. Do not claim the mechanism.
- `E4` passes and `E4n ≈ 0` → expected: the oracle helps, the rate-only router does not.
  Proceed to Phase 1 (frontier) and Phase 2 (no-oracle boundary arms). This is the green
  light for the deployable mechanism.
- `E4` passes and `E4n > 0` → the rate-only router already helps; verify `E5` and alignment
  before claiming P3. Treat `E4n` as a surprise, not the expected result.
- `E4` passes but `E1` fails → P1 is negative; P3 is viable. Proceed to Phase 1 with a
  bottleneck sweep (smaller `K`, `Q`, larger `stride`) and Phase 2.

The oracle is the cheapest **ceiling**; the no-oracle arm is the cheapest **null**. Both are
needed: one bounds the hypothesis, the other is the deployable baseline.

---

## 4. Phase 1 — absolute-compression frontier (HPO, ~4 h)

Run whenever a codec is healthy (`G3`/`G4`); P1 and P2 are independent of `E4`/`E5`.
Train the **fixed codec** without the router and sweep the absolute rate-distortion point.
There is no global `rho`: for each fixed codec, set `rho_dynamic = achieved(fixed)` and train
the dynamic codec to match it. This makes the mechanism comparison fair by construction and
avoids the v4 collapse at a target below the natural rate. The fixed codec's rate is
controlled by `lam_fixed`.

Search space:

| Knob | Values | Rationale |
|---|---|---|
| `stride` | 4, 8, 16 | chunk length; larger = fewer codes |
| `K` | 8, 16, 32, 64 | codebook capacity |
| `Q` | 1, 2, 3 | fine levels |
| `d` | 64, 96, 128 | capacity |
| `eh_layers` | 1, 2, 3 | rate-model strength |
| `beta` | 0.1, 0.25, 1.0 | commitment |
| `lam_fixed` | 0, 0.1, 0.25, 0.5, 1.0 | fixed-codec rate pressure |
| `level_dropout` | False, True | ablation |

Objective (stage 1, no priors):

```
score = total_proxy + penalties
total_proxy = recon + achieved          # conservative: prior ≤ entropy head
penalties = 10·max(0, 0.20 − min(util))
```

Accept only `util ≥ 0.20`, monotone RD, and `recon + achieved < unigram`. This is a
conservative screen: the AR prior is stronger than the entropy head, so a config that
passes the proxy will pass E1. A config that fails by less than 0.3 bpb is promoted anyway.

Protocol: a simple grid/random search is enough; Optuna is optional. 300 steps for the
first 80 trials; promote the top 20 % to 1200 steps, top 10 to Phase 2. Seed the search
with the v4 configs and the oracle config. Log every trial; do not select on the test set.

---

## 5. Phase 2 — router objective (no-oracle, the make-or-break, ~4 h)

The deployable arm is **no-oracle**. The oracle arm is the ceiling; it is optional and only
runs on the controls. If E4 fails, the ceiling is zero; stop the mechanism claim, but still
run the no-oracle arms to report the deployable baseline. A P1 failure does not block Phase 2.

Use the healthiest fixed codec from Phase 1; set `rho_dynamic = achieved(fixed)`. Try in
order; each is a separate arm, same codec/rate config:

0. **Rate-only (v4 null).** No boundary loss. Expected `I ≈ 0`. This is the baseline; it
   confirms the null and calibrates the noise.
1. **Byte-surprise boundary target (self-supervised).** A small causal byte model
   (`ByteGPT`, 1 layer, frozen) gives `s_t = −log p(x_t | x_<t)`. Normalize per window.
   Train the router with `BCE(logits, stopgrad(s_t))` plus the rate term. This peaks after
   spaces and at rare transitions; it transfers to shakespeare and uses no labels.
2. **Oracle-supervised router (control only).** On synseg/syngrid, train the router with
   `BCE(logits, true_onsets)`. This is an upper bound on the learned router. It must
   reproduce most of `I_oracle`; if it does not, the chunker parameterization is wrong.
   Never use this arm on shakespeare.
3. **CF-in-the-loop.** Alternate: freeze codec; train AR/CF priors; unfreeze router;
   minimize `recon + lam·NLL_CF` (or `NLL_CF − NLL_AR`); freeze priors; repeat 3×. Use a
   small prior (`gen=1`) for the loop, the full prior for evaluation. This directly optimizes
   the quantity of interest; it is the most expensive and the most likely to work.
4. **Bottleneck.** Reduce `K/Q`, increase `stride`, add a codebook utilization floor. A
   bottleneck makes content chunks more efficient; it is complementary to 1–3.

Chunker parameterization: replace the stride-group offset router with a **per-byte boundary
head** (`logits: (B,L)`) or keep the stride-group router but add the boundary loss. The
per-byte head is more expressive; constrain it with a length prior (geometric) and a
minimum chunk length. The geometry code already handles arbitrary lengths.

Acceptance:

- `E5`: `I_no-oracle ≥ max(0.5 × I_oracle, 2·SEM)` on synseg; `|I_no-oracle| ≤ 2·SEM` on syngrid.
- `align(no-oracle) ≥ grid + 0.05` on synseg.
- `E4n`: the rate-only arm is ≈ 0 (the null holds; a nonzero value is a surprise).
- E1/E2 and G3/G4 hold on the codec/prior pair.
- If no arm passes, report the oracle ceiling and the no-oracle negative. That is a valid
  paper: "content chunking helps, but the deployable router cannot find it."

---

## 6. Phase 3 — real corpus and product metric (~2 h)

Run when at least one of E1/E2, E3, or E5 is viable. `shakespeare`, 3 seeds, the selected
config, paired windows, bootstrap CIs. Report P1, P2, and P3 separately:

- P1: total bpb vs unigram and byte GPT; per-seed and mean ± SEM.
- P2: utility Pareto (`calls` vs byte-GPT bpb) for byte GPT, AR, CF (k=1,2,4),
  fixed/dynamic; `calls_AR / calls_CF` at matched quality, median and CI.
- P3: `I` per seed, across-seed mean − 2·SEM; control verdicts; alignment.
- Sample gallery, verbatim, with the same seeds. No cherry-picking.

---

## 7. Engineering (hncm5.py)

One file first. Split only when a second caller appears. No backwards compatibility.

```text
hncm5.py
  config      # frozen dataclass Cfg; PRESETS; HNCM_CFG JSON override
  data        # corpora, controls, onsets, oracle boundaries, windows
  layers      # Block, Stack, TokenFeats, geometry, layout
  chunkers    # Fixed, Oracle, RateOnly, Boundary, BPE (Protocol)
  codec       # RVQ, Codec; rate model
  priors      # Prior AR/CF, nll, sample
  baseline    # ByteGPT
  train       # fit, dual ascent, checkpoints, resume
  metrics     # components, RD, alignment, interaction, endpoints/gates
  experiment  # run_all, artifact store, fingerprints
  hpo         # optional: grid/random, successive halving
  cli         # argparse/main; env hooks
```

Rules:

- Plain frozen dataclass for `Cfg`; JSON at the boundary; no pydantic.
- `StrEnum`/`Literal` for modes when cheap; `Protocol` for `Chunker`/`RateModel`.
- `logging` or `print`; f-strings fine for trusted data; parameterize SQL.
- No module-level I/O; `main()`/`if __name__ == "__main__"`.
- One `ResultStore` for atomic writes/resume; one `Gate` registry.
- `uv` + `pyproject.toml`; `ruff format` always; `ruff check --fix` changed files; no
  pre-commit; no pyright gate.
- Tests: a small `_selftest()` for controls, cancellation, JSON, shared init, resume.
  No coverage target.

Dependencies: `torch`, `numpy`, `matplotlib`; `optuna` optional. No `pydantic`,
`structlog`, or `typer`.

CLI:

```bash
uv run python hncm5.py run --preset quick --corpora synseg,syngrid --chunker oracle --seeds 0,1
uv run python hncm5.py run --preset quick --corpora synseg,syngrid --chunker rate-only --seeds 0,1
uv run python hncm5.py hpo --stage 1 --trials 100
uv run python hncm5.py report --results runs/latest/results.json
```

`HNCM_CFG` is a JSON override, fingerprinted in `results.json`; resume refuses a changed
config or corpus sha. A changed config starts a new run directory.

---

## 8. Compute budget

| Phase | Trials | Steps | Time (3080) | Endpoint |
|---|---|---|---|---|
| 0 ceiling+null | 3 chunkers × 2 corpora × 2 seeds | 1500 | ~40 min | E4, E4n, E1 |
| 1 frontier | 100 → 20 → 10 | 300 → 1200 | ~4 h | E1/E2, G3 |
| 2 router | 5 arms × 2 seeds | 1200 | ~5 h | E5, E3, E4n |
| 3 real | 3 seeds | full | ~2 h | all |

Each claim has its own kill switch. A failed P1 does not cancel P2/P3; a failed P3 does
not cancel P1/P2. The oracle phase is the cheapest ceiling for P3; the frontier is the
cheapest kill switch for P1.

---

## 9. Risks and mitigations

| Risk | Signal | Mitigation |
|---|---|---|
| P3 false | `E4 ≈ 0` (oracle ceiling) | report P1/P2; stop router work |
| Oracle-only gain | `E4 > 0`, `E4n ≈ 0`, `E5 ≈ 0` | report ceiling; deployable P3 negative |
| No-oracle false positive | `E4n > 0` but `align ≈ grid` | verify E5; treat as surprise, not mechanism |
| P1 false | `total_oracle ≥ unigram` | bottleneck sweep; smaller K/Q, larger stride |
| Codebook collapse | `util < 0.2`, flat RD | lower `lam_fixed`; utilization floor; commitment |
| Router at chance | `align ≈ grid` | byte-surprise; CF-in-the-loop; oracle ceiling |
| False positive `I` | `I > 0` but `align` at chance | E5 mandatory; report oracle gap |
| P1 true, P2 false | depth ratio < 8 at matched quality | report P1/P3; tune k passes |
| Underpowered | wide CI | 3 seeds, paired windows, pre-registered ε |
| Compute burn | — | phase gates; successive halving; resume |

---

## 10. Deliverables

1. `hncm5.py` with the oracle/fixed/rate-only/boundary/BPE chunkers and one gate registry.
   Split only if a second caller appears.
2. `runs/<phase>/results.json` per trial; `hncm5.db` only if Optuna is used; Pareto plot.
3. `REPORT.md`: P1, P2, and P3 separately; oracle upper bound; negative results included.
4. A negative result is acceptable and publishable if the controls are clean.

## 11. Decisions and remaining questions

Decided:

1. Co-primary claims: P1 (absolute compression), P2 (parallelism dividend), P3 (mechanism).
   Report all three; do not select one post hoc.
2. No global `rho=1.0`. Match dynamic to fixed (`rho_dynamic = achieved(fixed)`); sweep the
   fixed codec's `lam_fixed` to trace the absolute frontier.
3. Start with one file, `hncm5.py`. Split only if a second caller appears.
4. Oracle onsets are allowed on the control corpora as a **ceiling** (`E4`), and as a
   supervised upper bound in Phase 2. The **deployable** arm is no-oracle (`E4n`, `E5`); never
   use oracle onsets on shakespeare.

Remaining:

5. Is a **BPE chunker** an acceptable reference arm? It is a strong, cheap control for
   "content chunking helps" without a learned router.
