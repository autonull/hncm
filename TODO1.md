# TODO1 — H-NCM v6: learn the cost, solve the structure

Status: `hncm5.py` is a diagnostic, not evidence. Its codec collapsed (`util = 1/64`),
its boundaries were free side information, and its corpora were mismatched to the
claims. This plan replaces it with `hncm6.py`: one shared codec, one cost field,
exact DP tiling, scalar quantization, a low-rank hierarchical prior, span-relative
decoding, and closed-loop generation. No backwards compatibility. `hncm5.py` stays as
the reference implementation.

## Progress (2026-11-23)

`hncm6.py` now implements the v6 architecture: one shared codec, scalar
quantization with universal noise, a low-rank hierarchical prior, a span-relative
decoder, exact shortest-path DP tiling, `fixed`/`BPE`/`IsoWarp`/`oracle`/
`oracle-costed`/`random` controls, byte AR/GPT/unigram/gzip baselines, staged
training, atomic `results.json`, fingerprint-checked resume, gates, report, and
`_selftest`. `ruff format` and `ruff check` pass.

Two measurement defects were fixed before the budget run. Both inflated the codec:

1. **The entropy model conditioned on the latent it was coding.** `ScalarQ` took
   `ctx = [z, mu, gain@Fh, hq]` and predicted the code from `z`. The decoder
   never has `z`, so the reported `code_bpb` was a fantasy: `CE=0.02` bpb on eval
   against an empirical code entropy of `H=2.96` bpb. `SpanEnt` now conditions
   only on what the decoder has: the span start, the span length, the transmitted
   global factor `h`, and the already-sent codes (previous span, earlier dims).
2. **The rate was charged per position, not per span.** `pool` packs one code per
   span at index `s < S`, but `quantize` charged `L` codes and `code_info` counted
   the zero-derived padding tail. For stride 4 that is a 4x overcharge, and the tail
   deflated `CE` because its codes are constants. The packed prefix is now the only
   thing charged, decoded, and measured (`span_layout`, `valid`).

Also fixed: `pos="span"` (the default decoder mode) computed the span position
embedding and never added it, so the main mode silently dropped the span geometry.
`constant-code`/`no-quant` now use the same position mode as the scalar codec, and
`rel-pos` is the main span-relative arm (train and eval agree).

## Redesign (2026-11-24): the code must be a dictionary index

The v6 plan's codec is retired by measurement, not by opinion. The span code is a
12-bit scalar quantization of a *mean*, and a mean is permutation-invariant: the
encoder's per-byte features recover the bytes exactly (a probe reaches 0.0 bpb),
but the mean cannot tell `abab` from `aabb`, so the code carries ~1 bpb while it
costs 3.1 bpb. A probe from the same code (a stronger decoder than the codec's)
reaches 3.42 bpb, so the decoder is not the limit. `total = code + residual` is
monotone in the code rate (slope 0.30, unchanged at 1200 steps, at every level,
`d_c`, stride, pool and tiling), and the optimizer's optimum is to collapse the
code (the attention pool does exactly that: `total=4.79`, no code). The oracle
tiling lies on the same R-D line as the grid.

`hncm7.py` changes what the code is: the *index* of a span in a *free*
dictionary. The decoder writes the exact bytes, so the residual is zero for
dictionary spans, and spans outside the dictionary pay the byte unigram. Only the
indices and the fallback are paid. The tiling is still solved by the exact DP,
with a greedy arm on the *same* cost model, so the DP/greedy gap isolates solving
from guessing (`runs/v7/results.json`):

| corpus | unigram | gzip | DP | greedy | oracle | fixed4 |
|---|---:|---:|---:|---:|---:|---:|
| synseg | 4.640 | 2.225 | **1.285** | 1.285 | 1.285 | 4.591 |
| syngrid | 3.911 | 2.886 | **2.051** | 2.051 | 2.051 | 2.051 |
| shakespeare | 4.777 | 3.218 | **2.709** | 2.738 | 2.776 | 4.728 |

The representation wins (below the unigram and gzip on all three corpora, with a
parallel decoder) and the tiling is decisive: the fixed grid destroys the value
(4.59/4.73 on synseg/shakespeare); on syngrid the grid *is* the oracle, so the
fixed arm ties the DP and the plan's negative control holds. The solver is *not*
the win: on a word dictionary the greedy already finds the words, so the DP ties
the greedy except on shakespeare (+0.029). The DP does beat the oracle on
shakespeare by splitting rare words into the cheaper byte fallback.

Caveat: the v7 vocabulary is built from the train word onsets (supervised); the
tiling is not (the DP uses only costs). Next: build the vocabulary without
onsets (a suffix automaton / BPE / a learned candidate generator), then test the
solving claim on an overlapping word+phrase vocabulary where the greedy is myopic.

## Step 1 (2026-11-24): the model is charged, and it is the frontier

The v7 dictionary is data, so it is charged: `512 + sum(len+3)` bytes, amortized
over the stream (train+val), and every arm pays the same index costs. The
dictionary is pruned by train frequency (`min_count`), so the model/rate trade is
explicit (`runs/v7/results.json`):

| corpus | unigram | gzip | DP rate | model | DP charged | crossover (gzip) |
|---|---:|---:|---:|---:|---:|---:|
| synseg | 4.640 | 2.225 | 1.288 | 0.041 | **1.329** | 0.05 MB |
| syngrid | 3.911 | 2.886 | 2.051 | 0.018 | **2.068** | 0.03 MB |
| shakespeare | 4.777 | 3.218 | 3.004 | 0.341 | 3.345 | 1.78 MB |

The win survives where the model is small (synseg/syngrid, 6.1/2.6 kB) and does
not survive on shakespeare at 1.1 MB: the best prune is 5,127 spans (47.6 kB) and
the charged total loses to gzip by 0.127 bpb, tying it at 1.78 MB. The budget
curve (shakespeare): min_count 1 -> 4,985; 2 -> 3,635; 4 -> 3,345; 8 -> 3,368;
16 -> 3,469; 32 -> 3,756. The optimum is a compromise, and the frontier is set by
the stream size.

The tiling stays decisive under the charge: the fixed grid costs
4.63/2.07/5.07. On syngrid the grid *is* the oracle (`align=1.000`), so the fixed
arm ties the DP: the negative control holds. The DP beats the oracle on synseg
(1.288 vs 1.347) by splitting rare words into the cheaper byte fallback. The
solver is *not* the win: `DP == greedy` on synseg/syngrid and 0.001 bpb apart on
shakespeare, so the falsifier is confirmed on a word dictionary.

Next: an unsupervised vocabulary; then an overlapping (word+phrase) vocabulary
where the greedy is myopic; then strong baselines (brotli/xz) and enwik8 with the
model charged; then the parallelism measurement at matched bpb.

## Step 2 (2026-11-24): the vocabulary is text-only, and the solver now matters

The onset supervision is removed: the candidate set is every substring of the
train text (<= 16 bytes, count >= 2, one pass per length with a rolling hash),
pruned by count and charged. The word dictionary stays as a baseline
(`runs/v7/onsets.json`). All arms pay the same model and index costs:

| corpus | vocab | dict | model | unigram | gzip | DP | charged | greedy | fixed4 | oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synseg | word | 894 | 0.047 | 4.640 | 2.225 | 1.692 | 1.739 | 1.692 | 4.605 | 1.692 |
| synseg | substr | 8,361 | 0.420 | 4.640 | 2.225 | **1.682** | **2.101** | 1.751 | 3.306 | 1.913 |
| syngrid | word | 558 | 0.018 | 3.911 | 2.886 | 2.551 | 2.568 | 2.551 | 2.551 | 2.551 |
| syngrid | substr | 2,478 | 0.100 | 3.911 | 2.886 | **2.530** | **2.630** | 2.551 | 2.551 | 2.551 |
| shakespeare | word | 5,127 | 0.341 | 4.777 | 3.218 | 3.390 | 3.731 | 3.400 | 4.760 | 3.444 |
| shakespeare | substr | 5,792 | 0.292 | 4.777 | 3.218 | **3.133** | 3.426 | 3.352 | 4.106 | 4.262 |

Results:

- **Text-only survives the charge on synseg/syngrid** (2.101 vs gzip 2.225;
  2.630 vs 2.886), so the earlier win did not depend on supervised word
  boundaries.
- **Shakespeare stays model-bound at 1.1 MB**: 3.426 vs gzip 3.218, crossing at
  3.85 MB (text-only) or ~1.8 MB (word).
- **The solver now matters**: on the word dictionary (no overlap) the DP vs
  greedy gain is +0.000/+0.000/+0.010; on the text-only overlapping vocabulary
  it is **+0.070/+0.021/+0.219 bpb**. That is P3 with clean controls: solving
  beats guessing exactly where guessing is myopic.
- The DP beats the fixed grid by 1.62/0.02/0.97 and the oracle by
  0.23/0.02/1.13. The oracle is a baseline, not a ceiling, for a substring
  vocabulary; `DP <= oracle` is an invariant of the optimal cover.

Gates: G0/G1/G2 (charged vs gzip/unigram) pass on synseg/syngrid and fail on
shakespeare; G5 (DP < fixed) passes on all three; G5c (DP <= greedy) passes on
all three, with a positive gain only on the overlapping vocabulary.

Next: one cost model for both vocabularies (a token-usage unigram with an EM
pass) so the comparison is exact; enwik8 with the model charged at 1/10/100 MB;
brotli/xz/zstd and a byte transformer; the parallelism dividend at matched bpb.

## Step 3 (2026-11-24): one cost model, and the honest verdict

Both vocabularies now use the same model: a token-usage unigram fitted by EM on
the train (the DP re-tokenizes, the usage becomes the cost, unused strings drop),
with the vocabulary size swept and the model charged. `runs/v7/results.json`
(text-only) and `runs/v7/onsets.json` (word):

| corpus | vocab | dict | model | unigram | gzip | DP | charged | greedy | fixed4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synseg | word | 768 | 0.038 | 4.640 | 2.225 | 1.290 | **1.328** | 1.290 | 4.592 |
| synseg | text | 4,352 | 0.463 | 4.640 | 2.225 | 1.880 | 2.343 | 1.978 | 4.626 |
| syngrid | word | 558 | 0.018 | 3.911 | 2.886 | 2.051 | **2.068** | 2.051 | 2.051 |
| syngrid | text | 512 | 0.018 | 3.911 | 2.886 | 2.573 | **2.590** | 2.573 | 3.906 |
| shakespeare | word | 4,352 | 0.285 | 4.777 | 3.218 | 3.062 | 3.347 | 3.062 | 4.725 |
| shakespeare | text | 2,304 | 0.156 | 4.777 | 3.218 | 3.299 | 3.455 | 3.382 | 4.647 |

- **The word vocabulary pays**: charged 1.328 vs gzip 2.225 (synseg) and 2.068 vs
  2.886 (syngrid), 0.90/0.82 bpb, with a 7.0/2.6 kB model.
- **The text vocabulary's solver gain does not buy back its model**: on synseg it
  gains +0.098 from solving but pays −0.425 bpb more model (4,352 vs 768 entries),
  so it loses (2.343 vs 1.328).
- **The solver is real but conditional**: DP − greedy = 0.000 on the word
  vocabulary (no overlap); +0.098 (synseg) and +0.084 (shakespeare) on the
  overlapping text vocabulary; 0.000 on syngrid (the grid is the structure).
- **Shakespeare is model-bound at 1.1 MB**: word 3.347 vs gzip 3.218 (0.129 loss,
  crossing ~2.0 MB); text 3.455 (0.237 loss). The budget optimum (word) is 4,352
  entries: below it the rate pays, above it the model pays.

The earlier `count`-normalized numbers were optimistic: the unified usage model
raises the word rate slightly and the text rate by ~0.2 bpb, which flips the text
synseg arm from a pass to a loss. That is the honest verdict: the representation
pays where the model amortizes, and the solver pays only where it must be paid
for with a bigger model.

Caveat: the vocabulary size is chosen on the charged total reported on the same
val (8 points, one knob) — a selection leak to remove with a held-out split.

Diagnostics only (no claim is established). The v6 R-D curve, honest, seed 0, 300
steps, `eval_windows=128`, synseg:


| levels | d_c | stride | recon | code | total | unigram |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 16 | 4.544 | 0.248 | 4.792 | 4.637 |
| 4 | 1 | 4 | 4.385 | 0.620 | 5.005 | 4.637 |
| 8 | 2 | 4 | 4.143 | 1.581 | 5.724 | 4.637 |
| 8 | 4 | 4 | 3.715 | 3.080 | 6.795 | 4.637 |
| 8 | 1 | 16 | 4.536 | 0.283 | 4.818 | 4.637 |
| 8 | 2 | 16 | 4.498 | 0.425 | 4.923 | 4.637 |

On syngrid the best point is 4.011 (levels 4, `d_c` 1, stride 16) against unigram
3.916. At 1200 steps the slope is unchanged (synseg, stride 4): `d_c=1`
4.323+0.847=5.170; `d_c=2` 4.106+1.570=5.676; `d_c=4` 3.586+3.063=6.648.
Reconstruction improves
by 0.3-0.5 bpb per extra bpb of code at every operating point, so `total = recon + code`
is monotone in the code budget and bounded below by the unigram (code 0). No
`lambda` and no tiling can pass G0 at this codec: the *code transfer*, not the
tiling, is the bottleneck (the code conveys ~4 of its 12 bits per span to the
reconstruction).

Full Phase 0, preset `quick`, 3 seeds, 512 windows, 400 steps, `format: 7`
(`runs/p0/quick-c3b7a8ac/`):

| corpus | arm | total | recon | code | H(code) | util |
|---|---|---:|---:|---:|---:|---:|
| synseg | rel-pos | 6.891 | 3.752 | 3.138 | 381.0 | 1.000 |
| synseg | abs-pos | 6.844 | 3.758 | 3.085 | 376.9 | 1.000 |
| synseg | constant-code | 4.642 | 4.642 | 0.000 | 0.0 | 0.0 |
| synseg | no-quant | 3.758 | 3.609 | 0.149 | 0.0 | 0.0 |
| synseg | unigram | 4.637 | 4.637 | 0.000 | - | - |
| syngrid | rel-pos | 5.975 | 2.863 | 3.112 | 377.0 | 1.000 |
| syngrid | abs-pos | 5.958 | 2.839 | 3.119 | 378.6 | 1.000 |
| syngrid | constant-code | 3.909 | 3.909 | 0.000 | 0.0 | 0.0 |
| syngrid | no-quant | 2.812 | 2.664 | 0.148 | 0.0 | 0.0 |
| syngrid | unigram | 3.908 | 3.908 | 0.000 | - | - |

Gates: **G0 fail** (total above unigram on both corpora; util, delta, H pass),
**G6 pass** (`rel-abs = -0.006/+0.024`), **G9 pass** (3 seeds). The code is fully
used and worth 0.89/1.05 bpb of reconstruction, but it costs 3.14/3.11 bpb. The
honest rate is ~`log2(levels) = 3` bits/symbol: the decodable context cannot
compress the code at all (`CE ≈ H`).

Artifacts: `runs/p0/quick-c3b7a8ac/`. Superseded and misleading:
`runs/p0/quick-dbe8d386/` (`format: 6`: per-position rate, entropy model
conditioned on `z`), `runs/p1/smoke-37d2450a/`, `runs/p2/smoke-bad03cb9/`.

Per the decision tree G0 fails, so Phase 1 was **not** run: the DP cannot be
interpreted on top of an uncompetitive codec. The negative is publishable: the
controls are clean and the accounting is now honest. The next decisive step is code
transfer efficiency, not the tiling: the reconstruction must gain >1 bpb per code bit
at the operating point (now ~0.29), e.g. a decoder that predicts bytes
conditionally from the *span content* (a wider code or an autoregressive decoder
inside the span), retrained end to end. The tiling only makes sense once the code
pays for itself.


---

## 0. What v5 established (and what it did not)

v5 established plumbing only: frozen `Cfg`, presets, JSON overrides, a
fingerprint-checked resume, an atomic `results.json`, a gate registry, a report, and a
`_selftest`. It did **not** establish the thesis. The quick run (L=128, d=96, Q=3,
K=64, stride=4, 300 codec steps) is the clearest evidence:

| corpus | unigram | byte GPT | fixed+CF | oracle+CF | rate-only+CF | util | align fixed/oracle/rate-only |
|---|---:|---:|---:|---:|---:|---:|---|
| synseg | 4.638 | 4.295 | 4.638 | 4.607 | 4.640 | 1/64 | 0.145 / 0.579 / 0.150 |
| syngrid | 3.911 | 2.785 | 4.123 | 4.142 | 3.950 | 1/64 | 1.000 / 1.000 / 0.258 |

Read this table literally:

- The codec ties unigram on synseg (`4.638` vs `4.638`) and loses to it on syngrid
  (`4.123` vs `3.911`). P1 fails.
- The codec has `util = 1/64`: each RVQ level uses exactly one code. The code is
  decorative.
- The codec is far worse than the byte GPT on both corpora (`4.638` vs `4.295`;
  `4.123` vs `2.785`). P1b fails.
- The oracle improves the total by only `0.031` bpb on synseg, and it *hurts* on
  syngrid (`+0.019`), where the oracle tiling *is* the grid.
- The oracle improves AR sample quality by `0.089` bpb but CF sample quality by only
  `0.116`; the interaction is `I = (5.072-5.188) - (5.036-5.125) = -0.027`. The
  *sample-quality* sign is negative, so the boundary is not the CF-specific bottleneck.
- The rate-only arm is not a null: its AR sample quality is `5.234`, so it is a
  different codec, not a different boundary.

### 0.1 Root causes

1. **Codec collapse.** The rate term is minimized by a constant code, and the decoder
   can reconstruct from absolute position alone, so nothing forces the code to carry
   information. The codebook collapses to one code per level.
2. **Absolute position is a free side channel.** `Codec.decode` uses
   `dpos.weight[:L]`; the decoder models the window from position. The code is not
   needed.
3. **Boundaries are free in every arm.** The tiling (`order`, `offset`, `length`) is
   passed to the decoder and never paid for in the rate. The oracle is free; the
   router is free; the fixed grid is free.
4. **Per-chunker codecs confound the boundary effect.** Each chunker trains its own
   codec, so `I` mixes boundary placement with codec capacity. The oracle helps AR
   more than CF partly because the AR codec is different.
5. **The rate target is on the wrong quantity.** The dual ascent targets the code rate
   (`rho=3.4`) with a shallow entropy head; this is a pessimistic proxy and it
   collapses the codebook.
6. **"Sample quality" is a judge.** `ref_bpb` is byte GPT cross-entropy, not the
   model's own description length. It is confounded by the judge's capacity.
7. **P1 is evaluated on synthetic corpora.** synseg is 8 topics x 40 words; syngrid is
   4-byte aligned. Neither is a compression benchmark; the unigram baseline is not
   representative.
8. **Calls are not compute.** A CF call is a full forward over the window; an AR call
   is one token. The ratio of calls is not a ratio of wall-clock or FLOPs.
9. **The baseline is not matched.** The byte GPT is a different size and budget from
   the codec/prior.
10. **Statistics are underpowered.** Two seeds; CI over windows only; no seed CI.
11. **The entropy model conditioned on the latent it coded.** `ScalarQ` predicted
    the code from `z`, which the decoder never has; the reported code rate was a
    fantasy (`CE=0.02` vs empirical `H=2.96` bpb). Measurement defect.
12. **The rate was charged per position, not per span.** One code per span is packed
    at `s < S`, but `L` positions were charged and the zero-derived padding tail
    was counted in `H(code)`; a 4x overcharge for stride 4, and the tail deflated
    `CE`. Measurement defect.
13. **The code transfer is the bottleneck.** Honest Phase 0: the code carries
    information (`util=1.0`, `H≈380` bits/window) and the decoder uses it
    (`delta_recon=0.89/1.05` bpb), but it costs 3.14/3.11 bpb and buys 0.29 bpb
    per bit. `total` is monotone in the code budget at every swept point and at 4x
    the steps, so no `lambda` or tiling can pass G0. The tiling cannot fix a code
    that does not pay for itself.

### 0.2 What is salvageable

- The gate registry, fingerprint/resume, atomic results, paired windows, seed
  factorial, `_selftest` discipline, and "report all claims, including negatives".
- The oracle and grid controls; the byte AR/GPT/unigram baselines.
- The capability registry and the provenance JSON.

---

## 1. Thesis and claims

### 1.1 The corrected thesis

> **H-NCM v6** learns a rate–distortion cost field over byte spans and solves the
> optimal tiling exactly with a shortest-path DP. The tiling is a **transmitted code**;
> the span codes are **scalar-quantized** with a **low-rank hierarchical prior**; the
> decoder is **span-relative** and the generator is **closed-loop, confidence-gated
> parallel commitment**. The parallelism dividend comes from **solving** the structure,
> not learning it.

The v5 thesis ("merge H-Net chunking with NCM RVQ + parallel decoding") is retired: it
learns the structure and the code, and both fail. v6 separates them: the codec is
learned; the tiling is inferred; the code is paid for; the decoder is bottlenecked.
v6 composes q2's exact tiling, q3's scalar/low-rank code, and v5's protocol.

### 1.2 Claims and corpora

| Claim | Corpus | Primary endpoint | Controls / baselines | Gates |
|---|---|---|---|---|
| P1 absolute compression | `shakespeare` | `total_bpb` | unigram, byte GPT (matched), gzip, fixed | G1, G2, G2b |
| P2 parallelism dividend | `shakespeare` | `calls` and wall-clock at matched `total_bpb` | fixed, DP, AR | G3, G3b |
| P3 boundary mechanism | `synseg`, `syngrid` | `I_structure` at matched `lambda` | fixed, oracle, oracle-costed, BPE, IsoWarp, random | G4, G4n, G5-G5d |
| Health / leak | `synseg`, `syngrid` | `util`, `code_info`, `delta_recon` | constant code, no quant, absolute position | G0, G6 |
| Quantizer / prior | `synseg`, `syngrid` | `I_quant`, `I_corr` | VQ, mean-field | G7, G8 |

P1 is not evaluated on synseg/syngrid. P3 is not claimed on shakespeare (no oracle
there). P1/P2/P3 are independent; a P1 failure does not invalidate P2/P3.

### 1.3 The factorial decomposition

Let `T(arm, lambda)` be the total bpb at rate knob `lambda`. At **matched lambda**:

```
I_structure = T(fixed, lambda) - T(DP, lambda)
I_oracle    = T(fixed, lambda) - T(oracle, lambda)      (free boundaries: ceiling)
I_quant     = T(VQ, lambda)    - T(scalar, lambda)
I_corr      = T(mean-field, lambda) - T(low-rank, lambda)
```

At **matched total bpb** (`|dT| <= 0.05`), compare `calls` and wall-clock for P2.

The README's `I = (D-C) - (B-A)` is the **free-boundary oracle interaction**; it is a
ceiling, not the deployable claim. The deployable claim is `I_structure` with a **shared
codec** and **paid tiling**.

### 1.4 Sign predictions

- `I_structure > 0` on synseg; `I_structure ~ 0` on syngrid (the grid is already the
  oracle; the DP cannot beat it). This is the negative control.
- `I_oracle > 0` on synseg; `I_oracle ~ 0` on syngrid.
- `align(DP) >= align(grid) + 0.05` on synseg; `align(DP) ~ align(grid)` on syngrid.
- `I_quant >= 0` if scalar quantization beats VQ; `I_corr >= 0` if the low-rank prior
  beats mean-field. Both are reportable, not assumed.
- **Falsifier:** if `I_structure <= 0` with a healthy codec and paid tiling, the DP does
  not help; the structure claim is false. Report it.

---

## 2. Architecture (`hncm6.py`)

One file. No backwards compatibility. Split only when a second caller appears.

### 2.1 Data flow

```
bytes x
  -> encoder E(x)                  # per-byte features, bidirectional or causal
  -> cost field C(E)               # boundary/length logits
  -> tiling T = argmin DP(C, lambda) # exact spans; transmitted; paid
  -> span means z = pool(E, T)       # O(1) via cumsum
  -> global factor h ~ N(0, I_r)      # transmitted once
  -> span codes y = quant(z; h)        # scalar symbols; entropy-coded
  -> decoder D(y, span-relative pos)   # reconstruct bytes, parallel within span
  -> total_bits = recon + code + boundary
```

The **shared codec** is trained once per corpus. The **tiling is inference**: fixed,
DP, BPE, IsoWarp, oracle, or random. The codec never sees the chunker's identity during
training; it sees **random tilings** (and optionally the DP tiling), so it is tiling
agnostic. This is the single most important experimental fix.

### 2.2 Scalar quantization

Replace RVQ with a per-dimension scalar quantizer:

```
y = z + U(-0.5, 0.5)                 # universal noise (training)
p(k | span) = softmax(MLP(span feature))  # conditional entropy model
rate = -log p(k | span)              # actual bits
dequant = (k + 0.5) / levels - 0.5     # or the conditional mean
```

- No codebook, no commitment, no restart, no STE, no `util` collapse in the RVQ sense.
- The code is a vector of `d_c` symbols; the rate is the sum of the per-symbol rates.
- `levels` in {4, 8, 16}; `d_c` in {4, 8}; ablation against VQ.

### 2.3 Low-rank hierarchical prior

```
h ~ N(0, I_r)                          # global factor, transmitted once
mu(span), gain(span), sigma(span)       # from the span feature
y = mu + gain @ (F h) + sigma * eps     # correlated span code
rate = -log p(h) + sum -log p(y | h)    # hierarchical code
```

- `r` in {0, 4, 8, 16}; `r=0` is the mean-field control.
- `h` is the **coarse code**; `y` is the **fine code**; this is NCM's coarse-to-fine,
  but with a *transmitted* `h` and a *scalar* code.
- The marginal (`h` integrated out) is a control: `rate = -log p(y)`.

### 2.4 Cost field and exact DP

```
cost[t, ell] = -log p(length=ell | h_t) - log p(boundary at t | h_t)
               + lambda * ell + code_rate(span) + recon(span)
best[t] = min_ell best[t-ell] + cost[t, ell]
```

- `dp_tile` is the shortest path in a DAG; optimal by induction on the prefix.
- `O(L * |lengths|)` with cumsum span pooling; `O(1)` per span mean.
- The gradient flows through the **min path** (envelope theorem); no STE on boundaries.
- The tiling is **transmitted**; its cost is `-log p(length) - log p(boundary)`.
  Fixed and oracle are free; DP/BPE/IsoWarp pay. This is the rate accounting v5 lacked.

### 2.5 Span-relative decoder

The decoder sees:

- the dequantized span code `y` (the content channel),
- the **within-span relative position** (0..ell-1),
- the **span length** and the **span index** (the coarse address; part of the tiling).

It does **not** see the absolute byte position. The leak gate (G6) measures the
difference: `delta_recon = recon(span-relative) - recon(absolute)`. If removing absolute
position costs more than `0.10` bpb, the code was decorative; stop and fix the decoder.

### 2.6 Closed-loop generation

```
canvas = MASK
repeat:
  cost = C(E(canvas))
  T = DP(cost, lambda)
  confidence = p(span | canvas)
  commit the high-confidence spans in parallel
  re-encode the canvas; freeze committed spans
until the canvas is covered or the step budget is exhausted
```

The NFE is content-adaptive: easy spans commit in bulk; hard spans get more rounds.
The ablation is one-shot commitment (all spans at once).

### 2.7 Training

Staged, because it is more stable:

1. **Codec stage.** Freeze the cost field; train the codec with **random tilings**
   (span lengths drawn from the prior); loss = `recon + code` (no boundary). This
   makes the codec tiling-agnostic and forces the code to carry information.
2. **Cost-field stage.** Freeze the codec; train the cost field with the **DP**; loss
   = `total` (recon + code + boundary). The min-path gradient updates the cost field.
3. Repeat 1-2.

Ablation: joint training (both stages at once). The staged protocol is the main.

### 2.8 What is deleted and kept

**Deleted:** RVQ, commitment, restarts, `RateOnlyChunker`, `BoundaryChunker`'s boundary
heads, the dual ascent, the ratio loss, the smoothing EMA, the gates, the nested
dropout, the corruption chains, the K-level weights, the per-chunker codec, the
absolute-position decoder, the free boundaries.

**Kept:** the oracle, BPE, IsoWarp, fixed, random, and constant-code controls; the
byte AR, byte GPT, unigram, and gzip baselines; the gate registry, fingerprint/resume,
atomic results, paired windows, seed factorial, `_selftest`, and provenance.

### 2.9 Module layout

```
hncm6.py
  cfg        # frozen Cfg; presets; HNCM_CFG JSON
  data       # synthetic + shakespeare + enwik8; onsets; word marginal; windows
  quant      # ScalarQ: levels, universal noise, entropy model
  prior      # LowRankPrior: global h, conditional span codes; mean-field control
  canvas     # encoder, pool, quantize, decode; span-relative
  tiling     # dp_tile; fixed; bpe; isowarp; oracle; oracle-costed; random
  codec      # encode/decode; total_bits; code_info; rate; util
  train      # staged cost/codec; joint ablation; lambda sweep; resume
  metrics    # total/recon/code/boundary; align; calls; wall; NFE; TV
  gates      # registry; decisions
  report     # results.json; REPORT.md; plots
  selftest   # DP vs brute force; cumsum pooling; exact cover; entropy round-trip; leak
  cli
```

### 2.10 Interface contract and AGENTS.md conformance

**Seams (`Protocol`).** `Tiling.boundaries(h, x) -> Boundary`,
`Quantizer.encode/decode`, `Prior.rate/sample`, `ResultStore.save/load`. Names are
`Literal`; `Cfg` is a frozen `dataclass(slots=True)`; only `Cfg`, `RESULT_FORMAT`,
and the read-only registries are global constants. No mutable global state.

**One source of truth.** `Cfg` owns every threshold (`g0_util`, `g0_code_info`,
`g1_margin`, ...); the gate registry owns every endpoint; metrics are defined once; the
capability table is read-only. No second table repeats a threshold.

**AGENTS.md rules.** One file; no backwards compatibility; no pydantic (JSON at the
boundary only); no import-time I/O; `main()` + CLI; atomic `results.json`; resume
refuses a changed `cfg` or corpus sha; seed/log every run; `uv` + `ruff format`;
`ruff check --fix` on changed files; tests only invariants that broke (DP optimality,
exact cover, entropy round-trip, resume, shared init). Use context managers for files
and GPU; catch specific exceptions; no bare `except`; guard clauses.

**Efficiency contract.** DP is `O(L * |lengths|)`; cumsum pooling is `O(1)` per span;
the decoder is parallel within spans; one codec per corpus; one shared prior. The
minimal decisive run and the full budget are in S5.7.

**Effectiveness contract.** Every claim has a gate and a stop rule; `total_bpb` is the
only primary metric; `I_structure` is measured at matched `lambda`; the oracle is a
ceiling; the controls are paid. Statistics are in S5.6.

---

## 3. Controls and baselines

| Control | Purpose | Free or paid? |
|---|---|---|
| `fixed` (stride 4/8/16) | no structure; the grid | free |
| `oracle` (true onsets) | ceiling; best tiling | free (ceiling) |
| `oracle-costed` | strong control; transmit the tiling | paid |
| `BPE` | heuristic tokenizer control | paid |
| `IsoWarp` | learned monotone warp control | paid |
| `random` | negative control; random tiling | paid |
| `constant-code` | code is decorative | free |
| `no-quant` | quantization ablation | free |
| `absolute-position` | leak ablation | free |
| `unigram` | entropy baseline | free |
| `byte AR` | matched AR baseline | free |
| `byte GPT` | matched neural baseline | free |
| `gzip` | real codec baseline | free |

The oracle is a ceiling, not a deployable arm. The BPE/IsoWarp are strong controls;
if the DP loses to them, report it.

**Cost accounting (one rule).** The fixed grid and the oracle are free (the decoder
knows them); the DP, `oracle-costed`, BPE, IsoWarp, and random tilings are paid:
`rate = -log p(tiling)`. The BPE transmits its token mask; IsoWarp transmits `rho`;
the random tiling transmits its lengths; the DP transmits its span addresses/lengths.
This is why the v5 comparison was invalid: all v5 arms had free boundaries.

---

## 4. Endpoints, gates, and decisions

### 4.1 Gate registry (one source: `Cfg`)

| Endpoint | Gate | Claim | Test | Threshold |
|---|---|---|---|---|
| E0 | G0 | health | codec carries info | `util >= 0.20`, `delta_recon(const-enc) >= 0.10`, `H(code) >= 2`, `recon+code < unigram` |
| E1 | G1 | P1a | total on shakespeare | `total < unigram - 0.10` |
| E2 | G2 | P1b | total vs byte GPT | `total <= byte GPT + 0.05` at matched params/steps |
| E2b | G2b | P1c | total vs gzip | `total <= gzip + 0.10` |
| E3 | G3 | P2a | calls at matched total | `calls_AR/calls_DP >= 8` at `|dTotal| <= 0.05` |
| E3b | G3b | P2b | wall-clock | `t_DP <= 0.5 t_AR` at matched total, batch=1 |
| E4 | G4 | P3a | oracle ceiling | `I_oracle > 0` on synseg; CI low > 0 |
| E4n | G4n | P3null | no-oracle null | `|I_DP| <= 2 SEM` on syngrid; `align ~ grid` |
| E5 | G5 | P3b | DP deployable | `I_DP >= max(0.5 I_oracle, 2 SEM)`; `align_DP >= grid + 0.05` |
| E5b | G5b | P3c | DP between | `total_oracle < total_DP < total_fixed` |
| E5c | G5c | P3d | DP vs BPE | `total_DP <= total_BPE` |
| E5d | G5d | P3e | DP vs IsoWarp | `total_DP <= total_IsoWarp` |
| E6 | G6 | leak | no absolute position | `delta_recon(abs - rel) <= 0.05`; fail if > 0.10 |
| E7 | G7 | quant | scalar vs VQ | `total_scalar <= total_VQ + 0.05`; else report |
| E8 | G8 | prior | low-rank vs mean-field | `total_lowrank <= total_meanfield + 0.05`; else report |
| - | G9 | stats | 3 seeds, paired windows | seed CI and window CI reported |

### 4.2 Decision tree

1. **G0 fails** -> codec unhealthy. Tune quantizer (levels, `d_c`), prior (rank,
   `sigma`), decoder (span-relative, capacity), and `lambda`. Do not run the DP.
2. **G6 fails** -> the code is decorative. Fix the decoder bottleneck (no absolute
   position), retrain. Do not run the DP.
3. **G0/G6 pass** -> run Phase 1.
4. **G4 fails** (`I_oracle ~ 0`) -> the ceiling is zero; no DP can pass P3. Report
   P1/P2 if G1/G3 hold; stop structure work.
5. **G4 passes, G5 fails** -> the oracle helps, the DP does not. Report the ceiling and
   the negative; stop.
6. **G4/G5/G5b/G5c/G5d pass** -> the structure holds; run Phase 2.
7. **G5c/G5d fail** -> the DP is worse than BPE/IsoWarp; report the heuristic; stop.
8. **Phase 2:** G1/G2 pass -> P1; G3/G3b pass -> P2. Report each independently; a
   P1 failure does not invalidate P2, and vice versa.
9. **Phase 3:** report P1/P2/P3 with all seeds; no cherry-picking.

---

## 5. Experiments

### 5.1 Phase 0 — codec health and leak (~20-40 min)

- Corpora: synseg, syngrid; seeds 0,1,2; 512 windows.
- Arms: `fixed4`, `fixed8`, `fixed16`, `constant-code`, `no-quant`, `abs-pos`,
  `rel-pos`.
- Train: 400 steps; batch 32; L=128; d=96; scalar levels 4/8; rank 0/4/8.
- Measure: E0, E6, plus `recon`, `code`, `util`, `H(code)`, `params`.
- Gates: G0, G6.
- **Stop if G0/G6 fail.** No DP.

### 5.2 Phase 1 — structure (~3 h)

- Corpora: synseg, syngrid; seeds 0,1,2; 512 windows.
- Arms: `fixed`, `DP`, `BPE`, `IsoWarp`, `oracle`, `oracle-costed`, `random`.
- `lambda` in {0, 1, 2, 4}; 400 steps; batch 32; L=128; **one shared codec**.
- Train: staged (codec with random tilings; cost field with DP); ablation joint.
- Successive halving: 300 steps for all arms; promote the top half to 1200. No HPO.
- Measure: E4, E4n, E5-E5d, `align`, `calls`, `wall`, `NFE`, `params`.
- Gates: G4, G4n, G5-G5d.
- **Stop if G4 fails.** Report ceiling.

### 5.3 Phase 2 — compression and parallelism (~4 h)

- Corpus: shakespeare; seeds 0,1,2; 2048 windows.
- Arms: `DP`, `fixed`, `AR`, `byte GPT`, `unigram`, `gzip`; `lambda` sweep to match
  total.
- Train: 1500 steps; L=256; d=192. Match the byte GPT at the same `d`, layers,
  steps, batch, and bytes; log its params. An unmatched P1b is not evidence.
- Measure: E1, E2, E2b, E3, E3b; total/recon/code/boundary; calls; wall.
- Gates: G1, G2, G2b, G3, G3b.
- **Report P1 and P2 independently.**

### 5.4 Phase 3 — scale and report (~2 h)

- Corpus: shakespeare (enwik8 optional); seeds 0,1,2; 1500 steps.
- Arms: best DP + controls; no oracle on shakespeare for the headline.
- Report: P1/P2/P3 separately; samples; plots; provenance.
- **A negative result is publishable if the controls are clean.**

### 5.5 Ablations (~1 h)

- Quantizer: scalar vs VQ.
- Prior: low-rank vs mean-field; rank in {0,4,8,16}.
- Position: absolute vs span-relative.
- Structure: fixed vs DP; BPE vs DP; IsoWarp vs DP.
- Training: staged vs joint.
- Generation: closed-loop vs one-shot.
- Lengths: {1,2,4,8} vs {1,4,16}.
- Cost field: causal vs bidirectional.

### 5.6 Statistics

- 3 seeds (0,1,2) for every decisive phase.
- Paired windows: the same window indices across arms; 512 (Phase 0/1), 2048 (Phase 2/3).
- Primary CI uses non-overlapping validation windows; overlapping windows are a sensitivity
  check, because random starts overlap and inflate confidence otherwise.
- Bootstrap 2000 over windows and 2000 over seeds; report **both** CIs (the larger).
- Pre-register epsilon = 0.05; report median and CI; no cherry-picking; no post-hoc
  selection.

### 5.7 Budget

Measure the step time first; then plan. On the 3080:

| Phase | Minimal decisive | Full | Steps | Minimal time |
|---|---|---|---|---|
| 0 | 3 arms x 3 seeds x 2 corpora | 7 x 3 x 2 | 400 | ~20 min |
| 1 | 3 arms x 3 seeds x 2 corpora x 2 lambda | 7 x 3 x 2 x 3 | 300 -> 1200 | ~2 h |
| 2 | 4 arms x 3 seeds x 1 corpus x 2 lambda | 6 x 3 x 1 x 3 | 1200 | ~2.5 h |
| 3 | 2 arms x 3 seeds | 2 x 3 | 1500 | ~1 h |
| Ablations | 2 x 3 | ~6 x 3 | 400 | ~30 min |

**Minimal decisive ~6 h; full ~10 h.** Phase 1 uses successive halving: 300 steps for
all arms, promote the top half to 1200; never HPO. Paired windows and 3 seeds reduce
variance more cheaply than more arms. Run the minimal first; expand only when the gates
pass. If the measured time exceeds the budget, cut an arm, not the seeds.

A **toy diagnostic** is separate: 4 arms x 1 seed x synseg, L=64, d=64, 1 layer, 150
steps; ~2-8 min measured. It answers one question only: does the code carry information
(`util`, `code_info`)? It cannot test P1/P2/P3; do not report it as preliminary
evidence. If the measured step time exceeds the budget, cut to 2 arms and 100 steps.

### 5.8 Artifacts, resume, and CLI

`results.json` schema (one `ResultStore`, atomic `os.replace`):

```
format, cfg_fingerprint, device, torch, corpus sha, started, finished,
runs: [{claim, corpus, seed, arm, lambda, total_bpb, recon_bpb, code_bpb,
        boundary_bpb, calls, wall_sec, NFE, align, util, params,
        samples: [...], provenance: {...}}]
```

Resume refuses a changed `cfg` or corpus sha; checkpoints are `<corpus>_<arm>_<seed>.pt`.
Plot/report reads only `results.json`; no hidden state. Data split: shakespeare 1.0 MB
train / 0.1 MB val; synseg 1.2 MB; syngrid 1.2 MB; L=128 (Phase 0/1), 256 (Phase 2/3).

```
python hncm6.py run --preset quick --phase 0 --corpora synseg,syngrid --seeds 0,1,2
python hncm6.py report --results runs/latest/results.json
```

---

## 6. Harvest table

| Idea | Source | v6 use | Status |
|---|---|---|---|
| Exact tiling DP | q2 | `tiling.dp_tile` | **main** |
| Cumsum span pooling | q2 | `canvas.pool` | **main** |
| Atomic `(length, code)` | q2 | `canvas.action` | **main** |
| Closed-loop canvas | q2 | `canvas.generate` | **main** |
| Confidence commit | q2 | `canvas.commit` | **main** |
| Min-path gradient | q2 | `canvas.loss` | **main** |
| Scalar quant + noise | q3 | `quant.ScalarQ` | **main** |
| Low-rank prior | q3 | `prior.LowRank` | **main** |
| One ELBO in bits | q3 | `codec.total` | **main** |
| IsoWarp | q3 | `tiling.isowarp` | control |
| Real codec (zlib) | q1/q3 | `baseline.gzip` | control |
| Router + STE | q1 | deleted | reference |
| RVQ | q1 | deleted | reference |
| Boundary heads | q1 | deleted | reference |
| Cached DC-state | q1 | deleted (no AR) | reference |
| Word-TV metric | q1 | `metrics.word_tv` | secondary |
| Seed factorial | q1 | all phases | protocol |
| Paired windows | q1 | all phases | protocol |
| Gate registry | v5 | all phases | protocol |
| Fingerprint resume | v5 | all phases | protocol |
| Capability registry | q1 | `CAPABILITIES` | doc |

---

## 7. Risks and mitigations

| Risk | Signal | Mitigation |
|---|---|---|
| Codec still ignores the code | G6 fails; `code_info < 0.10` | span-relative decoder; capacity; random tilings |
| DP cost field weak | `I_structure <= 0` | BPE/IsoWarp controls; report |
| Scalar worse than VQ | `I_quant < 0` | report RD; VQ control |
| Low-rank unstable | `I_corr < 0` | rank sweep; KL; mean-field control |
| DP loses to BPE/IsoWarp | G5c/G5d fail | report heuristic; stop DP |
| Oracle ceiling zero | G4 fails | report P1/P2; stop structure |
| Wall-clock ratio lower | G3b fails | report calls and wall-clock |
| P1 fails on shakespeare | G1/G2 fail | report negative; P2/P3 independent |
| Statistics underpowered | wide CI | 3 seeds; paired windows; both CIs |
| Budget overrun | measured > plan | cut search; minimal decisive run |

---

## 8. Implementation order and deliverables

1. **Implemented.** `SpanEnt` (decodable) + `LowRank` (transmitted global factor
   + per-span gains) + span-relative decoder; fixed grid; no DP. Full Phase 0, 3
   seeds, 512 windows: G0 fails on total only (util, delta, H pass), G6/G9 pass.
   The rate equals the empirical code entropy (`CE ≈ H`); the code is worth
   0.29 bpb per bit. No claim; the next step is transfer efficiency.
2. **Implemented, not run.** Cost field + `dp_tile`; staged training. The Phase 1
   smoke predates the honest accounting; do not interpret it. Do not rerun Phase 1
   before G0 passes.
3. **Implemented, not run.** Shakespeare P1/P2. The Phase 2 smoke predates the
   honest accounting; do not interpret it. No claim.
4. **Not run.** Scale/report; Phase 3.
5. **In progress.** The negative diagnostics are reported; `README.md`/`REPORT.md`
   carry the honest status and the next step; no claim.

Deliverables:

- `hncm6.py`, one file; `hncm5.py` reference.
- `runs/<phase>/results.json`; `REPORT.md`; `hncm6_<corpus>.png`.
- `TODO1.md` (this plan); updated `README.md`.
- A negative result is acceptable and publishable if the controls are clean.

---

## 9. Decisions and open questions

Decided:

1. The main architecture is DP + scalar + low-rank + closed-loop.
2. The codec is **shared**; the tiling is **inference**; per-chunker codecs are a
   confound and are forbidden for P3.
3. The boundary cost is **paid** for the DP/BPE/IsoWarp; the fixed grid and the oracle
   are free. The oracle is a ceiling.
4. The decoder is **span-relative**; no absolute position. G6 is the guard.
5. P1/P2 are on shakespeare; P3 is on synseg/syngrid. P1 is not evaluated on
   synthetic corpora.
6. `total_bits` is the only objective; `I_structure`, `I_quant`, `I_corr` are the
   factorial effects; the README's `I` is a ceiling.
7. 3 seeds; paired windows; both CIs; pre-registered epsilon.
8. The budget is measured; the minimal decisive run is ~7 h.
9. A negative result is publishable.

Open questions:

1. Does the DP beat BPE/IsoWarp? If not, the cost field is not better than a heuristic.
2. Does the global `h` help? Ablate hierarchical vs marginal.
3. Does the scalar quantizer beat VQ? Report both.
4. Is the wall-clock ratio >= 2x on a GPU at batch=1? Measure.
5. Does the causal cost field lose compression? Ablate causal vs bidirectional.
6. Does the shared codec close the per-chunker confound? It should; measure.
7. Is shakespeare large enough? Add enwik8 if the budget allows; keep the same gates.
