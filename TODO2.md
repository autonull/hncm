# TODO2 — H-NCM v7: pay for the concept, then cover the tail

Status: `hncm6.py` is the dense-latent negative, `hncm7.py` is the dictionary
redesign, and both are committed. This plan starts from the measured state and
runs the NCM part, which is still open. Do not repeat v6. No backwards
compatibility. One file per design.

## 1. What is established (measured, committed)

Commits: `ac9ef8b` (v6 honest accounting), `bbf4ed0` (v7 dictionary),
`bf5fe17` (charged budget), `c6c166c` (text-only candidates), `5795f02` (one cost
model), `7f83f4f` (coverage/calls).

**v6, the dense code fails.** The span code is a 12-bit scalar quantization of a
pooled *mean*. The mean is permutation-invariant, so it cannot carry byte order:
the code conveys ~1 bpb and costs 3.1. A probe from the same code with a stronger
decoder reaches 3.42, so the code, not the decoder, is the limit. `total = code +
residual` is monotone in the code rate (slope 0.30 at 300 and 1200 steps, every
level, `d_c`, stride, pool and tiling). The optimizer's optimum is to collapse
the code: the attention pool does (`total=4.79`, no code). The oracle tiling lies
on the same R-D line as the grid.

Full Phase 0, 3 seeds, 512 windows (`runs/p0/quick-c3b7a8ac/`):

| corpus | arm | total | recon | code | H(code) | util |
|---|---|---:|---:|---:|---:|---:|
| synseg | rel-pos | 6.891 | 3.752 | 3.138 | 381.0 | 1.000 |
| synseg | constant-code | 4.642 | 4.642 | 0.000 | 0.0 | 0.0 |
| synseg | unigram | 4.637 | 4.637 | 0.000 | - | - |
| syngrid | rel-pos | 5.975 | 2.863 | 3.112 | 377.0 | 1.000 |
| syngrid | constant-code | 3.909 | 3.909 | 0.000 | 0.0 | 0.0 |
| syngrid | unigram | 3.908 | 3.908 | 0.000 | - | - |

G0 fails on total only; G6/G9 pass. Phase 1 was not run.

**v7, the dictionary code pays where the model amortizes.** The code is a span
index into a *free* dictionary; the decoder writes the exact bytes, so the
residual is zero. The dictionary is a token-usage unigram fitted by EM, the
vocabulary size is swept, and the model is charged and amortized over train+val.
Both vocabularies use the same model (`runs/v7/results.json` text-only,
`runs/v7/onsets.json` word):

| corpus | vocab | dict | model | DP | charged | gzip | DP−greedy | covered | calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synseg | word | 768 | 0.038 | 1.290 | **1.328** | 2.225 | +0.000 | 0.996 | 485 |
| synseg | text | 4,352 | 0.463 | 1.880 | 2.343 | 2.225 | +0.098 | 0.779 | 26,560 |
| syngrid | word | 558 | 0.018 | 2.051 | **2.068** | 2.886 | +0.000 | 1.000 | 1 |
| syngrid | text | 512 | 0.018 | 2.573 | **2.590** | 2.886 | +0.000 | 0.554 | 53,494 |
| shakespeare | word | 4,352 | 0.285 | 3.062 | 3.347 | 3.218 | +0.000 | 0.667 | 37,168 |
| shakespeare | text | 2,304 | 0.156 | 3.299 | 3.455 | 3.218 | +0.084 | 0.607 | 43,887 |

Three findings:

1. The code must be a *dictionary index* (a discrete class with an exact inverse),
   not a dense latent. The dictionary pays by 0.90/0.82 bpb on synseg/syngrid and
   loses by 0.13 on shakespeare at 1.1 MB (`runs/v7/onsets.json`).
2. The *model* is the frontier. The budget optimum on shakespeare is 4,352
   entries (39.7 kB); below it the rate pays, above it the model pays.
3. The *parallelism dividend is bounded by coverage*. Calls = 1 + the uncovered
   bytes. On shakespeare the dictionary covers 0.667, so the tail needs 37,168
   calls against 128 for a byte-AR model: the dividend is 0.003x. On synseg
   `0.996` needs 485 calls: `0.26x`. Only syngrid (the grid *is* the structure)
   reaches 1 call.

So the dictionary is a *parallel* codec only where it covers; on real text the
tail dominates. **The NCM's remaining job is the tail.**

## 2. What the NCM part must do (`hncm.q4.md`)

`hncm.q4.md` (H-Net dynamic chunking × NCM multiscale RVQ) supplies the NCM
design. Take these ideas:

- **The coarse-to-fine generator** (`gen_concept`): level 0 global chunk code,
  then sub-chunk, then byte, each conditioned on the coarser reconstruction. This
  is the *tail coder*: a learned decoder that writes a span from its concept
  instead of looking it up.
- **The tied-router initialisation** (`W_k ← W_q`) and the **minimum-chunk-length
  prior**: both fix over-segmentation (1.2 bytes/chunk with a random router). The
  v7 DP prices the same effect *exactly* through the length cost, so these are
  control arms, not the main mechanism.
- **The factorial estimand** `I_perf = (D−C) − (B−A)` on −BPB, with arms A byte-AR,
  B H-Net-lite, C NCM-lite, D H-NCM. This is the clean merge test; adopt it.
- **The corruption-chain training** (`p_corr`) for the hierarchy: train fine
  levels against corrupted coarse codes.

Reject or re-scope:

- The RVQ hierarchy is 4 levels of 512/512/512/256 ≈ 27 bits/chunk. At ~3.4
  bytes/chunk that is ~9 bpb. `hncm.q4.md` measures D at 10.6 bpb and its own
  code term at 9.07: the same rate bottleneck as v6. A concept must be *cheap*
  (a few bits), or the total cannot pay.
- The interaction is −0.85 in q4 at 400 steps: still sub-additive. Do not read it
  as evidence until the code is cheaper than the reconstruction gain.

The NCM's measurable claim: **on the uncovered tail, a learned concept plus a
learned decoder must beat the byte unigram at 1 call per span.** It has no other
job: the dictionary already covers what a lookup covers.

## 3. Plan (priority order)

### P0 — coverage/calls frontier (~15 min)

Add `covered` and `calls` to the budget loop in `run` (`hncm7.py`), which currently
reports rate/charged only. Anchor:

```python
rate = bits_of(cover(va, dk, 0.0), dk) / len(va)
charged = rate + 8 * dk.model_bytes / stream
# add: toks = cover(...); single = sum(len(s) for s in toks if len(s) == 1)
#      covered = 1 - single/len(va); calls = 1 + single
```

Sweep vocab in `(256, 512, 1024, 2048, 4096, 8192, 16384, 32768)` for both
`--vocab onsets` and `--vocab substr`.

- Endpoint: `covered`, `calls`, `charged` per point; the calls ratio
  `L / (1 + uncovered)` at the charged optimum.
- Gate: the calls ratio ≥ 8 at the charged-optimal point, on at least one corpus.
- Expected: fail on shakespeare, pass on syngrid only. **Report it either way.**
- Stop: if no point passes, the parallel dividend is dead on real text; the tail
  is the only path (P3).

### P1 — held-out vocabulary size (~15 min)

The size is chosen on the charged total reported on the same val. Split the val
into two halves: select the size on half A, report on half B. Gate: the selected
size is within one grid step and the charged total changes by ≤ 0.05 bpb. If not,
the budget curve is a selection artifact.

### P2 — strong baselines (~30 min)

gzip is weak. Add `zlib`, `lzma`, `bz2` (already in the stdlib; no dependency),
`zstd` and `brotli` only if installed. Charge every model. Gate: the dictionary's
charged total ≤ the best real codec on shakespeare. Expected: xz ≈ 2.4–2.7 bpb, so
the dictionary (3.347) lose. Report it.

### P3 — the NCM tail coder (~2 h, the research step)

Take the *uncovered* val spans (a span not in the dictionary). Train a learned
concept plus decoder on the *train* spans:

```
span bytes (<= 16) -> order-aware span encoder (a 1-2 layer transformer)
                   -> z (d_c = 8..16) -> VQ concept (K = 256..4096)
concept + span length + within-span offset -> learned decoder -> 256 logits/byte
```

Train end to end; the rate is the concept's entropy under a *static/EMA* prior
(decodable, never conditioned on `z`). Arms on the tail:

| arm | what it is | calls |
|---|---|---|
| `uni` | the byte unigram (the current fallback) | 1 per byte |
| `concept` | the learned concept + decoder | 1 per span |
| `ar-span` | a causal byte model inside the span | L per span |

- Endpoint: the tail's residual bpb and the calls.
- Gate: the tail residual < the byte unigram (4.78 on shakespeare) *and* the calls
  ratio `L / (1 + uncovered) ≥ 8` at the *charged-optimal* vocabulary, and the
  total (concept + residual) < the dictionary's charged total on the tail.
- Stop: if the tail residual ≥ the byte unigram, the concept carries nothing on
  unseen spans. Report the negative; the NCM part is then a lookup, not a model.

### P4 — the factorial (~1 h)

Adopt q4's estimand on the new arms: A byte-AR, B dense codec (v6), C dictionary
plus DP (v7), D dictionary plus DP plus the NCM tail (P3). `I_perf = (D−C)−(B−A)`
on −BPB at matched budget. Report per-seed; 3 seeds. A negative is publishable
with clean controls.

### P5 — scale (~1 h)

`enwik8` at 10 MB and 100 MB with the model charged at each size. Gate: the
charged total at 100 MB is below the best real codec, and the calls ratio rises
with size. The representation should dominate; the crossover should disappear.

## 4. Rules not to repeat (each cost a run)

1. The entropy model must not condition on the latent it codes. The decoder must
   have every input of the model.
2. Charge per *transmitted* symbol, not per position. A packed tail is not a code.
3. Train length embeddings on *all* lengths the DP and oracle can emit, not only
   the DP set.
4. Cap a BPE merge at 16 bytes. A 512-byte "token" is memorisation.
5. The oracle is a *baseline*, not a ceiling, when the vocabulary is not
   word-aligned.
6. Normalise the index cost by the *token usage* (a unigram), not by positions.
   Both vocabularies must use one model.
7. Charge the model (the dictionary or codebook) and amortise it over the stream.
   Report the stream size and the charge.
8. Report coverage and calls together. Calls = 1 + the uncovered bytes.
9. Select the vocabulary size on a held-out split, never on the reported val.
10. `total_bpb` is the primary metric. A probe, not a judge, bounds the code.

## 5. Commands and artifacts

```bash
python hncm6.py selftest
python hncm6.py run --preset quick --phase 0 --seeds 0,1,2
python hncm7.py selftest
python hncm7.py run --corpora synseg,syngrid,shakespeare --vocab onsets
python hncm7.py run --corpora synseg,syngrid,shakespeare --vocab substr
```

Artifacts: `runs/v7/onsets.json`, `runs/v7/results.json`,
`runs/v7/REPORT.md`, `runs/p0/quick-c3b7a8ac/` (the v6 negative).
Docs: `README.md`, `REPORT.md`, `hncm.q4.md` (the NCM proposal).

## 6. Constraints (AGENTS.md)

Python 3.14; `uv` for the environment; `ruff format` always and
`ruff check --fix` on changed files; `_selftest` must cover every invariant that
broke (DP optimality, exact cover, entropy round-trip, resume, shared init, tail
invariance, decoder geometry); one file per design; no backwards compatibility;
no pydantic at the boundary; no import-time I/O; `main()` plus a CLI; atomic
writes; seed and log every run; a run without provenance is not evidence.

**Minimal decisive run: P0 + P1 + P2 (~1 h). The research step: P3 (~2 h).**
A negative with clean controls is publishable. Do not tune a dead arm.
