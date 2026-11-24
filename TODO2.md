# TODO2 — H-NCM v7: the concept LM is the solution; measure it

Status: `hncm6.py` is the dense-latent negative. `hncm7.py` is the dictionary
redesign. `concept_lm.py` is the measured solution path; `tail_coder.py` (M5) and
`transfer.py` (the amortization) measure the two remaining charges. All are
committed and all `selftest` pass. **M1–M3 are §8; M5 and the amortization are
§9.**

## 0. Start here (fresh session)

```bash
git log --oneline -5
python hncm6.py selftest
python hncm7.py selftest
python concept_lm.py run        # ~8 min, M1-M3 (§8)
```

M1, M2, M3 in §4 are done; §8 is the measured answer.

**One-sentence state.** A span *dictionary index* (an exact code) pays where the
model amortizes; a *first-order concept LM* over the indices cuts the rate by
0.40–0.67 bpb on every corpus, and at 1 MB only a 1–2k-entry table is
affordable, so the charge decides: the index pays against zlib on the two
structured corpora and against bz2/lzma there too, while on real text at 1 MB it
beats zlib by at most 0.13 bpb (or ties it under the held-out rule) and loses
0.45 to bz2. The tail (48% of the val, 80% of the bits) now pays: a 16.4 kB
*shared-parameter* model reaches 4.020 bpb there (1.03 below the byte unigram,
0.43 below a 256 kB count table), the whole pipeline is 2.723 / 3.006 charged
(2.677 / 2.960 with the space boundary, the tail side of a retokenization and
so a bound) with 6.5x fewer calls, and the model is paid *once*, so above ~five 22 kB files
the shared model beats bz2 on text.

## 1. Files, commits, artifacts

| file | what it is |
|---|---|
| `hncm6.py` | v6: dense-latent codec. The negative. `run --phase 0` |
| `hncm7.py` | v7: dictionary codec, EM cost model, exact DP. `--vocab onsets\|substr` |
| `concept_lm.py` | the solution probe: first-order Viterbi over dictionary indices |
| `hncm.q4.md` | H-Net × NCM v2 proposal (the NCM design; §3.3) |
| `TODO1.md` | the v6 plan and its progress (history) |
| `README.md`, `REPORT.md` | the headline status and tables |
| `runs/p0/quick-c3b7a8ac/` | v6 Phase 0, 3 seeds, 512 windows |
| `runs/v7/results.json` | v7 text-only vocabulary (charged) |
| `runs/v7/onsets.json` | v7 word vocabulary (charged) |
| `runs/v7/REPORT.md` | the generated v7 report |

Commits: `ac9ef8b` (v6 honest accounting), `bbf4ed0` (v7 dictionary),
`bf5fe17` (charged budget), `c6c166c` (text-only), `5795f02` (one cost model),
`7f83f4f` (coverage/calls), `d463a51` (the concept LM).

## 2. What is established (measured, do not re-derive)

### 2.1 v6: a dense latent code fails

The span code is a 12-bit scalar quantization of a *pooled mean*. A mean is
permutation-invariant, so it cannot carry byte order: the code conveys ~1 bpb and
costs 3.1. A probe from the same code with a stronger decoder reaches 3.42, so
the *code*, not the decoder, is the limit. `total = code + residual` is monotone
in the code rate (slope 0.30 at 300 and 1200 steps, every level, `d_c`, stride,
pool and tiling). The optimizer's optimum is to collapse the code; the attention
pool does (`total=4.79`, no code). The oracle tiling lies on the same R-D line as
the grid, so the tiling cannot rescue a dense code.

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

R-D sweep (honest, seed 0, 300 steps, synseg, unigram 4.637): levels 4,
`d_c` 1, stride 16 → 4.544 + 0.248 = 4.792; levels 8, `d_c` 2, stride 4 →
4.143 + 1.581 = 5.724; levels 8, `d_c` 4, stride 4 → 3.715 + 3.080 = 6.795.
At 1200 steps: 4.323 + 0.847 = 5.170; 4.106 + 1.570 = 5.676;
3.586 + 3.063 = 6.648. The slope stays ~0.3.

### 2.2 v7: a dictionary code pays where the model amortizes

The code is a span *index* into a dictionary; the decoder writes the exact bytes,
so the residual is zero. The dictionary is a token-usage unigram fitted by EM on
the train, the vocabulary size is swept, and the model (`512 + sum(len+3)` bytes)
is charged and amortized over train+val. Both vocabularies use the same model.

Best charged point per row (`runs/v7/onsets.json`, `runs/v7/results.json`):

| corpus | vocab | dict | model | rate | charged | gzip | DP−greedy | covered | calls |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synseg | word | 768 | 0.038 | 1.290 | **1.328** | 2.225 | +0.000 | 0.996 | 485 |
| synseg | text | 4,352 | 0.463 | 1.880 | 2.343 | 2.225 | +0.098 | 0.779 | 26,560 |
| syngrid | word | 558 | 0.018 | 2.051 | **2.068** | 2.886 | +0.000 | 1.000 | 1 |
| syngrid | text | 512 | 0.018 | 2.573 | **2.590** | 2.886 | +0.000 | 0.554 | 53,494 |
| shakespeare | word | 4,352 | 0.285 | 3.062 | 3.347 | 3.218 | +0.000 | 0.667 | 37,168 |
| shakespeare | text | 2,304 | 0.156 | 3.299 | 3.455 | 3.218 | +0.084 | 0.607 | 43,887 |

Budget curve (word, shakespeare, 1.12 MB stream):

| vocab | model kB | rate | charged |
|---:|---:|---:|---:|
| 256 | 2.6 | 3.877 | 3.896 |
| 512 | 4.8 | 3.756 | 3.791 |
| 1,024 | 9.5 | 3.435 | 3.504 |
| 2,048 | 19.3 | 3.237 | 3.375 |
| 4,096 | 39.7 | 3.062 | **3.347** |
| 8,192 | 82.5 | 2.905 | 3.497 |
| 16,384 | 175.0 | 2.816 | 4.071 |
| 32,768 | 315.6 | 2.709 | 4.972 |

### 2.3 v7: the parallelism dividend is bounded by coverage

Calls = 1 + the uncovered bytes. The uncovered bytes pay the byte unigram, one at
a time. Word dictionary, DP: synseg covered 0.996 / 485 calls; syngrid 1.000 / 1
call; shakespeare 0.667 / 37,168 calls against 128 for a byte-AR model. So the
dividend is 0.003x on shakespeare, 0.26x on synseg, 128x only on syngrid (where
the grid *is* the structure). The dictionary is a parallel codec only where it
covers.

### 2.4 The solution: a first-order concept LM (`concept_lm.py`)

v7 prices each concept by its *unigram*: no context. The alphabet is predictable,
and the previous concept is already decoded, so a *bigram over the indices* is a
decodable cost model. It also makes the DP a *first-order Viterbi*, which is
exact. Count bigram (the ceiling; a neural bigram is M4):

| corpus | vocab | dict | unigram | **bigram** | cut |
|---|---:|---:|---:|---:|---:|
| shakespeare | 2,048 | 2,304 | 3.237 | **2.655** | 0.58 |
| shakespeare | 4,096 | 4,352 | 3.062 | **2.538** | 0.52 |
| shakespeare | 8,192 | 8,448 | 2.905 | **2.450** | 0.46 |
| shakespeare | 16,384 | 16,640 | 2.816 | **2.400** | 0.42 |
| synseg | 2,048 | 894 | 1.285 | **0.888** | 0.40 |
| syngrid | 2,048 | 558 | 2.051 | **1.382** | 0.67 |

**The pitfall that hid it (do not repeat).** An *iterated* bigram — costs from
the previous token of the *unigram* tiling, then one more DP pass — is *worse*
than the unigram: 3.128 vs 2.905 on shakespeare; 1.728 vs 1.285; 2.801 vs 2.051.
The DP picks tokens under a model built for a different path. The model's cost
must condition on the *actual* state of the path. The first-order Viterbi does
that, and it wins.

**The charge.** The bigram table is data: ~6 bytes per entry (two 2-byte indices
and a 2-byte count). The rate gain is 0.4–0.7 bpb; a pruned table is ~0.2–2 MB,
so the table pays above roughly 1–5 MB of stream. At 10–100 MB the charge is
0.02–0.2 bpb. The crossover is `8 * model_bytes / (gzip − rate)`.

## 3. Findings and the solution path

1. **The code must be an index, not a dense latent.** A dense latent of 12 bits
   conveys ~1 bpb (the mean destroys order, the probe bounds it at 3.42); an index
   into a dictionary has an exact inverse, so the residual is zero.
2. **The model is the frontier.** The dictionary is free at the rate and charged
   as data: 6.1 kB on synseg, 39.7 kB on shakespeare. Below the budget optimum
   the rate pays, above it the model pays.
3. **The dividend is bounded by coverage.** Calls = 1 + uncovered bytes, so on
   real text the tail dominates unless the tail is covered by concepts.
4. **The solution is a first-order concept LM over the indices**, solved exactly
   by a first-order Viterbi: 0.40–0.67 bpb on all three corpora, with a decodable
   cost model and no fantasy. Its charge is the table and the dictionary.
5. **The NCM's job is now precise**: the *tail coder* (spans outside the
   dictionary, one call per span) and the *bigram generalizer* (unseen concept
   pairs versus the backoff). Both are rate endpoints; both are cheap to measure.

### 3.3 What to take from `hncm.q4.md`

Take: the *coarse-to-fine generator* (`gen_concept`: global chunk → sub-chunk →
byte, each conditioned on the coarser reconstruction) as the tail coder; the *tied
router* (`W_k ← W_q`) and the *minimum-chunk-length* prior (they fix
over-segmentation; the DP prices the same effect exactly through the length cost,
so they are control arms); the *corruption-chain* training for the hierarchy; and
the factorial `I_perf = (D−C) − (B−A)` as the merge estimand (M6).

Reject or re-scope: the RVQ hierarchy is 4 × (512, 512, 512, 256) ≈ 27
bits/chunk ≈ 9 bpb at 3.4 bytes/chunk. `hncm.q4.md` measures D at 10.6 bpb and
its code at 9.07: the same rate wall as v6. A concept must be *cheap*. Its
interaction is −0.85 at 400 steps: do not read it as evidence.

## 4. Remaining work (the measurements, in order) — superseded by `TODO3.md`

The measurements below are done (§8-§9). The next plan is `TODO3.md`: the
bitstream, scale, the tail's context, the concept table, the boundary, the
lattice, the factorial.

### M1 — charged bigram budget curve (~30 min) — done (§8)

*Question:* does the concept LM pay for its table at each stream size?
*How:* in `concept_lm.py`, sweep the pruned bigram size (top-N by count:
4k, 8k, 16k, 32k, 64k) times the dictionary size (2,048, 4,096, 8,192), with the
backoff for the pruned entries. Charge `dict_bytes + 6 * n_bigrams`. Report the
charged total and the crossing `N* = 8*model_bytes/(gzip − rate)`.
*Endpoint:* charged total versus gzip and xz at 1.12 MB; the crossing in MB.
*Gate:* the charged total < gzip 3.218 at the corpus size, or report the crossing.
*Stop:* none; this is the measurement that sets the claim. A positive here is a
real result; a crossing beyond the corpus means a scale-conditioned claim.

### M2 — the solver gain under the first-order cost (~10 min) — done (§8)

*Question:* does the exact DP beat a greedy on the *same* bigram costs?
*How:* add a greedy arm to `concept_lm.py`: at each position take the longest
dictionary span, pay the bigram cost of its sequence, then compare the sum to the
Viterbi's. Also add the fixed-4 grid and the unigram for reference.
*Endpoint:* `DP − greedy` in bpb, and `DP − fixed`.
*Gate:* DP < greedy. Expected positive, because the greedy is myopic for a bigram
cost. This is the *first honest test* of "solving beats guessing" with a payoff.

### M3 — strong baselines (~30 min) — done (§8)

*Question:* does the concept LM beat a real codec, not gzip?
*How:* add `lzma`, `bz2` and `zlib` (stdlib) on the same val bytes; `zstd` and
`brotli` only if installed. Charge every model (these include their tables in the
stream, so their size already covers it).
*Endpoint:* their bpb versus the concept LM's charged total.
*Gate:* the charged concept LM ≤ the best real codec. xz on shakespeare is ~2.5;
the concept LM is 2.4–2.5 before the charge, so this is the decisive test.

### M4 — the neural bigram (~1–2 h)

*Question:* does a *learned* concept LM generalize where the count table cannot?
*How:* train a small model over the dictionary indices (a 64-dim embedding and a
low-rank output; `p(i | prev) = softmax(E[prev] @ W)`). Measure its conditional
CE on *held-out* concept pairs (the val pairs not in the train bigram) versus the
count table's backoff. Charge its fitted parameters.
*Endpoint:* the held-out conditional CE and the charged total.
*Gate:* the neural CE < the table's backoff CE and its charged total ≤ the table's.
*Stop:* if the neural CE ≥ the backoff, the concept LM does not generalize;
report it. That is the NCM's *bigram generalizer* job, and it is a negative.

### M5 — the tail coder (~2–4 h, the last research step) — done (§9)

*Question:* on spans *outside* the dictionary, does a learned concept plus a
decoder beat the byte unigram at one call per span?
*How:* take the uncovered val spans. Train a span encoder (an order-aware
1–2 layer transformer, not a mean) → a VQ concept (K = 256–4,096) with a *static*
prior → a learned decoder (the concept + the span length + the within-span offset
→ 256 logits per byte). Arms: `uni` (the byte unigram, 1 call per byte),
`concept` (1 call per span), `ar-span` (a causal byte model inside the span).

*Run as:* the count version first (a hard k-means code over the span profile, a
Good-Turing byte table per context, a length prior), plus a `ceiling` arm with no
boundary. If a hard code with counts cannot pay, a learned decoder over the same
code can only smooth it; the ceiling separates the code from the decoder and the
charge from the boundary.
*Endpoint:* the tail residual in bpb and the calls.
*Gate:* the tail residual < the byte unigram (4.78) *and* the calls ratio
`L / (1 + uncovered) ≥ 8` at the charged-optimal vocabulary *and* the total
(concept + residual) < the dictionary's charged total on the tail.
*Stop:* if the tail residual ≥ the byte unigram, the concept carries nothing on
unseen spans; report and stop.

### M6 — the factorial (~1 h)

Adopt `hncm.q4.md`'s estimand on the current arms: A byte-AR, B the dense codec
(v6), C the dictionary plus unigram plus DP (v7), D the dictionary plus concept
LM plus DP. `I_perf = (D−C) − (B−A)` on −BPB at a matched budget; 3 seeds;
report per-seed. A negative is publishable with clean controls.

### M7 — scale (~1 h)

`enwik8` at 10 MB and 100 MB, with the model charged at each size. Gate: the
charged total at 100 MB is below the best real codec and the calls ratio rises
with size. The representation should dominate and the crossover should disappear.

**Minimal decisive run: M1 + M2 + M3 (~1 h). Research: M4, M5.**

## 5. Rules not to repeat (each cost a run)

1. The entropy model must not condition on the latent it codes. The decoder must
   have every input of the model.
2. Charge per *transmitted* symbol, not per position. A packed tail is not a code.
3. Train length embeddings on *all* lengths the DP and oracle can emit.
4. Cap a BPE merge at 16 bytes. A 512-byte "token" is memorisation.
5. The oracle is a *baseline*, not a ceiling, when the vocabulary is not
   word-aligned.
6. Normalise the index cost by the *token usage* (a unigram), not by positions.
   Both vocabularies must use one model.
7. Charge the model (the dictionary, the bigram table, the codebook) and amortise
   it over the stream. Report the stream size with the charge.
8. Report coverage and calls together. Calls = 1 + the uncovered bytes.
9. Select the vocabulary size on a held-out split, never on the reported val.
10. Evaluate the cost model on the *path it searches*. An iterated approximation
    with stale context is worse than no context (3.128 vs 2.905).
11. `total_bpb` is the primary metric. A probe, not a judge, bounds the code.
    "Sample quality" is not a rate.

## 6. Constraints (AGENTS.md)

Python 3.14; `uv`; `ruff format` always and `ruff check --fix` on changed files;
`_selftest` must cover every invariant that broke (DP optimality, exact cover,
entropy round-trip, resume, tail invariance, decoder geometry); one file per
design; no backwards compatibility; no pydantic at the boundary; no import-time
I/O; `main()` plus a CLI; atomic writes; seed and log every run. A run without
provenance is not evidence.

## 7. Commands

```bash
python hncm6.py selftest
python hncm6.py run --preset quick --phase 0 --seeds 0,1,2
python hncm7.py selftest
python hncm7.py run --corpora synseg,syngrid,shakespeare --vocab onsets
python hncm7.py run --corpora synseg,syngrid,shakespeare --vocab substr
python concept_lm.py selftest
python concept_lm.py run            # ~8 min, writes runs/v7/concept.json + CONCEPT.md
python concept_lm.py run --vocab onsets
```

## 8. Results: M1–M3 measured

Artifacts: `runs/v7/concept.json`, `runs/v7/CONCEPT.md`; `python concept_lm.py
selftest` passes; `run` takes ~8 min on CPU (`shakespeare` dominates: the
substring counts).

Setup. The dictionary and the table are fitted on the first 90% of each corpus.
The budget (vocabulary × dict size × table size) is chosen by a *rule* that sees
the last 10% of the train, which the model never fitted; the val is only
reported. `oracle` is the best grid point on the val plus the header that sends
the choice: the envelope of the representation, and the gap from the rule to it
is the cost of the rule. The table is refitted on the DP's own path (2 EM
rounds, chosen on the same 10%). The model is charged 6 bytes per bigram entry
over train+val. v7's `gzip` column is zlib/deflate; here both are measured.

| corpus | stream | zlib | gzip | bz2 | lzma | zstd | brotli | ppmd | rule | oracle | vs zlib | vs best | zlib crossing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shakespeare | 1.12 MB | 3.218 | 3.199 | 2.635 | 2.947 | 2.984 | 2.825 | **2.504** | 3.227 | 3.087 | −0.131 | +0.583 | 0.62 MB |
| synseg | 1.20 MB | 2.225 | 2.220 | 1.513 | 1.948 | 1.941 | 1.954 | **1.325** | **1.323** | −0.901 | −0.190 | 0.06 MB |
| syngrid | 1.20 MB | 2.886 | 2.866 | 2.129 | **2.005** | 2.453 | 2.295 | 2.063 | **1.939** | **1.939** | −0.947 | −0.066 | 0.32 MB |

**M1, the charge.** The dictionary (6.4 kB at 512 spans to 25.4 kB at 2,048) pays
above 0.06 MB (synseg), 0.32 MB (syngrid) and 0.62 MB (shakespeare) of stream.
At 1.12 MB the affordable table is 1,024–2,048 entries (6–12 kB) on real text
and 256–8,192 on the structured corpora; the *full* table is never affordable
at 1 MB (shakespeare: 58,981 entries, 354 kB, rate 2.870, charged 5.55). The
crossings vs lzma are 0.09 MB (synseg), 1.02 MB (syngrid) and 8.53 MB
(shakespeare); vs bz2 they are 0.24 MB, 0.77 MB and *never* on shakespeare,
because the concept LM's rate (2.92) is above bz2's (2.635) at every affordable
budget. The charge is what decides, and it is a scale question.

**M2, the solver.** At the charged optimum the DP equals the greedy to 0.001 bpb
on all three corpora: the table is too small at 1 MB for the myopic longest
match to lose. At the *full* table (v7's rate optimum) the DP beats the greedy
by 0.098 (shakespeare text) and 0.084 (synseg text), and ties on syngrid, where
the grid *is* the dictionary (the negative control). DP < fixed-4 everywhere
(rate 3.05/1.27/1.59 against 4.73/4.59/1.59) and DP < the unigram codec
everywhere (3.49/1.29/2.05): solving beats guessing, but at 1 MB the solver's
prize is smaller than the table's charge.

**M3, the strong codecs.** PPMd is the text bar: it wins shakespeare (2.504
against the oracle 3.087; bz2 2.635, brotli 2.825, lzma 2.947, zstd 2.984) and
synseg (1.500 vs 1.323), and xz wins syngrid by 0.066 (2.005 vs 1.939).
zlib/gzip are weak everywhere. So on real text the concept LM is 0.58 behind
PPMd at 1.12 MB.

**The decision.** The dictionary index pays its charge against *every* real
codec exactly where the source has a small exact vocabulary (synseg, syngrid:
0.06–0.19 bpb below bz2/lzma, 0.9 below zlib). On natural text at 1.1 MB it is
at parity with zlib at best (a 0.13 win at the envelope: 3.087 against 3.218;
3.227, i.e. a 0.01 loss, under the held-out rule) and 0.45 bpb behind bz2, and
56% of the val is
uncovered, i.e. the tail is the remaining job. The held-out rule costs 0.14 bpb
on shakespeare: the train tail is 0.29 bpb easier than the val, and at 1 MB the
budget choice is worth more than the model. So the claim is scale- and
source-conditioned, and it is a negative on real text at 1 MB.

Next: M4 (does the learned bigram generalize the concept table the way the
tail's low-rank model generalizes the bytes?), M6 (the factorial `I_perf`), and
M7 (enwik8 at 10/100 MB, where both models' charges fall below 0.02 bpb and the
deploy convention stops binding).

## 9. Results: M5 (the tail) and the amortization

Artifacts: `runs/v7/tail.json`, `runs/v7/TAIL.md`, `runs/v7/transfer.json`,
`runs/v7/TRANSFER.md`. Both `selftest` pass; `tail_coder.py run` is 1.6 min and
`transfer.py run` is 1.4 min, on the budget envelope of §8.

### M5: the tail pays with a shared-parameter model

shakespeare's uncovered 53,746 B (8,263 spans, 6.50 B/span) cost the byte unigram
5.048 bpb against 0.943 on the covered bytes: the tail is 48% of the val and
~80% of its bits. The `svd@r` arm is the rank-r factorization of the train
tail's 257x256 byte-context table (shared parameters); r is chosen on the
held-out 10% of the train, never the val. The count arms are the same table,
pruned.

| arm (shakespeare) | byte bpb | boundary | tail bpb | rate | charged | model kB | calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| uni | 5.048 | - | 5.048 | 2.921 | 3.087 | 23.1 | 53,747 |
| ar@256 | 4.724 | 0.616 | 5.340 | 3.062 | 3.503 | 61.6 | 8,264 |
| ar@full | 4.447 | 0.616 | 5.063 | 2.928 | 4.768 | 256.5 | 8,264 |
| svd@8 | 4.642 | 0.616 | 5.258 | 3.022 | 3.247 | 31.3 | 8,264 |
| **svd@16** | **4.020** | 0.616 | **4.636** | **2.723** | **3.006** | 39.5 | 8,264 |
| svd@32 | 3.937 | 0.616 | 4.553 | 2.682 | 3.084 | 55.9 | 8,264 |
| svd@16 + space | 4.542 | 0.000 | 4.542 | 2.677 | **2.960** | 39.5 | 8,264 |
| ceiling (no boundary) | 4.020 | - | 4.020 | 2.426 | 2.709 | 39.5 | 8,264 |

- **The model generalizes, and that is the win.** At 16.4 kB of parameters the
  rank-16 factorization reaches 4.020 bpb on the tail bytes: 1.03 bpb below the
  byte unigram, and 0.43 below the *unbounded* count table (256.5 kB) at 1/16
  the size. The unseen letter pairs of rare words are what a shared
  parameterization carries and counts cannot.
- **The tail now pays its charge:** 3.006 against 3.087 (a 0.08 win). With the
  boundary coded as the trailing space it is 2.960 (0.13), but the covered token
  that follows still pays its index, so that row is the *tail side* of a
  space-suffixed tokenizer: a bound, not a finished codec. Against zlib: 0.21.
- **The budget is measured, not tuned:** the held-out charge is U-shaped in r
  (2.727, 2.503, 2.440, 2.389, **2.310** at r=16, 2.386 at r=32, 2.623 at
  r=64). The val optimum agrees, but the rule is the sel minimum.
- **The boundary** costs 0.616 bpb as a length and 0.522 as the trailing space
  (a word ends at a space, which the byte model knows).
- **Calls: 6.5x** (53,747 -> 8,264) at +0.03 bpb: the synergy's second currency.
- **The hard k-means concept still adds nothing** (`prev` 4.795, k-means 4.974
  against `ar` 4.480): a hard code with counts is a negative; the low-rank
  *shared* model is the positive.
- **Pitfalls:** an add-alpha table is overconfident on sparse keys (-0.15 bpb)
  and `D = 0` gives a zero probability (Good-Turing, floored, fixes both);
  factoring the *log* probabilities is worse than factoring the counts (rank 2 <
  rank 1): the domain matters.
- **Controls:** syngrid has no tail (coverage 1.000), so the tail coder is a
  no-op; synseg's tail is 493 B and the arms tie.

The whole pipeline at 1.12 MB: **rate 2.723, archive charge 3.006** (dict 23.1 +
tail 16.4 kB) against zlib 3.218 and bz2 2.635; with the space bound it is
2.677 / 2.960. bz2's per-file mean is 2.942 at K=4, so the shared model now
crosses bz2 at K ~ 5 instead of K=8.

### The amortization: one model, K files; the curve is flat

The val is split into K contiguous shards; every codec compresses each shard on
its own, and the concept LM tiles every shard with the *same* dictionary and
table. `archive` charges the model over train + the K shards (the §8 convention),
`per-file` makes each shard pay it alone.

| corpus | PPMd K=1 -> 16 | bz2 K=1 -> 16 | concept rate | archive | per-file | PPMd / bz2 cross at (no tail) |
|---|---|---|---:|---:|---:|---|
| shakespeare | 2.504 -> 3.429 | 2.635 -> 3.443 | 2.921 -> 2.922 | 3.087 | 4.575 -> 29.385 | K = 8 / 8 |
| synseg | 1.500 -> 3.296 | 1.513 -> 3.130 | 1.275 -> 1.278 | 1.323 -> 1.326 | 1.757 -> 8.991 | K = 1 / 1 |
| syngrid | 2.063 -> 2.940 | 2.129 -> 2.956 | 1.594 -> 1.595 | 1.939 -> 1.940 | 5.045 -> 56.817 | K = 1 / 1 |

- **The shared model's curve is flat** (rate 2.921 -> 2.922; archive 3.087 ->
  3.087 from K=1 to 16) because the model is paid once. Every codec's curve
  rises (bz2 2.635 -> 3.443), because it pays its model per file.
- **So the K=1 comparison flatters every codec.** On shakespeare the crossing is
  K = 8 for PPMd and bz2: PPMd's per-file mean is 2.857 at K=4 and 3.113 at K=8
  against the archive charge 3.087. On synseg and syngrid it is K = 1 already.
- **The deploy convention is the harsh one:** if the model must be paid by the
  111 kB of shards alone, it is 1.65 bpb and nothing pays; the model needs a
  stream ~10x its own size. That is M7's question (10/100 MB), not a defect of
  the code.

Next: M4 (does the learned bigram generalize the concept table the way
the tail's low-rank model generalizes the bytes?), M6 (the factorial `I_perf`),
and M7 (enwik8 at 10/100 MB, where both models' charges fall below 0.02 bpb).
