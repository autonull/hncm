# H-NCM report

## v6: the dense-latent codec fails, and why

The span code is a 12-bit scalar quantization of a *mean*, which is
permutation-invariant: it cannot carry byte order, so it conveys ~1 bpb while it
costs 3.1. A probe from the same code (a stronger decoder) reaches 3.42, so the
code, not the decoder, is the limit. `total = code + residual` is monotone in the
code rate (slope 0.30 at 300 and 1200 steps, every level/`d_c`/stride/pool); the
optimizer's optimum is to collapse the code; the oracle lies on the same R-D line.

Honest Phase 0 (`runs/p0/quick-c3b7a8ac/`, 3 seeds, 512 windows):

| corpus | arm | total | recon | code | H(code) | util |
|---|---|---:|---:|---:|---:|---:|
| synseg | rel-pos | 6.891 | 3.752 | 3.138 | 381.0 | 1.000 |
| synseg | constant-code | 4.642 | 4.642 | 0.000 | 0.0 | 0.0 |
| synseg | unigram | 4.637 | 4.637 | 0.000 | - | - |
| syngrid | rel-pos | 5.975 | 2.863 | 3.112 | 377.0 | 1.000 |
| syngrid | constant-code | 3.909 | 3.909 | 0.000 | 0.0 | 0.0 |
| syngrid | unigram | 3.908 | 3.908 | 0.000 | - | - |

G0 fails on total only; G6/G9 pass. No Phase 1: the code does not pay.

## v7: the redesign, one cost model for both vocabularies

The code is the index of a span in a dictionary; the decoder writes the exact
bytes (residual zero). The dictionary is a token-usage unigram fitted by EM on the
train, the vocabulary size is swept, and the model is charged and amortized over
train+val. Both vocabularies use the same model, so their totals are comparable.

| corpus | vocab | dict | model | unigram | gzip | DP | charged | greedy | fixed4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synseg | word | 768 | 0.038 | 4.640 | 2.225 | 1.290 | **1.328** | 1.290 | 4.592 |
| synseg | text | 4,352 | 0.463 | 4.640 | 2.225 | 1.880 | 2.343 | 1.978 | 4.626 |
| syngrid | word | 558 | 0.018 | 3.911 | 2.886 | 2.051 | **2.068** | 2.051 | 2.051 |
| syngrid | text | 512 | 0.018 | 3.911 | 2.886 | 2.573 | **2.590** | 2.573 | 3.906 |
| shakespeare | word | 4,352 | 0.285 | 4.777 | 3.218 | 3.062 | 3.347 | 3.062 | 4.725 |
| shakespeare | text | 2,304 | 0.156 | 4.777 | 3.218 | 3.299 | 3.455 | 3.382 | 4.647 |

## What holds

- **The representation pays where the model amortizes.** Word vocabulary:
  charged 1.328 vs gzip 2.225 (synseg, 0.90 bpb) and 2.068 vs 2.886 (syngrid,
  0.82 bpb), with a 7.0/2.6 kB model. The dictionary is data and it is paid for.
- **The tiling is decisive.** The fixed grid costs 4.59/2.05/4.73 against the DP
  1.29/2.05/3.06. On syngrid the grid is the oracle by construction, so the
  fixed arm ties the DP and the negative control holds.
- **The solver earns its keep only where candidates overlap.** Word: DP − greedy
  = 0.000 on all three. Text-only: +0.098 (synseg), +0.084 (shakespeare), 0.000
  on syngrid. That is P3 with clean controls: solving beats guessing exactly
  where guessing is myopic.
- The DP beats the fixed grid by 3.30/0.00/1.66 bpb (word) and
  2.75/1.33/1.35 (text).

## What does not hold

- **The solver's gain does not buy back the text vocabulary's model.** On synseg
  the text point is 4,352 entries / 0.463 bpb of model against 768 / 0.038 for
  the word point: +0.098 bpb from the solver, −0.425 bpb from the model. Net:
  the text vocabulary loses (2.343 vs 1.328).
- **Shakespeare is model-bound at 1.1 MB** *at a fixed 4,352-entry dict*. Word:
  3.347 vs gzip 3.218, a 0.129 loss, crossing at ~2.0 MB. Text: 3.455, a 0.237
  loss. (The charge section below revises this: with the dict *and* the table
  chosen, the envelope reaches 3.087.)
- The text vocabulary wins only on syngrid (2.590 vs 2.886), where the grid is
  the structure and the solver adds nothing.

## The budget curve (shakespeare, word vocabulary, 1.12 MB stream)

| vocab | dict | model kB | rate | charged |
|---:|---:|---:|---:|---:|
| 256 | 512 | 2.6 | 3.877 | 3.896 |
| 512 | 768 | 4.8 | 3.756 | 3.791 |
| 1024 | 1280 | 9.5 | 3.435 | 3.504 |
| 2048 | 2304 | 19.3 | 3.237 | 3.375 |
| 4096 | 4352 | 39.7 | 3.062 | 3.347 |
| 8192 | 8448 | 82.5 | 2.905 | 3.497 |
| 16384 | 16640 | 175.0 | 2.816 | 4.071 |
| 32768 | 28949 | 315.6 | 2.709 | 4.972 |

The optimum is 4,352 entries: below it the rate pays, above it the model pays.

## v7: the charge of the model (M1-M3, measured)

`concept_lm.py run` (8 min) fits the dictionary and the table on the first 90%
of each corpus, chooses the budget (vocabulary × dict × table) with a rule that
sees the last 10% of the train, and charges 6 bytes per bigram entry over
train+val. `oracle` is the val-optimal grid point plus the header that sends the
choice: the envelope, and the gap from the rule is the cost of the rule. v7's
`gzip` column above is zlib/deflate; here both are measured.

| corpus | stream | zlib | gzip | bz2 | lzma | zstd | brotli | rule | oracle | vs zlib | vs best | crossing zlib |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shakespeare | 1.12 MB | 3.218 | 3.199 | **2.635** | 2.943 | 2.984 | 2.825 | 3.227 | 3.087 | -0.131 | +0.451 | 0.62 MB |
| synseg | 1.20 MB | 2.225 | 2.220 | 1.513 | 1.948 | 1.941 | 1.954 | **1.325** | **1.323** | -0.901 | -0.190 | 0.06 MB |
| syngrid | 1.20 MB | 2.886 | 2.866 | 2.129 | **1.999** | 2.453 | 2.295 | **1.939** | **1.939** | -0.947 | -0.060 | 0.32 MB |

- **M1, the charge.** The dictionary (6.4-25.4 kB) pays above 0.06/0.32/0.62 MB
  of stream. At 1.12 MB the affordable table is 1,024-2,048 entries on real text
  and 256-8,192 on the structured corpora; the full table is never affordable at
  1 MB (shakespeare: 58,981 entries, 354 kB, rate 2.870, charged 5.55). The
  crossing vs lzma is 8.5 MB and vs zstd 2.9 MB; vs bz2 it is *never*, because
  the concept LM's rate (2.92) is above bz2's (2.635) at every affordable
  budget.
- **M2, the solver.** At the charged optimum DP = greedy to 0.001 bpb on all
  three; at the full table the DP beats the greedy by 0.098/0.084 and ties on
  syngrid, where the grid *is* the dictionary (the negative control). DP <
  fixed-4 everywhere (3.05/1.27/1.59 against 4.73/4.59/1.59) and DP < the
  unigram everywhere (3.49/1.29/2.05).
- **M3, the strong codecs.** bz2 wins shakespeare (2.635 against the envelope
  3.087; brotli 2.825, lzma 2.943, zstd 2.984), bz2 wins synseg but loses to the
  index (1.513 vs 1.323), lzma wins syngrid by 0.06 (1.999 vs 1.939).
- **The decision.** The index pays against *every* codec where the source has a
  vocabulary smaller than the stream (synseg, syngrid), and is at best at parity
  with zlib on real text at 1 MB (a 0.13 win at the envelope, a 0.01 loss under
  the held-out rule) while bz2 wins there by 0.45. 56% of the shakespeare val is
  uncovered: its ~50k calls are the tail, i.e. M5's job. The claim is source- and
  scale-conditioned; at 1 MB it is a negative on natural text.

## Caveats

1. The budget rule (the charged total on the last 10% of the train) costs 0.14
   bpb on shakespeare, because the train tail is 0.29 bpb easier than the val.
   `oracle` is the val-optimal grid point plus an 8-bit header: it is the
   envelope for any rule, not a learned result.
2. The val is one contiguous 10% block (the last play of tinyshakespeare) and it
   is harder than the train average: every codec sees the same bytes, but the
   rates are this block's, not the corpus's.
3. The fallback is charged as the byte unigram, not as a switch: a span outside
   the dictionary costs its bytes, with no flag.
4. The budget grid uses the unigram-tiling table; the EM refit runs only at the
   chosen point.
5. The synthetic corpora are highly repetitive, so their win is the structure,
   not the net: enwik8 (a large vocabulary) is untested.

## Next

1. M4: does a learned bigram generalize the table on held-out concept pairs, at
   1 MB and 10 MB?
2. M5: the tail coder on the 56% uncovered shakespeare val: one call per span
   against the byte unigram's one call per byte.
3. M7: enwik8 at 10/100 MB, where the table's 1-2k-entry crossing says a bigger
   budget becomes affordable.

## Artifacts

- `runs/v7/results.json` (text-only), `runs/v7/onsets.json` (word), both with
  `REPORT.md` tables.
- `runs/v7/concept.json`, `runs/v7/CONCEPT.md` (M1-M3, the charge).
- `runs/p0/quick-c3b7a8ac/` (v6 honest negative).
