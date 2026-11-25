# H-NCM: Hierarchical Neural Concept Modeling

Status: `hncm6.py` is the dense-latent v6 negative. `hncm7.py` is the redesign:
the span code is a *dictionary index*, the dictionary is a *charged model*, and
the tiling is *solved*. **v8/N0 prices it end to end**: `codec.py` packs the
model as counts, derives the CDF, and range-codes the DP's path; the decoder
never reruns the DP, and every corpus round-trips. The recorded table corrects
the v7 estimate: the published tail account (2.723 rate / 3.006 charged on
shakespeare) is a lower bound by 0.07-0.28 bpb because `total - sum(byte
unigram)` subtracts 5.05 bpb for tail bytes the concept path had already priced
at the bigram row (3.83). The realizable codec at 1.12 MB is 3.47 charged on
shakespeare (0.25 behind gzip, 0.97 behind PPMd), 1.39 on synseg (0.11 ahead of
PPMd, 0.12 ahead of bz2), and 2.10 on syngrid (0.09 behind lzma).
The structure stands: the model is paid once; the tail arm makes 6.5x fewer
calls (8,264 vs 56,592) but pays 0.30 bpb more than the concept arm at 1 MB, so
the calls pay only when the tail fraction falls. **N1a measured (enwik8, 9 MB train / 1 MB val):** the dictionary covers
0.545/0.629/0.698/0.756 at 2k/4k/8k/16k substrings and 0.414-0.536 at word
onsets; the predicted total is 3.56 -> 3.14 (substr) and 3.59 -> 3.38 (onsets),
against PPMd 2.476 on the 1 MB val. The defect is the tail's context: the SVD
restarts at 256, but the decoder holds the real preceding bytes, and an order-2
table is 3.83 vs the SVD's 4.48 bpb on the 1 MB tail. That table costs 95 kB and
pays at 10 MB (0.088 charge for ~0.35 gain) but not at 1 MB (0.877). The NCM
(15 kB GRU, GPU, 10 epochs) reaches 4.61, so the learned tail loses to the counts
at this budget: the count context table is the NCM at scale. The complete model
is the *concept arm* (one normalized model: the H-Net rows give the tail its
context), 3.67 charged vs the hard split's 3.94 at 1 MB. The 10 MB concept arm is **3.035 charged** (8k dict / 16k table) against PPMd
**2.160**: the coverage is 0.756 and the tail (24% at 5.6 bpb) is the gap. The chunked DP and the real-context tail are implemented; the 25 kB
3-byte-context NCM reaches 4.307 on the 10 MB tail against the SVD's 4.459 (a
0.152 gain, +0.009 of charge, +0.04 net), but the count order-2 is 3.83 at
1 MB and is the better tail. The 0.875 gap to PPMd is structural on enwik8;
the 100 MB (the charge is 0.02 there) is the remaining scale test, and the
structured source (synseg 1.389 vs PPMd 1.500) is the win at 1.2 MB.

## The measured bottleneck (v6)

The span code is a 12-bit scalar quantization of a *pooled mean*. A mean is
permutation-invariant: it cannot tell `abab` from `aabb`, so the code carries
~1 bpb while it costs 3.1. A probe from the same code with a stronger decoder
reaches 3.42, so the code, not the decoder, is the limit. `total = code +
residual` is monotone in the code rate (slope 0.30, unchanged at 1200 steps and at
every level, `d_c`, stride, pool and tiling); the optimizer's optimum is to
collapse the code; the oracle tiling lies on the same R-D line. Phase 0:
`runs/p0/quick-c3b7a8ac/` (3 seeds, 512 windows), G0 fails on total only.

## The redesign (v7): one cost model

The code is the index of a span in a dictionary; the decoder writes the exact
bytes, so the residual is zero. The dictionary is a token-usage unigram fitted by
EM on the train (the DP re-tokenizes, the usage becomes the cost), the vocabulary
size is swept, and the model (`512 + sum(len+3)` bytes) is charged and amortized
over train+val. Both vocabularies use the *same* model, so their totals are
comparable. The DP solves the exact cover; the greedy runs the same costs with a
longest-match cover.

| corpus | vocab | dict | model | DP | charged | gzip | DP−greedy |
|---|---|---:|---:|---:|---:|---:|---:|
| synseg | word | 768 | 0.038 | 1.290 | **1.328** | 2.225 | +0.000 |
| synseg | text | 4,352 | 0.463 | 1.880 | 2.343 | 2.225 | +0.098 |
| syngrid | word | 558 | 0.018 | 2.051 | **2.068** | 2.886 | +0.000 |
| syngrid | text | 512 | 0.018 | 2.573 | **2.590** | 2.886 | +0.000 |
| shakespeare | word | 4,352 | 0.285 | 3.062 | 3.347 | 3.218 | +0.000 |
| shakespeare | text | 2,304 | 0.156 | 3.299 | 3.455 | 3.218 | +0.084 |

Read this literally:

- **The word vocabulary pays**: charged 1.328 vs gzip 2.225 (synseg) and 2.068 vs
  2.886 (syngrid), i.e. 0.90/0.82 bpb below gzip with the model paid. Both are
  under a 7.0/2.6 kB model, so the charge is 0.04/0.02 bpb.
- **Shakespeare is model-bound at 1.1 MB**: the word point charges 0.285 bpb and
  loses to gzip by 0.129, crossing at ~2.0 MB; the text point charges 0.156 and
  loses by 0.237.
- **The solver earns its keep only where candidates overlap.** Word (no
  overlap): DP − greedy = 0.000. Text-only (overlapping): +0.098 (synseg) and
  +0.084 (shakespeare), 0.000 on syngrid (the grid *is* the structure).
- **The tiling is decisive**: the fixed grid costs 4.59/2.05/4.73 (word) against
  the DP 1.29/2.05/3.06. On syngrid the grid is the oracle, so the fixed arm ties
  the DP: the negative control holds.
- The text-only vocabulary needs a *bigger* model for the same rate on synseg
  (4,352 vs 768 entries): it trades model bytes for the solver's gain, and loses
  on the charge. That is the honest frontier: the solver pays, but not enough to
  buy back the model.

## Caveats

1. The usage unigram is fitted on the train and evaluated on the val; the
   vocabulary size is chosen on the charged total *reported on the same val*
   (a selection leak of one knob; the ranking is over 8 sizes).
2. The fallback (a span outside the dictionary pays its bytes) is charged as the
   byte unigram, not as a switch.
3. gzip is weak (brotli/xz are ~2.4-2.7 bpb on shakespeare).
4. The synthetic corpora are highly repetitive; their candidate counts are huge
   (1.1M distinct substrings), so the EM's pruning is doing the heavy lifting.

## The solution: a first-order concept LM

v7 prices each concept by its unigram, so it leaves the context on the table.
The alphabet is predictable and the previous concept is already decoded, so a
bigram over the dictionary indices is a decodable cost model; it also makes the
DP a first-order Viterbi, which is exact (`concept_lm.py`):

| corpus | vocab | unigram | **bigram** | cut |
|---|---:|---:|---:|---:|
| shakespeare | 2,048 | 3.237 | **2.655** | 0.58 |
| shakespeare | 4,096 | 3.062 | **2.538** | 0.52 |
| shakespeare | 8,192 | 2.905 | **2.450** | 0.46 |
| synseg | 2,048 | 1.285 | **0.888** | 0.40 |
| syngrid | 2,048 | 2.051 | **1.382** | 0.67 |

A 0.40-0.67 bpb cut on all three corpora: on shakespeare 2.40-2.54 against gzip
3.218 and xz ~2.5. The pitfall that hid it: an *iterated* bigram is worse than
the unigram (3.128 vs 2.905) because its costs come from the previous token of a
different path. The cost model must be evaluated on the path it searches.

The bigram table is data and is charged (~6 bytes per entry), so it pays above
roughly 1-5 MB and 0.02-0.2 bpb at 10-100 MB. The NCM's jobs are now precise:
the tail coder (spans outside the dictionary at one call per span) and the bigram
generalizer (unseen concept pairs versus the backoff).

## The charge (M1-M3, measured)

`concept_lm.py run` (~8 min) fits the dictionary and the table on the first 90%
of each corpus, chooses the budget (vocabulary × dict × table) with a rule that
sees the last 10% of the train, and charges 6 bytes per bigram entry over
train+val. `oracle` is the val-optimal grid point plus the header that sends the
choice: the envelope, and the gap from the rule is the cost of the rule.

| corpus | stream | zlib | bz2 | lzma | zstd | brotli | ppmd | rule | oracle | vs zlib | vs best | zlib crossing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shakespeare | 1.12 MB | 3.218 | 2.635 | 2.947 | 2.984 | 2.825 | **2.504** | 3.227 | 3.087 | -0.131 | +0.583 | 0.62 MB |
| synseg | 1.20 MB | 2.225 | 2.220 | 1.513 | 1.948 | 1.941 | 1.954 | **1.325** | **1.323** | -0.901 | -0.190 | 0.06 MB |
| syngrid | 1.20 MB | 2.886 | 2.129 | **2.005** | 2.453 | 2.295 | 2.063 | **1.939** | **1.939** | -0.947 | -0.066 | 0.32 MB |

Read it literally:

- **Where the source is a dictionary, the index pays against every codec.** On
  synseg it is 0.18 bpb below PPMd (the strongest text codec) and 0.9 below
  zlib; on syngrid 0.066 below xz. On shakespeare PPMd wins (2.504 against the
  envelope 3.087): the gap is 0.58, not the 0.37 that bz2 (2.635) suggested.
- **On real text at 1.1 MB it is at best at parity.** The envelope (3.087, a
  0.13 win against zlib) needs a budget chosen on the target stream; the
  held-out rule lands at 3.227, a 0.01 loss, because at 1 MB the budget choice
  is worth more than the model (the train tail is 0.29 bpb easier than the val).
  bz2 wins by 0.45 and never loses: the concept LM's rate (2.92) is above bz2's
  (2.635) at every affordable budget, so no stream size rescues it at these
  rates. lzma's crossing is 8.5 MB.
- **The table pays only above 1-2k entries at 1 MB.** The full table (58,981
  entries, 354 kB) reaches rate 2.870 but charges 5.55 bpb; the affordable
  optimum is 1,024-2,048 entries. The dictionary (6-25 kB) pays above
  0.06-0.62 MB of stream.
- **The solver's prize is smaller than the table's charge.** At the charged
  optimum DP = greedy to 0.001 bpb; at the full table the DP beats the greedy by
  0.098/0.084 and ties on syngrid (where the grid *is* the dictionary). DP <
  fixed-4 and DP < the unigram everywhere.
- **The tail is the remaining job:** 56% of the shakespeare val is uncovered, so
  the concept LM makes ~50k calls there. M5 measures it below.

## The tail pays (M5, measured)

The dictionary covers a span exactly; the spans it lacks pay the byte unigram,
one byte at a time: 48% of the shakespeare val and ~80% of its bits. The tail
coder codes one call per span. The win is the *shared* model: the rank-16
factorization of the 257x256 byte-context table (16.4 kB, r chosen on the
held-out 10% of the train, never the val).

| arm | byte bpb | boundary | tail bpb | rate | charged | model kB | calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| uni | 5.048 | - | 5.048 | 2.921 | 3.087 | 23.1 | 53,747 |
| ar@full (counts) | 4.447 | 0.616 | 5.063 | 2.928 | 4.768 | 256.5 | 8,264 |
| svd@8 | 4.642 | 0.616 | 5.258 | 3.022 | 3.247 | 31.3 | 8,264 |
| **svd@16** | **4.020** | 0.616 | **4.636** | **2.723** | **3.006** | 39.5 | 8,264 |
| svd@16 + space | 4.542 | - | 4.542 | 2.677 | **2.960** | 39.5 | 8,264 |
| ceiling (no boundary) | 4.020 | - | 4.020 | 2.426 | 2.709 | 39.5 | 8,264 |

- 16.4 kB of shared parameters reach 4.020 bpb on the tail bytes: **1.03 below
  the byte unigram and 0.43 below the unbounded 256.5 kB count table**. The
  low-rank structure carries the unseen letter pairs of rare words; the counts
  cannot.
- The tail now pays its charge: **3.006 against 3.087**, i.e. 0.21 below zlib at
  1.12 MB. With the boundary coded as the trailing space: 2.960, the *tail side*
  of a space-suffixed tokenizer (a bound: the covered token would lose its
  space too).
- The boundary is 0.616 bpb as a length, 0.522 as the trailing space.
- Calls: **6.5x** (53,747 -> 8,264) at +0.03 bpb.
- Budget chosen on the held-out split (U-shaped: 2.727, 2.503, 2.440, 2.389,
  **2.310** at r=16, 2.386, 2.623); the val optimum agrees, but it is not the
  rule.
- The hard k-means concept adds nothing (`prev` 4.795, k-means 4.974 against
  `ar` 4.480): a hard code with counts is a negative; the shared *low-rank*
  model is the positive. Pitfalls: add-alpha is overconfident on sparse keys
  (-0.15 bpb), `D = 0` gives a zero probability, and factorizing the log
  probabilities is worse than the counts.

## One model, K files (measured)

The val is split into K shards; every codec compresses each shard on its own,
and the concept LM tiles every shard with the same model. `archive` charges the
model over train + the shards; `per-file` makes each shard pay it alone.

| corpus | PPMd K=1 -> 16 | bz2 K=1 -> 16 | concept rate | archive | per-file | PPMd / bz2 cross at (no tail) |
|---|---|---|---:|---:|---:|---|
| shakespeare | 2.504 -> 3.429 | 2.635 -> 3.443 | 2.921 -> 2.922 | 3.087 | 4.575 -> 29.385 | K = 8 / 8 |
| synseg | 1.500 -> 3.296 | 1.513 -> 3.130 | 1.275 -> 1.278 | 1.323 | 1.757 -> 8.991 | K = 1 / 1 |
| syngrid | 2.063 -> 2.940 | 2.129 -> 2.956 | 1.594 -> 1.595 | 1.939 | 5.045 -> 56.817 | K = 1 / 1 |

The shared model's curve is flat and every codec's curve rises, because a codec
pays its model per file. (The table is the concept LM alone; the tail comes next.)
With the tail coder the whole pipeline is rate 2.723 /
archive 3.006: bz2's per-file mean passes it between K=4 (2.942) and K=8 (3.158),
PPMd's between 2.857 and 3.113, so the crossings are K≈5 (bz2) and K≈6 (PPMd):
above ~six 22 kB files the shared concept model is cheaper than PPMd per file.
The deploy convention (the 111 kB of shards paying a 39.5 kB model) is 2.84 bpb
of charge: the model needs a stream ~10x its size (N1).

## Next

`TODO3.md` (rev 2) is the end-to-end plan. **PPMd** (`7z -m0=PPMd -mx9`, measured
2.504/1.500/2.063) is the text bar, not bz2, and it is in `codecs()` now.

1. **N0 the bitstream** (`codec.py`): a range coder over the token indices, the
   model packed and charged, the CDF derived from the charged counts, `--try 3`
   (the per-file budget, transmitted), the recorded size reconciled with the
   charge. The artifact.
2. **N1a the coverage probe** (15 min): fit, tile, report coverage — it decides
   the scale run.
3. **N1b scale** (enwik8 10/100 MB): coverage is 51.8% at 1.1 MB and the tail's
   71% share is a coverage artifact; target ≤ 1.80 charged at 10 MB, within 0.10
   of PPMd.
4. **N2 the tail's context** at the same 16.4 kB, **N3 the concept table's
   sharing plus a second-order context**, **N4 the boundary**, **N5 the
   lattice**, **N6 the factorial** `I_perf`.

## Commands

```bash
python hncm6.py selftest
python hncm7.py selftest
python hncm7.py run --corpora synseg,syngrid,shakespeare --vocab substr
python hncm7.py run --corpora synseg,syngrid,shakespeare --vocab onsets
python concept_lm.py selftest
python concept_lm.py run            # M1-M3, ~8 min
python tail_coder.py selftest
python tail_coder.py run            # M5, ~1.6 min
python transfer.py selftest
python transfer.py run              # amortization, ~1.4 min
```

## Artifacts

- `runs/v7/results.json` (text-only), `runs/v7/onsets.json` (word).
- `runs/v7/concept.json`, `runs/v7/CONCEPT.md` (M1-M3, the charge).
- `runs/v7/tail.json`, `runs/v7/TAIL.md` (M5, the tail).
- `runs/v7/transfer.json`, `runs/v7/TRANSFER.md` (the amortization).
- `runs/v7/REPORT.md`, `runs/p0/quick-c3b7a8ac/` (v6 negative).
- `hncm7.py`, `hncm6.py`; both `selftest` pass; `ruff format`/`ruff check` pass.
