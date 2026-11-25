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
- **The tail's estimate is not the codec.** 16.4 kB of rank-16 parameters reach
  4.020 bpb on the uncovered shakespeare bytes (1.03 below the byte unigram,
  0.43 below a 256.5 kB count table), but the 2.723 / 3.006 account that used it
  is a lower bound: `total - sum(byte unigram)` subtracts the unigram for bytes
  the concept path priced at the bigram row. The v8 bitstream (below) records
  3.47 charged on shakespeare, 1.39 on synseg and 2.10 on syngrid; the tail
  model's gain is eaten by the boundary and the type at 1 MB.

## v8: the bitstream (N0)

`codec.py` is the artifact: the model is packed as log-quantized counts, the CDF
is derived (`round(2^24 p)`), and a 32-bit arithmetic coder writes the DP's token
path. Two codecs are measured at each budget: `concept` (every token, the
segmentation implicit) and `tail` (a type, a length, the shared byte model).
`--try` transmits the budget choice; the decoder never reruns the DP.

| corpus | arm | budget | rate | ref | model kB | charged | best codec |
|---|---|---:|---:|---:|---:|---:|---|
| shakespeare | concept | 1024/2048 | 3.305 | 3.024 | 23.1 | 3.472 | gzip 3.218 |
| synseg | tail | 512/1024 | 1.290 | 1.221 | 14.7 | 1.389 | PPMd 1.500 |
| syngrid | tail | 302/8192 | 1.739 | 1.594 | 53.6 | 2.098 | lzma 2.005 |

Every corpus round-trips; the recorded size equals the code's entropy + the
charge within 0.0013 bpb in both the `shared` and the `self` conventions. The
published estimate is 0.07-0.28 bpb below the model it names. At 1 MB the model
pays on synseg only; the scale run (N1) is the next lever.

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

| corpus | stream | zlib | gzip | bz2 | lzma | zstd | brotli | ppmd | rule | oracle | vs zlib | vs best | crossing zlib |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shakespeare | 1.12 MB | 3.218 | 3.199 | 2.635 | 2.947 | 2.984 | 2.825 | **2.504** | 3.227 | 3.087 | -0.131 | +0.583 | 0.62 MB |
| synseg | 1.20 MB | 2.225 | 2.220 | 1.513 | 1.948 | 1.941 | 1.954 | **1.325** | **1.323** | -0.901 | -0.190 | 0.06 MB |
| syngrid | 1.20 MB | 2.886 | 2.866 | 2.129 | **2.005** | 2.453 | 2.295 | 2.063 | **1.939** | **1.939** | -0.947 | -0.066 | 0.32 MB |

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
- **M3, the strong codecs.** PPMd is the text bar: 2.504 on shakespeare (against
  the envelope 3.087), then bz2 2.635, brotli 2.825, lzma 2.947, zstd 2.984. PPMd
  also wins synseg (1.500) but loses to the index (1.323); xz wins syngrid by
  0.066 (2.005 vs 1.939). So on real text the concept LM is 0.58 behind PPMd at
  1.12 MB, not 0.37 behind bz2.
- **The decision.** The index pays against *every* codec where the source has a
  vocabulary smaller than the stream (synseg, syngrid), and is at best at parity
  with zlib on real text at 1 MB (a 0.13 win at the envelope, a 0.01 loss under
  the held-out rule) while bz2 wins there by 0.45. 56% of the shakespeare val is
  uncovered: its ~50k calls are the tail, i.e. M5's job. The claim is source- and
  scale-conditioned; at 1 MB it is a negative on natural text.

## v7: the tail pays (M5, measured)

The dictionary covers a span exactly; the spans it lacks pay the byte unigram,
one byte at a time. On shakespeare that tail is 48% of the val and ~80% of its
bits. The tail coder codes one call per span; the win is the *shared* model, the
rank-r factorization of the train tail's 257x256 byte-context table, with r
chosen on the held-out 10% of the train and never the val.

| arm | byte bpb | boundary | tail bpb | rate | charged | model kB | calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| uni | 5.048 | - | 5.048 | 2.921 | 3.087 | 23.1 | 53,747 |
| ar@256 (counts) | 4.724 | 0.616 | 5.340 | 3.062 | 3.503 | 61.6 | 8,264 |
| ar@full (counts) | 4.447 | 0.616 | 5.063 | 2.928 | 4.768 | 256.5 | 8,264 |
| svd@8 | 4.642 | 0.616 | 5.258 | 3.022 | 3.247 | 31.3 | 8,264 |
| **svd@16** | **4.020** | 0.616 | **4.636** | **2.723** | **3.006** | 39.5 | 8,264 |
| svd@16 + space | 4.542 | - | 4.542 | 2.677 | **2.960** | 39.5 | 8,264 |
| ceiling (no boundary) | 4.020 | - | 4.020 | 2.426 | 2.709 | 39.5 | 8,264 |

- **The model generalizes, and that is the win**: 16.4 kB of shared parameters
  reach 4.020 bpb on the tail bytes, 1.03 below the byte unigram and 0.43 below
  the *unbounded* 256.5 kB count table. The unseen letter pairs of rare words
  are what a shared parameterization carries and counts cannot.
- **The tail now pays its charge**: 3.006 against 3.087 (0.21 below zlib at
  1.12 MB); 2.960 with the trailing space, the tail side of a retokenization
  and so a bound.
- **The budget is measured**: the held-out charge is U-shaped in r (2.727,
  2.503, 2.440, 2.389, **2.310** at r=16, 2.386, 2.623). The val optimum
  agrees, but the rule is the sel minimum.
- **The boundary** is 0.616 bpb as a length and 0.522 as the trailing space.
- **Calls: 6.5x** (53,747 -> 8,264) at +0.03 bpb.
- **The hard k-means concept still adds nothing** (`prev` 4.795, k-means 4.974
  against `ar` 4.480): a hard code with counts is a negative; the low-rank
  shared model is the positive.
- **Pitfalls**: add-alpha is overconfident on sparse keys (-0.15 bpb) and `D=0`
  gives a zero probability (Good-Turing, floored, fixes both); factoring the
  *log* probabilities is worse than factoring the counts (rank 2 < rank 1).
- **Controls**: syngrid has no tail (coverage 1.000), so the tail coder is a
  no-op; synseg's tail is 493 B and the arms tie.

## v7: one model, K files (measured)

Every codec pays its model per file; a learned vocabulary and concept LM pay it
once. K contiguous shards of the val; every codec compresses each shard alone,
the concept LM tiles each shard with the same model. `archive` charges the model
over train + the K shards; `per-file` makes each shard pay it alone.

| corpus | PPMd K=1 -> 16 | bz2 K=1 -> 16 | concept rate | archive | per-file | PPMd / bz2 cross at (no tail) |
|---|---|---|---:|---:|---:|---|
| shakespeare | 2.504 -> 3.429 | 2.635 -> 3.443 | 2.921 -> 2.922 | 3.087 | 4.575 -> 29.385 | K = 8 / 8 |
| synseg | 1.500 -> 3.296 | 1.513 -> 3.130 | 1.275 -> 1.278 | 1.323 | 1.757 -> 8.991 | K = 1 / 1 |
| syngrid | 2.063 -> 2.940 | 2.129 -> 2.956 | 1.594 -> 1.595 | 1.939 | 5.045 -> 56.817 | K = 1 / 1 |

The shared curve is flat and every codec's curve rises. So the K=1 comparison
flatters bz2: on shakespeare the crossing is K=8, i.e. above eight 14 kB files
the shared concept model is cheaper than bz2 per file, and on synseg/syngrid it
is K=1. With the tail coder the whole pipeline is rate 2.723 / archive 3.006, so
bz2's crossing moves to K≈5 and PPMd's to K≈6 (above six 22 kB files). The deploy convention
(the 111 kB of shards paying a 39.5 kB model) is 2.84 bpb of charge: the model
needs a stream ~10x its size.

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
6. The tail's positive is a low-rank factorization fitted by SVD, not a trained
   decoder; it has no position context and no VQ. The negative is only for the
   hard k-means code with counts.
7. The transfer crossing (K=8) uses the envelope's model, charged over
   train+shards. In the deploy convention (the 111 kB of shards alone) the model
   is 1.65 bpb: it needs a stream ~10x its size.

## Next

`TODO3.md` (rev 2): the bitstream (`codec.py`), the coverage probe, scale
(enwik8 10/100 MB), the tail's context at the same 16.4 kB, the concept table's
sharing, the boundary, the factorial.

PPMd (`7z -m0=PPMd -mx9`) is the text bar — 2.504 on shakespeare, 1.500 on
synseg, 2.063 on syngrid — and it is now in `codecs()`, so every table of record
reports it.

## Artifacts

- `runs/v7/results.json` (text-only), `runs/v7/onsets.json` (word), both with
  `REPORT.md` tables.
- `runs/v7/concept.json`, `runs/v7/CONCEPT.md` (M1-M3, the charge).
- `runs/v7/tail.json`, `runs/v7/TAIL.md` (M5, the tail).
- `runs/v7/transfer.json`, `runs/v7/TRANSFER.md` (K files).
- `runs/p0/quick-c3b7a8ac/` (v6 honest negative).


## v8/N1a: the coverage probe and the tail's context

On the 9 MB/1 MB enwik8 split (cache: `~/.cache/hncm/counts-*`), the dictionary
is fitted on the train and the val tiled with the unigram cover:

| vocab | dict | coverage | covered bpb | tail bpb | model kB | predicted |
|---|---:|---:|---:|---:|---:|---:|
| substr | 2,048 | 0.545 | 1.997 | 5.321 | 57.8 | 3.557 |
| substr | 4,096 | 0.629 | 2.030 | 5.453 | 83.8 | 3.367 |
| substr | 8,192 | 0.698 | 2.032 | 5.606 | 138.1 | 3.223 |
| substr | 16,384 | 0.756 | 2.039 | 5.717 | 251.4 | 3.139 |
| onsets | 8,192 | 0.536 | 1.889 | 4.840 | 156.3 | 3.382 |

PPMd on the 1 MB val is 2.476 (zlib 3.027, bz2 2.653, lzma 2.723, brotli 2.449),
so the hard-split model is ~0.7-1.2 bpb behind at 10 MB's val. The probe locates
the defect: the SVD tail restarts at 256 and throws away the real preceding
bytes. Fitted on the train tail:

| context | fitted bpb | val ceiling bpb |
|---|---:|---:|
| order 1 (the SVD is 4.48) | 4.17 | 3.80 |
| order 2 | **3.83** | 2.75 |
| order 3 | 3.73 | 1.62 |

The order-2 table is 23,851 entries / 95 kB: 0.088 bpb of charge at 10 MB for
~0.35 bpb of tail gain, so it pays at 10 MB and not at 1 MB (0.877). The SVD was
a 1 MB size choice. The NCM at the same 15 kB loses to the counts: a GPU-trained
GRU (previous byte, concept class, position; 10 epochs) is 4.61 bpb against the
SVD's 4.48. The right NCM at scale is the count context table, not a 15 kB GRU.

The complete model is the *concept arm* (one normalized model: every token is
coded by the row of the previous token, so the H-Net rows are the tail's
context), not the hard split:

| arm | rate | model kB | charged |
|---|---:|---:|---:|
| concept | 3.452 | 27.3 | **3.671** |
| tail (hard split) | 3.457 | 60.2 | 3.939 |
| PPMd | 2.476 | in-stream | 2.476 |

The gap is the context order, not the representation. Next: the concept arm at
10 MB with the order-2 tail and the first-order concept table; the chunked DP
(the 10 MB `viterbi` allocates L+1 dicts) is the prerequisite.

## v8/N1b: enwik8 at 10 MB

Chunked DP, scaled tail, table chosen on the held-out 10%, charged over 10 MB:

| dict | table | concept rate | model kB | charged | PPMd |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 16,384 | 2.857 | 222.5 | **3.035** | 2.160 |
| 8,192 | 8,192 | 2.927 | 165.0 | 3.059 | 2.160 |
| 16,384 | 4,096 | 2.873 | 248.8 | 3.072 | 2.160 |
| 8,192 | 4,096 | 2.991 | 135.0 | 3.098 | 2.160 |

The concept arm beats the hard split by 0.09 at half the model; the selection
rule costs nothing. The gap to PPMd is the *tail*: the 16k dict covers 0.756 and
its covered tokens are ~2.0, below PPMd's 2.16; the 24% uncovered at 5.6 bpb
(the SVD cold start) is the whole deficit. Real-context order-2 is 3.83 vs the
SVD's 4.48 on the 1 MB tail. With that tail at 10 MB the total is ~2.6; at
100 MB the charge (0.02) and coverage give ~1.8-2.0 against PPMd's ~1.8. The
text claim is a 100 MB claim; the structured win (synseg 1.389 vs PPMd 1.500)
stands at 1.2 MB.

The NCM v2 (the real previous three bytes + the H-Net class + the position;
25 kB; 3 epochs on the GPU) is 4.307 bpb on the 10 MB DP tail against the
SVD's 4.459: +0.152 on the tail, +0.009 of charge, a +0.04 net. The context
is real and it beats the SVD, but the count order-2 (3.83 at 1 MB) is the
better tail; the 10 MB gap to PPMd is not closed by the NCM.

## v9/N8: the union vocabulary (measured)

The union is the merge of the two cached EM tilings (substrings and word onsets):
one candidate map, one unigram, one first-order DP. On enwik8_10m (9 MB train /
1 MB val, SHA `5985c81c...`) the unigram probe covers 0.699/0.766/0.815 at
8k/16k/32k dicts; the first-order DP rate is 2.607/2.499/2.425 at the 32k
table. The selection rule picks 16,384/16,384; the codec records:

| budget | arm | rate | model kB | shared | calls |
|---|---|---:|---:|---:|---:|
| union 16k/16k | concept | 2.729 | 309.6 | **2.977** | 253,853 |
| union 16k/16k | tail | 2.772 | 342.4 | 3.046 | 253,853 |
| substr 8k/16k (v8) | concept | 2.857 | 222.5 | 3.035 | 56,592 |

The union is a **0.058 win on the record** and 0.021 below zlib; PPMd is 2.160.
The estimate moved 0.284 (3.035 -> 2.751), the record 0.058. The 0.044 G2r gap
is the 2-byte count quantization. The N8 gate (recorded <= 2.94) misses by
0.037. The coverage gate's legs do not meet at one budget: 0.815 costs 347.7 kB
and its covered rate is 2.129.

## v9/N9: the order-2 tail (measured)

The pruned order-2 counts (the real two preceding bytes, order-1 backoff, top on
the held-out 10%, charged) beat the concept row on the same tail bytes:

| rect | tail bytes | ord2 kB | ord2 bpb | row bpb |
|---|---:|---:|---:|---:|
| union 16k/16k | 253,852 | 81.0 | **4.303** | 4.854 |
| substr 8k/4k | 317,741 | 81.2 | **4.140** | 4.721 |

On the old identical 310,677 substr bytes the NCM is 4.307 (27.4 kB); the
corrected counts are 4.140 at 81.2 kB, ahead by 0.167 on the rate and 53 kB
larger. The N9 gate (tail <= 4.0) misses by 0.303 / 0.140.

## v9/N10: the end-to-end proof (measured)

The 90 MB train / 10 MB val split. The seed is the 10 MB union vocabulary with
its counts refit on the 81 MB fit (one EM round); the full adds the 90 MB heavy
hitters. The table is fitted on 81 MB, chosen on 9 MB, charged over 100 MB. The
bitstream round-trips:

| vocab | model kB | val | payload B | rate | shared | calls | RT |
|---|---:|---:|---:|---:|---:|---:|---|
| seed | 300.9 | 1 MB | 327,389 | 2.619 | **2.643** | 233,504 | ok |
| seed | 300.9 | 10 MB | 3,300,098 | 2.640 | **2.664** | 2,393,110 | ok |
| full | 278.7 | 1 MB | 532,626 | 4.261 | 4.283 | 131,766 | ok |

PPMd on the same 10 MB val is 1.834 (zlib 2.903, bz2 2.281, lzma 2.144). The
seed is 0.830 behind PPMd; the 1.90 gate misses. The full vocabulary covers
0.912 but its table's unseen transitions cost 4.261: coverage is not the lever.
The text claim is a **negative at 100 MB**; the structured win (synseg 1.389
against PPMd 1.500) is the product. The earlier unnormalized estimate
(2.503 / 2.520) was 0.14 / 1.76 optimistic; the encoded, round-tripping
records are 2.643 / 4.283.

The 90 MB scan found 2,702,205 substrings with count >= 32 (64 MB cache,
`runs/v9/scan100.npz`, not committed). The GPU unique pass is the exact
replacement for the CPU `counts_substr` that would need ~10x the memory.
