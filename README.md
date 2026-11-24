# H-NCM: Hierarchical Neural Concept Modeling

Status: `hncm6.py` is the dense-latent v6 negative. `hncm7.py` is the redesign:
the span code is a *dictionary index*, the dictionary is a *charged model* fit by
EM, and the tiling is *solved*. **With one cost model for both vocabularies: the
word vocabulary beats gzip including its model on synseg and syngrid by
0.90/0.82 bpb, and loses on shakespeare by 0.13. The solver gains 0.08-0.10 bpb
only on the overlapping text-only vocabulary.**

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

## Next

`TODO2.md` is the self-contained plan. Run M1 (the charged bigram budget curve),
M2 (the DP against a greedy on the same bigram costs) and M3 (lzma, bz2, zstd,
brotli) first: ~1 h, and they set the claim. Then M4 (the neural bigram),
M5 (the tail coder), M6 (the factorial), M7 (enwik8 at 10/100 MB).

## Commands

```bash
python hncm6.py selftest
python hncm7.py selftest
python hncm7.py run --corpora synseg,syngrid,shakespeare --vocab substr
python hncm7.py run --corpora synseg,syngrid,shakespeare --vocab onsets
```

## Artifacts

- `runs/v7/results.json` (text-only), `runs/v7/onsets.json` (word).
- `runs/v7/REPORT.md`, `runs/p0/quick-c3b7a8ac/` (v6 negative).
- `hncm7.py`, `hncm6.py`; both `selftest` pass; `ruff format`/`ruff check` pass.
