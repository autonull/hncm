# H-NCM report

## v6: the dense-latent codec fails, and why

The span code is a 12-bit scalar quantization of a *mean*, which is
permutation-invariant, so it cannot carry byte order: it conveys ~1 bpb while it
costs 3.1. A probe from the same code (a stronger decoder) reaches 3.42, so the
code, not the decoder, is the limit. `total = code + residual` is monotone in the
code rate (slope 0.30 at 300 and 1200 steps, every level/`d_c`/stride/pool), the
optimizer's optimum is to collapse the code, and the oracle tiling lies on the
same R-D line as the grid.

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

## v7: the redesign, with the model charged and both vocabularies

The code is the index of a span in a dictionary; the decoder writes the exact
bytes (residual zero); the model is charged and amortized over train+val; the DP
solves the exact cover and the greedy runs the same cost model.

`runs/v7/results.json` (text-only, `substr`) and `runs/v7/onsets.json` (word
spans, the supervised baseline):

| corpus | vocab | dict | model | unigram | gzip | DP | charged | greedy | fixed4 | oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synseg | word | 894 | 0.047 | 4.640 | 2.225 | 1.692 | 1.739 | 1.692 | 4.605 | 1.692 |
| synseg | substr | 8,361 | 0.420 | 4.640 | 2.225 | **1.682** | **2.101** | 1.751 | 3.306 | 1.913 |
| syngrid | word | 558 | 0.018 | 3.911 | 2.886 | 2.551 | 2.568 | 2.551 | 2.551 | 2.551 |
| syngrid | substr | 2,478 | 0.100 | 3.911 | 2.886 | **2.530** | **2.630** | 2.551 | 2.551 | 2.551 |
| shakespeare | word | 5,127 | 0.341 | 4.777 | 3.218 | 3.390 | 3.731 | 3.400 | 4.760 | 3.444 |
| shakespeare | substr | 5,792 | 0.292 | 4.777 | 3.218 | **3.133** | 3.426 | 3.352 | 4.106 | 4.262 |

## What holds

- **The representation pays where the model is small.** Text-only, no onsets in
  the vocabulary: charged 2.101 vs gzip 2.225 (synseg, 0.12 bpb) and 2.630 vs
  2.886 (syngrid, 0.26 bpb). The dictionary is data and it is paid for.
- **The tiling is decisive.** The fixed grid costs 3.31/2.55/4.11 against the DP
  1.68/2.53/3.13. On syngrid the grid is the oracle by construction, so the
  fixed arm ties the DP and the negative control holds.
- **The solver earns its keep only when candidates overlap.** Word dictionary
  (no overlap): DP vs greedy +0.000/+0.000/+0.010. Text-only (overlapping): the
  DP gains **+0.070/+0.021/+0.219 bpb**. That is the plan's P3 claim with clean
  controls: solving beats guessing exactly where guessing is myopic.
- The DP beats the fixed grid by 1.62/0.02/0.97 bpb; it beats the oracle by
  0.23/0.02/1.13. The oracle is a baseline, not a ceiling, for a substring
  vocabulary: the words are not special, so the DP, as the optimal cover, wins.

## What does not hold

- **Shakespeare is model-bound at 1.1 MB.** The best text-only point (5,792
  spans, 40.7 kB, 0.292 bpb) charges 3.426 against gzip 3.218: a 0.208 bpb loss.
  The model pays at 3.85 MB (text-only) or ~1.8 MB (word). The word vocabulary
  was better there because the words are the right units and the model is smaller.
- The text-only vocabulary costs more model than the word vocabulary for the same
  rate (8,361 vs 894 entries on synseg), so the *charge* is the frontier, not
  the codec.

## Caveats

The index cost is the static substring unigram, `-log2(count/n)`; it is
conservative in the length normalizer, and the two vocabularies normalize
differently, so their charged totals are not directly comparable (each is
compared to the same baselines). gzip is weak (brotli/xz are ~2.4-2.7 bpb on
shakespeare). The fallback switch is not charged.

## Next

1. One cost model for both vocabularies: a token-usage unigram with an EM pass,
   so the comparison is exact.
2. enwik8 with the model charged at 1/10/100 MB, where the representation should
   dominate.
3. brotli, xz, zstd -19, a byte transformer.
4. The parallelism dividend at matched bpb: one pass (DP + lookup) against L
   passes for a byte-AR model.

## Artifacts

- `runs/v7/results.json`, `runs/v7/REPORT.md` (text-only).
- `runs/v7/onsets.json` (word vocabulary).
- `runs/p0/quick-c3b7a8ac/` (v6 honest negative).
