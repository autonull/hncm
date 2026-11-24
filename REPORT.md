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
- **Shakespeare is model-bound at 1.1 MB.** Word: 3.347 vs gzip 3.218, a 0.129
  loss, crossing at ~2.0 MB. Text: 3.455, a 0.237 loss.
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

## Caveats

The vocabulary size is chosen on the charged total reported on the same val (one
knob, 8 points): a selection leak. The fallback is charged as the byte unigram,
not as a switch. gzip is a weak baseline (brotli/xz are ~2.4-2.7 bpb on
shakespeare). The synthetic corpora are highly repetitive, so the EM's pruning
does the heavy lifting.

## Next

1. enwik8 with the model charged at 1/10/100 MB.
2. brotli, xz, zstd -19, a byte transformer.
3. The parallelism dividend at matched bpb: one pass against L for a byte-AR.
4. A held-out split for the vocabulary size.

## Artifacts

- `runs/v7/results.json` (text-only), `runs/v7/onsets.json` (word), both with
  `REPORT.md` tables.
- `runs/p0/quick-c3b7a8ac/` (v6 honest negative).
