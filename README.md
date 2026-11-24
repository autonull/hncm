# H-NCM: Hierarchical Neural Concept Modeling

Status: `hncm6.py` is the dense-latent v6 negative. `hncm7.py` is the redesign:
the span code is a *dictionary index*, the dictionary is a *charged model*, and
the tiling is *solved*. **The redesign beats gzip including its model on synseg
and syngrid, and loses on shakespeare at 1.1 MB. The solver beats the greedy only
on an overlapping (text-only) vocabulary.**

## The measured bottleneck (v6)

The span code is a 12-bit scalar quantization of a *pooled mean*. The mean is
permutation-invariant: it cannot tell `abab` from `aabb`. The encoder's per-byte
features recover the bytes exactly, but the mean destroys order, so the code
carries ~1 bpb while it costs 3.1. A probe from the same code with a stronger
decoder reaches 3.42, so the code, not the decoder, is the limit. `total = code +
residual` is monotone in the code rate (slope 0.30, unchanged at 1200 steps and at
every level, `d_c`, stride, pool and tiling); the optimizer's optimum is to
collapse the code; the oracle tiling lies on the same R-D line. Full Phase 0:
`runs/p0/quick-c3b7a8ac/` (3 seeds, 512 windows), G0 fails on total only.

## The redesign (v7)

The code is the index of a span in a dictionary; the decoder writes the exact
bytes, so the residual is zero. The dictionary is pruned by count, its model is
charged (`512 + sum(len+3)` bytes, amortized over train+val), and the DP solves
the exact cover. The greedy arm runs the *same* cost model with a longest-match
cover. All arms pay the same model and the same index costs.

| corpus | vocab | dict | model | unigram | gzip | DP | charged | greedy | fixed4 | oracle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synseg | word | 894 | 0.047 | 4.640 | 2.225 | 1.692 | 1.739 | 1.692 | 4.605 | 1.692 |
| synseg | substr | 8,361 | 0.420 | 4.640 | 2.225 | **1.682** | **2.101** | 1.751 | 3.306 | 1.913 |
| syngrid | word | 558 | 0.018 | 3.911 | 2.886 | 2.551 | 2.568 | 2.551 | 2.551 | 2.551 |
| syngrid | substr | 2,478 | 0.100 | 3.911 | 2.886 | **2.530** | **2.630** | 2.551 | 2.551 | 2.551 |
| shakespeare | word | 5,127 | 0.341 | 4.777 | 3.218 | 3.390 | 3.731 | 3.400 | 4.760 | 3.444 |
| shakespeare | substr | 5,792 | 0.292 | 4.777 | 3.218 | **3.133** | 3.426 | 3.352 | 4.106 | 4.262 |

Read this literally:

- **The representation pays for itself where the model is small.** With a
  text-only vocabulary the charged totals are 2.101 vs gzip 2.225 (synseg) and
  2.630 vs 2.886 (syngrid): the win survives the charge by 0.12/0.26 bpb, with no
  onsets in the vocabulary.
- **Shakespeare is model-bound at 1.1 MB**: the best text-only point is 5,792
  spans / 40.7 kB (0.292 bpb) and the charged total 3.426 loses to gzip by 0.208;
  it ties gzip at 3.85 MB. The word vocabulary is better there (3.731 in the same
  grid, gzip crossover ~1.8 MB) because the words are the right units and the
  dictionary can be smaller.
- **The solver earns its keep only on an overlapping vocabulary.** On the word
  dictionary the DP ties the greedy (+0.000/+0.000/+0.010). On the text-only
  vocabulary, where candidates overlap and the greedy is myopic, the DP gains
  **+0.070/+0.021/+0.219 bpb**. It also beats the fixed grid by 1.6/0.02/0.97.
- The oracle is a *baseline*, not a ceiling, for a substring vocabulary: the
  words are not special, so the DP beats it by 0.23/0.02/1.13 bpb. The DP is the
  optimal cover by construction, so `DP <= oracle` is an invariant, not a test.

## Honest caveats

1. The index cost is the static substring unigram, `-log2(count/n)`; it is
   conservative by the length normalizer. The word vocabulary normalizes over the
   kept token count instead, so the two vocabularies' charged totals are not
   directly comparable; each is compared to the same baselines.
2. The fallback switch is not charged (a span outside the dictionary pays its
   bytes; the decoder knows the index alphabet, so no flag is needed, but the
   single bytes share a separate normalizer).
3. gzip is a weak baseline (brotli/xz are ~2.4-2.7 bpb on shakespeare).
4. Shakespeare is 1.1 MB, so the model dominates; the crossover is the result.

## Next

1. One cost model for both vocabularies (a token-usage unigram with an EM pass),
   so the comparison is exact.
2. An enwik8 run with the model charged at 1/10/100 MB, where the representation
   should dominate.
3. Strong baselines: brotli, xz, zstd -19, a byte transformer.
4. The parallelism dividend at matched bpb: one pass (DP + lookup) against L
   passes for a byte-AR model.

## Commands

```bash
python hncm6.py selftest
python hncm7.py selftest
python hncm7.py run --corpora synseg,syngrid,shakespeare --vocab substr
python hncm7.py run --corpora synseg,syngrid,shakespeare --vocab onsets
```

## Artifacts

- `runs/v7/results.json`, `runs/v7/REPORT.md` (text-only vocabulary).
- `runs/v7/onsets.json` (word vocabulary baseline).
- `runs/p0/quick-c3b7a8ac/` (v6 honest negative).
- `hncm7.py`, `hncm6.py`; both `selftest` pass; `ruff format`/`ruff check` pass.
