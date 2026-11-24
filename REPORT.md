# H-NCM report

## v6: the dense-latent codec fails, and why

Two measurement defects were fixed first (an entropy model conditioned on the
latent, and a 4x per-position rate overcharge). The honest number then exposed the
real defect: the span code is a 12-bit scalar quantization of a *mean*, which is
permutation-invariant, so it carries ~1 bpb while costing 3.1 bpb.
`total = code + residual` is monotone in the code rate (slope 0.30 at 300 and 1200
steps, every level/`d_c`/stride/pool), the optimizer's optimum is to collapse the
code, and the oracle tiling lies on the same R-D line as the grid.

Honest Phase 0 (`runs/p0/quick-c3b7a8ac/`, 3 seeds, 512 windows):

| corpus | arm | total | recon | code | H(code) | util |
|---|---|---:|---:|---:|---:|---:|
| synseg | rel-pos | 6.891 | 3.752 | 3.138 | 381.0 | 1.000 |
| synseg | constant-code | 4.642 | 4.642 | 0.000 | 0.0 | 0.0 |
| synseg | unigram | 4.637 | 4.637 | 0.000 | - | - |
| syngrid | rel-pos | 5.975 | 2.863 | 3.112 | 377.0 | 1.000 |
| syngrid | constant-code | 3.909 | 3.909 | 0.000 | 0.0 | 0.0 |
| syngrid | unigram | 3.908 | 3.908 | 0.000 | - | - |

Gates: G0 fail on total only; G6 pass; G9 pass. No Phase 1: the code does not pay.

## v7: the redesign, with the model charged

The code is the index of a span in a dictionary; the decoder writes the exact
bytes, so the residual is zero. The dictionary is pruned by train frequency, so
its model is a real cost, amortized over the stream (train+val). The DP solves the
exact cover; the greedy runs the same cost model with a longest-match cover.

`runs/v7/results.json`:

| corpus | unigram | gzip | DP rate | model | DP charged | crossover (gzip) |
|---|---:|---:|---:|---:|---:|---:|
| synseg | 4.640 | 2.225 | 1.288 | 0.041 | **1.329** | 0.05 MB |
| syngrid | 3.911 | 2.886 | 2.051 | 0.018 | **2.068** | 0.03 MB |
| shakespeare | 4.777 | 3.218 | 3.004 | 0.341 | 3.345 | 1.78 MB |

The representation wins where the model is small: 1.33/2.07 bpb against gzip
2.23/2.89. On shakespeare the best prune (5,127 spans, 47.6 kB) charges 0.34 bpb
and the total loses to gzip by 0.127 bpb at 1.1 MB; it ties gzip at 1.78 MB.

The tiling is decisive: the fixed grid costs 4.63/2.07/5.07, destroying the
dictionary's value on synseg and shakespeare. On syngrid the grid is the oracle
(`align=1.000`), so the fixed arm ties the DP and the negative control holds.

The DP beats the oracle on synseg (1.288 vs 1.347) by splitting rare words into
the cheaper byte fallback. The solver is *not* the win: `DP == greedy` on
synseg/syngrid, and 0.001 bpb apart on shakespeare. On a word dictionary the
greedy longest-match already finds the words.

## The budget curve (shakespeare, 1.12 MB stream)

| min_count | dict | model kB | rate | charged |
|---:|---:|---:|---:|---:|
| 1 | 29,148 | 317.2 | 2.709 | 4.985 |
| 2 | 10,961 | 109.1 | 2.852 | 3.635 |
| 4 | 5,127 | 47.6 | 3.004 | 3.345 |
| 8 | 2,630 | 22.5 | 3.207 | 3.368 |
| 16 | 1,431 | 10.9 | 3.391 | 3.469 |
| 32 | 873 | 5.8 | 3.715 | 3.756 |

The optimum is a compromise: an unpruned dictionary is free at the rate but
expensive as a model; a pruned one pays at the rate. The frontier is set by the
stream size, not by the codec.

## Caveats

The vocabulary comes from train onsets (supervised); the tiling does not. The
index model normalizes over the kept words and the fallback switch is not charged
(optimistic). Shakespeare is 1.1 MB, and gzip is a weak baseline (brotli/xz are
~2.4-2.7 bpb).

## Next

1. An unsupervised vocabulary (suffix automaton / BPE / learned candidates), with
   the oracle kept only as a ceiling.
2. An overlapping vocabulary (words + phrases + subwords) where the greedy is
   myopic: the only setting where the solver can beat it.
3. Strong baselines (brotli, xz, zstd -19, a byte transformer) and enwik8, with the
   model charged at every size.
4. The parallelism dividend at matched bpb: one pass (DP + lookup) against L for a
   byte-AR model.

## Artifacts

- `runs/v7/results.json`, `runs/v7/REPORT.md` (v7, charged).
- `runs/p0/quick-c3b7a8ac/` (v6 honest negative).
- Superseded: `runs/p0/quick-dbe8d386/` (`format: 6`), `runs/p1/smoke-37d2450a/`,
  `runs/p2/smoke-bad03cb9/`.
