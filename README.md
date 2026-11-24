# H-NCM: Hierarchical Neural Concept Modeling

Status: `hncm6.py` is the dense-latent v6 negative; `hncm7.py` is the redesign
that follows from the measured bottleneck. **No v6 P1/P2/P3 claim is established.
The v7 redesign beats gzip *including its model* on synseg and syngrid; on
shakespeare it loses by 0.13 bpb at this corpus size.**

## The measured bottleneck (v6)

`hncm6.py` transmits a span code that is a 12-bit scalar quantization of a
*pooled span mean*. The mean is permutation-invariant: it cannot tell `abab` from
`aabb`. The encoder's per-byte features recover the bytes exactly, but the mean
destroys the byte order, so the code carries ~1 bpb while it costs 3.1 bpb. A
probe from the same 12-bit code (with a stronger decoder than the codec's) reaches
3.42 bpb, so the decoder is not the limit: the code is. `total = code + residual`
is monotone in the code rate (slope 0.30, unchanged at 1200 steps and at every
level, `d_c`, stride, pool and tiling), and the optimizer's optimum is to collapse
the code to a constant (`total=4.79`, i.e. no code). The oracle tiling lies on the
same R-D line as the grid, so the tiling cannot rescue a dense code.

Full v6 Phase 0: `runs/p0/quick-c3b7a8ac/` (3 seeds, 512 windows). G0 fails on
total only; util/delta/H pass; G6/G9 pass.

## The redesign (v7)

The code is the *index* of a span in a *dictionary*; the decoder writes the exact
bytes, so the residual is zero for dictionary spans and unknown spans pay their
bytes at the byte unigram. Only indices and fallback bytes are paid. The
dictionary is pruned by train frequency, so its *model* is a real cost. The tiling
is inference: a shortest-path DP solves the exact cover; the greedy arm runs the
*same* cost model with a longest-match cover. The oracle splits at the true word
onsets; the fixed grid is the chunker.

Charged, stream = train+val, `runs/v7/results.json`:

| corpus | unigram | gzip | DP rate | model | DP charged | verdict |
|---|---:|---:|---:|---:|---:|---|
| synseg | 4.640 | 2.225 | 1.288 | 0.041 | **1.329** | pass, by 0.90 |
| syngrid | 3.911 | 2.886 | 2.051 | 0.018 | **2.068** | pass, by 0.82 |
| shakespeare | 4.777 | 3.218 | 3.004 | 0.341 | 3.345 | fail, by 0.13; gzip crossover 1.78 MB |

Read this literally:

- The *representation* is the win. A pruned dictionary index plus a byte fallback
  costs 1.33/2.07/3.35 bpb against gzip 2.23/2.89/3.22, and against the unigram
  4.64/3.91/4.78. On the synthetic corpora the model is 6.1/2.6 kB, so the charge
  is 0.04/0.02 bpb; on shakespeare the best prune is 47.6 kB (5,127 spans) and the
  charge is 0.34 bpb at 1.1 MB.
- The *tiling* is decisive: the fixed 4-byte grid costs 4.63/2.07/5.07, i.e. it
  destroys the dictionary's value on synseg and shakespeare. On syngrid the grid
  *is* the oracle, so the fixed arm ties the DP: the plan's negative control
  holds, and `align=1.000` on both.
- The DP *beats the oracle* on synseg (1.288 vs 1.347): with a pruned dictionary
  the oracle's rare words pay the fallback, and the DP splits them into cheaper
  bytes. That is the fallback selection the solver earns.
- The *solver is not the win*: `DP == greedy` on synseg and syngrid, and on
  shakespeare the gap is 0.001 bpb. On a word dictionary the greedy longest-match
  already finds the words. The plan's falsifier is confirmed on this vocabulary.
- The `fixed`/`random`/`oracle`/`DP` arms all pay the same cost model; the fixed
  grid is free (its boundaries are the model), so this is conservative for it.

## Caveats

1. The v7 vocabulary is built from the *train* word onsets, so it is supervised;
   the *tiling* is not (the DP uses only costs). A deployable version must build
   the vocabulary from text alone (a suffix automaton, BPE, or a learned candidate
   generator).
2. The index model normalizes over the kept words; the fallback switch is not
   charged. That is optimistic by roughly the fraction of fallback tokens.
3. Shakespeare is only 1.1 MB. At this size the model dominates; the crossover is
   1.78 MB for gzip and 0.21 MB for the unigram. Scale up before comparing to
   strong codecs; gzip is a weak baseline, and brotli/xz are ~2.4-2.7 bpb.

## Next

1. Build the vocabulary without onsets; keep the oracle only as a ceiling.
2. Overlapping candidates (words + frequent phrases + subwords) so the greedy is
   myopic; that is the only setting where the solver can beat it.
3. Strong baselines (brotli, xz, zstd -19, a byte transformer) and enwik8, with the
   model charged at each size.
4. Measure the parallelism dividend at matched bpb: the v7 decoder is one pass (DP
   + lookup) against L passes for a byte-AR model.

## Commands

```bash
python hncm6.py selftest
python hncm6.py run --preset quick --phase 0 --seeds 0,1,2
python hncm7.py selftest
python hncm7.py run --corpora synseg,syngrid,shakespeare
```

## Artifacts

- `runs/v7/results.json`, `runs/v7/REPORT.md` (the redesign, charged).
- `runs/p0/quick-c3b7a8ac/` (v6 Phase 0, honest negative).
- `hncm7.py`, `hncm6.py`; both `selftest` pass; `ruff format`/`ruff check` pass.
- Superseded and misleading: `runs/p0/quick-dbe8d386/` (format 6),
  `runs/p1/smoke-37d2450a/`, `runs/p2/smoke-bad03cb9/`.
