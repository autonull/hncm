# H-NCM: Hierarchical Neural Concept Modeling

Status: `hncm6.py` is the dense-latent v6 negative; `hncm7.py` is the redesign
that follows from the measured bottleneck. **No v6 P1/P2/P3 claim is established.
The v7 redesign beats the byte unigram and gzip on all three corpora with a
*parallel* decoder.**

## The measured bottleneck (v6)

`hncm6.py` transmits a span code that is a 12-bit scalar quantization of a
*pooled span mean*. The mean is permutation-invariant: it cannot tell `abab` from
`aabb`. The encoder's per-byte features recover the bytes exactly, but the mean
destroys the byte order, so the code carries ~1 bpb while it costs 3.1 bpb
(`delta_recon=0.89/1.05`, `code=3.14/3.11`). A probe from the same 12-bit code
(with a stronger decoder than the codec's) reaches only 3.42 bpb, so the decoder
is not the limit: the code is. Consequently `total = code + residual` is monotone
in the code rate (slope 0.30, unchanged at 1200 steps and at every level, `d_c`,
stride and tiling), and the optimizer's optimum is to collapse the code to a
constant (`total=4.79`, i.e. no code). The oracle tiling lies on the same R-D
line as the grid, so the tiling cannot rescue a dense code.

Full v6 Phase 0: `runs/p0/quick-c3b7a8ac/` (3 seeds, 512 windows). G0 fails on
total only; util/delta/H pass; G6/G9 pass.

## The redesign (v7)

The code is the *index* of a span in a *free* dictionary (part of the model, like
the byte unigram table). The decoder writes the span's exact bytes, so the
residual is **zero** for dictionary spans; spans outside the dictionary pay their
bytes at the byte unigram. Only indices and fallback bytes are paid. The tiling is
inference: a shortest-path DP solves the exact cover with the index costs; the
greedy arm runs the *same* cost model with a longest-match cover, so the DP/greedy
gap isolates solving from guessing. The oracle splits at the true word onsets; the
fixed grid is the chunker. Every arm pays the same cost model.

Honest run (`runs/v7/results.json`), best lambda:

| corpus | unigram | gzip | DP | greedy | oracle | fixed4 |
|---|---:|---:|---:|---:|---:|---:|
| synseg | 4.640 | 2.225 | **1.285** | 1.285 | 1.285 | 4.591 |
| syngrid | 3.911 | 2.886 | **2.051** | 2.051 | 2.051 | 2.051 |
| shakespeare | 4.777 | 3.218 | **2.709** | 2.738 | 2.776 | 4.728 |

Read this literally:

- The *representation* is the win: the dictionary index plus a byte fallback is
  1.29/2.05/2.71 bpb, below the unigram (4.64/3.91/4.78) and below gzip
  (2.23/2.89/3.22) on all three corpora, with a *parallel* decoder (one call;
  the greedy is the same model).
- The *tiling* is decisive: the fixed 4-byte grid scores 4.59/2.05/4.73, i.e. it
  destroys the dictionary's value on synseg and shakespeare. On syngrid the grid
  is the oracle by construction (`align=1.000`), so the fixed arm ties the DP:
  that is the plan's negative control, and it holds.
- The DP recovers the true word boundaries from the index costs alone
  (`align=1.000` on synseg and syngrid) and beats the oracle on shakespeare
  (2.709 vs 2.776) by splitting rare words into the cheaper byte fallback.
- The DP beats the greedy only on shakespeare and only by 0.029 bpb. On a word
  dictionary the greedy longest-match already finds the words, so the solver adds
  little; the falsifier "the DP does not beat the greedy" is **confirmed** on two
  of three corpora.
- The `fixed` grid, `random`, `oracle` and `DP` all pay the same cost model; the
  fixed grid is *free* (its boundaries are part of the model), so this is a
  *conservative* comparison for it.

## Caveat

The v7 dictionary is built from the *train* word onsets, so the vocabulary is
supervised (the tiling is not: the DP uses only the costs). A deployable version
must build the vocabulary from the train text without onsets (a suffix automaton,
BPE, or a learned candidate generator). That is the next step. The dictionary is
free, as gzip's code and the unigram table are free; the JSON counts only the
indices and the fallback.

## Commands

```bash
python hncm6.py selftest
python hncm6.py run --preset quick --phase 0 --seeds 0,1,2
python hncm7.py selftest
python hncm7.py run --corpora synseg,syngrid,shakespeare
```

## Artifacts

- `runs/v7/results.json`, `runs/v7/REPORT.md` (the redesign).
- `runs/p0/quick-c3b7a8ac/` (v6 Phase 0, honest negative).
- `hncm7.py`, `hncm6.py`; both `selftest` pass; `ruff format`/`ruff check` pass.
- Superseded and misleading: `runs/p0/quick-dbe8d386/` (format 6),
  `runs/p1/smoke-37d2450a/`, `runs/p2/smoke-bad03cb9/`.
