# H-NCM report

## v6: the dense-latent codec fails, and why

Two measurement defects were fixed first (an entropy model conditioned on the
latent, and a 4x per-position rate overcharge). The honest number then exposed
the real defect: the span code is a 12-bit scalar quantization of a *mean*, which
is permutation-invariant, so it carries ~1 bpb while costing 3.1 bpb.
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

Gates: G0 fail on total only; G6 pass; G9 pass. No Phase 1 was run: the code
does not pay for itself.

## v7: the redesign

The code is the index of a span in a free dictionary; the decoder writes the
exact bytes, so the residual is zero and only indices and fallback bytes are paid.
The DP solves the exact cover with the index costs. The greedy is the same cost
model with a longest-match cover, so the gap isolates solving from guessing.

Best lambda, `runs/v7/results.json`:

| corpus | unigram | gzip | DP | greedy | oracle | fixed4 | align(DP/oracle) |
|---|---:|---:|---:|---:|---:|---:|---:|
| synseg | 4.640 | 2.225 | **1.285** | 1.285 | 1.285 | 4.591 | 1.000 |
| syngrid | 3.911 | 2.886 | **2.051** | 2.051 | 2.051 | 2.051 | 1.000 |
| shakespeare | 4.777 | 3.218 | **2.709** | 2.738 | 2.776 | 4.728 | 0.687 |

The representation wins: below the unigram and gzip on all three corpora, with a
parallel decoder. The tiling is decisive: the fixed grid destroys the value
(4.59/4.73 on synseg/shakespeare). On syngrid the grid *is* the oracle by
construction, so the fixed arm ties the DP: the plan's negative control holds.

The solver is not the win. On a word dictionary the greedy longest-match already
finds the words (`DP == greedy` on synseg/syngrid); the DP beats the greedy only
on shakespeare, by 0.029 bpb. The DP does beat the oracle on shakespeare
(2.709 vs 2.776) by splitting rare words into the cheaper byte fallback.

## Caveat and next step

The v7 vocabulary is built from the train word onsets, so it is supervised; the
*tiling* is not (the DP uses only costs). A deployable version must build the
vocabulary without onsets (a suffix automaton, BPE, or a learned candidate
generator), then rerun the arms. The second step is to make the solver matter: on
an overlapping (word + phrase) vocabulary the greedy is myopic and the DP can
merge, which is where the solving claim can be tested.

## Artifacts

- `runs/v7/results.json`, `runs/v7/REPORT.md` (v7).
- `runs/p0/quick-c3b7a8ac/` (v6 honest negative).
- Superseded: `runs/p0/quick-dbe8d386/` (`format: 6`), `runs/p1/smoke-37d2450a/`,
  `runs/p2/smoke-bad03cb9/`.
