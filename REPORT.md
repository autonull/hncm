# H-NCM v6 report

G0 fails on the codec, before the DP. Two measurement defects were fixed first;
both had inflated the codec:

1. The entropy model conditioned on the latent it was coding (`ctx = [z, mu,
   gain@Fh, hq]`). The decoder never has `z`, so the code rate was a fantasy
   (`CE=0.02` bpb against an empirical `H=2.96`). `SpanEnt` now conditions only on
   what the decoder has: the span start, the span length, the transmitted global
   factor, and the already-sent codes.
2. The rate was charged per position, not per span. `pool` packs one code per span
   at `s < S`, but `quantize` charged `L` codes and `code_info` counted the
   zero-derived padding tail. For stride 4 that is a 4x overcharge, and the tail
   deflated `CE`. Only the packed prefix is now charged, decoded, and measured.

Also fixed: `pos="span"` computed the span position embedding and never added it.

## Phase 0 at budget

Preset `quick`, 3 seeds, 512 windows, 400 steps, `format: 7`
(`runs/p0/quick-c3b7a8ac/`), mean of 3 seeds:

| corpus | arm | total | recon | code | H(code) | util |
|---|---|---:|---:|---:|---:|---:|
| synseg | rel-pos | 6.891 | 3.752 | 3.138 | 381.0 | 1.000 |
| synseg | abs-pos | 6.844 | 3.758 | 3.085 | 376.9 | 1.000 |
| synseg | constant-code | 4.642 | 4.642 | 0.000 | 0.0 | 0.0 |
| synseg | no-quant | 3.758 | 3.609 | 0.149 | 0.0 | 0.0 |
| synseg | unigram | 4.637 | 4.637 | 0.000 | - | - |
| syngrid | rel-pos | 5.975 | 2.863 | 3.112 | 377.0 | 1.000 |
| syngrid | abs-pos | 5.958 | 2.839 | 3.119 | 378.6 | 1.000 |
| syngrid | constant-code | 3.909 | 3.909 | 0.000 | 0.0 | 0.0 |
| syngrid | no-quant | 2.812 | 2.664 | 0.148 | 0.0 | 0.0 |
| syngrid | unigram | 3.908 | 3.908 | 0.000 | - | - |

Gates: **G0 fail** (total above the unigram on both corpora; `util`, `delta`,
`H` pass), **G6 pass** (`-0.006/+0.024`), **G9 pass** (3 seeds).

## Interpretation

The code is healthy: fully used (`util=1.0`), informative (`H≈380` bits/window =
2.97 bpb), and used by the decoder (`delta_recon=0.89/1.05` bpb). It is not worth
its bits: it costs 3.14/3.11 bpb and buys 0.89/1.05, i.e. 0.28-0.34 bpb per bpb of code.
The honest rate is ~`log2(levels)=3` bits/symbol; the decodable context cannot
compress the code (`CE ≈ H`).

The R-D curve is monotone at every swept point (levels 4/8, `d_c` 1/2/4, stride
4/8/16; 300 steps) and at 4x the steps (1200). `total = recon + code` is minimized
at code 0, where it equals the unigram. So no `lambda` and no tiling can pass G0:
the tiling cannot fix a code that does not pay for itself.

Per the decision tree G0 fails, so Phase 1 was not run and its smoke run is not
interpreted. The negative is publishable: the controls are clean.

## Next step

Code transfer efficiency, not the tiling. At the operating point the reconstruction
must gain > 1 bpb per code bit (now ~0.29): a decoder that predicts bytes
conditionally from the span content (a wider code, or an autoregressive decoder
inside the span), trained end to end, then rerun Phase 0. The DP only makes sense
once the code pays for itself.
