# H-NCM: Hierarchical Neural Concept Modeling

## Thesis
**H-NCM v6** learns a rate–distortion cost field over byte spans and solves the optimal tiling exactly with a shortest-path DP. The tiling is a transmitted code; span codes are scalar-quantized with a low-rank hierarchical prior; generation is closed-loop parallel commitment. The dividend comes from **solving** the structure, not learning it.

## Reference
- Cartesia H-Net: https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1#next-post-ii
- HazyResearch NCM: https://github.com/HazyResearch/ncm

## Limitations
* **H-Net:** the router has no reconstruction gradient; its boundaries are free side information. v5's router was at chance (`align 0.146` vs grid `0.145`).
* **NCM:** residual VQ collapses when the rate target is below the natural rate (`util 1/64`); the decoder's absolute position is a free side channel; coarse-to-fine is open-loop. v5's codec tied unigram (`4.638` vs `4.638` bpb) and its code carried nothing.
* **v5:** per-chunker codecs confound the boundary effect with codec capacity; boundaries are free in all arms; "sample quality" is a judge, not the model; P1 was evaluated on synthetic corpora.

## Hypothesis
Replacing the router with an exact DP over a learned cost field, and RVQ with scalar quantization + a low-rank prior, yields (i) `total < unigram` on a real corpus, (ii) `calls_AR/calls_DP ≥ 8` at matched total bits, and (iii) `I_structure > 0` at matched λ. The v5 interaction `I` is a ceiling, not the deployable claim.

## Implementation
- `hncm6.py` is the complete self-contained script (one file): shared codec,
  scalar quantization, low-rank prior, exact DP cost field, span-relative decoder,
  closed-loop generation, controls, baselines, staged training, resume, gates, and
  report.
- Status: smoke/quick diagnostics only. G0 fails on total/code rate; G6 passes;
  G4 passes but G5 fails on alignment; G1/G2 fail on shakespeare. **No claim is
  established.**
- Run the smallest decisive phase first; scale only after G0 and G6 pass.
- Commands:
  ```bash
  python hncm6.py selftest
  python hncm6.py run --preset quick --phase 0 --seeds 0
  python hncm6.py report --results runs/latest/results.json
  ```

## Ideas to consider
- Learn the cost; solve the structure. Do not learn the structure.
- Transmit the tiling; pay for it in bits. Only the fixed grid and the oracle are free.
- Share the codec across tilings; the tiling is inference, not training.
- Scalar quantization + low-rank prior instead of RVQ; the code must carry information.
- Span-relative decoding; no absolute position.
- Closed-loop, confidence-gated commitment; adaptive NFE.
- One ELBO in bits; no judge.
