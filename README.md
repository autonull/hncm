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
- Write the complete self-contained Google CoLab script (`hncm6.py`; one file).
- Demonstrate preliminary results on `shakespeare` and the controls; motivate further research.
- Start with small models; scale only after the codec-health and leak gates pass.

## Ideas to consider
- Learn the cost; solve the structure. Do not learn the structure.
- Transmit the tiling; pay for it in bits. Only the fixed grid and the oracle are free.
- Share the codec across tilings; the tiling is inference, not training.
- Scalar quantization + low-rank prior instead of RVQ; the code must carry information.
- Span-relative decoding; no absolute position.
- Closed-loop, confidence-gated commitment; adaptive NFE.
- One ELBO in bits; no judge.
