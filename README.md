 # H-NCM: Hierarchical Neural Concept Modeling

## Thesis
**H-NCM** merges **Cartesia's H-Net** (learned, dynamic, end-to-end tokenization directly from bytes) with **HazyResearch's Neural Concept Model (NCM)** (residual vector quantization and coarse-to-fine parallel decoding). 

## Reference
- Cartesia H-Net: https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1#next-post-ii
- HazyResearch NCM: https://github.com/HazyResearch/ncm

## Limitations
* **NCM:** Relies on a fixed stride length or heuristic boundaries for coarse tokens.
* **H-Net:** Generates sequentially next-byte in an autoregressive fashion.

## Hypothesis
Jointly training a model where hierarchical representations are learned end-to-end via a differentiable smoothing bridge using dynamic, semantic chunking will yield super-additive performance gains ($I = (D-C) - (B-A) > 0$) while maintaining near-constant, sub-quadratic inference passes relative to length.

## Implementation
- Write the complete self-contained Google CoLab script
- Demonstrate preliminary results
- Motivate further research
- Experiment starting with small models and scaling up.

## Ideas to consider
- To not simply combine H-Net's segmentation module with NCM's residual VQ; Develop a jointly learned, variable-rate hierarchical representation in which the model learns both what information to preserve and how many computational steps to spend reconstructing it.

