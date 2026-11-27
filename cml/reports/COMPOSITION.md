# Composition: typed holes and exact realization

- 4 templates, 171 B library, depth 3, seed 0
- banned edge `ADD->MUL`: train and seen never contain it
- `concept` codes the term stream; `bytes` is the order-2 byte arm on
  the same realizations. Both adapt on the split they code.
- `twin` is the unseen split with ADD -> SUB: the same shape and op
  count, so `unseen - twin` is the price of that one edge.
- `shared` charges the library once; `bpb` divides by realized bytes.

| split | terms | nodes | bytes | concept | shared | bytes arm |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,000 | 6,876 | 26,437 | 0.956 | 1.007 | 1.232 |
| seen | 1,000 | 6,950 | 26,769 | 0.952 | 1.004 | 1.226 |
| unseen | 1,000 | 10,255 | 39,732 | 0.931 | 0.965 | 1.161 |
| twin | 1,000 | 10,255 | 39,732 | 0.904 | 0.939 | 1.138 |

## Read

- every realization round-trips exactly; the term is the message and
  the bytes are derived, so the representation is editable.
- the novel edge costs +0.026 bpb for the term code and
  +0.023 bpb for the byte arm, at matched shape.
- Both arms generalize this edge: at this grain the term gain is
  symbolization, not composition. That is a pilot limit, not a gate.
- the term code still pays 0.230 bpb less than bytes on unseen
  compositions, with exact realization.
- This prices the representation, not reasoning: the encoder and
  decoder both hold the term. The separating test needs a harder
  composition task and a matched byte Transformer (P5e/P5f).
