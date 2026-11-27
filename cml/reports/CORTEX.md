# Cortex: predict the term stream, not the bytes

- 30000 terms per split, depth 4, seed 0
- both arms: d128 l4 h4 seq128, 3000 steps, batch 32, 810,496 params
- held-out edge `ADD->XOR`: train and seen never contain it;
  `twin` rewrites it to an allowed edge at the same shape
- `cortex` has a 22-symbol vocabulary, `byte` has 256; both code the
  same bytes and the table divides by realized bytes, so `bpb` is
  comparable at matched params, data and steps.

| split | terms | nodes | bytes | cortex bpb | byte bpb | delta |
|---|---:|---:|---:|---:|---:|---:|
| seen | 30,000 | 371,850 | 1,451,231 | 0.958 | 0.980 | -0.022 |
| unseen | 30,000 | 518,217 | 2,035,976 | 0.970 | 0.966 | +0.004 |
| twin | 30,000 | 518,217 | 2,035,976 | 0.938 | 0.959 | -0.021 |

## Read

- the novel edge costs +0.032 bpb for the cortex and
  +0.007 bpb for the byte arm, at matched shape.
- on unseen compositions the cortex is +0.004 bpb against the
  byte arm; its edge pay is +0.025 relative to bytes.
- The byte arm wins on both the rate and the novel edge.
- This is a synthetic arithmetic pilot with matched params and steps:
  it tests representation and prediction, not reasoning. It does not
  decide the program; a tie is a reason to fall back to
  `TODO8.compression.md`.

Training took 108 s on one GPU.
