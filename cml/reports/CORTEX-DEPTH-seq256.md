# Cortex: the depth sweep

- 20000 terms per split, 2500 steps, batch 32, seed 0
- both arms at d128 l4 h4 seq256, 826,880 params
- `edge pay` is the unseen minus twin cost: the price of the novel
  composition at matched shape. Lower is better.

| depth | seen delta | unseen delta | cortex edge | byte edge | gap |
|---:|---:|---:|---:|---:|---:|
| 6 | -0.001 | +0.013 | +0.012 | +0.003 | +0.009 |
| 7 | +0.001 | +0.011 | +0.009 | +0.002 | +0.008 |

## Read

- the edge gap moves -0.001 bpb from depth 6 to 7.
- A gap that falls with depth supports structure; a gap that rises
  or stays positive says the byte arm still computes the composition.
- This is a synthetic arithmetic sweep with matched params and steps,
  not code reasoning. No claim follows from one depth alone.

The sweep took 122 s on one GPU.
