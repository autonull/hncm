# Cortex: the depth sweep

- 20000 terms per split, 2500 steps, batch 32, seed 0
- both arms at d128 l4 h4 seq128, 810,496 params
- `edge pay` is the unseen minus twin cost: the price of the novel
  composition at matched shape. Lower is better.

| depth | seen delta | unseen delta | cortex edge | byte edge | gap |
|---:|---:|---:|---:|---:|---:|
| 3 | +0.020 | +0.050 | +0.038 | +0.014 | +0.024 |
| 4 | -0.017 | +0.008 | +0.028 | +0.008 | +0.020 |
| 5 | -0.027 | -0.011 | +0.018 | +0.005 | +0.013 |

## Read

- the edge gap moves -0.010 bpb from depth 3 to 5.
- A gap that falls with depth supports structure; a gap that rises
  or stays positive says the byte arm still computes the composition.
- This is a synthetic arithmetic sweep with matched params and steps,
  not code reasoning. No claim follows from one depth alone.

The sweep took 243 s on one GPU.
