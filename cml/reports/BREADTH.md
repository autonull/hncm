# Breadth sweep: hyperparameters, not verdicts

- corpus `code` sha `1ccd9ffc7f68`, held-out 70,000 B, 200 spans, seed 0
- pack 256 units; library and copy train on 1,010,246 elements
- A single config is not a kill. Each family gets a small grid; the
  best row is the direction to push next.

## Library order (past context only)

| order | exact | bits/span |
|---:|---:|---:|
| 1 | 0.030 | 2.869 |
| 2 | 0.110 | 3.347 |
| 3 | 0.195 | 4.697 |
| 4 | 0.180 | 5.481 |
| 5 | 0.205 | 5.970 |

## Template/hole (both sides known)

The target span is the hole; the units before and after it are
known. Back off from radius 3 to 1, then to the global count.

| radius | found | exact | bits/span |
|---:|---:|---:|---:|
| 1 | 0.980 | 0.220 | 3.553 |
| 2 | 0.980 | 0.340 | 4.937 |
| 3 | 0.980 | 0.340 | 5.026 |

## Copy: key x offset cap x confidence (frontier)

Copy predicts from the nearest prior match; the pointer predicts
otherwise. `tau` is the pointer confidence in the copy candidate
required to use it. `cap` bounds the offset code. Full grid in JSON.

| key | cap | tau | hit | exact | bits/span |
|---:|---:|---:|---:|---:|---:|
| 6 | - | 0.00 | 0.340 | 0.720 | 4.871 |
| 7 | - | 0.00 | 0.300 | 0.715 | 4.494 |
| 8 | - | 0.00 | 0.290 | 0.710 | 4.435 |
| 10 | - | 0.00 | 0.220 | 0.705 | 3.853 |
| 12 | - | 0.00 | 0.200 | 0.695 | 3.728 |
| 7 | 1024 | 0.00 | 0.175 | 0.690 | 3.137 |
| 8 | 1024 | 0.00 | 0.165 | 0.685 | 3.073 |
| 5 | 256 | 0.00 | 0.120 | 0.680 | 2.493 |
| 7 | 256 | 0.00 | 0.070 | 0.675 | 2.240 |
| 8 | 256 | 0.00 | 0.065 | 0.670 | 2.206 |
| 10 | 256 | 0.00 | 0.040 | 0.665 | 2.044 |
| 12 | 256 | 0.00 | 0.025 | 0.660 | 1.961 |
| 12 | 256 | 0.10 | 0.025 | 0.660 | 1.961 |
| 12 | 256 | 0.25 | 0.010 | 0.650 | 1.899 |

## Copy distribution: plurality of prior continuations

Rank the continuation, do not encode the absolute offset. The copy
distribution is the count of concept units that followed every prior
match of the key; `min hits` is the plurality required to use it.

| key | min hits | tau | hit | exact | bits/span |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 0.00 | 0.530 | 0.695 | 4.570 |
| 4 | 1 | 0.10 | 0.355 | 0.660 | 3.749 |
| 4 | 2 | 0.00 | 0.375 | 0.700 | 3.622 |
| 4 | 2 | 0.10 | 0.265 | 0.650 | 3.185 |
| 6 | 1 | 0.00 | 0.365 | 0.705 | 3.822 |
| 6 | 1 | 0.10 | 0.250 | 0.675 | 3.264 |
| 6 | 2 | 0.00 | 0.185 | 0.690 | 2.694 |
| 6 | 2 | 0.10 | 0.125 | 0.655 | 2.464 |
| 8 | 1 | 0.00 | 0.315 | 0.710 | 3.580 |
| 8 | 1 | 0.10 | 0.225 | 0.670 | 3.157 |
| 8 | 2 | 0.00 | 0.155 | 0.685 | 2.580 |
| 8 | 2 | 0.10 | 0.100 | 0.655 | 2.362 |
| 10 | 1 | 0.00 | 0.240 | 0.710 | 3.168 |
| 10 | 1 | 0.10 | 0.165 | 0.670 | 2.813 |
| 10 | 2 | 0.00 | 0.110 | 0.685 | 2.345 |
| 10 | 2 | 0.10 | 0.060 | 0.655 | 2.136 |
| 12 | 1 | 0.00 | 0.205 | 0.695 | 3.032 |
| 12 | 1 | 0.10 | 0.150 | 0.670 | 2.729 |
| 12 | 2 | 0.00 | 0.065 | 0.670 | 2.146 |
| 12 | 2 | 0.10 | 0.035 | 0.655 | 1.989 |

## Concept + copy (R2)

Copy when a prior key match passes the cap and confidence gate;
otherwise the concept head predicts. Copy costs the flag plus the
offset; the fallback costs the concept NLL. Frontier rows only.

| key | cap | tau | hit | exact | bits/span |
|---:|---:|---:|---:|---:|---:|
| 6 | - | 0.00 | 0.340 | 0.720 | 4.782 |
| 7 | - | 0.00 | 0.300 | 0.715 | 4.391 |
| 8 | - | 0.00 | 0.290 | 0.710 | 4.335 |
| 10 | - | 0.00 | 0.220 | 0.705 | 3.699 |
| 12 | - | 0.00 | 0.200 | 0.695 | 3.546 |
| 7 | 1024 | 0.00 | 0.175 | 0.690 | 3.057 |
| 8 | 1024 | 0.00 | 0.165 | 0.685 | 2.996 |
| 5 | 256 | 0.00 | 0.120 | 0.680 | 2.382 |
| 7 | 256 | 0.00 | 0.070 | 0.675 | 2.126 |
| 8 | 256 | 0.00 | 0.065 | 0.670 | 2.100 |
| 10 | 256 | 0.00 | 0.040 | 0.665 | 1.913 |
| 12 | 256 | 0.00 | 0.025 | 0.660 | 1.881 |
| 12 | 256 | 0.10 | 0.025 | 0.660 | 1.881 |
| 12 | 256 | 0.25 | 0.010 | 0.650 | 1.809 |

## Concept + copy distribution (R2)

Copy the plurality continuation when it passes the hit and
confidence gate; otherwise the concept head predicts. Copy costs
the continuation entropy; the fallback costs the concept NLL.
Frontier rows only.

| key | min hits | tau | hit | exact | bits/span |
|---:|---:|---:|---:|---:|---:|
| 10 | 1 | 0.00 | 0.240 | 0.710 | 2.767 |
| 12 | 1 | 0.00 | 0.205 | 0.695 | 2.651 |
| 6 | 2 | 0.00 | 0.185 | 0.690 | 2.404 |
| 10 | 2 | 0.00 | 0.110 | 0.685 | 2.074 |
| 12 | 2 | 0.00 | 0.065 | 0.670 | 1.956 |
| 12 | 2 | 0.10 | 0.035 | 0.655 | 1.889 |

## Copy gates (R2)

`blend` mixes the concept and copy distributions by `alpha`;
`margin` uses copy only when the plurality share passes `g`;
`stack` tries the high key, then the low key, then the concept head.
Frontier rows only.

| kind | key | params | hit | exact | bits/span |
|---|---|---|---:|---:|---:|
| blend | 6 | m1 a0.90 | 0.365 | 0.715 | 2.740 |
| blend | 10 | m1 a0.90 | 0.240 | 0.705 | 2.362 |
| blend | 10 | m1 a0.75 | 0.240 | 0.690 | 1.917 |
| blend | 6 | m1 a0.50 | 0.365 | 0.680 | 1.698 |
| blend | 6 | m1 a0.25 | 0.365 | 0.675 | 1.667 |

## Edit prefix

`base` is the same survivors under the full pack; `delta exact` is
the behavior change after the deletion.

| prefix | units | affected B | bpb delta | base | exact | delta exact | round-trip |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ` ` | 120 | 4,384 | +0.146 | 0.485 | 0.491 | +0.006 | OK |
| `
` | 224 | 6,904 | +0.228 | 0.439 | 0.367 | -0.072 | OK |
| `return` | 254 | 379 | +0.026 | 0.414 | 0.419 | +0.005 | OK |
| `self` | 254 | 40 | -0.000 | 0.415 | 0.415 | +0.000 | OK |
| `=` | 255 | 194 | +0.006 | 0.412 | 0.402 | -0.010 | OK |
| `    ` | 150 | 465 | +0.028 | 0.421 | 0.426 | +0.005 | OK |

## Speculative block

| K | accepted/call | all accepted | accept/step |
|---:|---:|---:|---:|
| 2 | 1.065 | 0.355 | 0.526 |
| 4 | 1.410 | 0.105 | 0.344 |
| 8 | 1.560 | 0.005 | 0.188 |
| 12 | 1.505 | 0.000 | 0.125 |
| 16 | 1.505 | 0.000 | 0.093 |

## Best directions

- library: order **5**, exact 0.205, 5.970 bits/span.
- template: radius **2**, exact 0.340, 4.937 bits/span.
- copy: key **6**, cap **None**, tau **0.00**, exact 0.720, 4.871 bits/span.
- copy distribution: key **10**, min hits **1**, tau **0.00**, exact 0.710, 3.168 bits/span.
- concept + copy: key **6**, cap **None**, tau **0.00**, exact 0.720, 4.782 bits/span.
- concept + copy dist: key **10**, min hits **1**, tau **0.00**, exact 0.710, 2.767 bits/span.
- best gate: **blend**, key **6**, exact 0.715, 2.740 bits/span.
- edit: prefix `self`, delta -0.000 bpb, exact change +0.000, round-trip OK.
- spec: K **8**, 1.560 accepted/call.

The concept-only baseline is 0.650 exact / 1.720 bits/span / one
decision. A family earns the next experiment only if its best grid
row approaches that frontier or reveals a new tradeoff.

Sweep took 240.6 s.
