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

## Copy: key x offset cap x confidence

Copy predicts from the nearest prior match; the pointer predicts
otherwise. `tau` is the pointer confidence in the copy candidate
required to use it. `cap` bounds the offset code.

| key | cap | tau | hit | exact | bits/span |
|---:|---:|---:|---:|---:|---:|
| 4 | 256 | 0.00 | 0.140 | 0.680 | 2.609 |
| 4 | 256 | 0.10 | 0.095 | 0.660 | 2.366 |
| 4 | 256 | 0.25 | 0.075 | 0.650 | 2.274 |
| 4 | 1024 | 0.00 | 0.270 | 0.685 | 3.731 |
| 4 | 1024 | 0.10 | 0.195 | 0.660 | 3.254 |
| 4 | 1024 | 0.25 | 0.155 | 0.650 | 3.006 |
| 4 | - | 0.00 | 0.435 | 0.705 | 5.570 |
| 4 | - | 0.10 | 0.320 | 0.665 | 4.692 |
| 4 | - | 0.25 | 0.245 | 0.650 | 4.065 |
| 5 | 256 | 0.00 | 0.120 | 0.680 | 2.493 |
| 5 | 256 | 0.10 | 0.080 | 0.665 | 2.279 |
| 5 | 256 | 0.25 | 0.055 | 0.650 | 2.161 |
| 5 | 1024 | 0.00 | 0.230 | 0.685 | 3.428 |
| 5 | 1024 | 0.10 | 0.165 | 0.660 | 3.020 |
| 5 | 1024 | 0.25 | 0.125 | 0.645 | 2.791 |
| 5 | - | 0.00 | 0.395 | 0.720 | 5.262 |
| 5 | - | 0.10 | 0.285 | 0.670 | 4.402 |
| 5 | - | 0.25 | 0.200 | 0.645 | 3.688 |
| 6 | 256 | 0.00 | 0.085 | 0.675 | 2.333 |
| 6 | 256 | 0.10 | 0.060 | 0.665 | 2.176 |
| 6 | 256 | 0.25 | 0.040 | 0.650 | 2.083 |
| 6 | 1024 | 0.00 | 0.185 | 0.685 | 3.177 |
| 6 | 1024 | 0.10 | 0.135 | 0.665 | 2.827 |
| 6 | 1024 | 0.25 | 0.100 | 0.650 | 2.622 |
| 6 | - | 0.00 | 0.340 | 0.720 | 4.871 |
| 6 | - | 0.10 | 0.245 | 0.675 | 4.068 |
| 6 | - | 0.25 | 0.165 | 0.650 | 3.371 |
| 7 | 256 | 0.00 | 0.070 | 0.675 | 2.240 |
| 7 | 256 | 0.10 | 0.055 | 0.665 | 2.146 |
| 7 | 256 | 0.25 | 0.035 | 0.650 | 2.053 |
| 7 | 1024 | 0.00 | 0.175 | 0.690 | 3.137 |
| 7 | 1024 | 0.10 | 0.130 | 0.665 | 2.805 |
| 7 | 1024 | 0.25 | 0.095 | 0.650 | 2.593 |
| 7 | - | 0.00 | 0.300 | 0.715 | 4.494 |
| 7 | - | 0.10 | 0.215 | 0.670 | 3.767 |
| 7 | - | 0.25 | 0.150 | 0.650 | 3.233 |
| 8 | 256 | 0.00 | 0.065 | 0.670 | 2.206 |
| 8 | 256 | 0.10 | 0.055 | 0.665 | 2.148 |
| 8 | 256 | 0.25 | 0.035 | 0.650 | 2.054 |
| 8 | 1024 | 0.00 | 0.165 | 0.685 | 3.073 |
| 8 | 1024 | 0.10 | 0.130 | 0.665 | 2.808 |
| 8 | 1024 | 0.25 | 0.095 | 0.650 | 2.595 |
| 8 | - | 0.00 | 0.290 | 0.710 | 4.435 |
| 8 | - | 0.10 | 0.215 | 0.670 | 3.774 |
| 8 | - | 0.25 | 0.150 | 0.650 | 3.239 |

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
- edit: prefix `self`, delta -0.000 bpb, exact change +0.000, round-trip OK.
- spec: K **8**, 1.560 accepted/call.

The concept-only baseline is 0.650 exact / 1.720 bits/span / one
decision. A family earns the next experiment only if its best grid
row approaches that frontier or reveals a new tradeoff.

Sweep took 212.5 s.
