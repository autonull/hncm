# Breadth sweep: hyperparameters, not verdicts

- corpus `code` sha `1ccd9ffc7f68`, held-out 70,000 B, 200 spans, seed 0
- pack 256 units; library and copy train on 1,010,246 elements
- A single config is not a kill. Each family gets a small grid; the
  best row is the direction to push next.

## Library order

| order | exact | bits/span |
|---:|---:|---:|
| 1 | 0.030 | 2.869 |
| 2 | 0.110 | 3.347 |
| 3 | 0.195 | 4.697 |
| 4 | 0.180 | 5.481 |
| 5 | 0.205 | 5.970 |

## Copy key length

| key | hit | exact | bits/span |
|---:|---:|---:|---:|
| 1 | 0.990 | 0.135 | 5.944 |
| 2 | 0.910 | 0.390 | 8.521 |
| 3 | 0.795 | 0.540 | 8.417 |
| 4 | 0.640 | 0.605 | 7.430 |
| 5 | 0.555 | 0.640 | 6.753 |
| 6 | 0.475 | 0.650 | 6.193 |

## Edit prefix

| prefix | units | affected B | bpb delta | round-trip |
|---|---:|---:|---:|:---:|
| ` ` | 120 | 4,384 | +0.146 | OK |
| `
` | 224 | 6,904 | +0.228 | OK |
| `return` | 254 | 379 | +0.026 | OK |
| `self` | 254 | 40 | -0.000 | OK |
| `=` | 255 | 194 | +0.006 | OK |
| `    ` | 150 | 465 | +0.028 | OK |

## Speculative block

| K | accepted/call | all accepted | accept/step |
|---:|---:|---:|---:|
| 2 | 1.065 | 0.355 | 0.526 |
| 4 | 1.410 | 0.105 | 0.344 |
| 8 | 1.560 | 0.005 | 0.188 |
| 16 | 1.505 | 0.000 | 0.093 |

## Best directions

- library: order **5**, exact 0.205, 5.970 bits/span.
- copy: key **6**, exact 0.650, hit 0.475.
- edit: prefix `self`, delta -0.000 bpb, round-trip OK.
- spec: K **8**, 1.560 accepted/call.

The concept-only baseline is 0.650 exact / 1.720 bits/span / one
decision. A family earns the next experiment only if its best grid
row approaches that frontier or reveals a new tradeoff.

Sweep took 164.9 s.
