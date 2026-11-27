# Causal edit curve (r5)

- corpus `code` sha `1ccd9ffc7f68`, held-out 70,000 B, 200 matched spans, seed 0
- pack 256 units / 2,771 B; remove every unit starting with ` `
- exact is scored only on targets that survive the edited pack; `base` is the same subset under the full pack.

| stage | units | n | removed | base | exact | delta | bits/span |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 256 | 200 | 0 | 0.415 | 0.415 | +0.000 | 3.133 |
| delete | 120 | 165 | 35 | 0.485 | 0.491 | +0.006 | 2.716 |
| restore64 | 184 | 180 | 20 | 0.461 | 0.456 | -0.006 | 2.796 |
| restore-all | 256 | 200 | 0 | 0.415 | 0.415 | +0.000 | 3.133 |

## Concept + blend edit (T3)

Survivors under the blend (key 10, alpha 0.90); `base` is the same
survivors under the full pack.

| stage | units | n | base | exact | delta | bits/span |
|---|---:|---:|---:|---:|---:|---:|
| full | 256 | 200 | 0.665 | 0.665 | +0.000 | 1.808 |
| delete | 120 | 165 | 0.745 | 0.733 | -0.012 | 1.434 |
| restore64 | 184 | 180 | 0.711 | 0.706 | -0.006 | 1.593 |
| restore-all | 256 | 200 | 0.665 | 0.665 | +0.000 | 1.808 |

## Concept draft acceptance

Pointer proposes a concept; the trunk is the target distribution; accept = min(1, p_target(c) / q_draft(c)).

| n | accept | exact |
|---:|---:|---:|
| 200 | 0.717 | 0.650 |

## Speculative block

Draft 4 concepts with the pointer, verify them with one trunk forward; accept until the first rejection.

| blocks | accepted/call | all accepted | accept/step |
|---:|---:|---:|---:|
| 200 | 1.310 | 0.080 | 0.334 |

## Concept + copy draft (T4)

The draft is the concept+copy blend; the trunk verifies each block.
`key 0` is the pointer-only control.

| key | alpha | temp | K | accepted/call | all accepted | accept/step |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00 | 1.0 | 8 | 1.480 | 0.000 | 0.182 |
| 0 | 0.00 | 2.0 | 8 | 2.415 | 0.035 | 0.303 |
| 0 | 0.00 | 4.0 | 8 | 3.795 | 0.160 | 0.476 |
| 0 | 0.00 | 4.0 | 16 | 4.450 | 0.045 | 0.280 |
| 0 | 0.00 | 4.0 | 4 | 2.720 | 0.490 | 0.687 |
| 10 | 0.90 | 1.0 | 8 | 1.480 | 0.000 | 0.182 |
| 10 | 0.90 | 2.0 | 8 | 2.400 | 0.020 | 0.303 |
| 10 | 0.90 | 4.0 | 8 | 3.835 | 0.200 | 0.480 |
| 10 | 0.90 | 4.0 | 16 | 4.650 | 0.040 | 0.289 |

## Read

Deletion is local: removed targets lose their exact concept by
construction; surviving targets stay at their full-pack base (±0.006).
The cost is the literal escape charged by the codec, not collateral
damage. The draft acceptance is the concept-level speculative signal:
one pointer proposal against one trunk target call. `accepted/call`
above 1 means a 4-concept draft can cut target calls per emitted
concept; it is not a wall-clock measurement.

Run took 94.9 s.
