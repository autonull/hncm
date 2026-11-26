# Capability screen: held-out span reconstruction

- corpus `code` sha `1ccd9ffc7f68`, held-out 70,000 B, 200 matched spans
- pack 256 units / 2,771 B, 500 fallback steps, seed 0
- exact = the predicted span equals the removed bytes; top-1 = the first symbol is right; decisions count emitted symbols.

| mechanism | family | exact span | top-1 | bits/span | decisions/span |
|---|---|---:|---:|---:|---:|
| trunk | r3/r4 control | 0.480 | 0.480 | 2.590 | 1.00 |
| pointer | r3/r4/r6 | 0.415 | 0.415 | 3.133 | 1.00 |
| byte | r1 baseline | 0.210 | 0.610 | 12.711 | 3.04 |
| library | r7 order-2 | 0.150 | 0.150 | 2.465 | 1.00 |
| counts | r1/r6 order-1 | 0.040 | 0.040 | 2.815 | 1.00 |

## Edit capability (r5)

Remove every unit starting with ` `; re-encode the same 70,000 B with the surviving pack and read it back.

| units before | units after | affected B | payload before | payload after | delta | round-trip |
|---:|---:|---:|---:|---:|---:|:---:|
| 256 | 120 | 4384 | 1.945 | 2.091 | +0.146 | OK |

## Most promising

`trunk`: exact 0.480, top-1 0.480, 2.59 bits/span, 1.00 decisions/span.

This is a capability comparison, not a gate. It measures next-span
prediction on held-out code plus one gradient-free pack removal.

Screen took 262.6 s.
