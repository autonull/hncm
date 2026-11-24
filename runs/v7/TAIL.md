# H-NCM v7: the tail coder (M5), bits and calls

The dictionary covers a span exactly; the uncovered spans pay the byte unigram, one byte at a time. The tail coder codes one call per span. The count arms use absolute discounting (Good-Turing, floored) and keep their top 4,096 keys; the `svd@r` arms are the rank-r factorization of the 257x256 byte-context table (shared parameters, 8.2 kB per rank step of 16), and r is chosen on the held-out 10% of the train, never the val. The cost is the span bytes plus the boundary H(L); `svd@space` codes the trailing space instead, but the covered token that follows still pays its index, so that row is the tail side of a space-suffixed tokenizer (a bound). The `ceiling` arm removes the boundary: the rate a boundary-free span code would reach.

| corpus | tail B | spans | B/span | covered bpb | tail bpb |
|---|---:|---:|---:|---:|---:|
| shakespeare | 53746 | 8263 | 6.50 | 0.943 | 5.048 |
| synseg | 493 | 489 | 1.01 | 1.254 | 6.536 |
| syngrid | 0 | 0 | 0.00 | 1.594 | 0.000 |

## shakespeare

| arm | len bpb | byte bpb | tail bpb | rate | charged | model kB | calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| uni | 0.000 | 5.048 | 5.048 | 2.921 | 3.087 | 23.1 | 53747 |
| pos | 0.616 | 4.884 | 5.500 | 3.139 | 3.614 | 66.2 | 8264 |
| ar | 0.616 | 4.480 | 5.096 | 2.944 | 4.201 | 175.3 | 8264 |
| prev | 0.616 | 4.795 | 5.412 | 3.096 | 4.361 | 176.3 | 8264 |
| prev+ar | 0.616 | 4.582 | 5.198 | 2.993 | 4.981 | 277.1 | 8264 |
| concept | 0.616 | 4.974 | 5.591 | 3.182 | 4.800 | 225.6 | 8264 |
| svd@1 | 0.616 | 5.367 | 5.983 | 3.372 | 3.545 | 24.1 | 8264 |
| svd@2 | 0.616 | 4.920 | 5.536 | 3.156 | 3.337 | 25.1 | 8264 |
| svd@4 | 0.616 | 4.917 | 5.533 | 3.155 | 3.350 | 27.2 | 8264 |
| svd@8 | 0.616 | 4.642 | 5.258 | 3.022 | 3.247 | 31.3 | 8264 |
| svd@16 * | 0.616 | 4.020 | 4.636 | 2.723 | 3.006 | 39.5 | 8264 |
| svd@32 | 0.616 | 3.937 | 4.553 | 2.682 | 3.084 | 55.9 | 8264 |
| svd@64 | 0.616 | 3.985 | 4.602 | 2.706 | 3.342 | 88.8 | 8264 |
| svd@space | 0.000 | 4.542 | 4.542 | 2.677 | 2.960 | 39.5 | 8264 |
| ar@256 | 0.616 | 4.724 | 5.340 | 3.062 | 3.503 | 61.6 | 8264 |
| ar@1024 | 0.616 | 4.601 | 5.218 | 3.003 | 3.783 | 108.7 | 8264 |
| ar@4096 | 0.616 | 4.480 | 5.096 | 2.944 | 4.201 | 175.3 | 8264 |
| ar@full | 0.616 | 4.447 | 5.063 | 2.928 | 4.768 | 256.5 | 8264 |
| ceiling | 0.000 | 4.020 | 4.020 | 2.426 | 2.709 | 39.5 | 8264 |

## synseg

| arm | len bpb | byte bpb | tail bpb | rate | charged | model kB | calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| uni | 0.000 | 6.536 | 6.536 | 1.275 | 1.323 | 7.2 | 494 |
| pos | 0.053 | 0.056 | 0.109 | 1.249 | 1.297 | 7.3 | 490 |
| ar | 0.053 | 0.055 | 0.109 | 1.249 | 1.297 | 7.3 | 490 |
| prev | 0.053 | 0.055 | 0.109 | 1.249 | 1.298 | 7.5 | 490 |
| prev+ar | 0.053 | 0.055 | 0.109 | 1.249 | 1.299 | 7.5 | 490 |
| concept | 0.053 | 0.108 | 0.161 | 1.249 | 1.407 | 23.7 | 490 |

## syngrid

| arm | len bpb | byte bpb | tail bpb | rate | charged | model kB | calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| uni | 0.000 | 0.000 | 0.000 | 1.594 | 1.939 | 51.8 | 1 |
| pos | 0.000 | 0.000 | 0.000 | 1.594 | 1.939 | 51.8 | 1 |
| ar | 0.000 | 0.000 | 0.000 | 1.594 | 1.939 | 51.8 | 1 |
| prev | 0.000 | 0.000 | 0.000 | 1.594 | 1.939 | 51.8 | 1 |
| prev+ar | 0.000 | 0.000 | 0.000 | 1.594 | 1.939 | 51.8 | 1 |
| concept | 0.000 | 0.000 | 0.000 | 1.594 | 1.941 | 52.1 | 1 |

## Gates

| gate | corpus | value | status |
|---|---|---|---|
| G4 | shakespeare | uni=5.048 best=svd@16 4.636 gain=+0.412 | **pass** |
| G5 | shakespeare | calls 53747 -> 8264 (x6.5, 6.5 B/span) | **fail** |
| G6 | shakespeare | best=3.006 current=3.087 gain=+0.081 | **pass** |
| G10 | shakespeare | space boundary 4.542 against len boundary 4.636 | **pass** |
| G9 | shakespeare | ceiling rate=2.426 best rate=2.723 uni rate=2.921 | **pass** |
| G4 | synseg | uni=6.536 best=uni 6.536 gain=+0.000 | **fail** |
| G5 | synseg | calls 494 -> 494 (x1.0, 1.0 B/span) | **fail** |
| G6 | synseg | best=1.323 current=1.323 gain=+0.000 | **fail** |
| G10 | synseg | no space boundary | **fail** |
| G9 | synseg | no tail: the tail coder is a no-op | **skip** |
| G4 | syngrid | uni=0.000 best=uni 0.000 gain=+0.000 | **fail** |
| G5 | syngrid | calls 1 -> 1 (x1.0, 0.0 B/span) | **fail** |
| G6 | syngrid | best=1.939 current=1.939 gain=+0.000 | **fail** |
| G10 | syngrid | no space boundary | **fail** |
| G9 | syngrid | no tail: the tail coder is a no-op | **skip** |

