# H-NCM v7: the tail coder (M5), bits and calls

The dictionary covers a span exactly; the uncovered spans pay the byte unigram, one byte at a time. The tail coder codes one call per span. The models are counts with absolute discounting (Good-Turing, floored), keys seen once back off, and each table keeps its top 4,096 keys. The cost is the span bytes plus the boundary H(L); the concept arm also transmits H(concept) and its centroids. The `ceiling` arm removes the boundary: it is the rate a boundary-free span code would reach.

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
| ar@256 | 0.616 | 4.724 | 5.340 | 3.062 | 3.503 | 61.6 | 8264 |
| ar@1024 | 0.616 | 4.601 | 5.218 | 3.003 | 3.783 | 108.7 | 8264 |
| ar@4096 | 0.616 | 4.480 | 5.096 | 2.944 | 4.201 | 175.3 | 8264 |
| ar@full | 0.616 | 4.447 | 5.063 | 2.928 | 4.768 | 256.5 | 8264 |
| ceiling | 0.000 | 4.718 | 4.718 | 2.762 | 3.297 | 74.5 | 8264 |

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
| G4 | shakespeare | uni=5.048 best=ar@256 5.340 gain=-0.292 | **fail** |
| G5 | shakespeare | calls 53747 -> 8264 (x6.5, 6.5 B/span) | **fail** |
| G6 | shakespeare | best=3.503 current=3.087 gain=-0.417 | **fail** |
| G9 | shakespeare | ceiling rate=2.762 best rate=3.062 uni rate=2.921 | **pass** |
| G4 | synseg | uni=6.536 best=pos 0.109 gain=+6.427 | **pass** |
| G5 | synseg | calls 494 -> 490 (x1.0, 1.0 B/span) | **fail** |
| G6 | synseg | best=1.297 current=1.323 gain=+0.026 | **pass** |
| G9 | synseg | no tail: the tail coder is a no-op | **skip** |
| G4 | syngrid | uni=0.000 best=pos 0.000 gain=+0.000 | **fail** |
| G5 | syngrid | calls 1 -> 1 (x1.0, 0.0 B/span) | **fail** |
| G6 | syngrid | best=1.939 current=1.939 gain=-0.000 | **fail** |
| G9 | syngrid | no tail: the tail coder is a no-op | **skip** |

