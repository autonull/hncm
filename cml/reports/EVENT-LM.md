# Event LM pilot: template events and held-out fills

- code corpus `7ae9de7b4a15`, train 2,000,000 B, held-out 100,000 B
- library 18,925 templates + 9,419 fills
- train 54,876 events; new fill combinations are held out by construction
- trunk d192 x 4 layers x 4 heads, seq 128, batch 32, 500 steps

| arm | fill accuracy | new combinations | bits/byte | params |
|---|---:|---:|---:|---:|
| template-event | 0.130 | 0.048 | 1.767 | 7,243,392 |
| slot-count | 0.120 | 0.001 | nan | 0 |
| type-count | 0.073 | 0.014 | nan | 0 |

## Read

New combinations are fills that occur in a template slot in test but never in that slot in train. The fill value itself appears in other slots, so the arm must reuse content, not memorize the pair.
The new-combination column has 1,080 fills.
- On new combinations the event arm scores 4.8%; type-count scores 1.4%. Fill IDs are expensive; the next lever is long fills (>=8 B) with a local copy pointer, not a larger fill softmax.

Took 24 s on one GPU.
