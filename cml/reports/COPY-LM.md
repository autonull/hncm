# ByteCopy LM: copy head on a frozen byte trunk

- trunk `codec-byte-model-code-756752-d448-l8-q2048`; 2048 B window; 2,000,000 B train, 100,000 B held-out
- self-supervised longest prior match: min 8 B, cap 64 B, depth 8
- head: pointer over the context, gate, run length; 405,701 params, frozen trunk

| gate | coverage | copy precision | copies | bytes/step |
|---:|---:|---:|---:|---:|
| 0.5 | 0.210 | 0.091 | 2,416 | 0.32x |
| 0.7 | 0.152 | 0.181 | 1,719 | 0.59x |
| 0.9 | 0.042 | 0.511 | 472 | 0.96x |

## Read

- best gate 0.7: copies **15.2%** of bytes at **18.1%** precision, **0.59x** bytes per sequential step.
- a copied span is exact by construction; the trunk only samples novel bytes. This is the model-side version of the exact memory: the pointer replaces a flat span id.

Took 50 s on one GPU.
