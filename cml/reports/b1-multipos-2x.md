# ByteCopy LM: joint last block + margin gate

- trunk `codec-byte-model-code-756752-d448-l8-q2048`; 2048 B window; 2,000,000 B train, 100,000 B held-out
- self-supervised longest prior match: min 8 B, cap 64 B, depth 8
- head: pointer over the context, gate with top-2 margin, run length conditioned on the chosen source; 407,494 head params; last block + head trained, rest frozen

| gate | coverage | span precision | source precision | copies | bytes/step |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.177 | 0.195 | 0.278 | 1,826 | 0.57x |
| 0.70 | 0.122 | 0.344 | 0.467 | 1,214 | 0.80x |
| 0.90 | 0.052 | 0.675 | 0.782 | 503 | 0.97x |
| 0.95 | 0.034 | 0.795 | 0.887 | 317 | 0.99x |
| 0.99 | 0.013 | 0.965 | 0.984 | 118 | 1.00x |

## Read

- best gate 0.7: copies **12.2%** of bytes at **34.4%** precision, **0.80x** bytes per sequential step.
- a copied span is exact by construction; the trunk only samples novel bytes. This is the model-side version of the exact memory: the pointer replaces a flat span id.

Took 373 s on one GPU.
