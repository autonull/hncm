# ByteCopy LM: joint last block + margin gate

- trunk `codec-byte-model-code-756752-d448-l8-q2048`; 2048 B window; 2,000,000 B train, 100,000 B held-out
- self-supervised longest prior match: min 8 B, cap 64 B, depth 8
- head: pointer over the context, gate with top-2 margin, run length; 407,494 head params; last block + head trained, rest frozen

| gate | coverage | span precision | source precision | copies | bytes/step |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.169 | 0.203 | 0.272 | 1,794 | 0.60x |
| 0.70 | 0.117 | 0.338 | 0.435 | 1,235 | 0.81x |
| 0.90 | 0.049 | 0.637 | 0.757 | 513 | 0.97x |
| 0.95 | 0.029 | 0.722 | 0.904 | 296 | 0.99x |
| 0.99 | 0.009 | 0.963 | 1.000 | 78 | 1.00x |

## Read

- best gate 0.7: copies **11.7%** of bytes at **33.8%** precision, **0.81x** bytes per sequential step.
- a copied span is exact by construction; the trunk only samples novel bytes. This is the model-side version of the exact memory: the pointer replaces a flat span id.

Took 267 s on one GPU.
