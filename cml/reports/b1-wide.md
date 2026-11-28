# ByteCopy LM: joint last block + margin gate

- trunk `codec-byte-model-code-756752-d448-l8-q2048`; 2048 B window; 2,000,000 B train, 100,000 B held-out
- self-supervised longest prior match: min 8 B, cap 64 B, depth 8
- head: 2.0x pointer over the context, gate with top-2 margin, run length conditioned on the chosen source; 808,902 head params; last block + head trained, rest frozen

| gate | coverage | span precision | source precision | copies | bytes/step |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.178 | 0.197 | 0.280 | 1,822 | 0.57x |
| 0.70 | 0.123 | 0.338 | 0.466 | 1,211 | 0.80x |
| 0.90 | 0.051 | 0.661 | 0.776 | 489 | 0.97x |
| 0.95 | 0.033 | 0.818 | 0.900 | 307 | 0.99x |
| 0.99 | 0.012 | 0.980 | 0.981 | 105 | 1.00x |

## Read

- best gate 0.7: copies **12.3%** of bytes at **33.8%** precision, **0.80x** bytes per sequential step.
- a copied span is exact by construction; the trunk only samples novel bytes. This is the model-side version of the exact memory: the pointer replaces a flat span id.

Took 370 s on one GPU.
