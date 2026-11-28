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

## Repair log

The first joint run was a control failure, not an interface verdict. Each
row changes one term.

| id | change | span precision @ gate 0.9 | coverage | bytes/step |
|---|---|---:|---:|---:|
| R0 | frozen trunk, copyability gate | 51.1% | 4.2% | 0.95x |
| R1 | joint, copyability gate, lr 1e-4 | no gate fire | 0% | 1.00x |
| R2 | joint, lr 1e-3, copyability gate | 27.7% | 11.0% | 0.77x |
| R3 | separate lrs, pointer-correctness gate | 45.8% | 5.4% | 0.93x |
| R4 | exact-span gate | 62.0% | 5.0% | 0.96x |
| R5 | run head at predicted source | 61.9% | 4.7% | 0.97x |
| R6 | run head conditioned on source | 63.7% | 4.9% | 0.97x |

R1 only shows that lr 1e-3 is too large for the pretrained block; the gate
fires at that rate. R3–R6 raise precision by making the gate and run head
predict the actual copy outcome.

At gate 0.99, R6 reaches **96.3% span precision** at 100% source
precision, but coverage is only **0.85%** and bytes/step is 1.00x. The
target precision exists; it is not yet at ≥5% coverage.

## Gate check

- Target: >90% span precision at ≥5% coverage and >1.2x bytes/step.
- Best precision at ≥5% coverage: about 34% at 11.7% coverage (gate 0.7),
  0.81x.
- 96.3% span precision at 0.85% coverage, 1.00x.

Next repair: pointer coverage. Raise pointer capacity or training data, or
change the pointer loss to accept all exact prior matches, before another
joint run. Do not change the run head again until source precision at 5%
coverage is above 90%.

