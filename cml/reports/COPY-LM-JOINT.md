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

## Repair log

Each row changes one term; the run head interface is fixed after R6.

| id | change | span precision @ gate 0.9 | coverage | bytes/step |
|---|---|---:|---:|---:|
| R0 | frozen trunk, copyability gate | 51.1% | 4.2% | 0.95x |
| R1 | joint, lr 1e-4 | no gate fire | 0% | 1.00x |
| R2 | joint, lr 1e-3, copyability gate | 27.7% | 11.0% | 0.77x |
| R3 | separate lrs, pointer-correctness gate | 45.8% | 5.4% | 0.93x |
| R4 | exact-span gate | 62.0% | 5.0% | 0.96x |
| R5 | run head at predicted source | 61.9% | 4.7% | 0.97x |
| R6 | run head conditioned on source | 63.7% | 4.9% | 0.97x |
| R7 | multi-positive pointer loss | 59.1% | 5.7% | 0.96x |
| R8 | R7 + 2x windows and steps | 67.5% | 5.2% | 0.97x |
| R9 | R8 + 2x pointer width | 66.1% | 5.1% | 0.97x |

R7 accepts every exact prior source in the pointer loss. R8 trains 3,000
steps on 400 windows. R9 widens the pointer query/key from 448 to 896.

## Read

- R9 reaches **98.0% span precision at 98.1% source precision**, but at
  **1.2% coverage** and 1.00x.
- At gate 0.9 the source is 77.6% and the span is 66.1% (85% run-length
  accuracy given a correct source). The gate cannot exceed source
  precision, so 90% at 5% coverage needs a better pointer, not a gate.
- Three independent repairs move the operating point only slightly:
  **66-67% span precision at ~5% coverage**, 0.97x. The pass gate
  (>90% at ≥5%, >1.2x) is missed.
- Stop the B1 repair loop. The next test is B2: put a retrieved exact pack
  in context and point into it. If B2 also misses, record the copy-LM
  interface as a stop in `TODO9.md`.

Took 370 s on one GPU for R9; 241 s for R7 and 373 s for R8.

Logs: `b1-multipos.log`, `b1-multipos-2x.log`, `b1-wide.log`.
