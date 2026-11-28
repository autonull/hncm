# `copy-lm` — a byte LM that copies exact spans

Research prototype of a model-side invention: a byte-level trunk predicts
novel bytes; a **copy head** points at an earlier context position and a
**run head** names the span length. The copied span is exact; the trunk
only computes what is new. No tokenizer, no flat span id, no pack charge.

```
byte h_t ──> gate ──> copy? ──> pointer (position in context)
   │                     │              │
   └──> byte head        └──> run head ──┘
```

## Why it can punch above its weight

- Code repeats long exact spans (identifiers, boilerplate, imports, diffs).
- A 20M byte model spends capacity memorizing those spans; the copy path
  delegates them to a pointer and keeps capacity for novel bytes.
- The pointer is a position the transformer already represents, so there
  is no id table, no count prior, and no per-copy id entropy.
- A copied span is byte-exact by construction; the digest is the guard.

Training is self-supervised: at each position the label is the longest
prior match in the context (min 8 B, cap 64 B). No edit labels, no
teacher model, no task-specific data.

## Prototype

Frozen `d448-l8`, `seq=2048` code trunk; 403k-parameter copy head trained
for 5,000 steps on 2 MB of code; evaluated on 100 kB held-out code.

| gate | coverage | copy precision | exact copies | bytes/step |
|---:|---:|---:|---:|---:|
| 0.5 | 21.0% | 9.1% | 2,416 | 0.32x |
| 0.7 | 15.2% | 18.1% | 1,719 | 0.59x |
| 0.9 | 4.2% | 51.1% | 472 | 0.96x |

The exact copy path works; the **gate is not calibrated on frozen last
states**. At gate 0.9 it reaches 0.96x bytes/step and every accepted copy
is exact.

## Joint repair loop

Frozen `d448-l8`, `seq=2048` code trunk. Each run changes one term on
2 MB of code and 100 kB held-out code.

| id | change | span precision @ 0.9 | coverage | bytes/step |
|---|---|---:|---:|---:|
| R0 | frozen trunk, copyability gate | 51.1% | 4.2% | 0.95x |
| R1 | joint, lr 1e-4 | no gate fire | 0% | 1.00x |
| R2 | joint, lr 1e-3, copyability gate | 27.7% | 11.0% | 0.77x |
| R3 | separate lrs, pointer-correctness gate | 45.8% | 5.4% | 0.93x |
| R4 | exact-span gate | 62.0% | 5.0% | 0.96x |
| R5 | run head at predicted source | 61.9% | 4.7% | 0.97x |
| R6 | run head conditioned on source | 63.7% | 4.9% | 0.97x |

R6 reaches **96.3% span precision at 100% source precision** at gate
0.99, but coverage is only **0.85%** and bytes/step is 1.00x. The
precision target exists; it is not at ≥5% coverage. The next repair is
pointer coverage, not the run head.

Report: `cml/reports/COPY-LM-JOINT.md`; repair logs in
`cml/reports/b1-*.log`.

## Run

```bash
uv run cml copy-lm --steps 5000 --train-bytes 2000000 \
  --eval-bytes 100000 --windows 200
uv run cml copy-lm --joint --steps 2000 --lr 0.001 --block-lr 1e-5 \
  --train-bytes 2000000 --eval-bytes 100000 --windows 200
```

Report: `cml/reports/COPY-LM.md`; joint repair:
`cml/reports/COPY-LM-JOINT.md`.
