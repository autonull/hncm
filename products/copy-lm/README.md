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

## Next experiment (one GPU job)

Train the copy head **jointly with the last block**, add a relative-position
bias and a top-2 pointer margin to the gate, and require a digest before a
copy. The target is >90% copy precision at ≥5% coverage; at mean length 10
that is >1.2x bytes/step, and the copied bytes are exact.

## Run

```bash
uv run cml copy-lm --steps 5000 --train-bytes 2000000 \
  --eval-bytes 100000 --windows 200
```

Report: `cml/reports/COPY-LM.md`.
