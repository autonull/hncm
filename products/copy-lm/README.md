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
| R7 | multi-positive pointer loss | 59.1% | 5.7% | 0.96x |
| R8 | R7 + 2x windows and steps | 67.5% | 5.2% | 0.97x |
| R9 | R8 + 2x pointer width | 66.1% | 5.1% | 0.97x |

These rows price every top-1 proposal as an emitted copy. R9 reaches
98.0% span precision at 98.1% source precision at gate 0.99, but only
1.2% coverage. The unverified pointer cannot reach 90% at 5%.

## Verified decode (R10)

The design verifies before it emits. The gate fires, the pointer names
the top-k prior sources, an exact match check accepts the longest span,
and a rejected proposal falls back to the byte head. A rejected proposal
costs a byte comparison, not a forward.

| gate | coverage | mean span | checks/B | bytes/step |
|---:|---:|---:|---:|---:|
| 0.10 | 38.6% | 14.6 | 3.41 | 1.56x |
| 0.50 | 26.6% | 16.1 | 0.46 | 1.33x |
| 0.70 | 19.3% | 17.4 | 0.19 | 1.22x |
| 0.90 | 9.0% | 18.0 | 0.05 | 1.09x |

**Gate PASS.** 38.6% coverage at **100% verified precision**, 1.56x.
The cheapest passing point is gate 0.7: 19.3% at 1.22x. Every copied
byte passes an exact check; a mismatch raises. The gate is a cost
control, not a correctness predictor.

## Ceilings: exact proposers

| proposer | coverage | checks/B | bytes/step |
|---|---:|---:|---:|
| hash over the 2 KB window | 47.0% | 0.15 | 1.78x |
| hash over a fixed 2 MB train pack | **78.4%** | 0.58 | **3.59x** |
| learned top-8 pointer | 38.6% | 3.41 | 1.56x |

Any exact copy of at least 8 B shares an 8-byte key, so a hash over that
key is a complete in-window proposer. The learned pointer is dominated.

A 2 MB pack of disjoint train code lifts coverage to **78.4% at 3.59x**.
B2 resolves to exact pack memory plus a hash, which `repo` already
supplies. The byte LM keeps the novel 21.6%. The 78.4% figure needs the
pack; it is not a standalone compression claim.

Report: `cml/reports/COPY-LM-JOINT.md`; logs in `cml/reports/b1-*.log`.

## Run

```bash
uv run cml copy-lm --steps 5000 --train-bytes 2000000 \
  --eval-bytes 100000 --windows 200
uv run cml copy-lm --joint --steps 2000 --lr 0.001 --block-lr 1e-5 \
  --train-bytes 2000000 --eval-bytes 100000 --windows 200
uv run cml copy-lm --joint --steps 3000 --lr 0.001 --block-lr 1e-5 \
  --pointer-scale 2 --verify-topk 8 --train-bytes 2000000 \
  --eval-bytes 100000 --windows 400
```

Report: `cml/reports/COPY-LM.md`; joint repair:
`cml/reports/COPY-LM-JOINT.md`.
